r"""Analyzes the overnight DeepSeek-V4.1 E065 "rescue" job (iterative correction + clean prefix-free rewrite) for epoch-2 SFT planning.

Reads the CNS outputs of `deepseek_v41_e065_correct_and_rewrite.py`
(`/cns/lu-d/home/arazdai/kaggle/arc_sft_data/e065_hardest_rollouts/corrected_rewritten/shards/`) and reports:

  1. Rescue job progress & solve rates (`summary_shard_*.json` + `correction_diagnostics_shard_*.jsonl`):
     overall, by correction attempt, by original failure category, by source, by rewrite source.
  2. Anatomy of the rescued (rewritten) trajectories (`e065_text_corrected_rewritten_shard_*.parquet`):
     assistant turns, tool calls, reasoning / code characters, estimated tokens, meta-phrase leak check,
     and an INDEPENDENT sandbox re-verification of the rewritten code on the ground-truth `train` + ALL `test` pairs.
  3. Original failed rollout vs rescued rollout side-by-side (turns / completion tokens / failure category).
  4. Printed sample trajectories (`--show_examples`).
  5. Leakage / overlap checks against the epoch-1 dataset manifests (`e065_img_text_v1/{train,val}_manifest.jsonl`):
     rescued task ids vs epoch-1 train tasks, rescued parents vs epoch-1 VAL parents.
  6. Optional Top-10K v8 verified-rollout inventory (column-projected scan of `verified_rollouts_shard_*.parquet`).

Writes `<out_dir>/rescue_analysis.json` and `<out_dir>/rescue_analysis.md` (also printed to stdout).

Example:
  blaze run //experimental/users/arazdai/kaggle/sft:analyze_e065_rescued_rollouts -- \
    --out_dir=/tmp/rescue_analysis --show_examples=3 --reverify_max=400
"""

from __future__ import annotations

import collections
import io
import json
import multiprocessing
import os
import re
import statistics
import sys
import time
from typing import Any

from absl import app
from absl import flags
import numpy as np
import pyarrow.parquet as pq

from google3.pyglib import gfile

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "rescue_shards_dir",
    "/cns/lu-d/home/arazdai/kaggle/arc_sft_data/e065_hardest_rollouts/corrected_rewritten/shards",
    "CNS directory with summary_*.json, correction_diagnostics_*.jsonl and e065_{images,text}_corrected_rewritten_*.parquet.",
)
flags.DEFINE_string(
    "non_test_passing_tasks_jsonl",
    "/cns/lu-d/home/arazdai/kaggle/arc_sft_data/e065_hardest_rollouts/non_test_passing/e065_images_non_test_passing_tasks.jsonl",
    "Ground-truth tasks + failure diagnostics of the original failed E065 rollouts.",
)
flags.DEFINE_string(
    "epoch1_data_dir",
    "/cns/lu-d/home/arazdai/kaggle/sft/data/e065_img_text_v1",
    "Epoch-1 tokenized dataset dir (train_manifest.jsonl / val_manifest.jsonl) for overlap & leakage checks.",
)
flags.DEFINE_string(
    "top10k_rollouts_dir",
    "/cns/lu-d/home/arazdai/kaggle/arc_sft_data/sft_prep_pipeline/deepseek_v41_rollouts_top10k_v8_pass1",
    "Top-10K v8 rollout dir (verified_rollouts_shard_*.parquet). Empty = skip the inventory.",
)
flags.DEFINE_integer("top10k_max_shards", 64, "Cap on Top-10K verified shards to scan (0 = skip).")
flags.DEFINE_string("out_dir", "/tmp/rescue_analysis", "Local directory for rescue_analysis.{json,md}.")
flags.DEFINE_integer("show_examples", 3, "Number of rescued trajectories to print per rewrite source.")
flags.DEFINE_integer("example_chars", 1800, "Max characters printed per reasoning / code block.")
flags.DEFINE_integer("reverify_max", 400, "Max rescued rollouts to independently re-verify in a sandbox (0 = skip).")
flags.DEFINE_float("reverify_timeout_s", 20.0, "Per-rollout sandbox timeout for re-verification.")

FORBIDDEN_META_PHRASES = (
    "previous attempt",
    "prior attempt",
    "earlier attempt",
    "failed attempt",
    "previous rollout",
    "reference solution",
    "provided solution",
    "given solution",
    "teacher hint",
    "rewrite",
    "verified implementation",
    "verified python",
    "verified solution",
    "solution trajectory",
    "instructions from user",
    "instructions say",
    "user instructions",
    "as if solving",
    "already discovered",
    "key reasoning insight",
    "do not mention",
    "internal",
    "the given code",
    "provided code",
    "given implementation",
)

_SHARD_RE = re.compile(r"shard_(\d+)of(\d+)")


# ----------------------------------------------------------------------------------------------
# IO helpers
# ----------------------------------------------------------------------------------------------
def _read_json(path: str) -> Any:
    with gfile.Open(path, "r") as f:
        return json.load(f)


def _read_jsonl(path: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with gfile.Open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return rows


def _read_parquet_rows(path: str, columns: list[str] | None = None) -> list[dict[str, Any]]:
    with gfile.Open(path, "rb") as f:
        data = f.read()
    table = pq.read_table(io.BytesIO(data), columns=columns)
    return table.to_pylist()


def _parquet_schema_names(path: str) -> list[str]:
    with gfile.Open(path, "rb") as f:
        data = f.read()
    return list(pq.ParquetFile(io.BytesIO(data)).schema_arrow.names)


def _pct(num: float, den: float) -> float:
    return round(100.0 * num / den, 2) if den else 0.0


def _stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"n": 0}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "n": int(arr.size),
        "mean": round(float(arr.mean()), 1),
        "p50": round(float(np.percentile(arr, 50)), 1),
        "p90": round(float(np.percentile(arr, 90)), 1),
        "max": round(float(arr.max()), 1),
        "min": round(float(arr.min()), 1),
    }


# ----------------------------------------------------------------------------------------------
# Trajectory parsing
# ----------------------------------------------------------------------------------------------
def _tool_call_code(tc: dict[str, Any]) -> tuple[str, str]:
    fn = tc.get("function") or {}
    name = str(fn.get("name", ""))
    args = fn.get("arguments", "")
    code = ""
    if isinstance(args, dict):
        code = str(args.get("code", ""))
    elif isinstance(args, str):
        try:
            parsed = json.loads(args)
            code = str(parsed.get("code", "")) if isinstance(parsed, dict) else args
        except json.JSONDecodeError:
            code = args
    return name, code


def parse_trajectory(messages: list[dict[str, Any]]) -> dict[str, Any]:
    """Extracts per-turn structure from an E065-schema message list (system, user, assistant/tool...)."""
    turns: list[dict[str, Any]] = []
    tool_outputs: list[str] = []
    for m in messages[2:]:
        role = m.get("role")
        if role == "assistant":
            reasoning = str(m.get("reasoning_content") or "")
            content = str(m.get("content") or "")
            calls = [_tool_call_code(tc) for tc in (m.get("tool_calls") or [])]
            turns.append({"reasoning": reasoning, "content": content, "calls": calls})
        elif role == "tool":
            tool_outputs.append(str(m.get("content") or ""))
    run_codes = [c for t in turns for (n, c) in t["calls"] if n in ("run_python", "python_exec") and c.strip()]
    submit_codes = [c for t in turns for (n, c) in t["calls"] if n == "submit_answer" and c.strip()]
    reasoning_chars = sum(len(t["reasoning"]) + len(t["content"]) for t in turns)
    code_chars = sum(len(c) for c in run_codes) + sum(len(c) for c in submit_codes)
    low = "\n".join(t["reasoning"] + "\n" + t["content"] for t in turns).lower()
    leaks = sorted({p for p in FORBIDDEN_META_PHRASES if p in low})
    # "internal" is a common ARC word (internal cells/borders); keep it out of the strong-leak count.
    strong_leaks = [p for p in leaks if p != "internal"]
    prior_ref = bool(
        re.search(r"\b(prev|previous(ly)? (failed|rule|code|attempt|approach)|old rule|diagnos\w*|why (the )?previous|earlier rule)\b", low)
    )
    return {
        "n_assistant_turns": len(turns),
        "n_tool_calls": sum(len(t["calls"]) for t in turns),
        "n_run_python": len(run_codes),
        "n_submit": len(submit_codes),
        "reasoning_chars": reasoning_chars,
        "code_chars": code_chars,
        "has_def_solve": any("def solve" in c for c in run_codes + submit_codes),
        "meta_leaks": leaks,
        "strong_leaks": strong_leaks,
        "prior_attempt_ref": prior_ref,
        "turns": turns,
        "tool_outputs": tool_outputs,
        "run_codes": run_codes,
        "submit_code": submit_codes[-1] if submit_codes else "",
        "tool_error": any(
            ("Traceback" in o or "Error" in o.split("\n")[-1][:80]) for o in tool_outputs[:-1]
        ),
    }


# ----------------------------------------------------------------------------------------------
# Independent sandbox re-verification (forked child, hard timeout)
# ----------------------------------------------------------------------------------------------
def _sandbox_worker(run_codes: list[str], submit_code: str, train: list[dict[str, Any]], tests: list[dict[str, Any]], conn: Any) -> None:
    import contextlib  # pylint: disable=g-import-not-at-top

    result: dict[str, Any] = {"train_pass": False, "test_pass_all": False, "n_test_ok": 0, "error": "", "cell_errors": 0}
    try:
        import numpy as _np  # pylint: disable=g-import-not-at-top

        ns: dict[str, Any] = {"np": _np, "numpy": _np, "train": train, "test_input": tests[0]["input"]}
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            for code in run_codes:
                # Mirror the persistent IPython session: a failing cell does not end the session.
                try:
                    exec(code, ns)  # pylint: disable=exec-used
                except Exception as cell_e:  # pylint: disable=broad-exception-caught
                    result["cell_errors"] += 1
                    result["error"] = f"cell:{type(cell_e).__name__}: {str(cell_e)[:120]}"

            def _to_grid(g: Any) -> Any:
                if hasattr(g, "tolist"):
                    g = g.tolist()
                return g

            def _predict(ti: list[list[int]]) -> Any:
                local_ns = dict(ns)
                local_ns["test_input"] = ti
                sub = (submit_code or "").strip()
                g = None
                if sub:
                    try:
                        g = eval(sub, local_ns)  # pylint: disable=eval-used
                    except SyntaxError:
                        exec(sub, local_ns)  # pylint: disable=exec-used
                        g = local_ns.get("answer", local_ns.get("output", local_ns.get("result")))
                    except NameError:
                        g = None
                if g is None and callable(local_ns.get("solve")):
                    g = local_ns["solve"](ti)
                return _to_grid(g)

            solve_fn = ns.get("solve")
            if callable(solve_fn):
                result["train_pass"] = all(_to_grid(solve_fn(p["input"])) == p["output"] for p in train)
            n_ok = 0
            for tp in tests:
                try:
                    if _predict(tp["input"]) == tp["output"]:
                        n_ok += 1
                except Exception:  # pylint: disable=broad-exception-caught
                    pass
            result["n_test_ok"] = n_ok
            result["test_pass_all"] = n_ok == len(tests) and bool(tests)
            if not callable(solve_fn):
                # No `solve` defined: fall back to "submit expression reproduces all test pairs" only.
                result["train_pass"] = result["test_pass_all"]
    except Exception as e:  # pylint: disable=broad-exception-caught
        result["error"] = f"{type(e).__name__}: {str(e)[:160]}"
    try:
        conn.send(result)
    except Exception:  # pylint: disable=broad-exception-caught
        pass
    conn.close()


def sandbox_reverify(
    run_codes: list[str],
    submit_code: str,
    train: list[dict[str, Any]],
    tests: list[dict[str, Any]],
    timeout_s: float,
) -> dict[str, Any]:
    ctx = multiprocessing.get_context("fork")
    parent_conn, child_conn = ctx.Pipe(duplex=False)
    proc = ctx.Process(target=_sandbox_worker, args=(run_codes, submit_code, train, tests, child_conn))
    proc.start()
    child_conn.close()
    result: dict[str, Any] = {"train_pass": False, "test_pass_all": False, "n_test_ok": 0, "error": "timeout"}
    if parent_conn.poll(timeout_s):
        try:
            result = parent_conn.recv()
        except EOFError:
            result["error"] = "eof"
    proc.join(0.5)
    if proc.is_alive():
        proc.kill()
        proc.join(1.0)
    return result


# ----------------------------------------------------------------------------------------------
# Section 1: rescue job progress
# ----------------------------------------------------------------------------------------------
def analyze_rescue_progress(shards_dir: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    names = sorted(gfile.ListDir(shards_dir))
    summaries = [n for n in names if n.startswith("summary_") and n.endswith(".json")]
    diags = [n for n in names if n.startswith("correction_diagnostics_") and n.endswith(".jsonl")]
    n_shards_total = 0
    for n in summaries + diags:
        m = _SHARD_RE.search(n)
        if m:
            n_shards_total = max(n_shards_total, int(m.group(2)))

    agg = collections.Counter()
    by_attempt: collections.Counter[str] = collections.Counter()
    wall: list[float] = []
    toks = 0
    tok_rate: list[float] = []
    for n in summaries:
        s = _read_json(os.path.join(shards_dir, n))
        agg["total_shard_tasks"] += int(s.get("total_shard_tasks", 0))
        agg["completed_tasks"] += int(s.get("completed_tasks", 0))
        agg["solved_and_rewritten_tasks"] += int(s.get("solved_and_rewritten_tasks", 0))
        agg["rewritten_image_parquet_rows"] += int(s.get("rewritten_image_parquet_rows", 0))
        agg["rewritten_text_parquet_rows"] += int(s.get("rewritten_text_parquet_rows", 0))
        for k, v in (s.get("solved_by_attempt") or {}).items():
            by_attempt[k] += int(v)
        wall.append(float(s.get("elapsed_wall_seconds", 0.0)))
        toks += int(s.get("decode_tokens_generated", 0))
        if s.get("avg_decode_tok_per_s"):
            tok_rate.append(float(s["avg_decode_tok_per_s"]))

    diag_rows: list[dict[str, Any]] = []
    for n in diags:
        diag_rows.extend(_read_jsonl(os.path.join(shards_dir, n)))

    by_cat: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    by_src: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    by_rw: collections.Counter[str] = collections.Counter()
    rw_turns: list[int] = []
    unsolved_last_diag: collections.Counter[str] = collections.Counter()
    unsolved_attempts: collections.Counter[int] = collections.Counter()
    for d in diag_rows:
        cat = str(d.get("orig_failure_category") or "unknown")
        src = str(d.get("source") or "unknown")
        solved = bool(d.get("solved"))
        by_cat[cat]["n"] += 1
        by_src[src]["n"] += 1
        if solved:
            by_cat[cat]["solved"] += 1
            by_src[src]["solved"] += 1
            by_rw[str(d.get("rewrite_source"))] += 1
            rw_turns.append(int(d.get("rewritten_turns") or 0))
        else:
            unsolved_attempts[int(d.get("prior_failed_attempts_count") or 0)] += 1
            for fa in d.get("failed_attempts") or []:
                unsolved_last_diag[str(fa.get("test_diag") or fa.get("stop_reason") or "?")[:48]] += 1

    out = {
        "shards_with_summary": len(summaries),
        "shards_with_diagnostics": len(diags),
        "num_shards_total": n_shards_total,
        "tasks_in_launched_shards": agg["total_shard_tasks"],
        "completed_tasks": agg["completed_tasks"],
        "solved_and_rewritten_tasks": agg["solved_and_rewritten_tasks"],
        "solve_rate_pct": _pct(agg["solved_and_rewritten_tasks"], agg["completed_tasks"]),
        "completion_pct_of_launched": _pct(agg["completed_tasks"], agg["total_shard_tasks"]),
        "solved_by_attempt": dict(sorted(by_attempt.items())),
        "rewritten_image_parquet_rows": agg["rewritten_image_parquet_rows"],
        "rewritten_text_parquet_rows": agg["rewritten_text_parquet_rows"],
        "decode_tokens_generated": toks,
        "avg_decode_tok_per_s_per_slice": round(statistics.mean(tok_rate), 1) if tok_rate else None,
        "wall_seconds_max": max(wall) if wall else 0.0,
        "wall_hours_max": round(max(wall) / 3600.0, 2) if wall else 0.0,
        "diag_rows": len(diag_rows),
        "by_failure_category": {
            k: {"n": v["n"], "solved": v["solved"], "solve_pct": _pct(v["solved"], v["n"])} for k, v in by_cat.items()
        },
        "by_source": {k: {"n": v["n"], "solved": v["solved"], "solve_pct": _pct(v["solved"], v["n"])} for k, v in by_src.items()},
        "rewrite_source": dict(by_rw),
        "rewritten_turns": _stats(rw_turns),
        "rewritten_turns_hist": dict(sorted(collections.Counter(rw_turns).items())),
        "unsolved_prior_attempt_counts": dict(sorted(unsolved_attempts.items())),
        "unsolved_attempt_diag_top": dict(unsolved_last_diag.most_common(10)),
    }
    return out, diag_rows


# ----------------------------------------------------------------------------------------------
# Section 2/3/4: rescued trajectory anatomy, original-vs-rescued, examples, re-verification
# ----------------------------------------------------------------------------------------------
def load_rescued_text_rows(shards_dir: str) -> list[dict[str, Any]]:
    names = sorted(
        n for n in gfile.ListDir(shards_dir) if n.startswith("e065_text_corrected_rewritten_") and n.endswith(".parquet")
    )
    rows: list[dict[str, Any]] = []
    for n in names:
        try:
            rows.extend(_read_parquet_rows(os.path.join(shards_dir, n)))
        except Exception as e:  # pylint: disable=broad-exception-caught
            print(f"  WARNING: could not read {n}: {e}", flush=True)
    return rows


def load_failure_records(path: str) -> dict[str, dict[str, Any]]:
    recs: dict[str, dict[str, Any]] = {}
    for r in _read_jsonl(path):
        recs[str(r.get("task_id"))] = r
    return recs


def _fmt_block(title: str, text: str, max_chars: int) -> str:
    text = text.strip()
    tail = "" if len(text) <= max_chars else f"\n... [{len(text) - max_chars} more chars]"
    return f"--- {title} ---\n{text[:max_chars]}{tail}\n"


def render_example(row: dict[str, Any], parsed: dict[str, Any], failure: dict[str, Any] | None, max_chars: int) -> str:
    lines = [
        f"### task `{row.get('task_id')}` (source={row.get('source')}, pair={row.get('pair')}, "
        f"orig_failure={failure.get('failure_category') if failure else '?'}, "
        f"orig_turns={failure.get('turns') if failure else '?'}, orig_completion_tokens={failure.get('completion_tokens') if failure else '?'})",
        f"rescued: assistant_turns={parsed['n_assistant_turns']}, tool_calls={parsed['n_tool_calls']}, "
        f"reasoning_chars={parsed['reasoning_chars']}, code_chars={parsed['code_chars']}, meta_leaks={parsed['meta_leaks']}",
        "",
    ]
    if failure and failure.get("last_python_code"):
        lines.append(_fmt_block("ORIGINAL (failed) code", str(failure["last_python_code"]), max_chars // 2))
    for i, t in enumerate(parsed["turns"], start=1):
        if t["reasoning"] or t["content"]:
            lines.append(_fmt_block(f"rescued turn {i} <think>", t["reasoning"] or t["content"], max_chars))
        for name, code in t["calls"]:
            lines.append(_fmt_block(f"rescued turn {i} tool_call `{name}`", code, max_chars))
        if i - 1 < len(parsed["tool_outputs"]):
            lines.append(_fmt_block(f"rescued turn {i} tool output", parsed["tool_outputs"][i - 1], max_chars // 3))
    return "\n".join(lines)


def analyze_rescued_rows(
    rows: list[dict[str, Any]],
    failures: dict[str, dict[str, Any]],
    show_examples: int,
    example_chars: int,
    reverify_max: int,
    reverify_timeout_s: float,
    rewrite_source_by_task: dict[str, str] | None = None,
) -> tuple[dict[str, Any], str]:
    # One rollout per task: keep pair==0 rows (multi-test tasks repeat the rollout per pair).
    pair0 = [r for r in rows if int(r.get("pair") or 0) == 0]
    by_task: dict[str, dict[str, Any]] = {}
    for r in pair0:
        by_task.setdefault(str(r["task_id"]), r)

    parsed_by_task: dict[str, dict[str, Any]] = {}
    for tid, r in by_task.items():
        msgs = json.loads(r["messages"]) if isinstance(r.get("messages"), str) else list(r.get("messages") or [])
        parsed_by_task[tid] = parse_trajectory(msgs)

    n = len(parsed_by_task)
    turns = [p["n_assistant_turns"] for p in parsed_by_task.values()]
    calls = [p["n_tool_calls"] for p in parsed_by_task.values()]
    reas = [p["reasoning_chars"] for p in parsed_by_task.values()]
    code = [p["code_chars"] for p in parsed_by_task.values()]
    est_tokens = [(p["reasoning_chars"] + p["code_chars"]) / 3.6 for p in parsed_by_task.values()]
    comp_tokens = [int(by_task[t].get("completion_tokens") or 0) for t in parsed_by_task]
    leaks = [t for t, p in parsed_by_task.items() if p["meta_leaks"]]
    leak_phrases = collections.Counter(ph for p in parsed_by_task.values() for ph in p["meta_leaks"])
    no_solve = [t for t, p in parsed_by_task.items() if not p["has_def_solve"]]
    tool_err = [t for t, p in parsed_by_task.items() if p["tool_error"]]

    # Original vs rescued.
    orig_turns, orig_tokens, orig_text_twin_pass = [], [], 0
    orig_cat: collections.Counter[str] = collections.Counter()
    for t in parsed_by_task:
        f = failures.get(t)
        if not f:
            continue
        orig_turns.append(int(f.get("turns") or 0))
        orig_tokens.append(int(f.get("completion_tokens") or 0))
        orig_cat[str(f.get("failure_category"))] += 1
        if f.get("text_twin_test_passing"):
            orig_text_twin_pass += 1

    # Independent re-verification.
    reverify: dict[str, Any] = {"n": 0}
    if reverify_max > 0:
        t0 = time.time()
        ok_train = ok_test = ok_both = n_err = n_timeout = n_with_cell_errors = 0
        ok_both_clean = 0
        checked = 0
        fail_examples: list[dict[str, Any]] = []
        for t, p in parsed_by_task.items():
            f = failures.get(t)
            if not f or not f.get("train") or not f.get("test"):
                continue
            if checked >= reverify_max:
                break
            res = sandbox_reverify(p["run_codes"], p["submit_code"], f["train"], f["test"], reverify_timeout_s)
            checked += 1
            has_cell_err = int(res.get("cell_errors") or 0) > 0
            n_with_cell_errors += int(has_cell_err)
            if res.get("error") == "timeout":
                n_timeout += 1
            elif res.get("error") and not str(res.get("error")).startswith("cell:"):
                n_err += 1
            ok_train += int(bool(res.get("train_pass")))
            ok_test += int(bool(res.get("test_pass_all")))
            if res.get("train_pass") and res.get("test_pass_all"):
                ok_both += 1
                ok_both_clean += int(not has_cell_err)
            elif len(fail_examples) < 8:
                fail_examples.append({"task_id": t, **{k: v for k, v in res.items()}})
        reverify = {
            "n": checked,
            "train_pass": ok_train,
            "train_pass_pct": _pct(ok_train, checked),
            "test_pass_all": ok_test,
            "test_pass_all_pct": _pct(ok_test, checked),
            "train_and_test_pass": ok_both,
            "train_and_test_pass_pct": _pct(ok_both, checked),
            "train_and_test_pass_without_any_error_cell": ok_both_clean,
            "trajectories_with_self_corrected_error_cells": n_with_cell_errors,
            "exec_errors_outside_cells": n_err,
            "timeouts": n_timeout,
            "elapsed_s": round(time.time() - t0, 1),
            "fail_examples": fail_examples,
        }

    rw_src = rewrite_source_by_task or {}
    rw_counts = collections.Counter(rw_src.get(t, "unknown") for t in parsed_by_task)
    stats = {
        "rescued_text_rows": len(rows),
        "rescued_unique_tasks": n,
        "multi_pair_rows": len(rows) - len(pair0),
        "assistant_turns": _stats(turns),
        "assistant_turns_hist": dict(sorted(collections.Counter(turns).items())),
        "tool_calls": _stats(calls),
        "reasoning_chars": _stats(reas),
        "code_chars": _stats(code),
        "est_output_tokens_chars_div_3p6": _stats(est_tokens),
        "completion_tokens_col": _stats(comp_tokens),
        "rewrite_source_counts": dict(rw_counts),
        "has_def_solve_pct": _pct(n - len(no_solve), n),
        "tasks_without_def_solve": no_solve[:20],
        "intermediate_tool_error_tasks": len(tool_err),
        "meta_phrase_leaks": {"n_tasks": len(leaks), "phrases": dict(leak_phrases), "examples": leaks[:15]},
        "leaks_by_rewrite_source": {
            src: {
                "n": sum(1 for t in parsed_by_task if rw_src.get(t, "unknown") == src),
                "strong_meta_leak": sum(1 for t, p in parsed_by_task.items() if rw_src.get(t, "unknown") == src and p["strong_leaks"]),
                "prior_attempt_reference": sum(
                    1 for t, p in parsed_by_task.items() if rw_src.get(t, "unknown") == src and p["prior_attempt_ref"]
                ),
                "either": sum(
                    1 for t, p in parsed_by_task.items()
                    if rw_src.get(t, "unknown") == src and (p["strong_leaks"] or p["prior_attempt_ref"])
                ),
                "intermediate_tool_error": sum(1 for t, p in parsed_by_task.items() if rw_src.get(t, "unknown") == src and p["tool_error"]),
                "turns_p50": float(np.median([p["n_assistant_turns"] for t, p in parsed_by_task.items() if rw_src.get(t, "unknown") == src] or [0])),
            }
            for src in sorted(rw_counts)
        },
        "clean_candidates_no_strong_leak_no_prior_ref": sum(
            1 for p in parsed_by_task.values() if not p["strong_leaks"] and not p["prior_attempt_ref"]
        ),
        "original_failed_rollouts": {
            "turns": _stats(orig_turns),
            "completion_tokens": _stats(orig_tokens),
            "failure_category": dict(orig_cat),
            "text_twin_already_test_passing": orig_text_twin_pass,
        },
        "independent_reverification": reverify,
    }

    # Examples: bucket by the diagnostics' rewrite source, original failure category, and leak status.
    md_parts: list[str] = []

    def _src(t: str) -> str:
        return str(rw_src.get(t, "unknown"))

    def _cat(t: str) -> str:
        return str((failures.get(t) or {}).get("failure_category"))

    buckets = {
        "model_clean_rewrite (try 1/2) / demo_overfit, no leak": [
            t for t in parsed_by_task
            if _src(t).startswith("model_clean_rewrite") and _cat(t).startswith("demo_overfit") and not parsed_by_task[t]["meta_leaks"]
        ],
        "model_clean_rewrite (try 1/2) / train_fail, no leak": [
            t for t in parsed_by_task
            if _src(t).startswith("model_clean_rewrite") and not _cat(t).startswith("demo_overfit") and not parsed_by_task[t]["meta_leaks"]
        ],
        "sandbox_verified_clean_synthesis (fallback 2-turn)": [t for t in parsed_by_task if _src(t) == "sandbox_verified_clean_synthesis"],
        "META-INSTRUCTION LEAK examples (should be scrubbed / dropped)": [t for t in parsed_by_task if parsed_by_task[t]["meta_leaks"]],
    }
    for bname, tids in buckets.items():
        md_parts.append(f"\n## Examples — {bname} ({len(tids)} tasks)\n")
        for t in tids[:show_examples]:
            md_parts.append(render_example(by_task[t], parsed_by_task[t], failures.get(t), example_chars))
    return stats, "\n".join(md_parts)


# ----------------------------------------------------------------------------------------------
# Section 5: overlap / leakage vs epoch-1 manifests
# ----------------------------------------------------------------------------------------------
def analyze_epoch1_overlap(epoch1_dir: str, rescued_tasks: list[str], rescued_parents: dict[str, str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    train_m = os.path.join(epoch1_dir, "train_manifest.jsonl")
    val_m = os.path.join(epoch1_dir, "val_manifest.jsonl")
    if not (gfile.Exists(train_m) and gfile.Exists(val_m)):
        return {"error": f"manifests not found under {epoch1_dir}"}
    train_rows = _read_jsonl(train_m)
    val_rows = _read_jsonl(val_m)
    train_tasks = {str(r["task_id"]) for r in train_rows}
    val_tasks = {str(r["task_id"]) for r in val_rows}
    train_parents = {str(r.get("parent_id") or r["task_id"]) for r in train_rows}
    val_parents = {str(r.get("parent_id") or r["task_id"]) for r in val_rows}
    res_set = set(rescued_tasks)
    res_par = {rescued_parents.get(t, t) for t in rescued_tasks}
    out.update(
        {
            "epoch1_train_records": len(train_rows),
            "epoch1_train_unique_tasks": len(train_tasks),
            "epoch1_val_records": len(val_rows),
            "epoch1_val_unique_tasks": len(val_tasks),
            "epoch1_train_by_modality": dict(collections.Counter(str(r.get("modality")) for r in train_rows)),
            "epoch1_train_by_batch": dict(collections.Counter(str(r.get("batch")) for r in train_rows)),
            "epoch1_train_tokens_total": int(sum(int(r.get("n_tokens") or 0) for r in train_rows)),
            "epoch1_train_supervised_tokens_total": int(sum(int(r.get("n_supervised_tokens") or 0) for r in train_rows)),
            "rescued_tasks_already_in_epoch1_train": len(res_set & train_tasks),
            "rescued_tasks_in_epoch1_val": len(res_set & val_tasks),
            "rescued_parents_in_epoch1_val_parents": len(res_par & val_parents),
            "rescued_parents_in_epoch1_train_parents": len(res_par & train_parents),
            "rescued_parents_new": len(res_par - train_parents - val_parents),
        }
    )
    return out


# ----------------------------------------------------------------------------------------------
# Section 6: Top-10K v8 verified inventory
# ----------------------------------------------------------------------------------------------
def analyze_top10k(top10k_dir: str, max_shards: int) -> dict[str, Any]:
    if not top10k_dir or max_shards <= 0 or not gfile.Exists(top10k_dir):
        return {"skipped": True}
    names = sorted(n for n in gfile.ListDir(top10k_dir) if n.startswith("verified_rollouts_shard_") and n.endswith(".parquet"))
    if not names:
        return {"skipped": True, "reason": "no verified_rollouts_shard_*.parquet"}
    names = names[:max_shards]
    wanted = [
        "task_id", "source", "batch", "dataset", "dataset_name", "learning_tier", "tier", "correct", "train_pass",
        "completion_tokens", "turns", "tool_calls", "verified_clean_sft", "all_3_levels_verified", "code_rescues",
        "cap_nudges", "pair",
    ]
    schema = _parquet_schema_names(os.path.join(top10k_dir, names[0]))
    cols = [c for c in wanted if c in schema]
    rows: list[dict[str, Any]] = []
    t0 = time.time()
    for i, n in enumerate(names):
        try:
            rows.extend(_read_parquet_rows(os.path.join(top10k_dir, n), columns=cols))
        except Exception as e:  # pylint: disable=broad-exception-caught
            print(f"  WARNING: top10k shard {n}: {e}", flush=True)
        if (i + 1) % 16 == 0:
            print(f"  top10k: {i + 1}/{len(names)} shards ({time.time() - t0:.0f}s)", flush=True)
    ds_col = "dataset" if "dataset" in cols else ("dataset_name" if "dataset_name" in cols else "source")
    tier_col = "learning_tier" if "learning_tier" in cols else ("tier" if "tier" in cols else None)

    def _tb(v: Any) -> bool:
        return v is True or (isinstance(v, str) and v.lower() == "true") or (isinstance(v, (int, np.integer)) and v == 1)

    clean = [r for r in rows if ("verified_clean_sft" not in cols) or _tb(r.get("verified_clean_sft"))]
    uniq = {str(r["task_id"]) for r in clean}
    return {
        "shards_scanned": len(names),
        "schema_columns": schema,
        "verified_rows": len(rows),
        "verified_clean_sft_rows": len(clean),
        "verified_clean_unique_tasks": len(uniq),
        "by_dataset": dict(collections.Counter(str(r.get(ds_col)) for r in clean).most_common()),
        "by_tier": dict(collections.Counter(str(r.get(tier_col)) for r in clean).most_common()) if tier_col else {},
        "completion_tokens": _stats([int(r.get("completion_tokens") or 0) for r in clean]),
        "turns": _stats([int(r.get("turns") or 0) for r in clean]),
        "rows_with_code_rescues_or_nudges": sum(
            1 for r in clean if int(r.get("code_rescues") or 0) > 0 or int(r.get("cap_nudges") or 0) > 0
        ),
    }


# ----------------------------------------------------------------------------------------------
# Markdown rendering
# ----------------------------------------------------------------------------------------------
def _kv_table(d: dict[str, Any], k_name: str = "metric", v_name: str = "value") -> str:
    lines = [f"| {k_name} | {v_name} |", "| :--- | ---: |"]
    for k, v in d.items():
        if isinstance(v, (dict, list)):
            v = json.dumps(v)
        lines.append(f"| `{k}` | {v} |")
    return "\n".join(lines)


def render_markdown(progress: dict[str, Any], anatomy: dict[str, Any], overlap: dict[str, Any], top10k: dict[str, Any], examples_md: str) -> str:
    p = progress
    md: list[str] = ["# E065 Rescue Job Analysis (DeepSeek-V4.1 iterative correction + clean rewrite)\n"]
    md.append("## 1. Rescue progress\n")
    md.append(
        _kv_table(
            {
                "shards launched (summary files)": f"{p['shards_with_summary']} / {p['num_shards_total']}",
                "tasks in launched shards": p["tasks_in_launched_shards"],
                "completed tasks": f"{p['completed_tasks']} ({p['completion_pct_of_launched']}% of launched)",
                "solved + rewritten tasks": f"{p['solved_and_rewritten_tasks']} ({p['solve_rate_pct']}% of completed)",
                "solved by correction attempt": p["solved_by_attempt"],
                "rewrite source": p["rewrite_source"],
                "rewritten assistant turns (hist)": p["rewritten_turns_hist"],
                "rewritten image rows / text rows": f"{p['rewritten_image_parquet_rows']} / {p['rewritten_text_parquet_rows']}",
                "decode tokens generated": f"{p['decode_tokens_generated']:,}",
                "avg decode tok/s per TPU slice": p["avg_decode_tok_per_s_per_slice"],
                "max shard wall time (h)": p["wall_hours_max"],
                "unsolved: prior failed attempts hist": p["unsolved_prior_attempt_counts"],
                "unsolved: top test diagnostics": p["unsolved_attempt_diag_top"],
            }
        )
    )
    md.append("\n### Solve rate by original failure category\n")
    md.append("| category | n | solved | solve % |\n| :--- | ---: | ---: | ---: |")
    for k, v in sorted(p["by_failure_category"].items(), key=lambda kv: -kv[1]["n"]):
        md.append(f"| `{k}` | {v['n']} | {v['solved']} | {v['solve_pct']} |")
    md.append("\n### Solve rate by source\n")
    md.append("| source | n | solved | solve % |\n| :--- | ---: | ---: | ---: |")
    for k, v in sorted(p["by_source"].items(), key=lambda kv: -kv[1]["n"]):
        md.append(f"| `{k}` | {v['n']} | {v['solved']} | {v['solve_pct']} |")

    a = anatomy
    md.append("\n## 2. Rescued trajectory anatomy (text twins)\n")
    md.append(
        _kv_table(
            {
                "rescued text rows / unique tasks": f"{a['rescued_text_rows']} / {a['rescued_unique_tasks']}",
                "assistant turns": a["assistant_turns"],
                "assistant turns hist": a["assistant_turns_hist"],
                "tool calls": a["tool_calls"],
                "reasoning chars": a["reasoning_chars"],
                "code chars": a["code_chars"],
                "est. output tokens (chars/3.6)": a["est_output_tokens_chars_div_3p6"],
                "`completion_tokens` column": a["completion_tokens_col"],
                "rewrite source (from diagnostics)": a["rewrite_source_counts"],
                "has `def solve` %": a["has_def_solve_pct"],
                "tasks with intermediate tool error": a["intermediate_tool_error_tasks"],
                "meta-phrase leaks": a["meta_phrase_leaks"],
                "leaks by rewrite source": a["leaks_by_rewrite_source"],
                "clean candidates (no strong leak, no prior-attempt ref)": a["clean_candidates_no_strong_leak_no_prior_ref"],
            }
        )
    )
    md.append("\n### Original failed rollout (same tasks) vs rescued\n")
    o = a["original_failed_rollouts"]
    md.append(
        _kv_table(
            {
                "original turns": o["turns"],
                "original completion tokens": o["completion_tokens"],
                "original failure categories": o["failure_category"],
                "original text twin already test-passing": o["text_twin_already_test_passing"],
                "rescued turns": a["assistant_turns"],
                "rescued est. output tokens": a["est_output_tokens_chars_div_3p6"],
            }
        )
    )
    md.append("\n### Independent sandbox re-verification of the rewritten code\n")
    md.append(_kv_table(a["independent_reverification"]))

    md.append("\n## 3. Overlap / leakage vs epoch-1 dataset (`e065_img_text_v1`)\n")
    md.append(_kv_table(overlap))

    md.append("\n## 4. Top-10K v8 verified rollout inventory\n")
    md.append(_kv_table({k: v for k, v in top10k.items() if k != "schema_columns"}))

    md.append("\n# Sample rescued trajectories\n")
    md.append(examples_md)
    return "\n".join(md)


def main(argv: list[str]) -> None:
    del argv
    os.makedirs(FLAGS.out_dir, exist_ok=True)
    t0 = time.time()

    print(f"[1/5] Rescue progress from {FLAGS.rescue_shards_dir} ...", flush=True)
    progress, diag_rows = analyze_rescue_progress(FLAGS.rescue_shards_dir)
    print(json.dumps({k: v for k, v in progress.items() if not isinstance(v, dict)}, indent=1), flush=True)

    print("[2/5] Loading ground-truth failure records ...", flush=True)
    failures = load_failure_records(FLAGS.non_test_passing_tasks_jsonl)
    print(f"  {len(failures)} original failed rollouts with GT grids", flush=True)

    print("[3/5] Loading rescued text parquet shards ...", flush=True)
    rows = load_rescued_text_rows(FLAGS.rescue_shards_dir)
    print(f"  {len(rows)} rescued text rows", flush=True)
    rewrite_source_by_task = {
        str(d.get("task_id")): str(d.get("rewrite_source")) for d in diag_rows if d.get("solved") and d.get("rewrite_source")
    }
    anatomy, examples_md = analyze_rescued_rows(
        rows,
        failures,
        FLAGS.show_examples,
        FLAGS.example_chars,
        FLAGS.reverify_max,
        FLAGS.reverify_timeout_s,
        rewrite_source_by_task=rewrite_source_by_task,
    )

    print("[4/5] Overlap vs epoch-1 manifests ...", flush=True)
    rescued_tasks = sorted({str(r["task_id"]) for r in rows})
    rescued_parents = {t: str((failures.get(t) or {}).get("parent") or t) for t in rescued_tasks}
    overlap = analyze_epoch1_overlap(FLAGS.epoch1_data_dir, rescued_tasks, rescued_parents)

    print("[5/5] Top-10K v8 inventory ...", flush=True)
    top10k = analyze_top10k(FLAGS.top10k_rollouts_dir, FLAGS.top10k_max_shards)

    md = render_markdown(progress, anatomy, overlap, top10k, examples_md)
    payload = {
        "generated_at_unix": time.time(),
        "elapsed_s": round(time.time() - t0, 1),
        "rescue_progress": progress,
        "rescued_anatomy": {k: v for k, v in anatomy.items()},
        "epoch1_overlap": overlap,
        "top10k_inventory": top10k,
        "rescued_task_ids": rescued_tasks,
        "unsolved_task_ids": sorted(str(d["task_id"]) for d in diag_rows if not d.get("solved")),
    }
    with open(os.path.join(FLAGS.out_dir, "rescue_analysis.json"), "w") as f:
        json.dump(payload, f, indent=1, default=str)
    with open(os.path.join(FLAGS.out_dir, "rescue_analysis.md"), "w") as f:
        f.write(md)
    print("\n" + md, flush=True)
    print(f"\nWrote {FLAGS.out_dir}/rescue_analysis.{{json,md}} in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    multiprocessing.set_start_method("fork", force=True)
    sys.setrecursionlimit(10000)
    app.run(main)
