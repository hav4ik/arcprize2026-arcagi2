"""E016: rescue E010 s20x60 attempts that hit the 60k hard cap with no answer.

For each cut-off attempt: load its trace, open a fresh sandbox session with the task data (as the harness
does), replay every executed tool cell in order, then send ONE thinking-disabled request with a "time is up,
submit now" message (plus at most one extra short turn if it ran code instead of submitting). Also a
zero-cost baseline: after replay, does the session already hold a correct `answer` or a 1-arg function that
solves the test input (best function by training-pair accuracy)?

Variants (how the time-up message is attached; both are accepted by the DeepSeek API):
  toolnote: drop the truncated final assistant turn (reasoning only, no content/tool calls) and append the
            time-up note to the last tool result, like the harness's budget notes (no new user turn).
  usermsg:  keep the trace exactly as sent (truncated assistant turn included) and append a user message.

Usage:
  .venv/bin/python experiments/E016_cutoff_continuation/continue_cutoffs.py --dry-run        # mock model
  .venv/bin/python experiments/E016_cutoff_continuation/continue_cutoffs.py --variants toolnote,usermsg
"""
import argparse
import copy
import gzip
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "evals"))
from run_deepseek_tools import (PROMPT_SUFFIXES, build_prompt, estimated_cost, lenient_parse, load_tasks,  # noqa: E402
                                parse, submit_answer, tool_output, turn, valid_grid, add_usage)
from sandbox_client import SandboxSession, check_health  # noqa: E402

RESULTS = ROOT / "evals/results/tools_persistent_grid_lean2sub_deepseek-flash_low_e010_full_s20x60.jsonl"
OUT_DIR = ROOT / "experiments/E016_cutoff_continuation/results"
TIME_UP = ("[Time is up: the output-token budget for this attempt is spent. Do not investigate further. Call "
           "submit_answer now with your best answer, e.g. solve(test_input) with your best function so far, or "
           "answer. If no function fits every training pair, submit the closest one anyway.]")
TIME_UP_AGAIN = "[Time is up. Call submit_answer now with your best answer; no more experiments.]"
HARD_CAP_USD = 3.0


def cutoff_records(path=RESULTS):
    recs = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return [r for r in recs if r.get("answer_source") is None and r["stop_reason"] == "output_budget"]


def load_trace(rec):
    with gzip.open(rec["trace_path"], "rt") as f:
        return json.load(f)


def strip_note(text):
    """Tool content as returned by the sandbox, without the harness's appended budget note."""
    i = text.find("\n[budget:")
    return text if i < 0 else text[:i]


def executed_calls(messages):
    """(name, code, original_tool_content) for every tool call the harness actually executed, in order."""
    results = {m["tool_call_id"]: m["content"] for m in messages if m["role"] == "tool"}
    calls = []
    for m in messages:
        for c in m.get("tool_calls", []) if m["role"] == "assistant" else []:
            content = results.get(c["id"])
            if content is None or content.startswith("[tool error:"):
                continue  # skipped (truncated turn) or malformed arguments: nothing ran
            code = json.loads(c["function"]["arguments"])["code"]
            calls.append((c["function"]["name"], code, content))
    return calls


def build_continuation(messages, variant):
    """Message list for the continuation request (deep copy of the exact trace + time-up message)."""
    msgs = copy.deepcopy(messages)
    variant = variant.removesuffix("_think")
    if variant == "usermsg":
        msgs.append({"role": "user", "content": TIME_UP})
        return msgs
    assert variant == "toolnote", variant
    if msgs[-1]["role"] == "assistant" and not msgs[-1].get("tool_calls"):
        msgs.pop()  # truncated reasoning-only turn
    if msgs[-1]["role"] == "tool":
        msgs[-1]["content"] += "\n" + TIME_UP
    else:  # no tool call ever ran: only the prompt remains
        assert msgs[-1]["role"] == "user"
        msgs.append({"role": "user", "content": TIME_UP})
    return msgs


def validate_messages(msgs):
    """OpenAI/DeepSeek structure: every assistant tool call is answered by a tool message right after."""
    assert msgs[0]["role"] == "user"
    i = 0
    while i < len(msgs):
        m = msgs[i]
        if m["role"] == "assistant" and m.get("tool_calls"):
            ids = [c["id"] for c in m["tool_calls"]]
            got = [msgs[i + 1 + k].get("tool_call_id") for k in range(len(ids))]
            assert got == ids, (i, ids, got)
            i += 1 + len(ids)
            continue
        assert m["role"] in {"user", "assistant"}, (i, m["role"])
        i += 1


def replay(session, messages):
    """Re-run executed cells in order. Returns stats including divergences from the original outputs."""
    stats = {"cells": 0, "errors": 0, "timeouts": 0, "diverged": 0, "status_diverged": 0, "state_lost": 0,
             "first_divergence": None}
    for name, code, original in executed_calls(messages):
        stats["cells"] += 1
        if name == "run_python":
            result = session.execute(code)
            new = tool_output(result, session.max_output)
            stats["errors"] += result["process_status"] == "error"
            stats["timeouts"] += result["process_status"] == "timeout"
            stats["state_lost"] += bool(result.get("state_reset"))
        elif name == "submit_answer":
            new, _ = submit_answer(session, code)
        else:
            continue
        orig = strip_note(original)
        if new != orig:
            stats["diverged"] += 1
            if new.split("\n", 1)[0] != orig.split("\n", 1)[0]:
                stats["status_diverged"] += 1
            if stats["first_divergence"] is None:
                stats["first_divergence"] = {"cell": stats["cells"], "orig": orig[:300], "new": new[:300]}
    return stats


PROBE = r'''
def _arc_probe():
    import contextlib, io, json, inspect, signal, types, copy
    class _T(Exception):
        pass
    def _h(*a):
        raise _T()
    old = signal.signal(signal.SIGALRM, _h)
    g = globals()
    inputs = json.loads(INPUTS)
    out = {"functions": []}
    def norm(v):
        if hasattr(v, "tolist"):
            v = v.tolist()
        return json.loads(json.dumps(v, default=int))
    try:
        if "answer" in g:
            try:
                out["answer"] = norm(g["answer"])
            except Exception as exc:
                out["answer_error"] = type(exc).__name__
        names = [n for n, v in list(g.items()) if isinstance(v, types.FunctionType) and not n.startswith("_")
                 and getattr(v, "__module__", None) == "__main__"]
        for n in names[-60:]:
            f = g[n]
            try:
                sig = inspect.signature(f)
                sig.bind(None)
            except (TypeError, ValueError):
                continue
            preds = []
            for grid in inputs:
                signal.setitimer(signal.ITIMER_REAL, 2.0)
                try:
                    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                        preds.append(norm(f(copy.deepcopy(grid))))
                except BaseException as exc:
                    preds.append({"error": type(exc).__name__})
                finally:
                    signal.setitimer(signal.ITIMER_REAL, 0)
            out["functions"].append({"name": n, "preds": preds})
    finally:
        signal.signal(signal.SIGALRM, old)
    print(MARKER + json.dumps(out))
try:
    _arc_probe()
finally:
    del _arc_probe
'''


def probe(session, test_output):
    """Zero-cost baseline from the replayed session. Test output is compared on the host only."""
    marker = f"ARC_PROBE_{uuid4().hex}:"
    inputs = [p["input"] for p in session.train] + [session.test_input]
    cell = PROBE.replace("INPUTS", repr(json.dumps(inputs))).replace("MARKER", repr(marker))
    old = session.timeout, session.max_output
    session.timeout, session.max_output = 300, 5_000_000  # probe output holds every prediction
    try:
        result = session.execute(cell)
    finally:
        session.timeout, session.max_output = old
    data = None
    for line in reversed(result.get("stdout", "").splitlines()):
        if line.startswith(marker):
            data = json.loads(line[len(marker):])
            break
    if data is None:
        return {"probe_error": result["process_status"] + ": " + (result.get("stderr") or result.get("stdout", ""))[-300:]}
    outs = [p["output"] for p in session.train]
    answer = data.get("answer")
    best = None
    for f in data["functions"]:
        train_ok = sum(p == o for p, o in zip(f["preds"][:-1], outs))
        test_pred = f["preds"][-1]
        key = (train_ok, valid_grid(test_pred))
        if best is None or key >= best["key"]:  # ties: the later-defined function wins
            best = {"key": key, "name": f["name"], "train_ok": train_ok, "test_valid": valid_grid(test_pred),
                    "test_correct": test_pred == test_output}
    full = [f["name"] for f in data["functions"] if all(p == o for p, o in zip(f["preds"][:-1], outs))]
    return {"answer_present": answer is not None, "answer_valid": valid_grid(answer),
            "answer_correct": answer == test_output if answer is not None else False,
            "n_functions": len(data["functions"]), "best_function": best and {k: v for k, v in best.items() if k != "key"},
            "n_train_train": len(outs), "functions_fitting_all_train": full,
            "any_function_correct": any(f["preds"][-1] == test_output for f in data["functions"])}


def continue_attempt(session, messages, call_fn, extra_turns=1):
    """Send the continuation; returns dict with pred, source, actions, usage."""
    total = {"prompt": 0, "cache_hit": 0, "completion": 0}
    actions, content, pred, source, raw_usage = [], "", None, None, []
    for k in range(1 + extra_turns):
        msg, usage, finish = call_fn(messages)
        messages.append(msg)
        add_usage(total, usage)
        raw_usage.append(usage)
        calls = msg.get("tool_calls", [])
        if not calls:
            content = msg.get("content") or ""
            actions.append(f"text({finish})")
            break
        accepted = None
        for call in calls:
            name = call["function"]["name"]
            try:
                code = json.loads(call["function"]["arguments"])["code"]
            except Exception:
                output, code = "[tool error: bad arguments]", None
            if code is None:
                pass
            elif finish == "length":
                output = "[tool error: Tool skipped: truncated response]"
            elif accepted is not None:
                output = "[skipped: solution already accepted]"
            elif name == "submit_answer":
                output, accepted = submit_answer(session, code)
            elif name == "run_python":
                output = tool_output(session.execute(code), session.max_output)
            else:
                output = f"[tool error: Unknown tool: {name}]"
            actions.append(f"{name}{'(ok)' if name == 'submit_answer' and accepted is not None else ''}"
                           + (f"[{code[:60]!r}]" if name == "submit_answer" and code else ""))
            messages.append({"role": "tool", "tool_call_id": call["id"], "content": output})
        if accepted is not None:
            pred, source = accepted, "submit"
            break
        if k < extra_turns:
            messages[-1]["content"] += "\n" + TIME_UP_AGAIN
    if pred is None:
        out, grid = submit_answer(session, "answer")
        if grid is not None:
            pred, source = grid, "session"
    if pred is None and content:
        cand = parse(content)
        pred = cand if valid_grid(cand) else lenient_parse(content)
        source = "parse" if pred is not None else None
    return {"pred": pred, "source": source, "actions": actions, "usage": total, "raw_usage": raw_usage,
            "final_content": content[:2000]}


def peak_now():
    now = datetime.now(timezone.utc)
    return now.weekday() < 5 and (1 <= now.hour < 4 or 6 <= now.hour < 10)


def mock_call(messages):
    """Dry run: always submit `answer` (no API)."""
    return ({"role": "assistant", "content": "", "reasoning_content": "",
             "tool_calls": [{"id": "call_mock", "type": "function",
                             "function": {"name": "submit_answer", "arguments": json.dumps({"code": "answer"})}}]},
            {"prompt_tokens": 0, "completion_tokens": 0, "prompt_cache_hit_tokens": 0}, "tool_calls")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default="toolnote")
    ap.add_argument("--sandbox-url", required=True)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--think-max-tokens", type=int, default=8192, help="per turn, *_think variants")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--dry-run", action="store_true", help="mock model, no API calls")
    ap.add_argument("--baseline", action="store_true", help="also run the zero-cost probe baseline")
    args = ap.parse_args()
    check_health(args.sandbox_url)
    recs = cutoff_records()[:args.limit]
    cfg = recs[0]["config"]
    tasks = load_tasks(Path(cfg["challenges"]))
    variants = [v for v in args.variants.split(",") if v]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / ("dryrun.jsonl" if args.dry_run else "continuations.jsonl")
    done = set()
    if out_path.exists() and not args.dry_run:
        for line in out_path.read_text().splitlines():
            r = json.loads(line)
            if "error" not in r:
                done.add((r["task_id"], r["pair"], r["attempt"], r["variant"]))
    lock = threading.Lock()
    spent = [0.0]
    client = None
    if not args.dry_run and any(v != "baseline" for v in variants):
        if peak_now():
            sys.exit("Peak pricing window (weekday 01-04 / 06-10 UTC): refusing to send paid requests")
        from openai import OpenAI
        from run_deepseek_arc2 import api_key
        client = OpenAI(api_key=api_key(), base_url="https://api.deepseek.com", timeout=600, max_retries=0)
    margs = SimpleNamespace(model=cfg["model"], effort=cfg["effort"], answer_tool=True, mode="grid",
                            tool_timeout=cfg["tool_timeout"])
    jobs = [(r, v) for r in recs for v in (variants + (["baseline"] if args.baseline else []))
            if (r["task_id"], r["pair"], r["attempt"], v) not in done]
    print(f"{len(recs)} cut-off attempts, {len(jobs)} jobs -> {out_path}", flush=True)

    def work(rec, variant):
        task = tasks[rec["task_id"]]
        test = task["test"][rec["pair"]]
        trace = load_trace(rec)
        messages = trace["messages"]
        expected_prompt = build_prompt(task["train"], test["input"]) + PROMPT_SUFFIXES[cfg["prompt"]]
        row = {"task_id": rec["task_id"], "pair": rec["pair"], "attempt": rec["attempt"], "variant": variant,
               "prompt_matches": messages[0]["content"] == expected_prompt,
               "orig_turns": rec["turns"], "orig_tool_calls": rec["tool_calls"]}
        t0 = time.monotonic()
        try:
            with SandboxSession(args.sandbox_url, task["train"], test["input"], cfg["tool_timeout"],
                                cfg["max_tool_output"]) as session:
                row["replay"] = replay(session, messages)
                if variant == "baseline":
                    row.update(probe(session, test["output"]))
                else:
                    if not args.dry_run and spent[0] >= HARD_CAP_USD:
                        raise RuntimeError("hard cap reached")
                    if not args.dry_run and peak_now():
                        raise RuntimeError("peak window started")
                    msgs = build_continuation(messages, variant)
                    validate_messages(msgs)
                    n0 = len(msgs)
                    think = variant.endswith("_think")  # comparison arm: thinking on, keeps prior reasoning
                    cap = args.think_max_tokens if think else args.max_tokens
                    call_fn = mock_call if args.dry_run else (
                        lambda m: turn(client, margs, m, cap, final=False, think=think))
                    res = continue_attempt(session, msgs, call_fn)
                    res["cost_usd"] = round(estimated_cost(res["usage"], cfg["model"], 1), 6)
                    res["correct"] = res["pred"] == test["output"]
                    res["new_messages"] = msgs[n0:]
                    row.update(res)
                    with lock:
                        spent[0] += res["cost_usd"]
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"[:500]
        row["seconds"] = round(time.monotonic() - t0, 1)
        with lock, out_path.open("a") as f:
            f.write(json.dumps(row) + "\n")
        brief = row.get("error") or (f"correct={row.get('correct')} src={row.get('source')} {row.get('actions')} "
                                     f"${row.get('cost_usd', 0):.4f}" if variant != "baseline" else
                                     f"answer={row.get('answer_present')} best={row.get('best_function')}")
        print(f"{rec['task_id']}.{rec['pair']}#{rec['attempt']} {variant} replay={row.get('replay', {}).get('diverged')}"
              f"/{row.get('replay', {}).get('cells')} {brief}  total ${spent[0]:.3f}", flush=True)

    with ThreadPoolExecutor(args.workers) as ex:
        list(ex.map(lambda j: work(*j), jobs))
    print(f"recorded spend ${spent[0]:.4f}")


if __name__ == "__main__":
    main()
