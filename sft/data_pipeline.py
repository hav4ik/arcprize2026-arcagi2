"""Data audit, contamination filtering, 90/10 parent split, and Qwen3.8 chat formatting for ARC-AGI-2 SFT."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import os
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from google3.pyglib import gfile


# Known contaminated tasks from havu/data_tools/CONTAMINATION.md:
# Training tasks 2753e76c and 81c0276b match ARC-AGI-2 eval task 89565ca0
# up to D8 symmetry + color permutation.
KNOWN_CONTAMINATED_TASKS: frozenset[str] = frozenset({"2753e76c", "81c0276b"})

# Official E031 / E028 system prompt and tool schemas (lean2sub + xhigh tier + minimal effort header).
LEAN2SUB_SYSTEM = (
    "You are solving an ARC-AGI-2 puzzle. Use `run_python` to inspect the "
    "training pairs, form a hypothesis, and test it on EVERY training pair "
    "before generating the test output. You have up to {max_turns} tool "
    "calls; do not submit until your rule reproduces all training outputs "
    "exactly. When verified, call `submit_answer` with the predicted 2D integer "
    "grid — or if two distinct hypotheses both fit the training pairs, call "
    "`submit_answer` with `[grid_1, grid_2]` to use both competition attempts."
)

TIER_SYSTEM_XHIGH = (
    "Reasoning effort is set to xhigh. Please think carefully through the task, "
    "validate key assumptions, consider plausible alternatives, and prioritize "
    "correctness, consistency, and clarity in the final answer."
)

MINIMAL_EFFORT_HEADER = (
    "Reasoning Effort: Minimal. Keep your thinking short and terse: write brief notes, not full sentences. "
    "Do not restate the task, do not re-read or re-derive what code has already printed, and do not re-verify "
    "what has already been checked. Test ideas with small code experiments instead of long deliberation, and "
    "submit as soon as the rule reproduces every training pair."
)

TOOL_SPECS_LEAN2SUB = [
    {
        "type": "function",
        "function": {
            "name": "run_python",
            "description": (
                "Execute a Python cell in your persistent IPython session. Variables/functions persist. "
                "Preloaded: train (list of {'input': grid, 'output': grid}), test_input (grid), numpy as np. "
                "scipy is available. No internet or host files. Timeout 30s. "
                "Use print() to inspect results. Check transformations on all training pairs."
            ),
            "parameters": {
                "type": "object",
                "properties": {"code": {"type": "string"}},
                "required": ["code"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit_answer",
            "description": (
                "End this attempt with your test output. `code` is a Python expression evaluated in your session, "
                "e.g. solve(test_input) or answer, giving a rectangular grid of ints 0-9 (list or numpy array); "
                "statements that set a variable `answer` also work. The harness reads the grid directly: do not "
                "retype it. Invalid grids return an error and the attempt continues."
            ),
            "parameters": {
                "type": "object",
                "properties": {"code": {"type": "string"}},
                "required": ["code"],
                "additionalProperties": False,
            },
        },
    },
]

# Token / chat markup constants for Qwen3.8-27B
IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
ASST_PREFIX = "<|im_start|>assistant\n"
USER_PREFIX = "<|im_start|>user\n"
SYS_PREFIX = "<|im_start|>system\n"
SUFFIX = "<|im_end|>\n"
LEAK_MARKERS = ("<|im_start|>", "<|im_end|>", "<tool_call>", "</tool_call>", "<function=")


def build_qwen_system_prompt(max_turns: int = 40) -> str:
    """Renders the exact Qwen3.8 system prompt block matching E031 / prompts_and_selector.py."""
    del max_turns
    parts = [
        f"{SYS_PREFIX}{TIER_SYSTEM_XHIGH}\n\n# Tools\n\n"
        "You have access to the following functions:\n\n<tools>"
    ]
    for tool in TOOL_SPECS_LEAN2SUB:
        parts.append("\n" + json.dumps(tool))
    parts.append(
        "\n</tools>\n\n"
        "If you choose to call a function ONLY reply in the following format with NO suffix:\n\n"
        "<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\n"
        "value_1\n</parameter>\n<parameter=example_parameter_2>\n"
        "This is the value for the second parameter\nthat can span\nmultiple lines\n"
        "</parameter>\n</function>\n</tool_call>\n\n"
        "<IMPORTANT>\n"
        "Reminder:\n"
        "- Function calls MUST follow the specified format: an inner <function=...></function> "
        "block must be nested within <tool_call></tool_call> XML tags\n"
        "- Required parameters MUST be specified\n"
        "- You may provide optional reasoning for your function call in natural language "
        "BEFORE the function call, but NOT after\n"
        "- If there is no function call available, answer the question like normal with your "
        "current knowledge and do not tell the user about function calls\n"
        "</IMPORTANT>\n\n"
        f"{MINIMAL_EFFORT_HEADER}"
    )
    parts.append(SUFFIX)
    return "".join(parts)


def get_parent_puzzle_id(task_id: str, parent_col: str | None = None) -> str:
    """Extracts the canonical parent puzzle ID so synthetic variants never split across train/val."""
    if parent_col and isinstance(parent_col, str) and parent_col.strip():
        return parent_col.strip().split("_")[0]
    return str(task_id).strip().split("_")[0]


def canonical_grid_hash(grid: Sequence[Sequence[int]], invariant_d8_color: bool = False) -> str:
    """Computes either an exact grid hash or a D8-symmetry + color-relabeling canonical hash."""
    arr = np.asarray(grid, dtype=np.int8)
    if not invariant_d8_color:
        return hashlib.sha256(arr.tobytes() + str(arr.shape).encode()).hexdigest()[:16]

    candidates: list[bytes] = []
    for k in range(4):
        rot = np.rot90(arr, k)
        for g in (rot, np.fliplr(rot)):
            flat = g.ravel()
            # Relabel colors in order of first appearance
            _, first_idx = np.unique(flat, return_index=True)
            order = flat[np.sort(first_idx)]
            lut = np.zeros(16, dtype=np.int8)
            for new_c, old_c in enumerate(order):
                if 0 <= old_c < 16:
                    lut[old_c] = new_c
            relabeled = lut[np.clip(g, 0, 15)]
            candidates.append(str(relabeled.shape).encode() + b":" + relabeled.tobytes())
    best = min(candidates)
    return hashlib.sha256(best).hexdigest()[:16]


def build_eval_contamination_index(eval_dir: str) -> dict[str, Any]:
    """Indexes all 120 official ARC-AGI-2 evaluation tasks by task_id, exact grid hash, and D8+color hash."""
    eval_task_ids: set[str] = set()
    exact_output_hashes: dict[str, str] = {}
    d8_pair_hashes: dict[str, str] = {}

    if gfile.Exists(eval_dir):
        for fname in sorted(gfile.ListDir(eval_dir)):
            if not fname.endswith(".json"):
                continue
            tid = fname[:-5]
            eval_task_ids.add(tid)
            fpath = os.path.join(eval_dir, fname)
            with gfile.Open(fpath, "r") as f:
                task_data = json.load(f)
            for split_name in ("train", "test"):
                for pair in task_data.get(split_name, []):
                    inp = pair.get("input")
                    out = pair.get("output")
                    if out is not None:
                        eh = canonical_grid_hash(out, invariant_d8_color=False)
                        exact_output_hashes[eh] = tid
                    if inp is not None and out is not None:
                        # Hash the (input, output) pair up to joint D8 + color relabeling
                        in_h = canonical_grid_hash(inp, invariant_d8_color=True)
                        out_h = canonical_grid_hash(out, invariant_d8_color=True)
                        d8_pair_hashes[f"{in_h}_{out_h}"] = tid

    return {
        "eval_task_ids": eval_task_ids,
        "exact_output_hashes": exact_output_hashes,
        "d8_pair_hashes": d8_pair_hashes,
    }


def is_task_contaminated(
    task_id: str,
    parent_id: str | None = None,
    eval_index: Mapping[str, Any] | None = None,
) -> tuple[bool, str]:
    """Checks if a task or its parent overlaps with ARC-AGI-2 evaluation tasks or known derivatives."""
    base_id = get_parent_puzzle_id(task_id, parent_id)
    if task_id in KNOWN_CONTAMINATED_TASKS or base_id in KNOWN_CONTAMINATED_TASKS:
        return True, f"known_d8_color_derivative({base_id})"
    if eval_index is not None:
        eval_ids = eval_index.get("eval_task_ids", set())
        if task_id in eval_ids or base_id in eval_ids:
            return True, f"direct_eval_task_id({base_id})"
    return False, ""


def audit_rollout_messages(
    messages: Sequence[Mapping[str, Any]],
    events: Sequence[Mapping[str, Any]],
) -> list[str]:
    """Audits a trajectory's messages and events for truncated turns, bad tool JSON, or markup leaks."""
    bad_reasons: list[str] = []
    ev_idx = 0
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        ev = events[ev_idx] if ev_idx < len(events) else {}
        ev_idx += 1
        if ev.get("finish_reason") == "length":
            bad_reasons.append("length_stop")

        reasoning = msg.get("reasoning_content") or ""
        content = msg.get("content") or ""
        for marker in LEAK_MARKERS:
            if marker in reasoning or marker in content:
                bad_reasons.append(f"markup_leak:{marker}")

        for tc in msg.get("tool_calls") or []:
            raw_args = tc.get("function", {}).get("arguments") or ""
            try:
                args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                if not isinstance(args, dict) or "code" not in args:
                    bad_reasons.append("missing_code_param")
                elif "<parameter=" in str(args["code"]) or "</function>" in str(args["code"]):
                    bad_reasons.append("xml_in_code")
            except Exception:
                bad_reasons.append("bad_tool_json")
        if not (msg.get("tool_calls") or content.strip()):
            bad_reasons.append("empty_assistant_turn")
    return bad_reasons


def render_assistant_turn(msg: Mapping[str, Any]) -> str:
    """Renders an assistant message (reasoning + content + tool_calls) into Qwen3.8 native format."""
    reasoning = msg.get("reasoning_content") or ""
    content = msg.get("content") or ""
    tool_calls = msg.get("tool_calls") or []

    tc_parts: list[str] = []
    for tc in tool_calls:
        fn = tc.get("function", {})
        name = fn.get("name", "")
        raw_args = fn.get("arguments") or "{}"
        args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
        tc_parts.append(f"<tool_call>\n<function={name}>")
        for k, v in args.items():
            val_str = v if isinstance(v, str) else json.dumps(v)
            tc_parts.append(f"\n<parameter={k}>\n{val_str}\n</parameter>")
        tc_parts.append("\n</function>\n</tool_call>")
    tc_block = "\n".join(tc_parts)

    body = f"<think>\n{reasoning}\n</think>\n\n"
    if content:
        body += content
        if tc_block:
            body += "\n" + tc_block
    else:
        body += tc_block
    return body + SUFFIX


def build_segments(messages: Sequence[Mapping[str, Any]], max_turns: int = 40) -> list[tuple[str, int]]:
    """Splits a trajectory into (text_chunk, loss_weight) segments.

    loss_weight == 0 -> masked in cross-entropy loss (labels = -100):
      - System prompt + tools header
      - User prompt & budget reminder notes
      - Tool execution responses (`<tool_response>...</tool_response>`)
      - Assistant turn header (`<|im_start|>assistant\\n`)
      - trailing `\\n` after assistant `<|im_end|>`
    loss_weight == 1 -> supervised in cross-entropy loss:
      - Assistant `<think>...\\n</think>\\n\\n{content}<tool_call>...</tool_call><|im_end|>`
    """
    segments: list[tuple[str, int]] = []
    assert messages and messages[0].get("role") == "system", "First message must be system"
    segments.append((build_qwen_system_prompt(max_turns=max_turns), 0))

    i = 1
    n = len(messages)
    while i < n:
        msg = messages[i]
        role = msg.get("role")
        if role == "user":
            segments.append((f"{USER_PREFIX}{msg.get('content', '')}{SUFFIX}", 0))
            i += 1
        elif role == "assistant":
            segments.append((ASST_PREFIX, 0))
            asst_body_with_suffix = render_assistant_turn(msg)
            # Supervise up to and including `<|im_end|>`, mask the trailing `\n`
            assert asst_body_with_suffix.endswith(SUFFIX)
            supervised_body = asst_body_with_suffix[:-1]  # ends with `<|im_end|>`
            segments.append((supervised_body, 1))
            segments.append(("\n", 0))
            i += 1
        elif role == "tool":
            tr_parts = [USER_PREFIX]
            first = True
            while i < n and messages[i].get("role") == "tool":
                if not first:
                    tr_parts.append("\n")
                first = False
                tr_parts.append(f"<tool_response>\n{messages[i].get('content', '')}\n</tool_response>")
                i += 1
            if i < n and messages[i].get("role") == "user":
                # Budget / turns note folded into the same user block right after </tool_response>
                tr_parts.append(f"\n{messages[i].get('content', '')}")
                i += 1
            tr_parts.append(SUFFIX)
            segments.append(("".join(tr_parts), 0))
        else:
            raise ValueError(f"Unexpected message role at index {i}: {role}")

    # Merge adjacent weight-0 segments so BPE merges across weight-0 boundaries match full-string tokenization
    merged: list[tuple[str, int]] = []
    for text, weight in segments:
        if merged and merged[-1][1] == weight == 0:
            merged[-1] = (merged[-1][0] + text, 0)
        else:
            merged.append((text, weight))
    return merged


def tokenize_and_verify_segments(
    segments: Sequence[tuple[str, int]],
    tokenizer: Any,
    im_start_id: int,
    im_end_id: int,
    asst_role_id: int,
    newline_id: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Tokenizes `(text, weight)` segments into `input_ids` and `loss_mask` and verifies exact boundaries."""
    ids_list: list[int] = []
    mask_list: list[int] = []
    for text, weight in segments:
        toks = tokenizer.encode(text, add_special_tokens=False)
        ids_list.extend(toks)
        mask_list.extend([weight] * len(toks))

    input_ids = np.asarray(ids_list, dtype=np.int32)
    loss_mask = np.asarray(mask_list, dtype=np.uint8)

    verify_token_loss_mask(
        input_ids=input_ids,
        loss_mask=loss_mask,
        im_start_id=im_start_id,
        im_end_id=im_end_id,
        asst_role_id=asst_role_id,
        newline_id=newline_id,
    )
    return input_ids, loss_mask


def verify_token_loss_mask(
    input_ids: np.ndarray,
    loss_mask: np.ndarray,
    im_start_id: int,
    im_end_id: int,
    asst_role_id: int,
    newline_id: int,
) -> int:
    """Independently reconstructs the assistant-only mask from raw token IDs and asserts exact equality."""
    assert input_ids.shape == loss_mask.shape, "input_ids and loss_mask must have the same shape"
    expected = np.zeros_like(loss_mask)
    starts = np.where(input_ids == im_start_id)[0]
    ends = np.where(input_ids == im_end_id)[0]
    if len(starts) != len(ends):
        raise ValueError(f"Mismatched <|im_start|> ({len(starts)}) and <|im_end|> ({len(ends)}) counts")

    asst_spans = 0
    for s, e in zip(starts, ends):
        if s >= e:
            raise ValueError(f"Invalid message span: start {s} >= end {e}")
        role_tok = int(input_ids[s + 1])
        if role_tok == asst_role_id:
            if int(input_ids[s + 2]) != newline_id:
                raise ValueError(f"Expected newline token {newline_id} at {s + 2}, got {input_ids[s + 2]}")
            # Supervise strictly after `<|im_start|>assistant\n` up to and including `<|im_end|>`
            expected[s + 3 : e + 1] = 1
            asst_spans += 1

    if not np.array_equal(loss_mask, expected):
        diff_idx = np.where(loss_mask != expected)[0]
        raise AssertionError(
            f"Token loss mask mismatch at {len(diff_idx)} positions (first at {diff_idx[0]})."
        )
    return asst_spans


def split_by_parent_puzzle_id(
    records: Sequence[Mapping[str, Any]],
    val_fraction: float = 0.10,
    seed: int = 20260405,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Splits trajectories ~90/10 by parent puzzle ID (`task_id.split('_')[0]`), stratified by source."""
    rng = np.random.default_rng(seed)

    # Group records by parent_id
    by_parent: dict[str, list[dict[str, Any]]] = {}
    parent_source: dict[str, str] = {}
    for rec in records:
        pid = get_parent_puzzle_id(rec["task_id"], rec.get("parent"))
        by_parent.setdefault(pid, []).append(dict(rec))
        # If a parent appears in arc2_train, label it arc2_train for stratification
        src = rec.get("source", "unknown")
        if pid not in parent_source or src == "arc2_train":
            parent_source[pid] = src

    # Group parent IDs by source so both arc2_train and nvarc_training are split ~90/10
    parents_by_source: dict[str, list[str]] = {}
    for pid, src in sorted(parent_source.items()):
        parents_by_source.setdefault(src, []).append(pid)

    val_parents: set[str] = set()
    train_parents: set[str] = set()
    for src, pids in sorted(parents_by_source.items()):
        pids_arr = list(pids)
        rng.shuffle(pids_arr)
        n_val = max(1, int(round(len(pids_arr) * val_fraction))) if len(pids_arr) > 1 else 0
        for p in pids_arr[:n_val]:
            val_parents.add(p)
        for p in pids_arr[n_val:]:
            train_parents.add(p)

    assert val_parents.isdisjoint(train_parents), "Parent IDs leaked across train and val!"

    train_records: list[dict[str, Any]] = []
    val_records: list[dict[str, Any]] = []
    for pid, group in by_parent.items():
        target = val_records if pid in val_parents else train_records
        for item in group:
            item_copy = dict(item)
            item_copy["parent_id"] = pid
            item_copy["split"] = "val" if pid in val_parents else "train"
            target.append(item_copy)

    # Verify zero parent overlap and zero task_id overlap
    train_pids = {r["parent_id"] for r in train_records}
    val_pids = {r["parent_id"] for r in val_records}
    assert train_pids.isdisjoint(val_pids), "Parent puzzle ID overlap detected!"

    split_stats = {
        "total_trajectories": len(train_records) + len(val_records),
        "train_trajectories": len(train_records),
        "val_trajectories": len(val_records),
        "val_trajectory_fraction": round(
            len(val_records) / max(1, len(train_records) + len(val_records)), 4
        ),
        "total_unique_parents": len(by_parent),
        "train_unique_parents": len(train_pids),
        "val_unique_parents": len(val_pids),
        "val_parent_fraction": round(len(val_pids) / max(1, len(by_parent)), 4),
        "parent_overlap_count": len(train_pids & val_pids),
        "train_unique_task_ids": len({r["task_id"] for r in train_records}),
        "val_unique_task_ids": len({r["task_id"] for r in val_records}),
    }
    return train_records, val_records, split_stats


def _read_rollout_rows_from_file(fpath: str) -> tuple[list[dict[str, Any]], list[str]]:
    """Reads rollout rows from either a .parquet or .jsonl file on CNS."""
    if fpath.endswith(".jsonl"):
        rows: list[dict[str, Any]] = []
        with gfile.Open(fpath, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        cols = list(rows[0].keys()) if rows else []
        return rows, cols
    import pyarrow as pa

    with gfile.Open(fpath, "rb") as f:
        data = f.read()
    table = pq.read_table(pa.BufferReader(data))
    df = table.to_pandas()
    return df.to_dict(orient="records"), list(df.columns)


def audit_rollout_file(
    fpath: str,
    eval_index: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Audits a single parquet or jsonl rollout file and returns summary stats + verified candidate records."""
    import time

    t0 = time.time()
    records, schema_cols = _read_rollout_rows_from_file(fpath)

    n_rows = len(records)
    unique_task_ids = {str(r["task_id"]) for r in records}
    unique_parents = {
        get_parent_puzzle_id(str(r["task_id"]), str(r.get("parent") or ""))
        for r in records
    }

    n_correct = sum(1 for r in records if r.get("correct") is True)
    n_train_pass = sum(1 for r in records if r.get("train_pass") is True)
    n_both = sum(
        1 for r in records if r.get("correct") is True and r.get("train_pass") is True
    )
    n_train_pass_but_wrong = sum(
        1 for r in records if r.get("train_pass") is True and r.get("correct") is not True
    )
    n_correct_but_train_fail = sum(
        1 for r in records if r.get("correct") is True and r.get("train_pass") is not True
    )

    answer_sources = Counter(str(r.get("answer_source") or "none") for r in records)
    stop_reasons = Counter(str(r.get("stop_reason") or "none") for r in records)

    contaminated_rows = 0
    contaminated_tasks: dict[str, str] = {}
    bad_turn_rows = 0
    bad_reason_counts: Counter[str] = Counter()
    verified_records: list[dict[str, Any]] = []

    for row in records:
        tid = str(row["task_id"])
        par = str(row.get("parent") or "")
        is_contam, reason = is_task_contaminated(tid, par, eval_index)
        if is_contam:
            contaminated_rows += 1
            contaminated_tasks[tid] = reason

        # Check verified correctness: correct == True AND train_pass == True AND answer_source == 'submit'
        if not (row.get("correct") is True and row.get("train_pass") is True):
            continue
        if row.get("answer_source") != "submit":
            continue
        if is_contam:
            continue

        messages = (
            json.loads(row["messages"])
            if isinstance(row["messages"], str)
            else row["messages"]
        )
        raw_events = row.get("events")
        events = (
            json.loads(raw_events)
            if isinstance(raw_events, str) and raw_events
            else (raw_events if isinstance(raw_events, list) else [])
        )
        bad_reasons = audit_rollout_messages(messages, events)
        if bad_reasons:
            bad_turn_rows += 1
            for r in set(bad_reasons):
                bad_reason_counts[r] += 1
            continue

        verified_records.append(
            {
                "task_id": tid,
                "pair": int(row["pair"]),
                "attempt": int(row["attempt"]),
                "batch": str(row["batch"]),
                "source": str(row["source"]),
                "parent": par,
                "parent_id": get_parent_puzzle_id(tid, par),
                "turns": int(row["turns"]),
                "tool_calls": int(row["tool_calls"]),
                "prompt_tokens": int(row["prompt_tokens"]),
                "completion_tokens": int(row["completion_tokens"]),
                "messages": messages,
            }
        )

    stats = {
        "file": os.path.basename(fpath),
        "path": fpath,
        "rows": n_rows,
        "unique_task_ids": len(unique_task_ids),
        "unique_parent_puzzles": len(unique_parents),
        "schema_columns": schema_cols,
        "correct_count": n_correct,
        "correct_rate": round(n_correct / max(1, n_rows), 4),
        "train_pass_count": n_train_pass,
        "train_pass_rate": round(n_train_pass / max(1, n_rows), 4),
        "correct_and_train_pass": n_both,
        "train_pass_but_test_wrong": n_train_pass_but_wrong,
        "false_positive_demo_rate": round(n_train_pass_but_wrong / max(1, n_train_pass), 4),
        "correct_but_train_fail": n_correct_but_train_fail,
        "answer_sources": dict(answer_sources),
        "stop_reasons": dict(stop_reasons),
        "contaminated_rows": contaminated_rows,
        "contaminated_tasks": contaminated_tasks,
        "bad_turn_rows_among_verified": bad_turn_rows,
        "bad_reason_counts": dict(bad_reason_counts),
        "verified_clean_count": len(verified_records),
        "elapsed_sec": round(time.time() - t0, 2),
    }
    return stats, verified_records


def discover_top10k_rollout_shards(top10k_dir: str) -> list[str]:
    """Discovers `rollouts_shard_*.jsonl` (or `.parquet` if `.jsonl` is absent) in `top10k_dir`."""
    if not top10k_dir or not gfile.Exists(top10k_dir):
        return []
    entries = sorted(gfile.ListDir(top10k_dir))
    jsonl_shards = [
        os.path.join(top10k_dir, f)
        for f in entries
        if f.startswith("rollouts_shard_") and f.endswith(".jsonl")
    ]
    if jsonl_shards:
        return jsonl_shards
    return [
        os.path.join(top10k_dir, f)
        for f in entries
        if f.startswith("rollouts_shard_") and f.endswith(".parquet")
    ]
