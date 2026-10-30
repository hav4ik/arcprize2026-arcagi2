"""Data audit, contamination filtering, 90/10 parent split, and Qwen3.8 chat formatting for ARC-AGI-2 SFT."""

from __future__ import annotations

from collections import Counter
import copy
import hashlib
import json
import os
import re
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
        # If a parent appears in arc2_train / kaggle_arc2_training, label it arc2_train for stratification
        src = rec.get("source", "unknown")
        if pid not in parent_source or src in ("arc2_train", "kaggle_arc2_training") or rec.get("batch") == "arc2_train":
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

        qual = analyze_trajectory_quality(messages, events)
        rec: dict[str, Any] = {
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
            "py_error_turns": qual["py_error_turns"],
            "ident_syntax_error_turns": qual["ident_syntax_error_turns"],
            "wasted_post_pass_turns": qual["wasted_post_pass_turns"],
            "post_pass_completion_tokens": qual["post_pass_completion_tokens"],
            "doubt_markers_count": qual["doubt_markers_count"],
            "messages": messages,
        }
        # Optional Top-10K v8 metadata (absent in Vu's E031-E038 parquet schema).
        for opt_key in (
            "learning_tier",
            "learning_signal_score",
            "dataset",
            "thinking_tokens",
            "code_rescues",
            "cap_nudges",
            "all_3_levels_verified",
            "verified_clean_sft",
        ):
            val = row.get(opt_key)
            if val is not None and not (isinstance(val, float) and np.isnan(val)):
                rec[opt_key] = val.item() if hasattr(val, "item") else val
        verified_records.append(rec)

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
        "zero_py_error_among_verified": sum(1 for r in verified_records if r["py_error_turns"] == 0),
        "zero_ident_syntax_error_among_verified": sum(
            1 for r in verified_records if r["ident_syntax_error_turns"] == 0
        ),
        "zero_wasted_post_pass_turns_among_verified": sum(
            1 for r in verified_records if r["wasted_post_pass_turns"] == 0
        ),
        "elapsed_sec": round(time.time() - t0, 2),
    }
    return stats, verified_records


def read_rollout_rows_from_file(fpath: str) -> tuple[list[dict[str, Any]], list[str]]:
    """Reads rollout rows from either a .parquet or .jsonl file on CNS."""
    return _read_rollout_rows_from_file(fpath)


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


def load_argon_ratings(argon_path: str) -> dict[str, dict[str, Any]]:
    """Loads Gemini 4 Argon difficulty ratings keyed by both puzzle_id and family_id."""
    ratings: dict[str, dict[str, Any]] = {}
    if not argon_path or not gfile.Exists(argon_path):
        return ratings
    with gfile.Open(argon_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                pid = str(row.get("puzzle_id") or row.get("representative_puzzle_id") or "")
                fid = str(row.get("family_id") or "")
                if pid:
                    ratings[pid] = row
                if fid and fid not in ratings:
                    ratings[fid] = row
            except Exception:
                continue
    return ratings


def filter_records_by_argon_difficulty(
    records: Sequence[Mapping[str, Any]],
    argon_ratings: Mapping[str, Mapping[str, Any]],
    min_qwen38_difficulty: int = 6,
    max_qwen38_difficulty: int = 9,
    only_argon_recommended: bool = True,
    keep_unrated_official_arc: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Filters SFT records using Gemini 4 Argon Qwen3.8-27B difficulty ratings."""
    kept: list[dict[str, Any]] = []
    stats = {
        "input_records": len(records),
        "matched_argon_rating": 0,
        "kept_hard_for_qwen38": 0,
        "kept_unrated_official": 0,
        "dropped_trivial_or_low_diff": 0,
        "dropped_not_recommended_or_ambiguous": 0,
    }
    for rec in records:
        tid = str(rec.get("task_id") or "")
        pid = get_parent_puzzle_id(tid, str(rec.get("parent") or ""))
        rating = argon_ratings.get(tid) or argon_ratings.get(pid)
        if rating is None:
            if keep_unrated_official_arc:
                rec_copy = dict(rec)
                kept.append(rec_copy)
                stats["kept_unrated_official"] += 1
            continue

        stats["matched_argon_rating"] += 1
        q_diff = int(rating.get("qwen38_difficulty_score", 5))
        rec_inc = bool(rating.get("sft_recommend_include", True))
        if q_diff < min_qwen38_difficulty:
            stats["dropped_trivial_or_low_diff"] += 1
            continue
        if q_diff > max_qwen38_difficulty or (only_argon_recommended and not rec_inc):
            stats["dropped_not_recommended_or_ambiguous"] += 1
            continue

        rec_copy = dict(rec)
        rec_copy["qwen38_difficulty_score"] = q_diff
        rec_copy["sft_value_score"] = float(rating.get("sft_value_score", 0.0))
        rec_copy["sft_priority_tier"] = str(rating.get("sft_priority_tier", ""))
        kept.append(rec_copy)
        stats["kept_hard_for_qwen38"] += 1

    return kept, stats


# ============================================================
# DeepSeek-V4.1 Trajectory Quality & Native DSML Tokenization
# ============================================================

PYERR_RE = re.compile(r"Traceback|\w+Error:|\w+Exception:")
IDENT_SYNTAX_ERR_RE = re.compile(
    r"\b(?:NameError|AttributeError|ImportError|ModuleNotFoundError|SyntaxError|IndentationError|UnboundLocalError):"
)
OUTPUT_REF_RE = re.compile(
    r"""\[\s*['"]output['"]\s*\]|\.get\(\s*['"]output['"]|\bexp(?:ected)?\b|\btarget\b|\bgt\b|\bwant\b"""
)
COMPARE_RE = re.compile(
    r"array_equal|==|!=|\.all\(\)|\ball\(|np\.sum\(|\.sum\(\)|\bcompare|\bcheck|\bverify|\bmatch"
)
NEG_VERDICT_RE = re.compile(
    r"\bFalse\b|\bFAIL|\bfail(?:ed|s)?\b|\bwrong\b|\bincorrect\b|\bmismatch(?:es)?\b(?!\D{0,4}\b0\b)"
    r"|✗|\bnot match|\bno match\b|\bdiffer(?:s|ent)?\b|\bMISMATCH"
)
NUM_BAD_RE = re.compile(
    r"\b(?:diff|diffs|ndiff|mismatch(?:es)?|errors?|wrong|bad|nwrong|ndiffs|diff_count|n_diff)\b[^\d\n]{0,12}?(\d+)",
    re.I,
)
POS_VERDICT_RE = re.compile(r"\bTrue\b|\bOK\b|\bPASS(?:ED)?\b|\bpassed\b|\bcorrect\b|✓")
ALL_LINE_RE = re.compile(r"\ball\b[^\n]{0,40}\b(?:True|match|pass|correct|OK|ok|equal)", re.I)
CORE_DOUBT_RE = re.compile(
    r"\b[Ww]ait\b|\b[Hh]m{2,}\b|\b[Aa]ctually\b|double[- ]?check|\bre-?check|\bre-?verify"
    r"|verify (?:it |this |that )?again|check (?:it |this |that )?again|[Ll]et me (?:also )?(?:re-?)?verify"
    r"|[Ll]et me confirm|sanity[- ]check|\b[Hh]old on\b|\b[Oo]ops\b|\b[Hh]ang on\b"
    r"|\bnot (?:entirely |fully |so |100% )?sure\b|\bunsure\b|\buncertain\b"
    r"|\breconsider|\bre-?think|\bre-?examine|\bre-?read"
)
WORD_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _detect_all_train_pass(code: str, out: str, n_train: int) -> bool:
    """Returns True if `code` checks training outputs and `out` reports all training pairs passing."""
    if not code or PYERR_RE.search(out) or not (OUTPUT_REF_RE.search(code) and COMPARE_RE.search(code)):
        return False
    body = out.split("[budget:")[0].replace("[completed]", "")
    lines = [ln for ln in body.split("\n") if not re.search(r"test", ln, re.I)]
    body = "\n".join(lines)
    if NEG_VERDICT_RE.search(body):
        return False
    for m in NUM_BAD_RE.finditer(body):
        if int(m.group(1)) > 0:
            return False
    pos_lines = sum(bool(POS_VERDICT_RE.search(ln) or NUM_BAD_RE.search(ln)) for ln in lines)
    if pos_lines >= max(1, n_train) or ALL_LINE_RE.search(body) or body.strip() == "True":
        return True
    if "all(" in code and any(ln.strip() == "True" for ln in lines):
        return True
    return False


def analyze_trajectory_quality(
    messages: Sequence[Mapping[str, Any]],
    events: Sequence[Mapping[str, Any]],
) -> dict[str, int]:
    """Extracts per-trajectory Python error counts, identifier/syntax typo counts, and post-pass tail stats."""
    user_text = next((str(m.get("content") or "") for m in messages if m.get("role") == "user"), "")
    n_train = max(1, len(re.findall(r"--Example \d+--", user_text)))
    usage_events = [e for e in events if isinstance(e, Mapping) and e.get("usage")]

    py_error_turns = 0
    ident_syntax_error_turns = 0
    doubt_markers_count = 0
    first_pass_turn: int | None = None
    turn_comp_tokens: list[int] = []
    turn_has_submit: list[bool] = []

    asst_idx = 0
    n_msgs = len(messages)
    for i, m in enumerate(messages):
        if m.get("role") != "assistant":
            continue
        think = str(m.get("reasoning_content") or "")
        content = str(m.get("content") or "")
        doubt_markers_count += len(CORE_DOUBT_RE.findall(think))

        code_parts: list[str] = []
        has_submit = False
        for tc in m.get("tool_calls") or []:
            fn = tc.get("function", {})
            fn_name = fn.get("name", "")
            raw_args = fn.get("arguments") or "{}"
            try:
                args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
            except Exception:
                args = {}
            c_str = str(args.get("code", "")) if isinstance(args, Mapping) else ""
            if fn_name == "submit_answer":
                has_submit = True
            else:
                code_parts.append(c_str)
        code = "\n".join(code_parts)

        outs: list[str] = []
        j = i + 1
        while j < n_msgs and messages[j].get("role") == "tool":
            outs.append(str(messages[j].get("content") or ""))
            j += 1
        out = "\n".join(outs)

        if out and PYERR_RE.search(out):
            py_error_turns += 1
            if IDENT_SYNTAX_ERR_RE.search(out):
                ident_syntax_error_turns += 1

        if first_pass_turn is None and _detect_all_train_pass(code, out, n_train):
            first_pass_turn = asst_idx

        ev = usage_events[asst_idx] if asst_idx < len(usage_events) else {}
        comp_tok = int((ev.get("usage") or {}).get("completion_tokens") or 0)
        if comp_tok <= 0:
            comp_tok = max(1, (len(think) + len(content) + len(code)) // 4)
        turn_comp_tokens.append(comp_tok)
        turn_has_submit.append(has_submit)
        asst_idx += 1

    n_asst = asst_idx
    if first_pass_turn is not None:
        if turn_has_submit[first_pass_turn]:
            wasted_post_pass_turns = 0
            post_pass_completion_tokens = 0
        else:
            # Ideal is calling submit_answer on turn `first_pass_turn + 1` (0 wasted turns).
            wasted_post_pass_turns = max(0, (n_asst - 1) - (first_pass_turn + 1))
            post_pass_completion_tokens = int(sum(turn_comp_tokens[first_pass_turn + 1 :]))
    else:
        wasted_post_pass_turns = 0
        post_pass_completion_tokens = 0

    return {
        "py_error_turns": py_error_turns,
        "ident_syntax_error_turns": ident_syntax_error_turns,
        "wasted_post_pass_turns": wasted_post_pass_turns,
        "post_pass_completion_tokens": post_pass_completion_tokens,
        "doubt_markers_count": doubt_markers_count,
    }


# DeepSeek-V4.1 special tokens & DSML formatting (byte-identical to encoding/encoding.py)
DS41_BOS_TOKEN = "<｜begin▁of▁sentence｜>"
DS41_EOS_TOKEN = "<｜end▁of▁sentence｜>"
DS41_SYSTEM_SP_TOKEN = "<｜System｜>"
DS41_USER_SP_TOKEN = "<｜User｜>"
DS41_ASSISTANT_SP_TOKEN = "<｜Assistant｜>"
DS41_THINK_START = "<think>"
DS41_THINK_END = "</think>"
DS41_TOKEN = "｜DSML｜"

DS41_TOOLS_TEMPLATE = """## Tools

You have access to a set of tools to help answer the user's question. You can invoke tools by writing a "<｜DSML｜ calls>" block like the following:

<｜DSML｜ calls>
<｜DSML｜ invoke name="$TOOL_NAME">
<｜DSML｜ parameter name="$PARAMETER_NAME" string="true|false">$PARAMETER_VALUE</｜DSML｜ parameter>
...
</｜DSML｜ invoke>
<｜DSML｜ invoke name="$TOOL_NAME2">
...
</｜DSML｜ invoke>
</｜DSML｜ calls>

String parameters should be specified as is and set `string="true"`. For all other types (numbers, booleans, arrays, objects), pass the value in JSON format and set `string="false"`.

If thinking_mode is enabled (triggered by <think>), you MUST output your complete reasoning inside <think>...</think> BEFORE any tool calls or final response.

Otherwise, output directly after </think> with tool calls or final response.

### Available Tool Schemas

{tool_schemas}

You MUST strictly follow the above defined tool name and parameter schemas to invoke tool calls.
"""


def _ds41_to_json(val: Any) -> str:
    try:
        return json.dumps(val, ensure_ascii=False)
    except Exception:
        return json.dumps(val, ensure_ascii=True)


def _ds41_encode_args_to_dsml(raw_arguments: Any) -> str:
    args = raw_arguments
    if not isinstance(args, dict):
        for _ in range(2):
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    break
            else:
                break
        if not isinstance(args, dict):
            args = {"arguments": raw_arguments}
    parts: list[str] = []
    for k, v in args.items():
        is_str = "true" if isinstance(v, str) else "false"
        val_str = v if isinstance(v, str) else _ds41_to_json(v)
        parts.append(f'<{DS41_TOKEN} parameter name="{k}" string="{is_str}">{val_str}</{DS41_TOKEN} parameter>')
    return "\n".join(parts)


def render_deepseek_v41_with_spans(
    messages: Sequence[Mapping[str, Any]],
    reasoning_effort: int = 50,
    tools: bool = True,
) -> dict[str, Any]:
    """Renders messages into DeepSeek-V4.1 DSML format and returns char spans for loss masking & weighting.

    Matches `tokenize_rollouts.py:render_with_roles` + `encoding/encoding.py` byte-for-byte.
    Returns dict with:
      - `text`: full rendered string starting with `<｜begin▁of▁sentence｜>`
      - `asst_spans`: list of `(start_char, end_char)` for every assistant message's generated suffix
      - `tc_spans`: list of `(start_char, end_char)` for every DSML tool-call block (`<｜DSML｜ calls>...`)
      - `final_submit_span`: `(start_char, end_char)` of the final assistant message
    """
    msgs = [copy.deepcopy(dict(m)) for m in messages]
    last_asst = max(i for i, m in enumerate(msgs) if m.get("role") == "assistant")
    msgs = msgs[: last_asst + 1]
    if tools:
        if msgs[0].get("role") != "system":
            msgs.insert(0, {"role": "system", "content": ""})
        msgs[0]["tools"] = copy.deepcopy(TOOL_SPECS_LEAN2SUB)

    # Merge tool messages into user content_blocks in call order (matching encoding.py)
    merged: list[dict[str, Any]] = []
    for msg in msgs:
        role = msg.get("role")
        if role == "tool":
            tool_block = {
                "type": "tool_result",
                "tool_use_id": msg.get("tool_call_id", ""),
                "content": msg.get("content", ""),
            }
            if merged and merged[-1].get("role") == "user" and "content_blocks" in merged[-1]:
                merged[-1]["content_blocks"].append(tool_block)
            else:
                merged.append({"role": "user", "content_blocks": [tool_block]})
        elif role == "user":
            c_blocks = msg.get("content_blocks")
            if c_blocks is None:
                c = msg.get("content", "")
                c_blocks = c if isinstance(c, list) else [{"type": "text", "text": c}]
            if merged and merged[-1].get("role") == "user" and "content_blocks" in merged[-1]:
                merged[-1]["content_blocks"].extend(c_blocks)
            else:
                new_m = dict(msg)
                new_m["content_blocks"] = c_blocks
                merged.append(new_m)
        else:
            merged.append(dict(msg))

    # Sort tool_result blocks by preceding assistant tool_calls order
    last_tc_order: dict[str, int] = {}
    for msg in merged:
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            last_tc_order = {}
            for idx, tc in enumerate(msg["tool_calls"]):
                tc_id = tc.get("id") or tc.get("function", {}).get("id", "")
                if tc_id:
                    last_tc_order[tc_id] = idx
        elif msg.get("role") == "user" and msg.get("content_blocks"):
            t_blocks = [b for b in msg["content_blocks"] if b.get("type") == "tool_result"]
            if len(t_blocks) > 1 and last_tc_order:
                s_blocks = sorted(t_blocks, key=lambda b: last_tc_order.get(b.get("tool_use_id", ""), 0))
                s_idx = 0
                new_b = []
                for b in msg["content_blocks"]:
                    if b.get("type") == "tool_result":
                        new_b.append(s_blocks[s_idx])
                        s_idx += 1
                    else:
                        new_b.append(b)
                msg["content_blocks"] = new_b

    text = DS41_BOS_TOKEN
    asst_spans: list[tuple[int, int]] = []
    tc_spans: list[tuple[int, int]] = []

    for idx, msg in enumerate(merged):
        role = msg.get("role")
        piece = ""
        if idx == 0:
            effort_str = (
                f"Reasoning Effort: {reasoning_effort} "
                "(range 1-100, the higher the value, the more thorough the reasoning)\n\n"
                if reasoning_effort > 0
                else ""
            )
            if effort_str or role == "system":
                piece += DS41_SYSTEM_SP_TOKEN
            piece += effort_str

        if role == "system":
            if idx > 0:
                piece += DS41_SYSTEM_SP_TOKEN
            piece += msg.get("content") or ""
            if msg.get("tools"):
                fn_schemas = [_ds41_to_json(dict(t["function"])) for t in msg["tools"]]
                piece += "\n\n" + DS41_TOOLS_TEMPLATE.format(tool_schemas="\n".join(fn_schemas))
            if idx + 1 >= len(merged) or merged[idx + 1].get("role") in ("assistant", "latest_reminder"):
                if idx > 0:
                    piece += DS41_ASSISTANT_SP_TOKEN + DS41_THINK_START
            text += piece

        elif role == "user":
            piece += DS41_USER_SP_TOKEN
            c_blocks = msg.get("content_blocks")
            if c_blocks:
                parts = []
                for b in c_blocks:
                    if b.get("type") == "text":
                        parts.append(b.get("text", ""))
                    elif b.get("type") == "tool_result":
                        parts.append(f"<tool_result>{b.get('content', '')}</tool_result>")
                piece += "\n\n".join(parts)
            else:
                piece += msg.get("content") or ""
            if idx + 1 >= len(merged) or merged[idx + 1].get("role") in ("assistant", "latest_reminder"):
                piece += DS41_ASSISTANT_SP_TOKEN + DS41_THINK_START
            text += piece

        elif role == "assistant":
            rc = msg.get("reasoning_content") or ""
            summary_content = msg.get("content") or ""
            thinking_part = f"{rc}{DS41_THINK_END}"
            tc_content = ""
            if msg.get("tool_calls"):
                tc_list = []
                for tc in msg["tool_calls"]:
                    fn = tc["function"]
                    name = fn["name"]
                    arg_str = _ds41_encode_args_to_dsml(fn.get("arguments", "{}"))
                    tc_list.append(
                        f'<{DS41_TOKEN} invoke name="{name}">\n{arg_str}\n</{DS41_TOKEN} invoke>'
                    )
                tc_joined = "\n".join(tc_list)
                tc_content = f"\n\n<{DS41_TOKEN} calls>\n{tc_joined}\n</{DS41_TOKEN} calls>"

            asst_start = len(text)
            body_before_tc = f"{thinking_part}{summary_content}"
            tc_start = asst_start + len(body_before_tc)
            piece = f"{body_before_tc}{tc_content}{DS41_EOS_TOKEN}"
            asst_end = asst_start + len(piece)
            asst_spans.append((asst_start, asst_end))
            if tc_content:
                tc_spans.append((tc_start, asst_end))
            text += piece
        else:
            raise ValueError(f"Unsupported role in DeepSeek-V4.1 render: {role}")

    return {
        "text": text,
        "asst_spans": asst_spans,
        "tc_spans": tc_spans,
        "final_submit_span": asst_spans[-1] if asst_spans else None,
    }


def tokenize_deepseek_v41_and_build_weights(
    messages: Sequence[Mapping[str, Any]],
    tokenizer: Any,
    reasoning_effort: int = 50,
    wic_alnum_weight: int = 10,
    tool_call_weight: int = 2,
    final_submit_weight: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    """Tokenizes a rollout in DeepSeek-V4.1 DSML format and builds `input_ids`, `loss_mask`, and `loss_weights`.

    `loss_weights` (`uint8`) assigns:
      - `0` on non-assistant tokens (system, user, tool_result, `<｜Assistant｜><think>` header)
      - `1` on standard assistant reasoning tokens
      - `tool_call_weight` (default 2) on DSML tool-call blocks (`<｜DSML｜ calls>...`)
      - `final_submit_weight` (default 5) on the final `submit_answer` turn
      - `wic_alnum_weight` (default 10) on word-internal alphanumeric continuation tokens (`WIC-alnum`)
        inside assistant spans, targeting the 75% BPE split-mixing typo concentration in pruned E128.
    """
    rendered = render_deepseek_v41_with_spans(messages, reasoning_effort=reasoning_effort, tools=True)
    text = rendered["text"]
    asst_spans = rendered["asst_spans"]
    tc_spans = rendered["tc_spans"]
    final_span = rendered["final_submit_span"]

    # Support both `tokenizers.Tokenizer` and `transformers.PreTrainedTokenizerFast`
    if hasattr(tokenizer, "encode") and not hasattr(tokenizer, "vocab_size"):
        enc = tokenizer.encode(text, add_special_tokens=False)
        ids = np.asarray(enc.ids, dtype=np.int32)
        offsets = enc.offsets
    else:
        enc_out = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        ids = np.asarray(enc_out["input_ids"], dtype=np.int32)
        offsets = enc_out["offset_mapping"]

    starts = np.asarray([s for s, _ in offsets], dtype=np.int64)
    loss_mask = np.zeros(len(ids), dtype=np.uint8)
    loss_weights = np.zeros(len(ids), dtype=np.uint8)

    for s, e in asst_spans:
        sel = (starts >= s) & (starts < e)
        loss_mask[sel] = 1
        loss_weights[sel] = 1

    for s, e in tc_spans:
        sel = (starts >= s) & (starts < e) & (loss_mask == 1)
        loss_weights[sel] = np.maximum(loss_weights[sel], tool_call_weight)

    if final_span is not None:
        fs, fe = final_span
        sel = (starts >= fs) & (starts < fe) & (loss_mask == 1)
        loss_weights[sel] = np.maximum(loss_weights[sel], final_submit_weight)

    # Identify WIC-alnum tokens (2nd+ token of a word [A-Za-z_][A-Za-z0-9_]* splitting two alphanumerics)
    words = [(m.start(), m.end()) for m in WORD_IDENT_RE.finditer(text)]
    wi = 0
    n_words = len(words)
    wic_alnum_count = 0
    for k, (a, b) in enumerate(offsets):
        if loss_mask[k] == 0 or a >= b:
            continue
        while wi < n_words and words[wi][1] <= a:
            wi += 1
        if wi < n_words and words[wi][0] < b and a < words[wi][1]:
            w_start = words[wi][0]
            if a > w_start and text[a - 1].isalnum() and text[a].isalnum():
                loss_weights[k] = max(int(loss_weights[k]), wic_alnum_weight)
                wic_alnum_count += 1

    tok_stats = {
        "n_asst_turns": len(asst_spans),
        "n_wic_alnum_tokens": wic_alnum_count,
        "n_tool_call_tokens": int(((loss_weights == tool_call_weight) | (loss_weights == wic_alnum_weight)).sum()),
    }
    return ids, loss_mask, loss_weights, tok_stats


