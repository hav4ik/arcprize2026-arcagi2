"""Iterative Reasoning Correction (up to 4 retries) + Clean Prefix-Free Solution Rewrite on E065 Non-Test-Passing Rollouts using Full DeepSeek-V4.1-Flash-1017 (384 Experts, BF16->FP32).

Workflow per non-test-passing E065 rollout:
1. Stage 1 — Iterative Reasoning Correction (up to `--max_correction_attempts=4` rounds):
   - Prompts the full DeepSeek-V4.1-Flash-1017 model with the puzzle (`train` pairs + `test_input`) and diagnostic feedback on why the original E065 rollout (plus any prior correction attempts 1..k-1) failed (`train_pass`, `train_frac`, previous code, and whether it overfitted training demos or failed training pairs, without leaking the ground-truth test output).
   - Runs the multi-turn IPython tool loop (`run_python` and `submit_answer`).
   - Checks both `train_pass == True` (via `run_3level_session_check`) and exact match against ground-truth test outputs across ALL test pairs of the task (`correct_all == True`).
   - Stops retrying as soon as a verified correct solution is found (`solved_on_attempt = k`).
2. Stage 2 — Clean Prefix-Free Solution Rewrite for SFT:
   - Once solved in Stage 1, runs the full DeepSeek model in a fresh `HavuSubprocessSession` to rewrite the solution from scratch as a clean, direct multi-turn trajectory (`<think>...</think>` -> `run_python` -> `tool` -> `<think>...</think>` -> `submit_answer` -> `tool`) with NO mention or prefix history of previous failed attempts.
   - Replaces `messages[0]` (system) and `messages[1]` (initial user prompt) with the exact original E065 `messages[0:2]` (preserving the inline `data:image/png;base64,...` PNG grid images for `e065_images_*` and text prompt for `e065_text_*`).
   - Saves E065-schema Parquet shards and JSONL checkpoints to CNS under `/cns/lu-d/home/arazdai/kaggle/arc_sft_data/e065_hardest_rollouts/corrected_rewritten/`.
"""

from __future__ import annotations

import copy
import hashlib
import io
import json
import os
import re
import time
from typing import Any

from absl import flags
import pyarrow as pa
import pyarrow.parquet as pq

from google3.experimental.users.arazdai.kaggle.arc_agi_2.pilot_1.setB.tpu_vllm import deepseek_v41_arc2_eval as ds_eval
from google3.experimental.users.arazdai.kaggle.arc_agi_2.pilot_1.setB.tpu_vllm import deepseek_v41_top10k_rollouts as top10k
from google3.experimental.users.arazdai.kaggle.arc_agi_2.pilot_1.setB.tpu_vllm import qwen_tpu_vllm_eval as base_eval
from google3.pyglib import gfile

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "non_test_passing_parquet_path",
    "/cns/lu-d/home/arazdai/kaggle/arc_sft_data/e065_hardest_rollouts/non_test_passing/e065_images_non_test_passing.parquet",
    "CNS path to e065_images_non_test_passing.parquet (containing original messages with inline base64 PNG images).",
)
flags.DEFINE_string(
    "non_test_passing_tasks_jsonl_path",
    "/cns/lu-d/home/arazdai/kaggle/arc_sft_data/e065_hardest_rollouts/non_test_passing/e065_images_non_test_passing_tasks.jsonl",
    "CNS path to e065_images_non_test_passing_tasks.jsonl (containing ground-truth train/test grids & failure diagnostics).",
)
flags.DEFINE_string(
    "difficulty_rankings_jsonl_path",
    "/cns/lu-d/home/arazdai/kaggle/arc_sft_data/e065_hardest_rollouts/difficulty_rankings/gemini38_flash_non_test_passing_ranked.jsonl",
    "Optional CNS path to Gemini Flash 3.8 difficulty rankings JSONL.",
)
flags.DEFINE_string(
    "e065_output_dir",
    "/cns/lu-d/home/arazdai/kaggle/arc_sft_data/e065_hardest_rollouts/corrected_rewritten",
    "CNS directory to write corrected & rewritten E065 SFT Parquet and JSONL shards.",
)
flags.DEFINE_integer(
    "max_correction_attempts",
    4,
    "Maximum number of full DeepSeek reasoning-correction attempts per failed rollout.",
)
flags.DEFINE_integer(
    "max_rewrite_attempts",
    2,
    "Maximum number of full DeepSeek clean-rewrite tries once a correct solution is found.",
)


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
)


def format_failure_feedback_block(
    failure_rec: dict[str, Any],
    prior_attempts: list[dict[str, Any]],
    difficulty_info: dict[str, Any] | None = None,
) -> str:
    """Formats diagnostic feedback on the original E065 failure + any prior retry attempts 1..k-1 without leaking GT test output."""
    lines: list[str] = [
        "=== DIAGNOSTIC FEEDBACK ON PREVIOUS INCORRECT ATTEMPT(S) ===",
        "A previous rollout on this exact puzzle produced an INCORRECT test output (`correct=False`).",
    ]

    orig_train_pass = bool(failure_rec.get("train_pass"))
    orig_train_frac = float(failure_rec.get("train_frac") or 0.0)
    orig_cat = str(failure_rec.get("failure_category") or "")
    orig_turns = int(failure_rec.get("turns") or 0)
    orig_stop = str(failure_rec.get("stop_reason") or "")

    if orig_train_pass:
        lines.append(
            f"- Original Rollout Diagnosis (`{orig_cat}`): The previous code PASSED all training examples "
            f"(`train_pass=True`, `train_frac=1.0`), but FAILED on the test input! "
            "This means the previous rule OVERFITTED a coincidence in the training examples or mishandled an edge case "
            "present in `test_input` (e.g., different grid dimensions, object counts, color roles, boundary clipping, "
            "symmetry axis, or tie-breaking order)."
        )
    elif orig_stop != "submitted":
        lines.append(
            f"- Original Rollout Diagnosis (`{orig_cat}`): The previous rollout stopped with `stop_reason={orig_stop}` "
            f"after {orig_turns} turns (`train_frac={orig_train_frac:.2f}`) without submitting a verified rule."
        )
    else:
        lines.append(
            f"- Original Rollout Diagnosis (`{orig_cat}`): The previous rollout submitted code that FAILED on the "
            f"training examples (`train_pass=False`, `train_frac={orig_train_frac:.2f}`) and failed on the test input."
        )

    # Summarize structural error on test pairs (shape match vs cell mismatch, WITHOUT revealing GT cell values)
    pair_notes = []
    for pc in failure_rec.get("pair_checks", []):
        p_idx = pc.get("pair", 0)
        ds = pc.get("diff_summary") or {}
        st = ds.get("status", "")
        if st == "shape_mismatch":
            pair_notes.append(
                f"test pair {p_idx}: predicted shape {ds.get('pred_shape')} did NOT match expected output shape"
            )
        elif st == "cell_mismatch":
            acc = ds.get("cell_accuracy", 0.0)
            pair_notes.append(
                f"test pair {p_idx}: output dimensions matched, but {ds.get('mismatched_cells')} cells were wrong (cell_accuracy={acc:.2f})"
            )
        elif st == "no_prediction":
            pair_notes.append(f"test pair {p_idx}: no valid grid was produced")
    if pair_notes:
        lines.append("- Test Output Structural Diagnostics: " + "; ".join(pair_notes))

    if difficulty_info and difficulty_info.get("reasoning_bottleneck"):
        lines.append(
            f"- Reasoning Bottleneck Analysis: {difficulty_info['reasoning_bottleneck']}"
        )

    orig_reas = (failure_rec.get("last_reasoning_excerpt") or "").strip()
    if orig_reas:
        lines.append(f"\n--- Previous Rollout Final Reasoning Excerpt ---\n{orig_reas[-1200:]}")

    orig_code = (
        failure_rec.get("last_submit_code") or failure_rec.get("last_python_code") or ""
    ).strip()
    if orig_code:
        lines.append(f"\n--- Previous Rollout Submitted / Executed Code (INCORRECT) ---\n```python\n{orig_code[:2400]}\n```")

    for idx, pa_info in enumerate(prior_attempts, start=1):
        lines.append(
            f"\n--- Prior Correction Retry #{idx} (ALSO INCORRECT — DO NOT REPEAT THIS HYPOTHESIS) ---\n"
            f"Retry #{idx} status: train_pass={pa_info.get('train_pass')}, train_frac={pa_info.get('train_frac')}, "
            f"stop_reason={pa_info.get('stop_reason')}, test_diag={pa_info.get('test_diag')}\n"
            f"Retry #{idx} code:\n```python\n{str(pa_info.get('code') or '')[:1600]}\n```"
        )

    lines.extend(
        [
            "\n=== YOUR TASK ===",
            "1. Carefully analyze why the previous reasoning and code failed (especially any hidden assumption that broke on `test_input` or failed on `train`).",
            "2. Use `run_python` to inspect `train` and `test_input` in detail, design a corrected general transformation `solve(g)`, and verify that `all(solve(p['input']) == p['output'] for p in train)` is True.",
            "3. Also inspect `solve(test_input)` in `run_python` (check its dimensions, background/foreground colors, connectivity, and edge cases) to make sure it generalizes cleanly to `test_input`.",
            "4. Once verified, call `submit_answer(code=\"solve(test_input)\")`.",
        ]
    )
    return "\n".join(lines)


def build_correction_initial_messages(
    task: dict[str, Any],
    test_input: list[list[int]],
    failure_rec: dict[str, Any],
    prior_attempts: list[dict[str, Any]],
    attempt_round: int,
    difficulty_info: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], int]:
    """Builds the initial `[system, user]` messages for Correction Attempt `attempt_round` (1..4)."""
    init_msgs, eff_val = ds_eval.build_havu_initial_messages(
        task=task,
        test_input=test_input,
        attempt_idx=attempt_round - 1,
        compact_grids=False,
    )
    base_user_prompt = str(init_msgs[1]["content"])
    feedback_block = format_failure_feedback_block(
        failure_rec=failure_rec,
        prior_attempts=prior_attempts,
        difficulty_info=difficulty_info,
    )
    init_msgs[1]["content"] = f"{base_user_prompt}\n\n{feedback_block}"
    return init_msgs, eff_val


def build_clean_rewrite_initial_messages(
    task: dict[str, Any],
    test_input: list[list[int]],
    verified_solve_code: str,
    verified_reasoning_summary: str,
) -> tuple[list[dict[str, Any]], int]:
    """Builds the teacher prompt asking DeepSeek to rewrite the verified solution cleanly without prior-attempt history."""
    init_msgs, eff_val = ds_eval.build_havu_initial_messages(
        task=task,
        test_input=test_input,
        attempt_idx=0,
        compact_grids=False,
    )
    base_user_prompt = str(init_msgs[1]["content"])
    rewrite_instructions = (
        "=== CLEAN SOLUTION REWRITE INSTRUCTIONS (INTERNAL — DO NOT MENTION IN YOUR OUTPUT) ===\n"
        "You have already discovered the exact, verified transformation rule for this puzzle. "
        "Below is the verified Python implementation that passes 100% of training and test pairs:\n"
        f"```python\n{verified_solve_code.strip()[:4000]}\n```\n"
        + (
            f"Key reasoning insight:\n{verified_reasoning_summary.strip()[:1200]}\n\n"
            if verified_reasoning_summary.strip()
            else "\n"
        )
        + "Now write a CLEAN, DIRECT, SELF-CONTAINED multi-turn solution trajectory from scratch as if solving the puzzle fresh on the first try:\n"
        "1. In Turn 1: Inside `<think>...</think>`, clearly analyze the training examples, state the exact visual/geometric transformation hypothesis, and then call `run_python` to define `solve(g)`, verify `all(solve(p['input']) == p['output'] for p in train)`, and inspect `solve(test_input)`.\n"
        "2. In Turn 2: After seeing the `run_python` verification output, briefly confirm in `<think>...</think>` that all training pairs pass and `solve(test_input)` is valid, and call `submit_answer(code=\"solve(test_input)\")` (or the exact verified submission expression).\n"
        "3. IMPORTANT: Do NOT mention any 'previous attempt', 'failed attempt', 'reference solution', or 'rewrite' anywhere in your `<think>` reasoning or tool calls."
    )
    init_msgs[1]["content"] = f"{base_user_prompt}\n\n{rewrite_instructions}"
    return init_msgs, eff_val


def evaluate_all_test_pairs_in_session(
    session: ds_eval.HavuSubprocessSession,
    pair0_pred: list[list[int]] | None,
    submit_code: str | None,
    test_pairs: list[dict[str, Any]],
) -> dict[str, Any]:
    """Evaluates whether the session's submitted rule produces the exact ground-truth output on ALL test pairs."""
    if not test_pairs:
        return {
            "correct_all": False,
            "preds_by_pair": [],
            "pair_correct": [],
            "test_diag": "no_test_pairs",
        }

    # Check pair 0 first
    gt0 = test_pairs[0].get("output")
    if pair0_pred is None or gt0 is None or pair0_pred != gt0:
        if pair0_pred is None:
            diag = "pair0_no_prediction"
        elif (len(pair0_pred), len(pair0_pred[0])) != (len(gt0), len(gt0[0])):
            diag = f"pair0_shape_mismatch_pred_{len(pair0_pred)}x{len(pair0_pred[0])}_vs_gt_{len(gt0)}x{len(gt0[0])}"
        else:
            mismatches = sum(
                1
                for r in range(len(gt0))
                for c in range(len(gt0[0]))
                if pair0_pred[r][c] != gt0[r][c]
            )
            diag = f"pair0_cell_mismatch_{mismatches}_cells"
        return {
            "correct_all": False,
            "preds_by_pair": [pair0_pred] + [None] * (len(test_pairs) - 1),
            "pair_correct": [False] * len(test_pairs),
            "test_diag": diag,
        }

    if len(test_pairs) == 1:
        return {
            "correct_all": True,
            "preds_by_pair": [pair0_pred],
            "pair_correct": [True],
            "test_diag": "all_1_test_pairs_exact_match",
        }

    # Multi-pair task: evaluate test pairs 1..N-1 in the IPython session using the same expression/function
    extra_inputs = [p["input"] for p in test_pairs[1:]]
    probe_script = (
        "import json as _json\n"
        f"_extra_test_inputs = {json.dumps(extra_inputs)}\n"
        f"_sub_code = {json.dumps(submit_code or '')}\n"
        "_outs = []\n"
        "for _ti in _extra_test_inputs:\n"
        "    _g = None\n"
        "    try:\n"
        "        _ns = dict(globals())\n"
        "        _ns['test_input'] = _ti\n"
        "        if _sub_code:\n"
        "            try:\n"
        "                _g = eval(_sub_code, _ns)\n"
        "            except SyntaxError:\n"
        "                exec(_sub_code, _ns)\n"
        "                _g = _ns.get('answer', _ns.get('output', _ns.get('result')))\n"
        "        if _g is None and 'solve' in globals() and callable(globals()['solve']):\n"
        "            _g = globals()['solve'](_ti)\n"
        "        if hasattr(_g, 'tolist'):\n"
        "            _g = _g.tolist()\n"
        "        _outs.append(_g)\n"
        "    except Exception:\n"
        "        _outs.append(None)\n"
        "print('__MULTI_TEST_PREDS__:' + _json.dumps(_outs))\n"
    )
    if getattr(session, "_pid", None) is not None:
        _, out_text = session.run_python(probe_script)
    else:
        tmp_sess = ds_eval.HavuSubprocessSession(
            train=getattr(session, "train", []),
            test_input=getattr(session, "test_input", test_pairs[0]["input"]),
            timeout_s=15.0,
            style="ipython",
        )
        try:
            for fn_name, cell_src in getattr(session, "executed", []):
                if fn_name == "run_python" and cell_src:
                    tmp_sess.run_python(cell_src)
            _, out_text = tmp_sess.run_python(probe_script)
        finally:
            tmp_sess.close()
    extra_preds: list[Any] = [None] * len(extra_inputs)
    for line in (out_text or "").splitlines():
        if line.startswith("__MULTI_TEST_PREDS__:"):
            try:
                parsed = json.loads(line.split("__MULTI_TEST_PREDS__:", 1)[1])
                if isinstance(parsed, list) and len(parsed) == len(extra_inputs):
                    extra_preds = parsed
            except Exception:  # pylint: disable=broad-exception-caught
                pass

    def _norm_grid(g: Any) -> list[list[int]] | None:
        if not isinstance(g, list) or not g:
            return None
        out_g: list[list[int]] = []
        w = None
        for r in g:
            if not isinstance(r, list) or not r:
                return None
            if w is None:
                w = len(r)
            elif len(r) != w:
                return None
            row_ints: list[int] = []
            for v in r:
                if isinstance(v, bool) or not isinstance(v, (int, float)):
                    return None
                iv = int(v)
                if iv != v:
                    return None
                row_ints.append(iv)
            out_g.append(row_ints)
        return out_g

    preds_by_pair: list[list[list[int]] | None] = [pair0_pred]
    pair_correct: list[bool] = [True]
    for idx, raw_p in enumerate(extra_preds, start=1):
        norm_p = _norm_grid(raw_p) if raw_p is not None else None
        gt_p = test_pairs[idx].get("output")
        is_ok = bool(norm_p is not None and gt_p is not None and norm_p == gt_p)
        preds_by_pair.append(norm_p)
        pair_correct.append(is_ok)

    all_ok = all(pair_correct)
    diag = (
        f"all_{len(test_pairs)}_test_pairs_exact_match"
        if all_ok
        else f"multi_pair_partial_{sum(pair_correct)}of{len(test_pairs)}"
    )
    return {
        "correct_all": all_ok,
        "preds_by_pair": preds_by_pair,
        "pair_correct": pair_correct,
        "test_diag": diag,
    }


def extract_self_contained_solve_code(messages: list[dict[str, Any]], fallback_code: str = "") -> str:
    """Extracts the self-contained Python code (defining `solve` or helpers) from a successful trajectory."""
    code_blocks: list[str] = []
    submit_expr = ""
    for m in messages:
        if m.get("role") != "assistant":
            continue
        for tc in m.get("tool_calls") or []:
            fn = tc.get("function") or {}
            name = fn.get("name", "")
            args_raw = fn.get("arguments", "")
            code_str = ""
            if isinstance(args_raw, dict):
                code_str = str(args_raw.get("code", ""))
            elif isinstance(args_raw, str):
                try:
                    parsed = json.loads(args_raw)
                    if isinstance(parsed, dict):
                        code_str = str(parsed.get("code", ""))
                except Exception:  # pylint: disable=broad-exception-caught
                    code_str = args_raw
            if name in ("run_python", "python_exec") and code_str.strip():
                code_blocks.append(code_str.strip())
            elif name == "submit_answer" and code_str.strip():
                submit_expr = code_str.strip()

    # Prefer the last code block that defines `def solve`
    for blk in reversed(code_blocks):
        if "def solve" in blk:
            return blk
    if submit_expr and "def " in submit_expr:
        return submit_expr
    if code_blocks:
        return code_blocks[-1]
    return fallback_code or submit_expr


def has_forbidden_meta_language(messages: list[dict[str, Any]]) -> bool:
    """Checks whether assistant reasoning/content mentions prior failed attempts or teacher hints."""
    for m in messages:
        if m.get("role") != "assistant":
            continue
        txt = (
            str(m.get("reasoning_content") or "") + "\n" + str(m.get("content") or "")
        ).lower()
        for phrase in FORBIDDEN_META_PHRASES:
            if phrase in txt:
                return True
    return False


def scrub_meta_phrases(text: str) -> str:
    """Removes any residual meta-references from reasoning text if fallback synthesis is used."""
    out_lines = []
    for line in text.splitlines():
        low = line.lower()
        if any(p in low for p in FORBIDDEN_META_PHRASES):
            continue
        out_lines.append(line)
    cleaned = "\n".join(out_lines).strip()
    return cleaned if cleaned else "Let's analyze the input-output grids in `train`, determine the transformation rule, verify it on all training examples, and apply it to `test_input`."


def synthesize_verified_clean_trajectory_in_sandbox(
    train_pairs: list[dict[str, Any]],
    test_input: list[list[int]],
    verified_solve_code: str,
    submit_code: str,
    reasoning_summary: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[list[int]] | None, dict[str, Any]]:
    """Executes the verified solution in a fresh IPython sandbox to guarantee a clean 2-turn prefix-free trajectory if needed."""
    session = ds_eval.HavuSubprocessSession(
        train=train_pairs,
        test_input=test_input,
        timeout_s=15.0,
        style="ipython",
    )
    try:
        clean_think_1 = scrub_meta_phrases(reasoning_summary)
        verify_code = verified_solve_code.strip()
        if "for " not in verify_code and "train" not in verify_code and "def solve" in verify_code:
            verify_code = (
                f"{verify_code}\n\n"
                "for idx, p in enumerate(train):\n"
                "    pred = solve(p['input'])\n"
                "    assert pred == p['output'], f'Mismatch on train {idx}'\n"
                "print('All', len(train), 'training pairs passed!')"
            )
        _, tool_out_1 = session.run_python(verify_code)
        sub_expr = submit_code.strip() if submit_code.strip() else "solve(test_input)"
        tool_out_2, grid = session.submit_answer(sub_expr)
        train_labels = top10k.run_3level_session_check(
            session=session,
            pred=grid,
            code=sub_expr,
            extra_pairs=[],
        )
    finally:
        session.close()

    asst_and_tool_messages = [
        {
            "role": "assistant",
            "content": "",
            "reasoning_content": clean_think_1,
            "tool_calls": [
                {
                    "id": "call_1_0",
                    "type": "function",
                    "function": {
                        "name": "run_python",
                        "arguments": json.dumps({"code": verify_code}),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_1_0",
            "content": tool_out_1,
        },
        {
            "role": "assistant",
            "content": "",
            "reasoning_content": (
                "The transformation function `solve` passes all training pairs cleanly. "
                "Now let's submit `solve(test_input)`."
            ),
            "tool_calls": [
                {
                    "id": "call_2_0",
                    "type": "function",
                    "function": {
                        "name": "submit_answer",
                        "arguments": json.dumps({"code": sub_expr}),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_2_0",
            "content": tool_out_2,
        },
    ]
    events = [
        {
            "turn": 1,
            "output_tokens": max(32, len(clean_think_1) // 4 + len(verify_code) // 4),
            "thinking_tokens": max(16, len(clean_think_1) // 4),
            "tool_use_tokens": max(16, len(verify_code) // 4),
            "answer_tokens": 0,
            "finish_reason": "stop",
            "num_tool_calls": 1,
            "is_canonical": True,
        },
        {
            "turn": 2,
            "output_tokens": 48,
            "thinking_tokens": 24,
            "tool_use_tokens": 24,
            "answer_tokens": 0,
            "finish_reason": "stop",
            "num_tool_calls": 1,
            "is_canonical": True,
        },
    ]
    return asst_and_tool_messages, events, grid, train_labels


def assemble_e065_rewritten_parquet_rows(
    orig_group_rows: list[dict[str, Any]],
    clean_asst_and_tool_messages: list[dict[str, Any]],
    clean_events: list[dict[str, Any]],
    preds_by_pair: list[list[list[int]] | None],
    train_labels: dict[str, Any],
    solved_on_attempt: int,
    rewrite_source: str,
    elapsed_s: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Assembles both `e065_images` (preserving inline base64 PNGs in `messages[1]`) and `e065_text` rows."""
    r0 = orig_group_rows[0]
    orig_msgs = json.loads(r0["messages"]) if isinstance(r0["messages"], str) else list(r0["messages"])

    # Preserve exact original E065 system + user prefix (including base64 PNG images in messages[1])
    img_prefix = copy.deepcopy(orig_msgs[:2])
    img_full_messages = img_prefix + copy.deepcopy(clean_asst_and_tool_messages)
    img_messages_json = json.dumps(img_full_messages)

    # Also build text-only twin messages (stripping image_url parts from messages[1])
    txt_prefix = copy.deepcopy(orig_msgs[:2])
    if len(txt_prefix) > 1 and isinstance(txt_prefix[1].get("content"), list):
        text_parts = [
            str(p.get("text", ""))
            for p in txt_prefix[1]["content"]
            if isinstance(p, dict) and p.get("type") == "text"
        ]
        txt_prefix[1]["content"] = "\n".join(text_parts)
    txt_full_messages = txt_prefix + copy.deepcopy(clean_asst_and_tool_messages)
    txt_messages_json = json.dumps(txt_full_messages)

    events_json = json.dumps(clean_events)
    n_turns = sum(1 for m in clean_asst_and_tool_messages if m.get("role") == "assistant")
    n_tool_calls = sum(
        len(m.get("tool_calls") or [])
        for m in clean_asst_and_tool_messages
        if m.get("role") == "assistant"
    )
    comp_toks = sum(int(ev.get("output_tokens") or 0) for ev in clean_events)
    reas_toks = sum(int(ev.get("thinking_tokens") or 0) for ev in clean_events)
    tool_toks = sum(int(ev.get("tool_use_tokens") or 0) for ev in clean_events)

    img_out_rows: list[dict[str, Any]] = []
    txt_out_rows: list[dict[str, Any]] = []

    for idx, orig_row in enumerate(orig_group_rows):
        p_idx = int(orig_row.get("pair") or idx)
        pred_grid = preds_by_pair[p_idx] if p_idx < len(preds_by_pair) else preds_by_pair[0]
        pred_json = json.dumps(pred_grid) if pred_grid is not None else None

        base_row = copy.deepcopy(orig_row)
        base_row["correct"] = True
        base_row["pred"] = pred_json
        base_row["answer_source"] = "submit"
        base_row["stop_reason"] = "submitted"
        base_row["turns"] = n_turns if p_idx == 0 else 0
        base_row["tool_calls"] = n_tool_calls if p_idx == 0 else 0
        base_row["completion_tokens"] = comp_toks if p_idx == 0 else 0
        base_row["reasoning_tokens"] = reas_toks if p_idx == 0 else 0
        base_row["tool_use_tokens"] = tool_toks if p_idx == 0 else 0
        base_row["answer_tokens"] = 0
        base_row["elapsed_s"] = round(elapsed_s, 2) if p_idx == 0 else 0.0
        base_row["events"] = events_json if p_idx == 0 else "[]"
        base_row["error"] = None
        base_row["train_pass"] = True
        base_row["train_frac"] = 1.0
        base_row["train_trace"] = str(train_labels.get("train_trace") or "expr")
        base_row["train_paths"] = json.dumps(train_labels.get("train_paths") or ["expr"])
        base_row["verified_code"] = str(train_labels.get("verified_code") or "solve(test_input)")
        base_row["train_check_s"] = float(train_labels.get("train_check_seconds") or 0.05)

        img_row = copy.deepcopy(base_row)
        img_row["messages"] = img_messages_json
        img_out_rows.append(img_row)

        txt_row = copy.deepcopy(base_row)
        txt_row["messages"] = txt_messages_json
        txt_out_rows.append(txt_row)

    return img_out_rows, txt_out_rows


def _write_parquet_rows_to_cns(rows: list[dict[str, Any]], cns_path: str) -> bool:
    if not rows:
        return False
    tbl = pa.Table.from_pylist(rows)
    buf = io.BytesIO()
    pq.write_table(tbl, buf, compression="zstd")
    tmp_path = f"{cns_path}.tmp.{os.getpid()}"
    with gfile.GFile(tmp_path, "wb") as f:
        f.write(buf.getvalue())
    gfile.Rename(tmp_path, cns_path, overwrite=True)
    return True


def load_failed_e065_work_items(
    non_test_passing_parquet_path: str,
    non_test_passing_tasks_jsonl_path: str,
    difficulty_rankings_jsonl_path: str = "",
) -> list[dict[str, Any]]:
    """Loads uncontaminated non-test-passing E065 rollouts paired with their original parquet rows and GT grids."""
    # 1. Load original parquet rows grouped by task_id
    with gfile.GFile(non_test_passing_parquet_path, "rb") as f:
        raw_pq = f.read()
    table = pq.read_table(io.BytesIO(raw_pq))
    orig_rows_by_task: dict[str, list[dict[str, Any]]] = {}
    for row in table.to_pylist():
        tid = str(row["task_id"])
        orig_rows_by_task.setdefault(tid, []).append(row)
    for tid in orig_rows_by_task:
        orig_rows_by_task[tid].sort(key=lambda r: int(r.get("pair") or 0))

    # 2. Load optional Gemini Flash 3.8 difficulty rankings (including live checkpoint)
    diff_by_task: dict[str, dict[str, Any]] = {}
    cand_paths: list[str] = []
    if difficulty_rankings_jsonl_path:
        ckpt_sibling = os.path.join(
            os.path.dirname(difficulty_rankings_jsonl_path),
            "gemini38_flash_checkpoint.jsonl",
        )
        cand_paths.extend([ckpt_sibling, difficulty_rankings_jsonl_path])
    for dpath in cand_paths:
        if dpath and gfile.Exists(dpath):
            with gfile.GFile(dpath, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        tid = str(obj.get("task_id", ""))
                        if tid:
                            diff_by_task[tid] = obj
                    except Exception:  # pylint: disable=broad-exception-caught
                        continue

    # 3. Load non_test_passing_tasks.jsonl and filter out contaminated tasks
    work_items: list[dict[str, Any]] = []
    with gfile.GFile(non_test_passing_tasks_jsonl_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("is_contaminated"):
                continue
            tid = str(rec["task_id"])
            orig_rows = orig_rows_by_task.get(tid, [])
            if not orig_rows or not rec.get("train") or not rec.get("test"):
                continue
            diff_info = diff_by_task.get(tid)
            f_cat = str(rec.get("failure_category", ""))
            cat_rank = 0 if f_cat == "demo_overfit_train_pass_test_fail" else 1
            comp_toks = int(rec.get("completion_tokens") or 0)
            work_items.append(
                {
                    "task_id": tid,
                    "failure_rec": rec,
                    "orig_group_rows": orig_rows,
                    "difficulty_info": diff_info,
                    "hardness_sort_key": (cat_rank, comp_toks, tid),
                }
            )

    # Sort deterministically by task_id for invariant sharding across workers
    work_items.sort(key=lambda w: w["task_id"])
    return work_items


def main(argv: list[str]) -> None:
    if len(argv) > 1:
        raise RuntimeError(f"Unrecognized positional arguments: {argv[1:]}")

    from google3.experimental.users.arazdai.kaggle.arc_agi_2.pilot_1.setB.tpu_vllm import deepseek_v41_tpu_vllm_traces as ds_traces
    from google3.experimental.users.arazdai.kaggle.sft import export_and_quantize
    import jax
    import tqdm

    num_tpus = jax.device_count()
    device_kind = jax.devices()[0].device_kind if num_tpus > 0 else "unknown"
    if num_tpus not in (8, 16):
        raise RuntimeError(
            f"Expected 8 or 16 TensorCores for ds41_full_precision, got num_tpus={num_tpus} ({device_kind})"
        )
    shard_idx = max(0, int(FLAGS.shard_idx))
    num_shards = max(1, int(FLAGS.num_shards))
    shard_tag = f"shard_{shard_idx:03d}of{num_shards:03d}"

    out_dir = FLAGS.e065_output_dir
    shards_dir = os.path.join(out_dir, "shards")
    for d in (out_dir, shards_dir):
        export_and_quantize.ensure_cns_quota_dir(d, group="deepmind-resources-colossus")

    all_items = load_failed_e065_work_items(
        non_test_passing_parquet_path=FLAGS.non_test_passing_parquet_path,
        non_test_passing_tasks_jsonl_path=FLAGS.non_test_passing_tasks_jsonl_path,
        difficulty_rankings_jsonl_path=FLAGS.difficulty_rankings_jsonl_path,
    )
    shard_items = all_items[shard_idx::num_shards]
    # Within each shard, process demo-overfit tasks first (ordered by completion_tokens asc)
    shard_items.sort(key=lambda w: (w["hardness_sort_key"], w["task_id"]))
    if FLAGS.limit_tasks > 0:
        shard_items = shard_items[: FLAGS.limit_tasks]

    diag_jsonl_path = os.path.join(shards_dir, f"correction_diagnostics_{shard_tag}.jsonl")
    img_jsonl_path = os.path.join(shards_dir, f"e065_images_corrected_rewritten_{shard_tag}.jsonl")
    txt_jsonl_path = os.path.join(shards_dir, f"e065_text_corrected_rewritten_{shard_tag}.jsonl")
    img_parquet_path = os.path.join(
        shards_dir, f"e065_images_corrected_rewritten_{shard_tag}.parquet"
    )
    txt_parquet_path = os.path.join(
        shards_dir, f"e065_text_corrected_rewritten_{shard_tag}.parquet"
    )
    summary_shard_path = os.path.join(shards_dir, f"summary_{shard_tag}.json")

    # Resume from existing shard files if present
    done_task_ids: set[str] = set()
    diag_records: list[dict[str, Any]] = []
    img_rewritten_rows: list[dict[str, Any]] = []
    txt_rewritten_rows: list[dict[str, Any]] = []

    if gfile.Exists(diag_jsonl_path):
        with gfile.GFile(diag_jsonl_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    tid = str(obj.get("task_id", ""))
                    if tid and tid not in done_task_ids:
                        done_task_ids.add(tid)
                        diag_records.append(obj)
                except Exception:  # pylint: disable=broad-exception-caught
                    continue

    if gfile.Exists(img_jsonl_path):
        with gfile.GFile(img_jsonl_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    img_rewritten_rows.append(json.loads(line))

    if gfile.Exists(txt_jsonl_path):
        with gfile.GFile(txt_jsonl_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    txt_rewritten_rows.append(json.loads(line))

    pending_items = [w for w in shard_items if w["task_id"] not in done_task_ids]

    wave_bsz = max(2, int(FLAGS.pair_batch_size))
    vllm_mod: Any = ds_traces.CompatVllmNamespace
    llm = ds_eval.NativeDeepSeekV41FullPrecisionLLM(
        raw_checkpoint_dir=FLAGS.ds41_raw_checkpoint_dir,
        ocdbt_tp8_dir=FLAGS.ds41_ocdbt_tp8_dir,
        ocdbt_tp16_dir=FLAGS.ds41_ocdbt_tp16_dir,
        max_seq_len=FLAGS.max_model_len,
        batch_size=wave_bsz,
        steps_per_call=12,
        temperature=FLAGS.temperature,
        top_k=FLAGS.top_k,
    )

    print("=" * 88, flush=True)
    print(
        "DeepSeek-V4.1-Flash-1017 Full Model (384 Experts) E065 Iterative Correction (<=4x) + Clean Rewrite",
        flush=True,
    )
    print(f"  TPU Hardware         : {num_tpus}x {device_kind} (wave_bsz={wave_bsz})", flush=True)
    print(
        f"  Shard                : {shard_idx + 1}/{num_shards} "
        f"(total={len(shard_items)}, resumed={len(done_task_ids)}, pending={len(pending_items)})",
        flush=True,
    )
    print(
        f"  Max Correction Tries : {FLAGS.max_correction_attempts} | Max Clean Rewrite Tries: {FLAGS.max_rewrite_attempts}",
        flush=True,
    )
    print(f"  Output Directory     : {out_dir}", flush=True)
    print("=" * 88, flush=True)

    if not pending_items:
        _write_parquet_rows_to_cns(img_rewritten_rows, img_parquet_path)
        _write_parquet_rows_to_cns(txt_rewritten_rows, txt_parquet_path)
        return

    start_wall = time.time()
    cum_gen_tokens = 0
    cum_gen_seconds = 0.0
    num_batches = (len(pending_items) + wave_bsz - 1) // wave_bsz

    for b_idx in range(num_batches):
        batch_slice = pending_items[b_idx * wave_bsz : (b_idx + 1) * wave_bsz]
        prior_attempts_by_task: dict[str, list[dict[str, Any]]] = {
            w["task_id"]: [] for w in batch_slice
        }
        solved_by_task: dict[str, dict[str, Any]] = {}
        task_wall_start = time.time()

        # ---- STAGE 1: Up to `max_correction_attempts` (4) rounds of reasoning correction ----
        for corr_round in range(1, int(FLAGS.max_correction_attempts) + 1):
            unsolved_slice = [w for w in batch_slice if w["task_id"] not in solved_by_task]
            if not unsolved_slice:
                break

            corr_runtimes: list[top10k.Top10kRolloutRuntime] = []
            for w in unsolved_slice:
                tid = w["task_id"]
                f_rec = w["failure_rec"]
                train_pairs = f_rec["train"]
                test_pairs = f_rec["test"]
                test_input_0 = test_pairs[0]["input"]
                task_dict = {
                    "task_id": tid,
                    "train": train_pairs,
                    "test": test_pairs,
                }
                init_msgs, eff_val = build_correction_initial_messages(
                    task=task_dict,
                    test_input=test_input_0,
                    failure_rec=f_rec,
                    prior_attempts=prior_attempts_by_task[tid],
                    attempt_round=corr_round,
                    difficulty_info=w.get("difficulty_info"),
                )
                st = top10k.RolloutAttemptState(
                    task_id=tid,
                    pair_idx=0,
                    attempt_idx=corr_round - 1,
                    soft_target=FLAGS.soft_token_target,
                    hard_budget=FLAGS.total_token_budget,
                )
                session = ds_eval.HavuSubprocessSession(
                    train=train_pairs,
                    test_input=test_input_0,
                    timeout_s=float(FLAGS.sandbox_timeout_s),
                    max_output_chars=16000,
                    style="ipython",
                    mem_mb=4096,
                    grace_s=5.0,
                )
                seed_base = int(
                    hashlib.sha1(f"{tid}_corr_{corr_round}".encode()).hexdigest()[:8], 16
                )
                rt = top10k.Top10kRolloutRuntime(
                    state=st,
                    messages=init_msgs,
                    reasoning_effort=eff_val,
                    session=session,
                    base_seed=seed_base,
                    candidate_meta=task_dict,
                    extra_verification_pairs=test_pairs[1:],
                )
                corr_runtimes.append(rt)

            w_toks, w_sec = top10k.run_top10k_rollout_wave_batched(
                llm=llm,
                vllm_mod=vllm_mod,
                runtimes=corr_runtimes,
                wave_desc=f"Batch {b_idx + 1}/{num_batches} CorrRound {corr_round}/{FLAGS.max_correction_attempts}",
            )
            cum_gen_tokens += w_toks
            cum_gen_seconds += w_sec

            for w, rt in zip(unsolved_slice, corr_runtimes):
                tid = w["task_id"]
                train_pairs = w["failure_rec"]["train"]
                test_pairs = w["failure_rec"]["test"]
                try:
                    # Note: `run_top10k_rollout_wave_batched` already populated `rt.train_labels`
                    # while `rt.session` was open and then closed `rt.session`.
                    if not rt.train_labels:
                        if rt.accepted_grid is not None and getattr(rt.session, "_pid", None) is not None:
                            rt.train_labels = top10k.run_3level_session_check(
                                session=rt.session,
                                pred=rt.accepted_grid,
                                code=rt.submit_code,
                                extra_pairs=test_pairs[1:],
                            )
                        else:
                            rt.train_labels = {"train_pass": False, "train_frac": 0.0}

                    multi_eval = evaluate_all_test_pairs_in_session(
                        session=rt.session,
                        pair0_pred=rt.accepted_grid,
                        submit_code=rt.submit_code,
                        test_pairs=test_pairs,
                    )
                finally:
                    rt.session.close()

                solve_code = extract_self_contained_solve_code(
                    rt.uncompacted_messages, fallback_code=str(rt.submit_code or "")
                )
                train_ok = bool(rt.train_labels.get("train_pass") is True)
                test_ok = bool(multi_eval["correct_all"])

                # If the model got the exact test answer (`test_ok=True`) and defined `def solve` in an
                # earlier cell, but submitted a precomputed variable (`answer`) that wasn't traced by
                # `run_3level_session_check`, verify `solve_code` directly on `train_pairs` in a fresh sandbox.
                if test_ok and not train_ok and solve_code:
                    chk_sess = ds_eval.HavuSubprocessSession(
                        train=train_pairs,
                        test_input=test_pairs[0]["input"],
                        timeout_s=15.0,
                        style="ipython",
                    )
                    try:
                        for fn_name, cell_src in getattr(rt.session, "executed", []):
                            if fn_name == "run_python" and cell_src:
                                chk_sess.run_python(cell_src)
                        sub_check = str(rt.submit_code or "solve(test_input)")
                        if "test_input" not in sub_check and "def solve" in solve_code:
                            sub_check = "solve(test_input)"
                        recheck = top10k.run_3level_session_check(
                            session=chk_sess,
                            pred=rt.accepted_grid,
                            code=sub_check,
                            extra_pairs=test_pairs[1:],
                        )
                        if recheck.get("train_pass") is True:
                            rt.train_labels = recheck
                            train_ok = True
                            if "test_input" in sub_check:
                                rt.submit_code = sub_check
                    finally:
                        chk_sess.close()

                submitted_ok = bool(
                    rt.answer_source in ("submit", "session", "session_solve")
                    or (rt.answer_source or "").startswith("salvage:")
                )

                if train_ok and test_ok and submitted_ok:
                    solved_by_task[tid] = {
                        "solved_on_attempt": corr_round,
                        "verified_solve_code": solve_code,
                        "submit_code": str(rt.submit_code or "solve(test_input)"),
                        "verified_reasoning_summary": rt.state.thought_text[-2000:],
                        "preds_by_pair": multi_eval["preds_by_pair"],
                        "train_labels": rt.train_labels,
                    }
                else:
                    prior_attempts_by_task[tid].append(
                        {
                            "attempt_round": corr_round,
                            "train_pass": train_ok,
                            "train_frac": rt.train_labels.get("train_frac"),
                            "correct_all": test_ok,
                            "stop_reason": rt.stop_reason,
                            "test_diag": multi_eval["test_diag"],
                            "code": solve_code,
                        }
                    )

        # ---- STAGE 2: Clean Solution Rewrite for all tasks solved in Stage 1 ----
        solved_slice = [w for w in batch_slice if w["task_id"] in solved_by_task]
        rewritten_by_task: dict[str, dict[str, Any]] = {}

        for rw_try in range(1, int(FLAGS.max_rewrite_attempts) + 1):
            need_rw = [w for w in solved_slice if w["task_id"] not in rewritten_by_task]
            if not need_rw:
                break

            rw_runtimes: list[top10k.Top10kRolloutRuntime] = []
            for w in need_rw:
                tid = w["task_id"]
                s_info = solved_by_task[tid]
                f_rec = w["failure_rec"]
                train_pairs = f_rec["train"]
                test_pairs = f_rec["test"]
                test_input_0 = test_pairs[0]["input"]
                task_dict = {"task_id": tid, "train": train_pairs, "test": test_pairs}

                rw_msgs, eff_val = build_clean_rewrite_initial_messages(
                    task=task_dict,
                    test_input=test_input_0,
                    verified_solve_code=s_info["verified_solve_code"],
                    verified_reasoning_summary=s_info["verified_reasoning_summary"],
                )
                st = top10k.RolloutAttemptState(
                    task_id=tid,
                    pair_idx=0,
                    attempt_idx=rw_try - 1,
                    soft_target=FLAGS.soft_token_target,
                    hard_budget=FLAGS.total_token_budget,
                )
                session = ds_eval.HavuSubprocessSession(
                    train=train_pairs,
                    test_input=test_input_0,
                    timeout_s=float(FLAGS.sandbox_timeout_s),
                    max_output_chars=16000,
                    style="ipython",
                    mem_mb=4096,
                    grace_s=5.0,
                )
                seed_base = int(
                    hashlib.sha1(f"{tid}_rewrite_{rw_try}".encode()).hexdigest()[:8], 16
                )
                rt = top10k.Top10kRolloutRuntime(
                    state=st,
                    messages=rw_msgs,
                    reasoning_effort=eff_val,
                    session=session,
                    base_seed=seed_base,
                    candidate_meta=task_dict,
                    extra_verification_pairs=test_pairs[1:],
                )
                rw_runtimes.append(rt)

            w_toks, w_sec = top10k.run_top10k_rollout_wave_batched(
                llm=llm,
                vllm_mod=vllm_mod,
                runtimes=rw_runtimes,
                wave_desc=f"Batch {b_idx + 1}/{num_batches} CleanRewrite {rw_try}/{FLAGS.max_rewrite_attempts}",
            )
            cum_gen_tokens += w_toks
            cum_gen_seconds += w_sec

            for w, rt in zip(need_rw, rw_runtimes):
                tid = w["task_id"]
                test_pairs = w["failure_rec"]["test"]
                try:
                    if not rt.train_labels:
                        if rt.accepted_grid is not None and getattr(rt.session, "_pid", None) is not None:
                            rt.train_labels = top10k.run_3level_session_check(
                                session=rt.session,
                                pred=rt.accepted_grid,
                                code=rt.submit_code,
                                extra_pairs=test_pairs[1:],
                            )
                        else:
                            rt.train_labels = {"train_pass": False, "train_frac": 0.0}

                    multi_eval = evaluate_all_test_pairs_in_session(
                        session=rt.session,
                        pair0_pred=rt.accepted_grid,
                        submit_code=rt.submit_code,
                        test_pairs=test_pairs,
                    )
                finally:
                    rt.session.close()

                asst_and_tool_msgs = rt.uncompacted_messages[2:]
                rw_ok = (
                    bool(rt.train_labels.get("train_pass") is True)
                    and bool(multi_eval["correct_all"])
                    and rt.answer_source == "submit"
                    and rt.stop_reason == "submitted"
                    and rt.code_rescues == 0
                    and rt.cap_nudges == 0
                    and not has_forbidden_meta_language(asst_and_tool_msgs)
                )
                if rw_ok:
                    rewritten_by_task[tid] = {
                        "clean_asst_and_tool_messages": asst_and_tool_msgs,
                        "clean_events": rt.events,
                        "preds_by_pair": multi_eval["preds_by_pair"],
                        "train_labels": rt.train_labels,
                        "rewrite_source": f"model_clean_rewrite_try_{rw_try}",
                        "elapsed_s": rt.elapsed_seconds,
                    }

        # Fallback sandbox-verified clean synthesis for any solved task whose rewrite try had meta text / nudge
        for w in solved_slice:
            tid = w["task_id"]
            if tid in rewritten_by_task:
                continue
            s_info = solved_by_task[tid]
            f_rec = w["failure_rec"]
            asst_msgs, evs, _, tr_labels = synthesize_verified_clean_trajectory_in_sandbox(
                train_pairs=f_rec["train"],
                test_input=f_rec["test"][0]["input"],
                verified_solve_code=s_info["verified_solve_code"],
                submit_code=s_info["submit_code"],
                reasoning_summary=s_info["verified_reasoning_summary"],
            )
            rewritten_by_task[tid] = {
                "clean_asst_and_tool_messages": asst_msgs,
                "clean_events": evs,
                "preds_by_pair": s_info["preds_by_pair"],
                "train_labels": tr_labels if tr_labels.get("train_pass") else s_info["train_labels"],
                "rewrite_source": "sandbox_verified_clean_synthesis",
                "elapsed_s": round(time.time() - task_wall_start, 2),
            }

        # ---- STAGE 3: Persist new corrected & rewritten rows + diagnostics to CNS ----
        new_diag: list[dict[str, Any]] = []
        new_img_rows: list[dict[str, Any]] = []
        new_txt_rows: list[dict[str, Any]] = []

        for w in batch_slice:
            tid = w["task_id"]
            done_task_ids.add(tid)
            if tid in solved_by_task and tid in rewritten_by_task:
                s_info = solved_by_task[tid]
                rw_info = rewritten_by_task[tid]
                img_rows, txt_rows = assemble_e065_rewritten_parquet_rows(
                    orig_group_rows=w["orig_group_rows"],
                    clean_asst_and_tool_messages=rw_info["clean_asst_and_tool_messages"],
                    clean_events=rw_info["clean_events"],
                    preds_by_pair=rw_info["preds_by_pair"],
                    train_labels=rw_info["train_labels"],
                    solved_on_attempt=s_info["solved_on_attempt"],
                    rewrite_source=rw_info["rewrite_source"],
                    elapsed_s=rw_info["elapsed_s"],
                )
                img_rewritten_rows.extend(img_rows)
                txt_rewritten_rows.extend(txt_rows)
                new_img_rows.extend(img_rows)
                new_txt_rows.extend(txt_rows)
                diag_obj = {
                    "task_id": tid,
                    "source": w["failure_rec"].get("source"),
                    "orig_failure_category": w["failure_rec"].get("failure_category"),
                    "solved": True,
                    "solved_on_attempt": s_info["solved_on_attempt"],
                    "rewrite_source": rw_info["rewrite_source"],
                    "rewritten_turns": sum(
                        1
                        for m in rw_info["clean_asst_and_tool_messages"]
                        if m.get("role") == "assistant"
                    ),
                    "prior_failed_attempts_count": len(prior_attempts_by_task[tid]),
                }
            else:
                diag_obj = {
                    "task_id": tid,
                    "source": w["failure_rec"].get("source"),
                    "orig_failure_category": w["failure_rec"].get("failure_category"),
                    "solved": False,
                    "solved_on_attempt": None,
                    "rewrite_source": None,
                    "rewritten_turns": 0,
                    "prior_failed_attempts_count": len(prior_attempts_by_task[tid]),
                    "failed_attempts": prior_attempts_by_task[tid],
                }
            diag_records.append(diag_obj)
            new_diag.append(diag_obj)

        with gfile.GFile(diag_jsonl_path, "a") as f:
            for d_obj in new_diag:
                f.write(json.dumps(d_obj) + "\n")
        if new_img_rows:
            with gfile.GFile(img_jsonl_path, "a") as f:
                for r_obj in new_img_rows:
                    f.write(json.dumps(r_obj) + "\n")
        if new_txt_rows:
            with gfile.GFile(txt_jsonl_path, "a") as f:
                for r_obj in new_txt_rows:
                    f.write(json.dumps(r_obj) + "\n")

        _write_parquet_rows_to_cns(img_rewritten_rows, img_parquet_path)
        _write_parquet_rows_to_cns(txt_rewritten_rows, txt_parquet_path)

        n_done = len(diag_records)
        n_solved = sum(1 for d in diag_records if d.get("solved"))
        by_attempt: dict[str, int] = {}
        for d in diag_records:
            if d.get("solved"):
                k_str = f"attempt_{d['solved_on_attempt']}"
                by_attempt[k_str] = by_attempt.get(k_str, 0) + 1

        summary_payload = {
            "shard_index": shard_idx,
            "num_shards": num_shards,
            "total_shard_tasks": len(shard_items),
            "completed_tasks": n_done,
            "solved_and_rewritten_tasks": n_solved,
            "solve_rate": round(n_solved / max(1, n_done), 4),
            "solved_by_attempt": by_attempt,
            "rewritten_image_parquet_rows": len(img_rewritten_rows),
            "rewritten_text_parquet_rows": len(txt_rewritten_rows),
            "decode_tokens_generated": cum_gen_tokens,
            "avg_decode_tok_per_s": round(cum_gen_tokens / max(cum_gen_seconds, 1e-6), 1),
            "elapsed_wall_seconds": round(time.time() - start_wall, 1),
            "img_parquet_path": img_parquet_path,
            "txt_parquet_path": txt_parquet_path,
            "diag_jsonl_path": diag_jsonl_path,
        }
        with gfile.GFile(summary_shard_path, "w") as sf:
            json.dump(summary_payload, sf, indent=2)
        print(
            f"[Shard {shard_idx:03d}/{num_shards:03d}] Batch {b_idx + 1}/{num_batches}: "
            f"solved={n_solved}/{n_done} ({100.0 * n_solved / max(1, n_done):.1f}%), "
            f"by_attempt={by_attempt}",
            flush=True,
        )


if __name__ == "__main__":
    from absl import app as app_runner

    app_runner.run(main)

