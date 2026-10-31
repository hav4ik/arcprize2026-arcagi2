"""Audits E065 DeepSeek-V4.1-Flash-1017 rollouts against ground-truth solutions and splits test-passing vs non-test-passing rollouts on CNS.

Specifically:
1. Loads all ground-truth tasks and test output solutions for `arc2_train`, `nvarc_b0`, and `nvarc_b1`.
2. Independently double-checks every rollout's predicted grid (`pred`) on every test pair against the ground-truth test output grid (`exact_grid_match`), cross-checking against the parquet `correct` column, `train_pass`, `answer_source`, `stop_reason`, `error`, and eval contamination (`2753e76c`, `81c0276b`, and the 120 ARC-AGI-2 evaluation tasks).
3. Saves separately to CNS (with `deepmind-resources-colossus` quota group):
   - `test_passing/e065_images_test_passing.parquet` (all test-passing image rollouts)
   - `test_passing/e065_images_test_passing_uncontaminated.parquet` (uncontaminated test-passing image rollouts)
   - `test_passing/e065_images_verified_3gate.parquet` (`correct_all & train_pass & submit` uncontaminated image rollouts)
   - Per-split test-passing parquets (`e065_images_{arc2_train,nvarc_b0,nvarc_b1}_test_passing.parquet`)
   - `test_passing/e065_text_test_passing.parquet` and `test_passing/e065_text_rescued_for_failed_images.parquet`
   - `non_test_passing/e065_images_non_test_passing.parquet` (all non-test-passing image rollouts, preserving original `messages` with inline base64 PNG images)
   - Per-split non-test-passing parquets (`e065_images_{arc2_train,nvarc_b0,nvarc_b1}_non_test_passing.parquet`)
   - `non_test_passing/e065_images_non_test_passing_tasks.jsonl` (self-contained task + failure diagnostics + full `train`/`test` grids for full DeepSeek iterative correction & clean rewrite)
   - `all_rollouts_telemetry.jsonl` (compact telemetry + reasoning/code summary for all 5,994 tasks for Gemini Flash 3.8 difficulty ranking)
   - `split_audit_summary.json`
"""

from __future__ import annotations

from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import io
import json
import os
import time
from typing import Any

from absl import app
from absl import flags
import pyarrow as pa
import pyarrow.parquet as pq

from google3.experimental.users.arazdai.kaggle.sft import data_pipeline
from google3.experimental.users.arazdai.kaggle.sft import export_and_quantize
from google3.pyglib import gfile

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "e065_root_dir",
    "/cns/lu-d/home/arazdai/kaggle/arc_sft_data/livobereshka_e065_rollouts",
    "CNS root directory of livobereshka/arc-agi2-deepseek-flash1017-rollouts-e065.",
)
flags.DEFINE_string(
    "e065_arc2_train_dir",
    "/cns/lu-d/home/arazdai/kaggle/ARC-AGI-2/data/training",
    "CNS directory containing the 1,000 official ARC-AGI-2 training JSON tasks.",
)
flags.DEFINE_string(
    "e065_eval_dir",
    "/cns/lu-d/home/arazdai/kaggle/ARC-AGI-2/data/evaluation",
    "CNS directory containing the 120 official ARC-AGI-2 evaluation JSON tasks.",
)
flags.DEFINE_string(
    "e065_split_output_cns_dir",
    "/cns/lu-d/home/arazdai/kaggle/arc_sft_data/e065_hardest_rollouts",
    "CNS directory where split test-passing and non-test-passing artifacts are saved.",
)
flags.DEFINE_string(
    "e065_cns_quota_group",
    "deepmind-resources-colossus",
    "CNS capacity_quota_user group to set on created CNS directories.",
)


E065_IMAGE_FILES = (
    "e065_images_arc2_train.parquet",
    "e065_images_nvarc_b0.parquet",
    "e065_images_nvarc_b1.parquet",
)

E065_TEXT_FILES = (
    "e065_text_arc2_train.parquet",
    "e065_text_nvarc_b0.parquet",
    "e065_text_nvarc_b1.parquet",
)


def _normalize_grid(grid: Any) -> list[list[int]] | None:
    """Normalizes a 2D integer grid or returns None if malformed."""
    if not isinstance(grid, list) or not grid:
        return None
    out: list[list[int]] = []
    w = None
    for row in grid:
        if not isinstance(row, list) or not row:
            return None
        if w is None:
            w = len(row)
        elif len(row) != w:
            return None
        norm_row: list[int] = []
        for v in row:
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                return None
            iv = int(v)
            if iv != v:
                return None
            norm_row.append(iv)
        out.append(norm_row)
    return out


def _parse_pred_json(pred_raw: Any) -> list[list[int]] | None:
    if pred_raw is None:
        return None
    if isinstance(pred_raw, list):
        return _normalize_grid(pred_raw)
    if isinstance(pred_raw, str):
        s = pred_raw.strip()
        if not s:
            return None
        try:
            parsed = json.loads(s)
        except Exception:  # pylint: disable=broad-exception-caught
            return None
        return _normalize_grid(parsed)
    return None


def load_all_ground_truth_tasks(
    e065_root_dir: str,
    arc2_train_dir: str,
) -> dict[str, dict[str, Any]]:
    """Loads ground-truth `train` and `test` pairs (with `output`) for all E065 tasks."""
    tasks: dict[str, dict[str, Any]] = {}

    # 1. Load manifest.jsonl if present for metadata
    manifest_meta: dict[str, dict[str, Any]] = {}
    manifest_path = os.path.join(e065_root_dir, "tasks", "manifest.jsonl")
    if gfile.Exists(manifest_path):
        with gfile.GFile(manifest_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                tid = str(obj.get("task_id", ""))
                if tid:
                    manifest_meta[tid] = obj

    # 2. Load official ARC-AGI-2 training tasks in parallel
    if arc2_train_dir and gfile.Exists(arc2_train_dir):
        json_fnames = [f for f in sorted(gfile.ListDir(arc2_train_dir)) if f.endswith(".json")]

        def _load_one_train(fname: str) -> tuple[str, dict[str, Any]]:
            tid = fname[:-5]
            fpath = os.path.join(arc2_train_dir, fname)
            with gfile.GFile(fpath, "r") as f:
                raw = json.load(f)
            train_pairs = [
                {
                    "input": _normalize_grid(p["input"]),
                    "output": _normalize_grid(p["output"]),
                }
                for p in raw.get("train", [])
            ]
            test_pairs = [
                {
                    "input": _normalize_grid(p["input"]),
                    "output": _normalize_grid(p.get("output")),
                }
                for p in raw.get("test", [])
            ]
            m = manifest_meta.get(tid, {})
            return tid, {
                "task_id": tid,
                "parent": m.get("parent") or tid.split("_")[0],
                "batch": m.get("batch", "arc2_train"),
                "source": m.get("source", "kaggle_arc2_training"),
                "train": train_pairs,
                "test": test_pairs,
            }

        with ThreadPoolExecutor(max_workers=64) as pool:
            for tid, entry in pool.map(_load_one_train, json_fnames):
                tasks[tid] = entry

    # 3. Load NVARC b0 and b1 challenges + solutions
    tasks_dir = os.path.join(e065_root_dir, "tasks")
    for batch_name in ("nvarc_b0", "nvarc_b1"):
        chal_path = os.path.join(tasks_dir, f"{batch_name}_challenges.json")
        sol_path = os.path.join(tasks_dir, f"{batch_name}_solutions.json")
        if not (gfile.Exists(chal_path) and gfile.Exists(sol_path)):
            continue
        with gfile.GFile(chal_path, "r") as f:
            challenges = json.load(f)
        with gfile.GFile(sol_path, "r") as f:
            solutions = json.load(f)
        for tid, chal in challenges.items():
            sol_list = solutions.get(tid, [])
            train_pairs = [
                {
                    "input": _normalize_grid(p["input"]),
                    "output": _normalize_grid(p["output"]),
                }
                for p in chal.get("train", [])
            ]
            raw_test = chal.get("test", [])
            test_pairs = []
            for p_idx, p in enumerate(raw_test):
                gt_out = p.get("output")
                if gt_out is None and p_idx < len(sol_list):
                    gt_out = sol_list[p_idx]
                test_pairs.append(
                    {
                        "input": _normalize_grid(p["input"]),
                        "output": _normalize_grid(gt_out),
                    }
                )
            m = manifest_meta.get(tid, {})
            tasks[tid] = {
                "task_id": tid,
                "parent": m.get("parent") or tid.split("_")[0],
                "batch": m.get("batch", batch_name),
                "source": m.get("source", batch_name),
                "train": train_pairs,
                "test": test_pairs,
            }

    return tasks


def extract_rollout_summary_signals(messages_json_or_list: Any) -> dict[str, Any]:
    """Extracts reasoning excerpts, last python code, last submit code, and error turns from `messages`."""
    if isinstance(messages_json_or_list, str):
        try:
            msgs = json.loads(messages_json_or_list)
        except Exception:  # pylint: disable=broad-exception-caught
            msgs = []
    elif isinstance(messages_json_or_list, list):
        msgs = messages_json_or_list
    else:
        msgs = []

    first_reasoning = ""
    last_reasoning = ""
    last_python_code = ""
    last_submit_code = ""
    last_tool_output = ""
    py_error_turns = 0
    num_assistant_turns = 0
    num_images_in_prompt = 0

    for m in msgs:
        role = m.get("role", "")
        if role == "user":
            content = m.get("content")
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "image_url":
                        num_images_in_prompt += 1
        elif role == "assistant":
            num_assistant_turns += 1
            rc = (m.get("reasoning_content") or "").strip()
            if rc:
                if not first_reasoning:
                    first_reasoning = rc[:1500]
                last_reasoning = rc[-2000:]
            tcs = m.get("tool_calls") or []
            for tc in tcs:
                fn = tc.get("function") or {}
                fn_name = fn.get("name", "")
                args_raw = fn.get("arguments", "")
                code_str = ""
                if isinstance(args_raw, dict):
                    code_str = str(args_raw.get("code", ""))
                elif isinstance(args_raw, str):
                    try:
                        parsed_args = json.loads(args_raw)
                        if isinstance(parsed_args, dict):
                            code_str = str(parsed_args.get("code", ""))
                    except Exception:  # pylint: disable=broad-exception-caught
                        code_str = args_raw
                if fn_name in ("run_python", "python_exec") and code_str:
                    last_python_code = code_str
                elif fn_name == "submit_answer" and code_str:
                    last_submit_code = code_str
        elif role == "tool":
            t_out = str(m.get("content") or "")
            if t_out:
                last_tool_output = t_out[:1500]
            if (
                "Traceback (most recent call last)" in t_out
                or "SyntaxError:" in t_out
                or "Execution timed out" in t_out
                or "[ERROR]" in t_out
            ):
                py_error_turns += 1

    return {
        "first_reasoning_excerpt": first_reasoning,
        "last_reasoning_excerpt": last_reasoning,
        "last_python_code": last_python_code[:6000],
        "last_submit_code": last_submit_code[:6000],
        "last_tool_output": last_tool_output,
        "py_error_turns": py_error_turns,
        "num_assistant_turns": num_assistant_turns,
        "num_images_in_prompt": num_images_in_prompt,
    }


def _grid_diff_summary(
    pred: list[list[int]] | None,
    gt: list[list[int]] | None,
) -> dict[str, Any]:
    """Summarizes how a predicted grid differs from the ground-truth grid."""
    if gt is None:
        return {"status": "missing_gt"}
    gt_h, gt_w = len(gt), len(gt[0])
    if pred is None:
        return {
            "status": "no_prediction",
            "gt_shape": [gt_h, gt_w],
            "pred_shape": None,
            "shape_match": False,
            "cell_accuracy": 0.0,
            "mismatched_cells": gt_h * gt_w,
        }
    pr_h, pr_w = len(pred), len(pred[0])
    if (pr_h, pr_w) != (gt_h, gt_w):
        return {
            "status": "shape_mismatch",
            "gt_shape": [gt_h, gt_w],
            "pred_shape": [pr_h, pr_w],
            "shape_match": False,
            "cell_accuracy": 0.0,
            "mismatched_cells": max(gt_h * gt_w, pr_h * pr_w),
        }
    total = gt_h * gt_w
    matches = 0
    sample_diffs = []
    for r in range(gt_h):
        for c in range(gt_w):
            if pred[r][c] == gt[r][c]:
                matches += 1
            elif len(sample_diffs) < 12:
                sample_diffs.append({"r": r, "c": c, "pred": pred[r][c], "expected_gt": gt[r][c]})
    mismatches = total - matches
    return {
        "status": "exact_match" if mismatches == 0 else "cell_mismatch",
        "gt_shape": [gt_h, gt_w],
        "pred_shape": [pr_h, pr_w],
        "shape_match": True,
        "cell_accuracy": round(matches / float(total), 4) if total > 0 else 1.0,
        "mismatched_cells": mismatches,
        "total_cells": total,
        "sample_diffs": sample_diffs,
    }


def _read_parquet_table_from_cns(path: str) -> pa.Table:
    with gfile.GFile(path, "rb") as f:
        raw = f.read()
    return pq.read_table(io.BytesIO(raw))


def _write_parquet_table_to_cns(table: pa.Table, path: str) -> None:
    buf = io.BytesIO()
    pq.write_table(table, buf, compression="zstd")
    with gfile.GFile(path, "wb") as f:
        f.write(buf.getvalue())


def audit_and_split_modality(
    rollouts_dir: str,
    file_names: tuple[str, ...],
    gt_tasks: dict[str, dict[str, Any]],
    eval_index: dict[str, Any],
) -> dict[str, Any]:
    """Audits all rollout files for a modality (`images` or `text`) and returns split tables & records."""
    per_file_stats: dict[str, Any] = {}
    test_passing_tables_by_file: dict[str, pa.Table] = {}
    test_passing_uncontam_tables_by_file: dict[str, pa.Table] = {}
    verified_3gate_tables_by_file: dict[str, pa.Table] = {}
    non_test_passing_tables_by_file: dict[str, pa.Table] = {}

    rollout_status_by_task: dict[str, dict[str, Any]] = {}
    non_test_passing_records: list[dict[str, Any]] = []
    all_telemetry_records: list[dict[str, Any]] = []

    total_Discrepancies = 0

    for fname in file_names:
        fpath = os.path.join(rollouts_dir, fname)
        table = _read_parquet_table_from_cns(fpath)
        rows = table.to_pylist()

        # Group row indices by (task_id, attempt)
        groups: dict[tuple[str, int], list[tuple[int, dict[str, Any]]]] = defaultdict(list)
        for idx, row in enumerate(rows):
            tid = str(row["task_id"])
            att = int(row.get("attempt") or 0)
            groups[(tid, att)].append((idx, row))

        pass_row_indices: list[int] = []
        pass_uncontam_row_indices: list[int] = []
        verified_3gate_row_indices: list[int] = []
        fail_row_indices: list[int] = []

        file_counts = Counter()

        for (tid, att), indexed_rows in groups.items():
            indexed_rows.sort(key=lambda x: int(x[1].get("pair") or 0))
            group_indices = [i for i, _ in indexed_rows]
            group_rows = [r for _, r in indexed_rows]
            r0 = group_rows[0]

            parent = str(r0.get("parent") or tid.split("_")[0])
            is_contam, contam_reason = data_pipeline.is_task_contaminated(tid, parent, eval_index)
            gt_info = gt_tasks.get(tid)
            gt_test_pairs = gt_info["test"] if gt_info else []

            pair_checks = []
            all_correct_col = True
            all_exact_gt_match = True
            discrepancy_on_task = False

            for r in group_rows:
                p_idx = int(r.get("pair") or 0)
                col_corr = bool(r.get("correct") is True)
                pred_grid = _parse_pred_json(r.get("pred"))
                gt_grid = (
                    gt_test_pairs[p_idx]["output"]
                    if p_idx < len(gt_test_pairs)
                    else None
                )
                if gt_grid is not None:
                    exact_match = bool(pred_grid is not None and pred_grid == gt_grid)
                    if exact_match != col_corr:
                        discrepancy_on_task = True
                        total_Discrepancies += 1
                else:
                    exact_match = col_corr

                if not col_corr:
                    all_correct_col = False
                if not exact_match:
                    all_exact_gt_match = False

                diff_info = _grid_diff_summary(pred_grid, gt_grid)
                pair_checks.append(
                    {
                        "pair": p_idx,
                        "correct_col": col_corr,
                        "exact_gt_match": exact_match,
                        "pred": pred_grid,
                        "diff_summary": diff_info,
                    }
                )

            # Check multi-pair completeness against ground truth test pairs
            if gt_test_pairs and len(group_rows) < len(gt_test_pairs):
                all_exact_gt_match = False
                all_correct_col = False

            test_passing = bool(all_exact_gt_match and all_correct_col)
            train_pass = all(r.get("train_pass") is True for r in group_rows)
            train_frac_vals = [
                float(r["train_frac"])
                for r in group_rows
                if r.get("train_frac") is not None
            ]
            train_frac = min(train_frac_vals) if train_frac_vals else 0.0
            has_error = any(r.get("error") is not None for r in group_rows)
            submitted_ok = all(
                r.get("answer_source") == "submit" and r.get("stop_reason") == "submitted"
                for r in group_rows
            )
            verified_3gate = bool(
                test_passing and train_pass and submitted_ok and not has_error and not is_contam
            )

            file_counts["total_rollouts"] += 1
            file_counts["total_rows"] += len(group_rows)
            if len(group_rows) > 1:
                file_counts["multi_pair_rollouts"] += 1
            if discrepancy_on_task:
                file_counts["gt_vs_col_discrepancies"] += 1
            if is_contam:
                file_counts["contaminated_rollouts"] += 1
            if has_error:
                file_counts["error_rollouts"] += 1

            # Determine failure category if non-test-passing
            if test_passing:
                failure_category = "test_passing_train_pass" if train_pass else "test_passing_train_fail"
                file_counts["test_passing_rollouts"] += 1
                file_counts["test_passing_rows"] += len(group_rows)
                pass_row_indices.extend(group_indices)
                if not is_contam:
                    file_counts["test_passing_uncontaminated_rollouts"] += 1
                    file_counts["test_passing_uncontaminated_rows"] += len(group_rows)
                    pass_uncontam_row_indices.extend(group_indices)
                if verified_3gate:
                    file_counts["verified_3gate_rollouts"] += 1
                    file_counts["verified_3gate_rows"] += len(group_rows)
                    verified_3gate_row_indices.extend(group_indices)
                elif not train_pass:
                    file_counts["test_passing_but_train_fail_rollouts"] += 1
            else:
                file_counts["non_test_passing_rollouts"] += 1
                file_counts["non_test_passing_rows"] += len(group_rows)
                fail_row_indices.extend(group_indices)
                if not is_contam:
                    file_counts["non_test_passing_uncontaminated_rollouts"] += 1
                    file_counts["non_test_passing_uncontaminated_rows"] += len(group_rows)
                if has_error:
                    failure_category = "api_or_runtime_error"
                    file_counts["fail_api_or_runtime_error"] += 1
                elif r0.get("answer_source") != "submit" or r0.get("stop_reason") != "submitted":
                    failure_category = "budget_or_turn_exhausted_no_submit"
                    file_counts["fail_budget_or_turn_exhausted"] += 1
                elif train_pass:
                    failure_category = "demo_overfit_train_pass_test_fail"
                    file_counts["fail_demo_overfit_train_pass"] += 1
                else:
                    failure_category = "train_fail_submitted"
                    file_counts["fail_train_fail_submitted"] += 1

            sig = extract_rollout_summary_signals(r0.get("messages"))
            status_entry = {
                "task_id": tid,
                "parent": parent,
                "attempt": att,
                "source": str(r0.get("source") or ""),
                "source_file": fname,
                "n_test_pairs": len(group_rows),
                "test_passing": test_passing,
                "train_pass": train_pass,
                "train_frac": train_frac,
                "train_trace": r0.get("train_trace"),
                "verified_3gate": verified_3gate,
                "is_contaminated": is_contam,
                "contamination_reason": contam_reason,
                "failure_category": failure_category,
                "answer_source": r0.get("answer_source"),
                "stop_reason": r0.get("stop_reason"),
                "turns": int(r0.get("turns") or 0),
                "tool_calls": int(r0.get("tool_calls") or 0),
                "prompt_tokens": int(r0.get("prompt_tokens") or 0),
                "completion_tokens": int(r0.get("completion_tokens") or 0),
                "reasoning_tokens": int(r0.get("reasoning_tokens") or 0),
                "elapsed_s": float(r0.get("elapsed_s") or 0.0),
                "error": r0.get("error"),
                "pair_checks": pair_checks,
                **sig,
            }
            rollout_status_by_task[tid] = status_entry
            all_telemetry_records.append(status_entry)

            if not test_passing:
                non_test_passing_records.append(
                    {
                        **status_entry,
                        "train": gt_info["train"] if gt_info else [],
                        "test": gt_info["test"] if gt_info else [],
                    }
                )

        test_passing_tables_by_file[fname] = table.take(pa.array(pass_row_indices, type=pa.int64()))
        test_passing_uncontam_tables_by_file[fname] = table.take(
            pa.array(pass_uncontam_row_indices, type=pa.int64())
        )
        verified_3gate_tables_by_file[fname] = table.take(
            pa.array(verified_3gate_row_indices, type=pa.int64())
        )
        non_test_passing_tables_by_file[fname] = table.take(
            pa.array(fail_row_indices, type=pa.int64())
        )
        per_file_stats[fname] = dict(file_counts)

    return {
        "per_file_stats": per_file_stats,
        "total_discrepancies": total_Discrepancies,
        "test_passing_tables_by_file": test_passing_tables_by_file,
        "test_passing_uncontam_tables_by_file": test_passing_uncontam_tables_by_file,
        "verified_3gate_tables_by_file": verified_3gate_tables_by_file,
        "non_test_passing_tables_by_file": non_test_passing_tables_by_file,
        "rollout_status_by_task": rollout_status_by_task,
        "non_test_passing_records": non_test_passing_records,
        "all_telemetry_records": all_telemetry_records,
    }


def run_audit_and_split(
    e065_root_dir: str,
    arc2_train_dir: str,
    eval_dir: str,
    output_cns_dir: str,
    cns_quota_group: str = "deepmind-resources-colossus",
) -> dict[str, Any]:
    """Runs the full E065 audit & split and writes all artifacts to CNS."""
    t0 = time.time()
    rollouts_dir = os.path.join(e065_root_dir, "rollouts")

    # 1. Ensure CNS output directories exist with deepmind-resources-colossus quota
    test_passing_dir = os.path.join(output_cns_dir, "test_passing")
    non_test_passing_dir = os.path.join(output_cns_dir, "non_test_passing")
    difficulty_dir = os.path.join(output_cns_dir, "difficulty_rankings")
    corrected_dir = os.path.join(output_cns_dir, "corrected_rewritten")
    for d in (output_cns_dir, test_passing_dir, non_test_passing_dir, difficulty_dir, corrected_dir):
        export_and_quantize.ensure_cns_quota_dir(d, group=cns_quota_group)

    # 2. Load ground-truth tasks + solutions and eval contamination index
    gt_tasks = load_all_ground_truth_tasks(e065_root_dir, arc2_train_dir)
    eval_task_ids = (
        {
            fname[:-5]
            for fname in gfile.ListDir(eval_dir)
            if fname.endswith(".json")
        }
        if eval_dir and gfile.Exists(eval_dir)
        else set()
    )
    eval_index = {"eval_task_ids": eval_task_ids}

    # 3. Audit & split e065_images_* and e065_text_*
    img_res = audit_and_split_modality(rollouts_dir, E065_IMAGE_FILES, gt_tasks, eval_index)
    txt_res = audit_and_split_modality(rollouts_dir, E065_TEXT_FILES, gt_tasks, eval_index)

    # 4. Cross-reference image vs text twins
    img_status = img_res["rollout_status_by_task"]
    txt_status = txt_res["rollout_status_by_task"]

    both_pass = 0
    img_only_pass = 0
    txt_only_pass = 0
    both_fail = 0
    txt_rescued_task_ids: set[str] = set()

    for tid, ist in img_status.items():
        tst = txt_status.get(tid)
        t_pass = bool(tst and tst["test_passing"])
        t_train_pass = bool(tst and tst["train_pass"])
        t_turns = int(tst["turns"]) if tst else 0
        t_tokens = int(tst["completion_tokens"]) if tst else 0
        ist["text_twin_test_passing"] = t_pass
        ist["text_twin_train_pass"] = t_train_pass
        ist["text_twin_turns"] = t_turns
        ist["text_twin_completion_tokens"] = t_tokens

        if ist["test_passing"] and t_pass:
            both_pass += 1
        elif ist["test_passing"] and not t_pass:
            img_only_pass += 1
        elif not ist["test_passing"] and t_pass:
            txt_only_pass += 1
            txt_rescued_task_ids.add(tid)
        else:
            both_fail += 1

    for rec in img_res["non_test_passing_records"]:
        tid = rec["task_id"]
        tst = txt_status.get(tid)
        rec["text_twin_test_passing"] = bool(tst and tst["test_passing"])
        rec["text_twin_train_pass"] = bool(tst and tst["train_pass"])
        rec["text_twin_turns"] = int(tst["turns"]) if tst else 0
        rec["text_twin_completion_tokens"] = int(tst["completion_tokens"]) if tst else 0
        rec["text_twin_last_submit_code"] = str(tst.get("last_submit_code", "")) if tst else ""

    # 5. Write per-file and combined Parquet tables to CNS
    for fname, tbl in img_res["test_passing_tables_by_file"].items():
        stem = fname[:-8]  # strip .parquet
        _write_parquet_table_to_cns(
            tbl, os.path.join(test_passing_dir, f"{stem}_test_passing.parquet")
        )
    for fname, tbl in img_res["non_test_passing_tables_by_file"].items():
        stem = fname[:-8]
        _write_parquet_table_to_cns(
            tbl, os.path.join(non_test_passing_dir, f"{stem}_non_test_passing.parquet")
        )

    combined_img_pass = pa.concat_tables(
        list(img_res["test_passing_tables_by_file"].values()),
        promote_options="default",
    )
    combined_img_pass_uncontam = pa.concat_tables(
        list(img_res["test_passing_uncontam_tables_by_file"].values()),
        promote_options="default",
    )
    combined_img_3gate = pa.concat_tables(
        list(img_res["verified_3gate_tables_by_file"].values()),
        promote_options="default",
    )
    combined_img_fail = pa.concat_tables(
        list(img_res["non_test_passing_tables_by_file"].values()),
        promote_options="default",
    )
    combined_txt_pass = pa.concat_tables(
        list(txt_res["test_passing_tables_by_file"].values()),
        promote_options="default",
    )

    _write_parquet_table_to_cns(
        combined_img_pass,
        os.path.join(test_passing_dir, "e065_images_test_passing.parquet"),
    )
    _write_parquet_table_to_cns(
        combined_img_pass_uncontam,
        os.path.join(test_passing_dir, "e065_images_test_passing_uncontaminated.parquet"),
    )
    _write_parquet_table_to_cns(
        combined_img_3gate,
        os.path.join(test_passing_dir, "e065_images_verified_3gate.parquet"),
    )
    _write_parquet_table_to_cns(
        combined_txt_pass,
        os.path.join(test_passing_dir, "e065_text_test_passing.parquet"),
    )
    _write_parquet_table_to_cns(
        combined_img_fail,
        os.path.join(non_test_passing_dir, "e065_images_non_test_passing.parquet"),
    )

    # Also save text twin rollouts that passed test on tasks where e065_images failed
    txt_pass_rows = combined_txt_pass.to_pylist()
    rescued_indices = [
        idx
        for idx, r in enumerate(txt_pass_rows)
        if str(r["task_id"]) in txt_rescued_task_ids
        and not data_pipeline.is_task_contaminated(
            str(r["task_id"]), str(r.get("parent") or ""), eval_index
        )[0]
    ]
    rescued_txt_table = combined_txt_pass.take(pa.array(rescued_indices, type=pa.int64()))
    _write_parquet_table_to_cns(
        rescued_txt_table,
        os.path.join(test_passing_dir, "e065_text_rescued_for_failed_images.parquet"),
    )

    # 6. Write non_test_passing_tasks.jsonl and all_rollouts_telemetry.jsonl
    non_pass_jsonl_path = os.path.join(
        non_test_passing_dir, "e065_images_non_test_passing_tasks.jsonl"
    )
    with gfile.GFile(non_pass_jsonl_path, "w") as f:
        for rec in img_res["non_test_passing_records"]:
            f.write(json.dumps(rec) + "\n")

    telemetry_jsonl_path = os.path.join(output_cns_dir, "all_rollouts_telemetry.jsonl")
    with gfile.GFile(telemetry_jsonl_path, "w") as f:
        for tid in sorted(img_status.keys()):
            rec = dict(img_status[tid])
            gt_info = gt_tasks.get(tid, {})
            train_grids = gt_info.get("train", [])
            test_grids = gt_info.get("test", [])
            rec["n_train_pairs"] = len(train_grids)
            rec["train_shapes"] = [
                {
                    "in": [len(p["input"]), len(p["input"][0])] if p.get("input") else None,
                    "out": [len(p["output"]), len(p["output"][0])] if p.get("output") else None,
                }
                for p in train_grids
            ]
            rec["test_shapes"] = [
                {
                    "in": [len(p["input"]), len(p["input"][0])] if p.get("input") else None,
                    "out": [len(p["output"]), len(p["output"][0])] if p.get("output") else None,
                }
                for p in test_grids
            ]
            f.write(json.dumps(rec) + "\n")

    # 7. Build summary JSON
    img_totals = Counter()
    for st in img_res["per_file_stats"].values():
        for k, v in st.items():
            img_totals[k] += v
    txt_totals = Counter()
    for st in txt_res["per_file_stats"].values():
        for k, v in st.items():
            txt_totals[k] += v

    summary = {
        "created_at_unix": time.time(),
        "elapsed_s": round(time.time() - t0, 2),
        "e065_root_dir": e065_root_dir,
        "output_cns_dir": output_cns_dir,
        "ground_truth_tasks_loaded": len(gt_tasks),
        "eval_tasks_in_contamination_index": len(eval_index["eval_task_ids"]),
        "e065_images_totals": dict(img_totals),
        "e065_images_per_file": img_res["per_file_stats"],
        "e065_images_gt_vs_col_discrepancies": img_res["total_discrepancies"],
        "e065_text_totals": dict(txt_totals),
        "e065_text_per_file": txt_res["per_file_stats"],
        "e065_text_gt_vs_col_discrepancies": txt_res["total_discrepancies"],
        "cross_modality_comparison": {
            "total_unique_tasks": len(img_status),
            "both_images_and_text_pass": both_pass,
            "images_pass_text_fail": img_only_pass,
            "text_pass_images_fail_rescued": txt_only_pass,
            "both_images_and_text_fail": both_fail,
            "union_either_modality_passes": both_pass + img_only_pass + txt_only_pass,
        },
        "saved_cns_files": {
            "images_test_passing_parquet": os.path.join(
                test_passing_dir, "e065_images_test_passing.parquet"
            ),
            "images_test_passing_uncontaminated_parquet": os.path.join(
                test_passing_dir, "e065_images_test_passing_uncontaminated.parquet"
            ),
            "images_verified_3gate_parquet": os.path.join(
                test_passing_dir, "e065_images_verified_3gate.parquet"
            ),
            "text_test_passing_parquet": os.path.join(
                test_passing_dir, "e065_text_test_passing.parquet"
            ),
            "text_rescued_for_failed_images_parquet": os.path.join(
                test_passing_dir, "e065_text_rescued_for_failed_images.parquet"
            ),
            "images_non_test_passing_parquet": os.path.join(
                non_test_passing_dir, "e065_images_non_test_passing.parquet"
            ),
            "images_non_test_passing_tasks_jsonl": non_pass_jsonl_path,
            "all_rollouts_telemetry_jsonl": telemetry_jsonl_path,
        },
    }

    summary_path = os.path.join(output_cns_dir, "split_audit_summary.json")
    with gfile.GFile(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    return summary


def main(argv: list[str]) -> None:
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")
    summary = run_audit_and_split(
        e065_root_dir=FLAGS.e065_root_dir,
        arc2_train_dir=FLAGS.e065_arc2_train_dir,
        eval_dir=FLAGS.e065_eval_dir,
        output_cns_dir=FLAGS.e065_split_output_cns_dir,
        cns_quota_group=FLAGS.e065_cns_quota_group,
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    app.run(main)
