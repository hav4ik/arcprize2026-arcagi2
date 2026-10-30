"""Evaluates ARC-AGI-2 / Top-10K puzzle difficulty and Qwen3.8-27B SFT value using Gemini 4 Argon.

Combines:
1. Empirical rollout telemetry from DeepSeek-V4.1 Pass@1 (`v8`) rollouts
   (turns, tool_calls, thinking_tokens, L1/L2/L3 verification, and verified
   Python `submitted_solve_code`) plus 1K empirical screening signals.
2. Gemini 4 Argon (`gemini-4-argon`) deep structural & algorithmic difficulty
   analysis tailored specifically to what is hard for Qwen3.8-27B and will
   improve Qwen3.8-27B SFT performance when included (while filtering out
   trivial 1-op synthetic tasks and over-long token-budget-blowing traces).
"""

from __future__ import annotations

import collections
from concurrent import futures
import json
import os
import subprocess
import time
from typing import Any, Callable, Mapping, Sequence

from absl import app
from absl import flags
import numpy as np

from google3.experimental.users.arazdai.kaggle.sft import data_pipeline
from google3.pyglib import gfile

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "candidates_path",
    "/cns/lu-d/home/arazdai/kaggle/arc_sft_data/sft_prep_pipeline/top_10k_deepseek_candidates.jsonl",
    "Path to Top-10K candidates JSONL on CNS.",
)
flags.DEFINE_string(
    "top10k_rollouts_dir",
    "/cns/lu-d/home/arazdai/kaggle/arc_sft_data/sft_prep_pipeline/deepseek_v41_rollouts_top10k_v8_pass1",
    "CNS directory containing Top-10K DeepSeek-V4.1 v8 rollout shards.",
)
flags.DEFINE_string(
    "output_cns_dir",
    "/cns/lu-d/home/arazdai/kaggle/arc_sft_data/sft_prep_pipeline/argon_qwen38_difficulty_eval",
    "CNS directory to write Gemini 4 Argon difficulty ratings and curated hard-for-Qwen3.8 manifests.",
)
flags.DEFINE_string(
    "generate_bin",
    "/google/bin/releases/gemini-agents-generate/generate",
    "Path to the Gemini generate CLI binary.",
)
flags.DEFINE_string(
    "model",
    "gemini-4-argon",
    "Primary Gemini model to use for difficulty evaluation.",
)
flags.DEFINE_string(
    "fallback_model",
    "gemini-3.1-pro-preview,gemini-3-flash-preview",
    "Comma-separated fallback models when gemini-4-argon endpoint returns out-of-capacity (UTA resource_exhausted).",
)
flags.DEFINE_enum(
    "thinking_level",
    "LOW",
    ["LOW", "MEDIUM", "HIGH"],
    "Thinking level for gemini-4-argon (supports LOW, MEDIUM, HIGH).",
)
flags.DEFINE_integer(
    "max_workers",
    16,
    "Number of concurrent Gemini requests.",
)
flags.DEFINE_integer(
    "max_families_to_eval",
    0,
    "Maximum number of unique parent families/tasks to evaluate (0 = all).",
)
flags.DEFINE_integer(
    "min_qwen38_difficulty",
    6,
    "Minimum Qwen3.8-27B difficulty score (1-10) for curated hard SFT inclusion.",
)
flags.DEFINE_integer(
    "max_qwen38_difficulty",
    9,
    "Maximum Qwen3.8-27B difficulty score (1-10) to avoid ambiguous/unsolvable outliers (10).",
)
flags.DEFINE_integer(
    "max_sft_tokens",
    36000,
    "Maximum total trajectory tokens for high-efficiency Qwen3.8-27B SFT inclusion.",
)
flags.DEFINE_bool(
    "prioritize_verified_rollouts",
    True,
    "If True, evaluates families with verified clean SFT rollouts first.",
)

QWEN38_FAILURE_MODES = (
    "multi_step_composition",
    "marker_relative_indexing",
    "dynamic_output_size_inference",
    "odd_even_parity_offset",
    "object_graph_connectivity",
    "conditional_color_logic",
    "periodic_phase_extrapolation",
    "occlusion_or_layering",
    "non_greedy_hypothesis_revision",
    "symmetry_or_dihedral_group",
    "gravity_or_raycast_simulation",
)


def get_candidate_puzzle_id(cand: Mapping[str, Any]) -> str:
    """Returns the canonical task/puzzle ID from a Top-10K candidate dict."""
    return str(
        cand.get("deepseek_task_id")
        or cand.get("puzzle_id")
        or cand.get("original_task_id")
        or cand.get("sample_id")
        or ""
    )


def get_candidate_dataset(cand: Mapping[str, Any]) -> str:
    """Returns the source dataset name from a Top-10K candidate dict."""
    return str(
        cand.get("dataset")
        or cand.get("source_dataset")
        or cand.get("source_group")
        or "unknown"
    )


def summarize_grid_compact_safe(
    grid: Sequence[Sequence[int]], max_rows: int = 10, max_cols: int = 14
) -> str:
    """Formats a 2D integer grid into a compact ASCII summary with shape, palette, and preview."""
    arr = np.asarray(grid, dtype=np.int32)
    if arr.ndim != 2:
        return f"invalid_shape={arr.shape}"
    h, w = arr.shape
    vals, counts = np.unique(arr, return_counts=True)
    palette = ", ".join(f"{int(v)}:{int(c)}" for v, c in zip(vals, counts))
    rows_str: list[str] = []
    for r in range(min(h, max_rows)):
        row_digits = "".join(str(int(x)) for x in arr[r, :max_cols])
        if w > max_cols:
            row_digits += "..."
        rows_str.append(row_digits)
    if h > max_rows:
        rows_str.append("...")
    return f"shape={h}x{w} palette=[{palette}] preview=[{' / '.join(rows_str)}]"


summarize_grid_compact = summarize_grid_compact_safe


def build_argon_difficulty_prompt(
    candidate: Mapping[str, Any],
    rollout_row: Mapping[str, Any] | None = None,
) -> str:
    """Builds the structured evaluation prompt for `gemini-4-argon`."""
    puzzle = candidate.get("puzzle") or candidate
    train_pairs = puzzle.get("train") or candidate.get("train") or []
    test_pairs = puzzle.get("test") or candidate.get("test") or []

    pair_lines: list[str] = []
    for idx, pair in enumerate(train_pairs[:4]):
        inp = pair.get("input") or []
        out = pair.get("output") or []
        pair_lines.append(
            f"  Demo {idx+1}: IN({summarize_grid_compact_safe(inp)}) -> OUT({summarize_grid_compact_safe(out)})"
        )
    for idx, pair in enumerate(test_pairs[:2]):
        inp = pair.get("input") or []
        out = pair.get("output")
        out_str = summarize_grid_compact_safe(out) if out is not None else "held_out"
        pair_lines.append(
            f"  Test {idx+1}: IN({summarize_grid_compact_safe(inp)}) -> OUT({out_str})"
        )

    emp = candidate.get("empirical_difficulty") or {}
    emp_summary = (
        f"screening_bucket={emp.get('bucket_3level', 'unknown')}, "
        f"empirical_pass_rate={emp.get('pass_rate_3level', 'unknown')}, "
        f"has_direct_screening={emp.get('has_direct_empirical', False)}"
    )

    rollout_summary = "No DeepSeek-V4.1 rollout available yet."
    solve_code_snippet = "None"
    if rollout_row is not None:
        turns = rollout_row.get("turns", 0)
        tool_calls = rollout_row.get("tool_calls", 0)
        prompt_toks = rollout_row.get("prompt_tokens", 0)
        comp_toks = rollout_row.get("completion_tokens", 0)
        think_toks = rollout_row.get("thinking_tokens", 0)
        l1 = rollout_row.get("train_pass", False)
        l2 = rollout_row.get("test_exact", rollout_row.get("correct", False))
        l3 = rollout_row.get("level3_pass", False)
        all3 = rollout_row.get("all_3_levels_verified", False)
        rollout_summary = (
            f"turns={turns}, tool_calls={tool_calls}, "
            f"total_tokens={int(prompt_toks) + int(comp_toks)} (thinking={think_toks}), "
            f"L1_train_pass={l1}, L2_test_exact={l2}, L3_holdout={l3}, all_3_verified={all3}"
        )
        code = str(rollout_row.get("submitted_solve_code") or "").strip()
        if code:
            solve_code_snippet = code[:1800]

    puzzle_id = get_candidate_puzzle_id(candidate)
    dataset_name = get_candidate_dataset(candidate)

    prompt = f"""You are evaluating an ARC-AGI-2 reasoning task and its solver trajectory to curate a high-leverage Supervised Fine-Tuning (SFT) dataset for **Qwen3.8-27B** (a 27B hybrid Gated-DeltaNet + Full-Attention model running with an IPython `run_python` tool under a strict 6-hour Kaggle 4xL4 token budget).

### Why This Evaluation Matters for Qwen3.8-27B
- **What Qwen3.8-27B already solves without SFT (DO NOT PRIORITIZE)**:
  Trivial 1-step synthetic tasks (1D-ARC, simple color swaps, single crop/flip/transpose, trivial global flood-fill) where Qwen3.8-27B already achieves >90% accuracy. Including too many trivial 1-op tasks dilutes SFT gradients.
- **Where Qwen3.8-27B fails and benefits MOST from SFT (HIGH PRIORITY)**:
  1. `multi_step_composition`: Tasks requiring 2 to 4 interacting spatial/logical stages (e.g. detect objects -> infer relation/graph -> transform/project -> resolve occlusion).
  2. `marker_relative_indexing`: Indexing or cropping relative to dynamic markers/anchors rather than fixed absolute coordinates.
  3. `dynamic_output_size_inference`: Inferring non-constant output grid dimensions from object counts, bounding boxes, or scale factors.
  4. `odd_even_parity_offset`: Handling odd/even grid centers, reflection axes, or +/-1 boundary offsets accurately.
  5. `object_graph_connectivity`: Connected components, topological containment (inside/outside holes), or ray/bridge routing between objects.
  6. `conditional_color_logic`: Rules where transformation depends on object shape, symmetry, size rank, or neighbor count.
  7. `periodic_phase_extrapolation`: Tiling/wallpaper/diagonal periodicity where the phase must be anchored to existing non-zero cells.
  8. `occlusion_or_layering`: Ordering overlapping sprites/frames by z-order or majority vote.
  9. `non_greedy_hypothesis_revision`: Tasks where a naive first guess fails on one demo pair and requires systematic verification across all demos.

### Task Metadata & Grid Structure
- puzzle_id: `{puzzle_id}`
- family_id: `{candidate.get('family_id')}`
- source_dataset: `{dataset_name}`
- empirical_screening: {emp_summary}
- DeepSeek-V4.1 Pass@1 Rollout Telemetry: {rollout_summary}
- Grid Pairs:
{chr(10).join(pair_lines)}

### Verified Solver Python Code (if solved by DeepSeek-V4.1)
```python
{solve_code_snippet}
```

### Instructions
Return ONLY a valid JSON object (no markdown fences, no extra commentary) with these exact fields:
{{
  "qwen38_difficulty_score": <int 1-10: 1-3 = trivial 1-step op Qwen3.8 already solves; 4-5 = moderate 2-step op; 6-9 = hard multi-step/relational/structural task where Qwen3.8-27B struggles and SFT provides high gain; 10 = ambiguous/cryptic/noisy synthetic artifact>,
  "algorithmic_complexity": <one of "trivial_1op", "moderate_2op", "hard_multi_step", "complex_search_or_graph">,
  "qwen38_failure_modes_targeted": <list of 1-4 strings chosen from {list(QWEN38_FAILURE_MODES)}>,
  "transformation_rule_summary": <concise 1-sentence description of the exact latent rule>,
  "solution_code_quality": <int 1-5: 5 = clean, generalizable algorithmic Python code; 3 = acceptable code or no rollout yet; 1 = hardcoded constants/overfitted hack>,
  "sft_recommend_include": <bool: true ONLY if qwen38_difficulty_score is 6-9 (or a super-clean multi-step 5) AND the rule is unambiguous AND solution_code_quality >= 3>,
  "sft_priority_tier": <one of "tier1_high_value_hard_for_qwen", "tier2_medium_value_composition", "tier3_skip_trivial_or_redundant", "tier4_skip_ambiguous_or_bloated">,
  "qwen38_sft_rationale": <concise 1-sentence explanation of why this task will or will not improve Qwen3.8-27B>
}}"""
    return prompt


def parse_argon_json_response(raw_text: str) -> dict[str, Any]:
    """Extracts and validates the JSON object from `gemini-4-argon` output."""
    text = raw_text.strip()
    if "```" in text:
        # Strip markdown fences if present
        parts = text.split("```")
        for part in parts:
            p = part.strip()
            if p.startswith("json"):
                p = p[4:].strip()
            if p.startswith("{") and p.endswith("}"):
                text = p
                break
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"No JSON object found in response: {raw_text[:200]!r}")
    obj = json.loads(text[start : end + 1])

    q_diff = int(np.clip(int(obj.get("qwen38_difficulty_score", 5)), 1, 10))
    code_qual = int(np.clip(int(obj.get("solution_code_quality", 3)), 1, 5))
    alg = str(obj.get("algorithmic_complexity", "moderate_2op"))
    if alg not in ("trivial_1op", "moderate_2op", "hard_multi_step", "complex_search_or_graph"):
        alg = "hard_multi_step" if q_diff >= 6 else "moderate_2op"

    modes_raw = obj.get("qwen38_failure_modes_targeted") or []
    modes = [str(m) for m in modes_raw if isinstance(m, str)][:4]
    if not modes:
        modes = ["multi_step_composition"] if q_diff >= 6 else []

    tier = str(obj.get("sft_priority_tier", ""))
    valid_tiers = (
        "tier1_high_value_hard_for_qwen",
        "tier2_medium_value_composition",
        "tier3_skip_trivial_or_redundant",
        "tier4_skip_ambiguous_or_bloated",
    )
    if tier not in valid_tiers:
        if 6 <= q_diff <= 9 and code_qual >= 4:
            tier = "tier1_high_value_hard_for_qwen"
        elif 5 <= q_diff <= 8 and code_qual >= 3:
            tier = "tier2_medium_value_composition"
        elif q_diff <= 4:
            tier = "tier3_skip_trivial_or_redundant"
        else:
            tier = "tier4_skip_ambiguous_or_bloated"

    recommend = bool(obj.get("sft_recommend_include", 6 <= q_diff <= 9 and code_qual >= 3))
    return {
        "qwen38_difficulty_score": q_diff,
        "algorithmic_complexity": alg,
        "qwen38_failure_modes_targeted": modes,
        "transformation_rule_summary": str(obj.get("transformation_rule_summary", ""))[:400],
        "solution_code_quality": code_qual,
        "sft_recommend_include": recommend,
        "sft_priority_tier": tier,
        "qwen38_sft_rationale": str(obj.get("qwen38_sft_rationale", ""))[:400],
    }


def compute_composite_sft_value_score(
    argon_rating: Mapping[str, Any],
    rollout_row: Mapping[str, Any] | None = None,
    candidate: Mapping[str, Any] | None = None,
    max_sft_tokens: int = 36000,
) -> float:
    """Computes a 0-100 composite SFT value score combining Gemini 4 Argon rating + rollout telemetry."""
    q_diff = int(argon_rating.get("qwen38_difficulty_score", 5))
    code_qual = int(argon_rating.get("solution_code_quality", 3))
    recommend = bool(argon_rating.get("sft_recommend_include", False))
    modes = argon_rating.get("qwen38_failure_modes_targeted") or []

    # 1. Base score from Qwen3.8 difficulty sweet spot (peak at 7-9, penalty for <=3 or 10)
    diff_weights = {
        1: 5.0,
        2: 10.0,
        3: 18.0,
        4: 32.0,
        5: 50.0,
        6: 72.0,
        7: 88.0,
        8: 96.0,
        9: 92.0,
        10: 25.0,
    }
    score = diff_weights.get(q_diff, 50.0)

    # 2. Code quality & failure mode diversity bonus
    score += (code_qual - 3) * 5.0
    score += min(len(modes), 3) * 2.5
    if not recommend:
        score -= 20.0

    # 3. Empirical screening signal (hard 0/2 or frontier 1/2 in 1K screening is higher value than trivial 2/2)
    if candidate is not None:
        emp = candidate.get("empirical_difficulty") or {}
        bucket = str(emp.get("bucket_3level") or "")
        if bucket == "0/2":
            score += 6.0
        elif bucket == "1/2":
            score += 5.0
        elif bucket == "2/2" and q_diff <= 4:
            score -= 8.0

    # 4. Rollout verification & token-efficiency adjustment
    if rollout_row is not None:
        all3 = bool(
            rollout_row.get("all_3_levels_verified")
            or (rollout_row.get("train_pass") and rollout_row.get("correct"))
        )
        total_toks = int(rollout_row.get("prompt_tokens", 0)) + int(
            rollout_row.get("completion_tokens", 0)
        )
        tool_calls = int(rollout_row.get("tool_calls", 0))
        if all3:
            score += 8.0
        else:
            score -= 25.0
        if 0 < total_toks <= 20000:
            score += 5.0
        elif total_toks > max_sft_tokens:
            score -= 18.0
        if 2 <= tool_calls <= 12:
            score += 3.0
        elif tool_calls > 22:
            score -= 8.0

    return round(float(np.clip(score, 0.0, 100.0)), 2)


def call_gemini_argon_cli(
    prompt: str,
    generate_bin: str = "/google/bin/releases/gemini-agents-generate/generate",
    model: str = "gemini-4-argon",
    fallback_model: str = "gemini-3.1-pro-preview,gemini-3-flash-preview",
    thinking_level: str = "LOW",
    timeout_sec: int = 60,
    max_retries: int = 2,
) -> dict[str, Any]:
    """Calls `gemini-4-argon` via `generate -prod -json`, falling back to `fallback_model` on endpoint capacity exhaustion."""
    models_to_try = [model]
    for fb in (fallback_model or "").split(","):
        fb_clean = fb.strip()
        if fb_clean and fb_clean not in models_to_try:
            models_to_try.append(fb_clean)

    last_err = ""
    for active_model in models_to_try:
        req_payload = {
            "model": (
                f"models/{active_model}"
                if not active_model.startswith("models/")
                else active_model
            ),
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generation_config": {
                "thinking_config": {"thinking_level": thinking_level},
            },
        }
        req_bytes = json.dumps(req_payload).encode("utf-8")
        # Try 1 attempt on primary model first; if UTA out-of-capacity, immediately switch to fallback
        attempts_for_model = 1 if (active_model == model and len(models_to_try) > 1) else max_retries

        for attempt in range(attempts_for_model):
            try:
                proc = subprocess.run(
                    [generate_bin, "-prod", "-json"],
                    input=req_bytes,
                    capture_output=True,
                    timeout=timeout_sec,
                    check=False,
                )
                if proc.returncode == 0 and proc.stdout:
                    resp = json.loads(proc.stdout.decode("utf-8", errors="replace"))
                    candidates = resp.get("candidates") or []
                    if candidates:
                        parts = (candidates[0].get("content") or {}).get("parts") or []
                        text_chunks = [
                            str(p.get("text", ""))
                            for p in parts
                            if not p.get("thought", False) and p.get("text")
                        ]
                        raw_text = "\n".join(text_chunks).strip()
                        parsed = parse_argon_json_response(raw_text)
                        usage = resp.get("usage_metadata") or {}
                        parsed["_argon_meta"] = {
                            "model": active_model,
                            "requested_model": model,
                            "thinking_level": thinking_level,
                            "prompt_token_count": usage.get("prompt_token_count", 0),
                            "candidates_token_count": usage.get("candidates_token_count", 0),
                            "thoughts_token_count": usage.get("thoughts_token_count", 0),
                            "attempt": attempt + 1,
                        }
                        return parsed
                last_err = (
                    proc.stderr.decode("utf-8", errors="replace")[:300]
                    or proc.stdout.decode("utf-8", errors="replace")[:300]
                )
            except Exception as e:
                last_err = f"{type(e).__name__}: {e}"
            if attempt + 1 < attempts_for_model:
                time.sleep(1.0 * (attempt + 1))

    raise RuntimeError(f"Gemini difficulty call failed ({models_to_try}): {last_err}")


def evaluate_family_batch(
    family_items: Sequence[tuple[str, dict[str, Any], dict[str, Any] | None]],
    caller_fn: Callable[[str], dict[str, Any]],
    max_workers: int = 16,
    max_sft_tokens: int = 36000,
    checkpoint_path: str | None = None,
) -> dict[str, dict[str, Any]]:
    """Evaluates a batch of `(family_id, representative_candidate, representative_rollout)` in parallel."""
    results: dict[str, dict[str, Any]] = {}
    if checkpoint_path and gfile.Exists(checkpoint_path):
        with gfile.Open(checkpoint_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                    fid = str(row.get("family_id") or "")
                    # Ignore stale entries from before candidate schema fix where representative_puzzle_id was null
                    if fid and row.get("representative_puzzle_id"):
                        results[fid] = row
                except Exception:
                    continue

    pending = [item for item in family_items if item[0] not in results]
    if not pending:
        return results

    if checkpoint_path:
        gfile.MakeDirs(os.path.dirname(checkpoint_path))
    ckpt_file = gfile.Open(checkpoint_path, "a") if checkpoint_path else None

    def _worker(item: tuple[str, dict[str, Any], dict[str, Any] | None]) -> tuple[str, dict[str, Any]]:
        fid, cand, roll = item
        prompt = build_argon_difficulty_prompt(cand, roll)
        t0 = time.time()
        rating = caller_fn(prompt)
        elapsed = round(time.time() - t0, 2)
        sft_score = compute_composite_sft_value_score(
            rating, rollout_row=roll, candidate=cand, max_sft_tokens=max_sft_tokens
        )
        record = {
            "family_id": fid,
            "representative_puzzle_id": get_candidate_puzzle_id(cand),
            "source_dataset": get_candidate_dataset(cand),
            "has_verified_rollout": bool(
                roll and (roll.get("all_3_levels_verified") or (roll.get("train_pass") and roll.get("correct")))
            ),
            "sft_value_score": sft_score,
            "elapsed_sec": elapsed,
            **rating,
        }
        return fid, record

    t_start = time.time()
    done_count = 0
    try:
        with futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
            future_map = {pool.submit(_worker, item): item[0] for item in pending}
            for fut in futures.as_completed(future_map):
                fid = future_map[fut]
                try:
                    _, rec = fut.result()
                    results[fid] = rec
                    if ckpt_file is not None:
                        ckpt_file.write(json.dumps(rec) + "\n")
                        ckpt_file.flush()
                except Exception as e:
                    print(f"WARNING: family {fid} failed evaluation: {e}")
                done_count += 1
                if done_count % 25 == 0 or done_count == len(pending):
                    rate = done_count / max(0.1, time.time() - t_start)
                    print(
                        f"  [Gemini Difficulty Eval] Evaluated {done_count}/{len(pending)} families "
                        f"(total cached={len(results)}, {rate:.2f} fam/s)"
                    )
    finally:
        if ckpt_file is not None:
            ckpt_file.close()

    return results


def load_candidates_and_rollouts(
    candidates_path: str,
    top10k_rollouts_dir: str,
) -> tuple[
    list[dict[str, Any]],
    dict[str, dict[str, Any]],
    list[tuple[str, dict[str, Any], dict[str, Any] | None]],
]:
    """Loads Top-10K candidates and available DeepSeek-V4.1 v8 rollout shards, grouped by family_id."""
    candidates: list[dict[str, Any]] = []
    with gfile.Open(candidates_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                candidates.append(json.loads(line))

    rollouts_by_task: dict[str, dict[str, Any]] = {}
    shard_paths = data_pipeline.discover_top10k_rollout_shards(top10k_rollouts_dir)
    for spath in shard_paths:
        try:
            rows, _ = data_pipeline.read_rollout_rows_from_file(spath)
            for r in rows:
                tid = str(r.get("task_id") or "")
                if not tid:
                    continue
                prev = rollouts_by_task.get(tid)
                is_ver = bool(
                    r.get("all_3_levels_verified") or (r.get("train_pass") and r.get("correct"))
                )
                prev_ver = bool(
                    prev
                    and (
                        prev.get("all_3_levels_verified")
                        or (prev.get("train_pass") and prev.get("correct"))
                    )
                )
                if prev is None or (is_ver and not prev_ver):
                    rollouts_by_task[tid] = r
        except Exception:
            continue

    by_family: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for c in candidates:
        pid = get_candidate_puzzle_id(c)
        fid = str(
            c.get("family_id")
            or c.get("seed_task_id")
            or data_pipeline.get_parent_puzzle_id(pid)
        )
        by_family[fid].append(c)

    family_items: list[tuple[str, dict[str, Any], dict[str, Any] | None]] = []
    for fid, group in by_family.items():
        best_cand = group[0]
        best_roll = None
        for c in group:
            pid = get_candidate_puzzle_id(c)
            r = rollouts_by_task.get(pid) or rollouts_by_task.get(str(c.get("original_task_id") or ""))
            if r is not None:
                if best_roll is None:
                    best_cand = c
                    best_roll = r
                elif (
                    r.get("all_3_levels_verified") or (r.get("train_pass") and r.get("correct"))
                ) and not (
                    best_roll.get("all_3_levels_verified")
                    or (best_roll.get("train_pass") and best_roll.get("correct"))
                ):
                    best_cand = c
                    best_roll = r
        family_items.append((fid, best_cand, best_roll))

    return candidates, rollouts_by_task, family_items


DATASET_PRIORITY_ORDER = {
    "ARC-Heavy": 0,
    "MindsAI-Synthetic-27Gen": 1,
    "NVARC-Synthetic-103k": 2,
    "Nemotron-RL-ARC-AGI-10k": 3,
    "ConceptARC": 4,
    "RE-ARC": 5,
    "ARC-1.5": 6,
    "Mini-ARC": 7,
    "synth_riddles": 8,
    "ARC-GEN-100K": 9,
    "arc-dataset-tama": 10,
    "PQA": 11,
    "Sequence_ARC": 12,
    "ARC_synthetic_extend": 13,
    "Sort-of-ARC": 14,
    "arc-dataset-diva": 18,
    "1D-ARC": 19,
    "arc_1d": 20,
}


def _family_priority_key(
    item: tuple[str, dict[str, Any], dict[str, Any] | None],
) -> tuple[int, int, int, str]:
    fid, cand, roll = item
    ds = get_candidate_dataset(cand)
    ds_rank = DATASET_PRIORITY_ORDER.get(ds, 15)
    emp = cand.get("empirical_difficulty") or {}
    bucket = str(emp.get("bucket_3level") or "")
    emp_rank = 0 if bucket in ("0/2", "1/2") else 1

    if roll is None:
        return (2, ds_rank, emp_rank, str(fid))
    if roll.get("all_3_levels_verified") or (roll.get("train_pass") and roll.get("correct")):
        # Prefer rollouts that required >=3 tool calls (non-trivial) within the same dataset tier
        tc = int(roll.get("tool_calls", 0))
        tc_rank = 0 if tc >= 3 else 1
        return (0, ds_rank, tc_rank + emp_rank, str(fid))
    return (1, ds_rank, emp_rank, str(fid))


def main(argv: list[str]) -> None:
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")

    print(
        f"Loading Top-10K candidates from {FLAGS.candidates_path} and rollouts from {FLAGS.top10k_rollouts_dir}..."
    )
    candidates, rollouts_by_task, family_items = load_candidates_and_rollouts(
        FLAGS.candidates_path, FLAGS.top10k_rollouts_dir
    )
    matched_fams = sum(1 for _, _, r in family_items if r is not None)
    verified_fams = sum(
        1
        for _, _, r in family_items
        if r is not None
        and (r.get("all_3_levels_verified") or (r.get("train_pass") and r.get("correct")))
    )
    print(
        f"Loaded {len(candidates)} candidates across {len(family_items)} unique parent families; "
        f"matched {len(rollouts_by_task)} rollout tasks ({matched_fams} families with rollouts, "
        f"{verified_fams} families with verified clean SFT rollouts)."
    )

    if FLAGS.prioritize_verified_rollouts:
        family_items.sort(key=_family_priority_key)

    if FLAGS.max_families_to_eval > 0:
        family_items = family_items[: FLAGS.max_families_to_eval]
        print(f"Limiting evaluation to first {len(family_items)} prioritized families.")

    gfile.MakeDirs(FLAGS.output_cns_dir)
    family_ckpt_path = os.path.join(FLAGS.output_cns_dir, "argon_family_ratings.jsonl")

    caller = lambda prompt: call_gemini_argon_cli(
        prompt=prompt,
        generate_bin=FLAGS.generate_bin,
        model=FLAGS.model,
        fallback_model=FLAGS.fallback_model,
        thinking_level=FLAGS.thinking_level,
    )

    family_ratings = evaluate_family_batch(
        family_items=family_items,
        caller_fn=caller,
        max_workers=FLAGS.max_workers,
        max_sft_tokens=FLAGS.max_sft_tokens,
        checkpoint_path=family_ckpt_path,
    )

    # Propagate family ratings + task-specific rollout telemetry to all 10,000 candidates
    task_ratings_path = os.path.join(FLAGS.output_cns_dir, "argon_top10k_task_ratings.jsonl")
    hard_sft_tasks_path = os.path.join(
        FLAGS.output_cns_dir, "argon_hard_for_qwen38_sft_candidates.jsonl"
    )

    tier_counts: collections.Counter[str] = collections.Counter()
    diff_hist: collections.Counter[int] = collections.Counter()
    mode_counts: collections.Counter[str] = collections.Counter()
    model_used_counts: collections.Counter[str] = collections.Counter()
    by_dataset_stats: dict[str, dict[str, int]] = collections.defaultdict(
        lambda: {"total_rated": 0, "hard_recommended": 0, "verified_hard_sft": 0, "trivial_skipped": 0}
    )

    rated_task_count = 0
    hard_recommended_count = 0
    verified_hard_sft_count = 0

    for fam_r in family_ratings.values():
        meta = fam_r.get("_argon_meta") or {}
        model_used_counts[str(meta.get("model") or FLAGS.model)] += 1

    with gfile.Open(task_ratings_path, "w") as f_all, gfile.Open(hard_sft_tasks_path, "w") as f_hard:
        for cand in candidates:
            pid = get_candidate_puzzle_id(cand)
            fid = str(
                cand.get("family_id")
                or cand.get("seed_task_id")
                or data_pipeline.get_parent_puzzle_id(pid)
            )
            fam_rating = family_ratings.get(fid)
            if fam_rating is None:
                continue

            rated_task_count += 1
            roll = rollouts_by_task.get(pid) or rollouts_by_task.get(str(cand.get("original_task_id") or ""))
            task_sft_score = compute_composite_sft_value_score(
                fam_rating,
                rollout_row=roll,
                candidate=cand,
                max_sft_tokens=FLAGS.max_sft_tokens,
            )
            q_diff = int(fam_rating["qwen38_difficulty_score"])
            recommend = bool(fam_rating["sft_recommend_include"])
            tier = str(fam_rating["sft_priority_tier"])
            ds = get_candidate_dataset(cand)

            is_verified = bool(
                roll
                and (
                    roll.get("all_3_levels_verified")
                    or (roll.get("train_pass") and roll.get("correct"))
                )
            )
            total_toks = (
                int(roll.get("prompt_tokens", 0)) + int(roll.get("completion_tokens", 0))
                if roll is not None
                else 0
            )
            is_hard_for_qwen = (
                FLAGS.min_qwen38_difficulty <= q_diff <= FLAGS.max_qwen38_difficulty
                and recommend
                and tier in ("tier1_high_value_hard_for_qwen", "tier2_medium_value_composition")
            )
            is_verified_hard_sft = (
                is_hard_for_qwen
                and is_verified
                and (total_toks == 0 or total_toks <= FLAGS.max_sft_tokens)
            )

            tier_counts[tier] += 1
            diff_hist[q_diff] += 1
            for m in fam_rating.get("qwen38_failure_modes_targeted") or []:
                mode_counts[m] += 1

            by_dataset_stats[ds]["total_rated"] += 1
            if q_diff <= 4:
                by_dataset_stats[ds]["trivial_skipped"] += 1
            if is_hard_for_qwen:
                hard_recommended_count += 1
                by_dataset_stats[ds]["hard_recommended"] += 1
            if is_verified_hard_sft:
                verified_hard_sft_count += 1
                by_dataset_stats[ds]["verified_hard_sft"] += 1

            task_row = {
                "puzzle_id": pid,
                "family_id": fid,
                "source_dataset": ds,
                "qwen38_difficulty_score": q_diff,
                "algorithmic_complexity": fam_rating["algorithmic_complexity"],
                "qwen38_failure_modes_targeted": fam_rating["qwen38_failure_modes_targeted"],
                "transformation_rule_summary": fam_rating["transformation_rule_summary"],
                "solution_code_quality": fam_rating["solution_code_quality"],
                "sft_recommend_include": recommend,
                "sft_priority_tier": tier,
                "qwen38_sft_rationale": fam_rating["qwen38_sft_rationale"],
                "sft_value_score": task_sft_score,
                "is_hard_for_qwen38": is_hard_for_qwen,
                "has_verified_rollout": is_verified,
                "is_verified_hard_sft": is_verified_hard_sft,
                "rollout_total_tokens": total_toks,
            }
            f_all.write(json.dumps(task_row) + "\n")
            if is_hard_for_qwen:
                f_hard.write(json.dumps(task_row) + "\n")

    summary = {
        "primary_model": FLAGS.model,
        "fallback_model": FLAGS.fallback_model,
        "models_used_by_family": dict(model_used_counts),
        "thinking_level": FLAGS.thinking_level,
        "unique_families_evaluated": len(family_ratings),
        "tasks_rated_via_family_and_rollout": rated_task_count,
        "rollouts_matched": len(rollouts_by_task),
        "families_with_rollouts": matched_fams,
        "families_with_verified_rollouts": verified_fams,
        "hard_for_qwen38_recommended_tasks": hard_recommended_count,
        "verified_hard_for_qwen38_sft_trajectories": verified_hard_sft_count,
        "min_qwen38_difficulty": FLAGS.min_qwen38_difficulty,
        "max_qwen38_difficulty": FLAGS.max_qwen38_difficulty,
        "max_sft_tokens": FLAGS.max_sft_tokens,
        "difficulty_score_histogram": {
            str(k): int(v) for k, v in sorted(diff_hist.items())
        },
        "priority_tier_counts": dict(tier_counts),
        "qwen38_failure_modes_targeted_counts": dict(mode_counts.most_common()),
        "by_dataset": dict(sorted(by_dataset_stats.items(), key=lambda x: -x[1]["total_rated"])),
        "family_ratings_path": family_ckpt_path,
        "task_ratings_path": task_ratings_path,
        "hard_sft_tasks_path": hard_sft_tasks_path,
    }
    summary_path = os.path.join(FLAGS.output_cns_dir, "argon_difficulty_summary.json")
    with gfile.Open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote difficulty evaluation summary to {summary_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    app.run(main)

