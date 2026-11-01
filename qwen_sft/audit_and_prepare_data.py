"""Audits Vu's E031-E038 DeepSeek rollouts (+ Top-10K v8 traces) and builds the verified 90/10 SFT dataset on CNS.

Round-2 additions (all opt-in via flags; defaults reproduce the round-1 E031 dataset):
  * `--max_completion_tokens` / `--max_tool_calls`: drop trajectories that could not finish under the inference budget.
  * `--drop_rescued_or_nudged`: drop Top-10K rows where the harness rescued code or injected cap nudges.
  * `--max_per_parent_vu` / `--max_per_task_top10k`: cap near-duplicate trajectories, keeping the SHORTEST verified ones.
  * `--upsample_arc2_train`: replicate real ARC-AGI-2 training-task trajectories in the TRAIN split only.
  * `--tier_weights`: stochastic per-tier up/down-sampling of Top-10K rows in the TRAIN split only.

E065 text+image rollouts (`--subset=e065_images|e065_text|e065_all`, DeepSeek-V4.1 only):
  * rows are de-duplicated per rollout (`pair==0`, `correct_all` over all test pairs), `error` rows dropped, and
    verified := correct_all & train_pass & answer_source=='submit' & stop_reason=='submitted';
  * image prompts are rendered with one `<｜deepseek_image｜>` placeholder per image, expanded to the full image
    span at tokenization time (`--max_seq_len` applies to the EXPANDED length; `--max_images_per_sample` caps images);
  * `{split}_tokens.npz` gains a flat int8 `token_types` array (-1 text, 0..3 image span types) and a sidecar
    `{split}_images.pkl` (list aligned with the split's trajectories; each element a list of PNG bytes).
"""

from __future__ import annotations

from collections import Counter
import io
import json
import os
import pickle
import re
import time
from typing import Any

from absl import app
from absl import flags
import numpy as np
import transformers

from google3.experimental.users.arazdai.kaggle.sft import data_pipeline
from google3.experimental.users.arazdai.kaggle.sft import dsv41_image_tokens
from google3.pyglib import gfile

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "rollouts_dir",
    "/cns/lu-d/home/arazdai/kaggle/arc_sft_data/livobereshka_e031_e036_rollouts/rollouts",
    "CNS directory containing Vu's rollout parquet files.",
)
flags.DEFINE_string(
    "manifest_path",
    "/cns/lu-d/home/arazdai/kaggle/arc_sft_data/livobereshka_e031_e036_rollouts/tasks/manifest.jsonl",
    "Path to tasks/manifest.jsonl for parent & pair metadata.",
)
flags.DEFINE_string(
    "eval_dir",
    "/cns/lu-d/home/arazdai/kaggle/ARC-AGI-2/data/evaluation",
    "Path to official ARC-AGI-2 120 evaluation tasks.",
)
flags.DEFINE_string(
    "tokenizer_path",
    "/cns/it-d/home/ads-video-creation-auto-plx/gaoxiao/models/Qwen3.8-27B",
    "Path to Qwen3.8-27B tokenizer on CNS.",
)
flags.DEFINE_string(
    "output_cns_dir",
    "/cns/lu-d/home/arazdai/kaggle/sft/data/e031_pilot",
    "CNS directory to write tokenized SFT train/val shards and manifests.",
)
flags.DEFINE_string(
    "summary_json_path",
    "",
    "Optional local path to write the JSON audit summary.",
)
flags.DEFINE_string(
    "top10k_rollouts_dir",
    "/cns/lu-d/home/arazdai/kaggle/arc_sft_data/sft_prep_pipeline/deepseek_v41_rollouts_top10k_v8_pass1",
    "CNS directory containing Top-10K Pass@1 DeepSeek-V4.1 v8 rollout shards (rollouts_shard_*.parquet / .jsonl).",
)
flags.DEFINE_enum(
    "subset",
    "e031",
    ["e031", "all", "top10k_v8", "all_with_top10k_v8", "e065_images", "e065_text", "e065_all", "e065_all_with_top10k_v8"],
    "Which rollout files to include in the SFT dataset: 'e031', 'all', 'top10k_v8', 'all_with_top10k_v8', "
    "the E065 text+image sets 'e065_images', 'e065_text', 'e065_all' (images + text twins), or "
    "'e065_all_with_top10k_v8' (E065 images + text twins + E082 text parts + Top-10K v8 + optional extra shards).",
)
flags.DEFINE_string(
    "e065_rollouts_dir",
    "/cns/lu-d/home/arazdai/kaggle/arc_sft_data/livobereshka_e065_rollouts/rollouts",
    "CNS directory containing the six E065 parquet files (e065_{images,text}_{arc2_train,nvarc_b0,nvarc_b1}.parquet).",
)
flags.DEFINE_string(
    "e065_manifest_path",
    "/cns/lu-d/home/arazdai/kaggle/arc_sft_data/livobereshka_e065_rollouts/tasks/manifest.jsonl",
    "E065 tasks/manifest.jsonl (parent fallback, n_train, expected test pairs). Empty or missing = not used.",
)
flags.DEFINE_string(
    "e065_extra_rollouts_dir",
    "",
    "Optional CNS directory containing corrected+rewritten E065 rollout parquet shards "
    "(e.g. /cns/lu-d/home/arazdai/kaggle/arc_sft_data/e065_hardest_rollouts/corrected_rewritten/shards) "
    "to include alongside e065_rollouts_dir when --subset in ('e065_images', 'e065_all', 'e065_all_with_top10k_v8').",
)
flags.DEFINE_list(
    "e065_text_part_dirs",
    [],
    "Extra directories of E065-schema TEXT parquet parts (e.g. .../rollouts/e082_text_nvarc_b2) audited like "
    "the e065_text_* files when --subset includes E065 text.",
)
flags.DEFINE_enum(
    "e065_extra_modality",
    "both",
    ["both", "text", "image"],
    "Which modality of the corrected+rewritten extra shards to keep (each rescued task has an image row and a text twin).",
)
flags.DEFINE_bool(
    "e065_extra_skip_existing_modality_tasks",
    True,
    "Skip an extra (rescued) row when E065 already has a verified rollout of the same task AND modality "
    "(the rescue was run on image failures; 30% of those tasks have a passing text twin).",
)
flags.DEFINE_bool(
    "e065_extra_drop_meta_leaks",
    True,
    "Drop rescued rows whose assistant reasoning refers to the hidden rewrite instructions / verified solution "
    "('verified implementation', 'instructions say', ...) or to a previous attempt ('prev', 'previous failed', "
    "'diagnostic', ...), and the canned sandbox-synthesis 2-turn rows.",
)
flags.DEFINE_integer(
    "e065_extra_upsample",
    1,
    "Number of copies of each kept rescued (corrected+rewritten) trajectory in the TRAIN split (1 = no upsampling).",
)
flags.DEFINE_string(
    "pin_split_from_data_dir",
    "",
    "Optional previous-round dataset dir (with train_manifest.jsonl / val_manifest.jsonl): its train parents are "
    "forced to train and its val parents to val so only NEW parents are shuffled (no train/val leakage across rounds).",
)
flags.DEFINE_integer(
    "max_seq_len",
    65536,
    "Maximum token length; overlong trajectories are dropped rather than truncated mid-turn.",
)
flags.DEFINE_float(
    "val_fraction",
    0.10,
    "Fraction of parent puzzle IDs held out for validation.",
)
flags.DEFINE_integer(
    "seed",
    20260405,
    "Deterministic RNG seed for 90/10 parent split.",
)
# ---- Round-2 data-recipe flags (defaults = round-1 behaviour) ----
flags.DEFINE_integer(
    "max_completion_tokens",
    0,
    "Drop trajectories whose total completion_tokens exceed this (0 = off). Keeps solutions that fit the inference budget.",
)
flags.DEFINE_integer("max_tool_calls", 0, "Drop trajectories with more than this many tool calls (0 = off).")
flags.DEFINE_bool(
    "drop_rescued_or_nudged",
    False,
    "Drop Top-10K rows with code_rescues > 0 or cap_nudges > 0 (harness interventions the student cannot reproduce).",
)
flags.DEFINE_bool(
    "top10k_require_verified_clean_sft",
    True,
    "For rows carrying a `verified_clean_sft` column, require it to be True.",
)
flags.DEFINE_integer(
    "max_per_parent_vu",
    0,
    "Cap verified trajectories per parent puzzle for Vu's synthetic batches (nvarc/e038; never arc2_train), keeping the shortest (0 = off).",
)
flags.DEFINE_integer(
    "max_per_task_top10k",
    0,
    "Cap verified trajectories per task_id for Top-10K rows, keeping the shortest (0 = off).",
)
flags.DEFINE_integer(
    "upsample_arc2_train",
    1,
    "Number of copies of each arc2_train trajectory in the TRAIN split (1 = no upsampling).",
)
flags.DEFINE_string(
    "tier_weights",
    "",
    "Comma-separated Top-10K learning-tier weights applied to the TRAIN split, e.g. 'tier1:1,tier2:1.5,tier3:0.5'. "
    "Weight w keeps floor(w) copies plus one more with probability frac(w).",
)
flags.DEFINE_enum(
    "model_family",
    "qwen38",
    ["qwen38", "deepseek_v41"],
    "Target model family for chat formatting and tokenization ('qwen38' or 'deepseek_v41').",
)
flags.DEFINE_string(
    "deepseek_tokenizer_file",
    "/usr/local/google/home/arazdai/.gemini/jetski/brain/d6923b28-7f5c-466e-a899-eb04a6ff1140/scratch/kaggle_e128_meta/tokenizer.json",
    "Path to DeepSeek-V4.1 tokenizer.json file (used when --model_family=deepseek_v41).",
)
flags.DEFINE_integer(
    "deepseek_reasoning_effort",
    50,
    "Numeric reasoning effort prefix for DeepSeek-V4.1 thinking mode (50 = 'low').",
)
flags.DEFINE_integer(
    "max_py_error_turns",
    -1,
    "Drop trajectories with more than this many Python error turns (-1 = off, 0 = zero Python errors).",
)
flags.DEFINE_integer(
    "max_ident_syntax_error_turns",
    -1,
    "Drop trajectories with more than this many NameError/AttributeError/ImportError/SyntaxError turns (-1 = off, 0 = zero typos).",
)
flags.DEFINE_integer(
    "max_wasted_post_pass_turns",
    -1,
    "Drop trajectories that spend more than this many extra assistant turns after all training pairs pass (-1 = off, 0 = submit immediately on next turn).",
)
flags.DEFINE_integer(
    "max_post_pass_tokens",
    -1,
    "Drop trajectories that generate more than this many completion tokens after all training pairs pass (-1 = off).",
)
flags.DEFINE_integer(
    "wic_alnum_weight",
    10,
    "Token-level loss weight multiplier for word-internal alphanumeric continuation (WIC-alnum) tokens in DeepSeek-V4.1 SFT.",
)
flags.DEFINE_integer(
    "tool_call_weight",
    2,
    "Token-level loss weight multiplier for DSML tool-call blocks in DeepSeek-V4.1 SFT.",
)
flags.DEFINE_integer(
    "final_submit_weight",
    5,
    "Token-level loss weight multiplier for the final submit_answer turn in DeepSeek-V4.1 SFT.",
)
flags.DEFINE_integer(
    "max_images_per_sample",
    32,
    "Drop rollouts carrying more than this many images (serving cap of the image harness; 0 = off).",
)
flags.DEFINE_integer(
    "max_records_per_file",
    0,
    "Smoke-run aid: only read the first N rows of every rollout file (0 = all rows).",
)

E031_PARQUETS = (
    "e031_arc2_train.parquet",
    "e031_nvarc_b0.parquet",
)

ALL_PARQUETS = (
    "e031_arc2_train.parquet",
    "e031_nvarc_b0.parquet",
    "e036_nvarc_b1.parquet",
    "e037_openrouter_deepinfra.parquet",
    "e038_openrouter_deepinfra_b1r.parquet",
)

# E065: E058 image harness rollouts ("images") and their E046 text-only twins ("text") of the same tasks.
E065_IMAGE_PARQUETS = (
    "e065_images_arc2_train.parquet",
    "e065_images_nvarc_b0.parquet",
    "e065_images_nvarc_b1.parquet",
)

E065_TEXT_PARQUETS = (
    "e065_text_arc2_train.parquet",
    "e065_text_nvarc_b0.parquet",
    "e065_text_nvarc_b1.parquet",
)

# E065_SUBSETS & friends are defined below next to the rescued-row helpers.

TIER_ALIASES = {
    "tier1": "tier1_frontier_1of2",
    "tier2": "tier2_hard_unsolved_0of2",
    "tier3": "tier3_verified_anchor_2of2",
}


audit_parquet_file = data_pipeline.audit_rollout_file
discover_top10k_rollout_shards = data_pipeline.discover_top10k_rollout_shards
discover_e065_extra_parquet_shards = data_pipeline.discover_e065_extra_parquet_shards


def parse_tier_weights(spec: str) -> dict[str, float]:
    """Parses 'tier1:1,tier2:1.5' into {'tier1_frontier_1of2': 1.0, ...}."""
    weights: dict[str, float] = {}
    for part in (spec or "").split(","):
        part = part.strip()
        if not part:
            continue
        name, _, val = part.partition(":")
        name = name.strip()
        weights[TIER_ALIASES.get(name, name)] = float(val)
    return weights


E065_SUBSETS = ("e065_images", "e065_text", "e065_all", "e065_all_with_top10k_v8")
E065_SUBSETS_WITH_IMAGES = ("e065_images", "e065_all", "e065_all_with_top10k_v8")
E065_SUBSETS_WITH_TEXT = ("e065_text", "e065_all", "e065_all_with_top10k_v8")
TOP10K_SUBSETS = ("top10k_v8", "all_with_top10k_v8", "e065_all_with_top10k_v8")

# Rescued (corrected+rewritten) rows: phrases that reveal the hidden rewrite instructions / verified solution.
RESCUE_META_LEAK_PHRASES = (
    "previous attempt", "prior attempt", "earlier attempt", "failed attempt", "previous rollout",
    "reference solution", "reference insight", "reference solve", "reference code", "reference implementation",
    "provided solution", "given solution", "teacher hint", "rewrite", "verified implementation", "verified python",
    "verified solution", "verified code", "solution trajectory", "instructions from user", "instructions say",
    "user instructions", "as if solving", "already discovered", "key reasoning insight", "do not mention",
    "the given code", "provided code", "given implementation",
)
# Sandbox-synthesis fallback rows carry the correction-stage thought and this canned turn-2 sentence.
RESCUE_SYNTHESIS_SENTENCE = "The transformation function `solve` passes all training pairs cleanly."
_RESCUE_PRIOR_REF_RE = re.compile(
    r"\b(prev|previous(ly)? (failed|rule|code|attempt|approach)|old rule|diagnos\w*|why (the )?previous|earlier rule)\b"
)


def rescued_row_leak_reasons(messages: list[dict[str, Any]]) -> list[str]:
    """Returns why a rescued (corrected+rewritten) trajectory should be dropped (empty = clean)."""
    texts: list[str] = []
    for m in messages[2:]:
        if m.get("role") == "assistant":
            texts.append(str(m.get("reasoning_content") or ""))
            texts.append(str(m.get("content") or ""))
    joined = "\n".join(texts)
    low = joined.lower()
    reasons: list[str] = []
    if RESCUE_SYNTHESIS_SENTENCE in joined:
        reasons.append("sandbox_synthesis")
    for ph in RESCUE_META_LEAK_PHRASES:
        if ph in low:
            reasons.append(f"meta:{ph}")
            break
    if _RESCUE_PRIOR_REF_RE.search(low):
        reasons.append("prior_attempt_reference")
    return reasons


def is_rescued_extra(rec: dict[str, Any]) -> bool:
    return "corrected_rewritten" in str(rec.get("rollout_file", ""))


def discover_text_part_parquets(part_dir: str) -> list[str]:
    """Lists `*.parquet` parts of an E065-schema text batch directory (e.g. e082_text_nvarc_b2/part-*.parquet)."""
    if not part_dir or not gfile.Exists(part_dir):
        return []
    return [
        os.path.join(part_dir, f)
        for f in sorted(gfile.ListDir(part_dir))
        if f.endswith(".parquet") and not f.startswith(".")
    ]


def load_pinned_parents(data_dir: str) -> tuple[set[str], set[str]]:
    """Reads parent_ids of a previous round's train/val manifests (empty sets when absent)."""
    out: list[set[str]] = []
    for split in ("train", "val"):
        path = os.path.join(data_dir, f"{split}_manifest.jsonl") if data_dir else ""
        parents: set[str] = set()
        if path and gfile.Exists(path):
            with gfile.Open(path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    parents.add(str(row.get("parent_id") or data_pipeline.get_parent_puzzle_id(str(row["task_id"]), row.get("parent"))))
        out.append(parents)
    return out[0], out[1]


def apply_record_filters(
    records: list[dict[str, Any]],
    max_completion_tokens: int,
    max_tool_calls: int,
    drop_rescued_or_nudged: bool,
    require_verified_clean_sft: bool,
    max_py_error_turns: int = -1,
    max_ident_syntax_error_turns: int = -1,
    max_wasted_post_pass_turns: int = -1,
    max_post_pass_tokens: int = -1,
    max_images_per_sample: int = 0,
) -> tuple[list[dict[str, Any]], Counter[str]]:
    """Applies row-level filters; returns kept records and drop-reason counts."""
    kept: list[dict[str, Any]] = []
    drops: Counter[str] = Counter()
    for rec in records:
        if require_verified_clean_sft and "verified_clean_sft" in rec and not bool(rec["verified_clean_sft"]):
            drops["not_verified_clean_sft"] += 1
            continue
        if drop_rescued_or_nudged and (
            int(rec.get("code_rescues", 0) or 0) > 0
            or int(rec.get("cap_nudges", 0) or 0) > 0
            or int(rec.get("auto_repairs", 0) or 0) > 0
        ):
            drops["rescued_or_nudged"] += 1
            continue
        if max_ident_syntax_error_turns >= 0 and int(rec.get("ident_syntax_error_turns", 0)) > max_ident_syntax_error_turns:
            drops["ident_syntax_errors_over_cap"] += 1
            continue
        if max_py_error_turns >= 0 and int(rec.get("py_error_turns", 0)) > max_py_error_turns:
            drops["py_errors_over_cap"] += 1
            continue
        if max_wasted_post_pass_turns >= 0 and int(rec.get("wasted_post_pass_turns", 0)) > max_wasted_post_pass_turns:
            drops["wasted_post_pass_turns_over_cap"] += 1
            continue
        if max_post_pass_tokens >= 0 and int(rec.get("post_pass_completion_tokens", 0)) > max_post_pass_tokens:
            drops["post_pass_tokens_over_cap"] += 1
            continue
        if max_completion_tokens > 0 and int(rec.get("completion_tokens", 0)) > max_completion_tokens:
            drops["completion_tokens_over_cap"] += 1
            continue
        if max_tool_calls > 0 and int(rec.get("tool_calls", 0)) > max_tool_calls:
            drops["tool_calls_over_cap"] += 1
            continue
        if max_images_per_sample > 0 and int(rec.get("n_images", 0) or 0) > max_images_per_sample:
            drops["images_over_cap"] += 1
            continue
        kept.append(rec)
    return kept, drops


def cap_group_keep_shortest(
    records: list[dict[str, Any]],
    group_key_fn,
    max_per_group: int,
    applies_fn,
) -> tuple[list[dict[str, Any]], int]:
    """Keeps at most `max_per_group` records per group (shortest completion first) for records where `applies_fn` is True."""
    if max_per_group <= 0:
        return list(records), 0
    groups: dict[str, list[dict[str, Any]]] = {}
    passthrough: list[dict[str, Any]] = []
    for rec in records:
        if applies_fn(rec):
            groups.setdefault(group_key_fn(rec), []).append(rec)
        else:
            passthrough.append(rec)
    kept: list[dict[str, Any]] = list(passthrough)
    dropped = 0
    for _, group in sorted(groups.items()):
        group_sorted = sorted(group, key=lambda r: (int(r.get("completion_tokens", 0)), int(r.get("turns", 0)), r["task_id"]))
        kept.extend(group_sorted[:max_per_group])
        dropped += max(0, len(group_sorted) - max_per_group)
    return kept, dropped


def is_top10k(rec: dict[str, Any]) -> bool:
    return str(rec.get("batch", "")).startswith("top10k")


def is_arc2_train(rec: dict[str, Any]) -> bool:
    """Real ARC-AGI-2 training-task trajectories (Vu's e031_arc2_train: batch='arc2_train', source='kaggle_arc2_training')."""
    return str(rec.get("batch", "")) == "arc2_train" or str(rec.get("source", "")) in (
        "arc2_train",
        "kaggle_arc2_training",
    )


def is_vu_synthetic(rec: dict[str, Any]) -> bool:
    return (not is_top10k(rec)) and not is_arc2_train(rec)


def is_e065(rec: dict[str, Any]) -> bool:
    """E065 records carry a `modality` ('image' | 'text') set by `audit_e065_rollout_file`."""
    return "modality" in rec


def vu_parent_cap_key(rec: dict[str, Any]) -> str:
    """Group key for `--max_per_parent_vu`: parent puzzle, and for E065 also the modality.

    With `--subset=e065_all` every nvarc task has an image rollout and a text twin; capping per
    (parent, modality) keeps up to N shortest of EACH instead of letting the (usually shorter) text
    twins crowd out the image rollouts of the same parent.
    """
    if is_e065(rec):
        return f"{rec['parent_id']}|{rec.get('modality', 'text')}"
    return str(rec["parent_id"])


def main(argv: list[str]) -> None:
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")

    print(f"Building ARC-AGI-2 evaluation contamination index from {FLAGS.eval_dir}...")
    eval_index = data_pipeline.build_eval_contamination_index(FLAGS.eval_dir)
    print(f"Indexed {len(eval_index['eval_task_ids'])} official evaluation tasks.")

    all_file_stats: list[dict[str, Any]] = []
    e031_verified: list[dict[str, Any]] = []
    all_verified: list[dict[str, Any]] = []
    top10k_verified: list[dict[str, Any]] = []

    if FLAGS.subset in ("e031", "all", "all_with_top10k_v8"):
        for fname in ALL_PARQUETS:
            fpath = os.path.join(FLAGS.rollouts_dir, fname)
            if not gfile.Exists(fpath):
                print(f"WARNING: {fpath} not found, skipping.")
                continue
            print(f"Auditing {fname}...")
            fstats, recs = audit_parquet_file(fpath, eval_index)
            all_file_stats.append(fstats)
            print(
                f"  -> rows={fstats['rows']}, unique_tasks={fstats['unique_task_ids']}, "
                f"unique_parents={fstats['unique_parent_puzzles']}, "
                f"correct={fstats['correct_count']} ({fstats['correct_rate']*100:.1f}%), "
                f"train_pass={fstats['train_pass_count']} ({fstats['train_pass_rate']*100:.1f}%), "
                f"train_pass_but_wrong={fstats['train_pass_but_test_wrong']} "
                f"({fstats['false_positive_demo_rate']*100:.1f}% of train_pass), "
                f"verified_clean={fstats['verified_clean_count']}"
            )
            if fname in E031_PARQUETS:
                e031_verified.extend(recs)
            if fname != "e037_openrouter_deepinfra.parquet":
                all_verified.extend(recs)

    if FLAGS.subset in TOP10K_SUBSETS:
        shard_paths = discover_top10k_rollout_shards(FLAGS.top10k_rollouts_dir)
        print(f"Auditing {len(shard_paths)} Top-10K v8 rollout shards from {FLAGS.top10k_rollouts_dir}...")
        t_sh = time.time()
        for s_idx, spath in enumerate(shard_paths):
            try:
                fstats, recs = audit_parquet_file(spath, eval_index)
            except Exception as e:  # pylint: disable=broad-exception-caught  (a shard may be mid-write while the rollout job runs)
                print(f"  WARNING: skipping shard {spath}: {e}")
                continue
            all_file_stats.append(fstats)
            top10k_verified.extend(recs)
            if (s_idx + 1) % 16 == 0:
                print(f"  ... {s_idx + 1}/{len(shard_paths)} shards, verified so far={len(top10k_verified)} ({time.time() - t_sh:.0f}s)")
        print(f"  -> Top-10K v8 verified clean trajectories: {len(top10k_verified)}")

    e065_verified: list[dict[str, Any]] = []
    extra_drop_reasons: Counter[str] = Counter()
    if FLAGS.subset in E065_SUBSETS:
        if FLAGS.model_family != "deepseek_v41":
            raise app.UsageError("E065 subsets are DeepSeek-V4.1 only (--model_family=deepseek_v41).")
        e065_manifest = data_pipeline.load_e065_manifest(FLAGS.e065_manifest_path)
        print(f"Loaded E065 manifest: {len(e065_manifest)} tasks from {FLAGS.e065_manifest_path or '<none>'}")
        e065_paths: list[str] = []
        if FLAGS.subset in E065_SUBSETS_WITH_IMAGES:
            e065_paths += [os.path.join(FLAGS.e065_rollouts_dir, f) for f in E065_IMAGE_PARQUETS]
        if FLAGS.subset in E065_SUBSETS_WITH_TEXT:
            e065_paths += [os.path.join(FLAGS.e065_rollouts_dir, f) for f in E065_TEXT_PARQUETS]
            for part_dir in FLAGS.e065_text_part_dirs:
                parts = discover_text_part_parquets(part_dir)
                print(f"E065-schema text parts in {part_dir}: {len(parts)}")
                e065_paths += parts
        for fpath in e065_paths:
            fname = os.path.basename(fpath)
            if not gfile.Exists(fpath):
                print(f"WARNING: {fpath} not found, skipping.")
                continue
            print(f"Auditing E065 {fname}...")
            fstats, recs = data_pipeline.audit_e065_rollout_file(
                fpath, eval_index, manifest=e065_manifest, max_rows=FLAGS.max_records_per_file
            )
            if fname.startswith("part-"):
                # Disambiguate E082-style part files in the per-file stats / rollout_file field.
                tag = os.path.basename(os.path.dirname(fpath))
                fstats["file"] = f"{tag}/{fname}"
                for r in recs:
                    r["rollout_file"] = f"{tag}/{fname}"
            all_file_stats.append(fstats)
            print(
                f"  -> rows={fstats['rows']}, rollouts={fstats['rollouts']} (multi-pair {fstats['multi_pair_rollouts']}), "
                f"unique_tasks={fstats['unique_task_ids']}, unique_parents={fstats['unique_parent_puzzles']}, "
                f"correct_all={fstats['correct_all_rollouts']}, error_rows={fstats['error_rows']}, "
                f"verified_clean={fstats['verified_clean_count']} (with images {fstats['verified_with_images']}, "
                f"images/rollout mean {fstats['images_per_verified_rollout_mean']} max {fstats['images_per_verified_rollout_max']}), "
                f"drops={fstats['drops']}, {fstats['elapsed_sec']}s"
            )
            e065_verified.extend(recs)
        if FLAGS.e065_extra_rollouts_dir and FLAGS.subset in E065_SUBSETS_WITH_IMAGES:
            extra_shards = discover_e065_extra_parquet_shards(FLAGS.e065_extra_rollouts_dir)
            print(f"Auditing {len(extra_shards)} extra corrected+rewritten E065 shards from {FLAGS.e065_extra_rollouts_dir}...")
            existing_by_modality = {(r["task_id"], r.get("modality", "text")) for r in e065_verified}
            seen_extra: set[tuple[str, str]] = set()
            added_extra = 0
            for spath in extra_shards:
                try:
                    fstats, recs = data_pipeline.audit_e065_rollout_file(
                        spath, eval_index, manifest=e065_manifest, max_rows=FLAGS.max_records_per_file
                    )
                except Exception as e:  # pylint: disable=broad-exception-caught
                    print(f"  WARNING: skipping extra E065 shard {spath}: {e}")
                    continue
                all_file_stats.append(fstats)
                for r in recs:
                    mod = str(r.get("modality", "text"))
                    key = (str(r["task_id"]), mod)
                    if FLAGS.e065_extra_modality != "both" and mod != FLAGS.e065_extra_modality:
                        extra_drop_reasons["modality_filtered"] += 1
                        continue
                    if key in seen_extra:
                        extra_drop_reasons["duplicate_extra_row"] += 1
                        continue
                    if FLAGS.e065_extra_skip_existing_modality_tasks and key in existing_by_modality:
                        extra_drop_reasons["task_already_verified_in_e065_same_modality"] += 1
                        continue
                    if FLAGS.e065_extra_drop_meta_leaks:
                        reasons = rescued_row_leak_reasons(r["messages"])
                        if reasons:
                            for rs in reasons:
                                extra_drop_reasons[rs] += 1
                            extra_drop_reasons["_dropped_for_leaks_total"] += 1
                            continue
                    seen_extra.add(key)
                    e065_verified.append(r)
                    added_extra += 1
            print(
                f"  -> Added {added_extra} corrected+rewritten E065 rollouts (modality={FLAGS.e065_extra_modality}); "
                f"extra drop reasons={dict(extra_drop_reasons)}; total e065_verified={len(e065_verified)}."
            )

    if FLAGS.subset == "e031":
        selected_records = e031_verified
    elif FLAGS.subset == "all":
        selected_records = all_verified
    elif FLAGS.subset == "top10k_v8":
        selected_records = top10k_verified
    elif FLAGS.subset == "e065_all_with_top10k_v8":
        selected_records = e065_verified + top10k_verified
    elif FLAGS.subset in E065_SUBSETS:
        selected_records = e065_verified
    else:
        selected_records = all_verified + top10k_verified

    n_selected_raw = len(selected_records)
    print(
        f"\nSelected subset '{FLAGS.subset}': {n_selected_raw} verified clean trajectories "
        f"prior to round-2 filters, tokenization & length filter."
    )

    def _breakdown(recs: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "n": len(recs),
            "by_batch": dict(Counter(str(r.get("batch")) for r in recs)),
            "by_source": dict(Counter(str(r.get("source")) for r in recs)),
            "by_tier": dict(Counter(str(r.get("learning_tier", "n/a")) for r in recs)),
            "unique_parents": len({r["parent_id"] for r in recs}),
            "unique_task_ids": len({r["task_id"] for r in recs}),
        }

    stage_stats: dict[str, Any] = {"0_selected_raw": _breakdown(selected_records)}

    # ---- Round-2 row-level filters ----
    selected_records, filter_drops = apply_record_filters(
        selected_records,
        max_completion_tokens=FLAGS.max_completion_tokens,
        max_tool_calls=FLAGS.max_tool_calls,
        drop_rescued_or_nudged=FLAGS.drop_rescued_or_nudged,
        require_verified_clean_sft=FLAGS.top10k_require_verified_clean_sft,
        max_py_error_turns=FLAGS.max_py_error_turns,
        max_ident_syntax_error_turns=FLAGS.max_ident_syntax_error_turns,
        max_wasted_post_pass_turns=FLAGS.max_wasted_post_pass_turns,
        max_post_pass_tokens=FLAGS.max_post_pass_tokens,
        max_images_per_sample=FLAGS.max_images_per_sample,
    )
    stage_stats["1_after_row_filters"] = {**_breakdown(selected_records), "drops": dict(filter_drops)}
    print(f"After row filters: {len(selected_records)} (drops={dict(filter_drops)})")

    # ---- Per-group caps (shortest first) ----
    selected_records, n_cap_vu = cap_group_keep_shortest(
        selected_records,
        group_key_fn=vu_parent_cap_key,
        max_per_group=FLAGS.max_per_parent_vu,
        applies_fn=is_vu_synthetic,
    )
    selected_records, n_cap_top = cap_group_keep_shortest(
        selected_records,
        group_key_fn=lambda r: r["task_id"],
        max_per_group=FLAGS.max_per_task_top10k,
        applies_fn=is_top10k,
    )
    stage_stats["2_after_group_caps"] = {
        **_breakdown(selected_records),
        "dropped_vu_per_parent_cap": n_cap_vu,
        "dropped_top10k_per_task_cap": n_cap_top,
    }
    print(f"After group caps: {len(selected_records)} (vu per-parent cap dropped {n_cap_vu}, top10k per-task cap dropped {n_cap_top})")

    # Load tokenizer for the target model family
    im_start_id = im_end_id = asst_role_id = newline_id = -1
    if FLAGS.model_family == "deepseek_v41":
        print(f"Loading DeepSeek-V4.1 fast tokenizer from {FLAGS.deepseek_tokenizer_file}...")
        local_tok_file = FLAGS.deepseek_tokenizer_file
        if not os.path.exists(local_tok_file) and gfile.Exists(local_tok_file):
            import tempfile
            with gfile.Open(local_tok_file, "rb") as f_in, tempfile.NamedTemporaryFile(
                suffix=".json", delete=False
            ) as f_out:
                f_out.write(f_in.read())
                local_tok_file = f_out.name
        tokenizer = transformers.PreTrainedTokenizerFast(tokenizer_file=local_tok_file)
        print(f"DeepSeek-V4.1 tokenizer loaded (vocab_size={len(tokenizer)}).")
    else:
        print(f"Loading Qwen3.8-27B tokenizer from {FLAGS.tokenizer_path}...")
        tokenizer = transformers.AutoTokenizer.from_pretrained(FLAGS.tokenizer_path, trust_remote_code=True)
        im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
        im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
        asst_hdr_ids = tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
        assert len(asst_hdr_ids) == 3 and asst_hdr_ids[0] == im_start_id, f"Unexpected assistant header IDs: {asst_hdr_ids}"
        asst_role_id = asst_hdr_ids[1]
        newline_id = asst_hdr_ids[2]
        print(
            f"Tokenizer verified: <|im_start|>={im_start_id}, <|im_end|>={im_end_id}, "
            f"assistant={asst_role_id}, newline={newline_id}"
        )

    # Tokenize and verify assistant-only loss masks for every selected trajectory
    tokenized_records: list[dict[str, Any]] = []
    overlong_dropped = 0
    render_failed = 0
    total_tokens_list: list[int] = []
    supervised_tokens_list: list[int] = []
    wic_alnum_tokens_list: list[int] = []

    t_tok = time.time()
    for idx, rec in enumerate(selected_records):
        try:
            if FLAGS.model_family == "deepseek_v41":
                input_ids, loss_mask, loss_weights, tok_stats = (
                    data_pipeline.tokenize_deepseek_v41_and_build_weights(
                        messages=rec["messages"],
                        tokenizer=tokenizer,
                        reasoning_effort=FLAGS.deepseek_reasoning_effort,
                        wic_alnum_weight=FLAGS.wic_alnum_weight,
                        tool_call_weight=FLAGS.tool_call_weight,
                        final_submit_weight=FLAGS.final_submit_weight,
                    )
                )
            else:
                segments = data_pipeline.build_segments(rec["messages"], max_turns=40)
                input_ids, loss_mask = data_pipeline.tokenize_and_verify_segments(
                    segments=segments,
                    tokenizer=tokenizer,
                    im_start_id=im_start_id,
                    im_end_id=im_end_id,
                    asst_role_id=asst_role_id,
                    newline_id=newline_id,
                )
                loss_weights = loss_mask.copy()
                tok_stats = {"n_wic_alnum_tokens": 0}
        except (AssertionError, ValueError, KeyError, TypeError) as e:
            render_failed += 1
            if render_failed <= 5:
                print(f"  WARNING: render/mask failure for {rec.get('task_id')} ({rec.get('batch')}): {e}")
            continue
        n_tok = int(len(input_ids))
        n_sup = int(loss_mask.sum())
        if n_tok > FLAGS.max_seq_len:
            overlong_dropped += 1
            continue

        total_tokens_list.append(n_tok)
        supervised_tokens_list.append(n_sup)
        wic_alnum_tokens_list.append(int(tok_stats.get("n_wic_alnum_tokens", 0)))

        rec_copy = dict(rec)
        # Drop raw messages from memory once tokenized
        del rec_copy["messages"]
        rec_copy["n_tokens"] = n_tok
        rec_copy["n_supervised_tokens"] = n_sup
        rec_copy["n_wic_alnum_tokens"] = int(tok_stats.get("n_wic_alnum_tokens", 0))
        rec_copy["supervised_ratio"] = round(n_sup / max(1, n_tok), 4)
        rec_copy["n_images"] = int(tok_stats.get("n_images", rec.get("n_images", 0)) or 0)
        rec_copy["n_image_tokens"] = int(tok_stats.get("n_image_tokens", 0) or 0)
        rec_copy["input_ids"] = input_ids
        rec_copy["loss_mask"] = loss_mask
        rec_copy["loss_weights"] = loss_weights
        token_types = tok_stats.get("token_types")
        rec_copy["token_types"] = (
            np.asarray(token_types, dtype=np.int8)
            if token_types is not None
            else np.full(n_tok, dsv41_image_tokens.TEXT, dtype=np.int8)
        )
        rec_copy["image_blobs"] = list(tok_stats.get("image_blobs") or [])
        if len(rec_copy["token_types"]) != n_tok:
            raise AssertionError(
                f"token_types length {len(rec_copy['token_types'])} != input_ids length {n_tok} "
                f"for {rec.get('task_id')} ({rec.get('batch')})"
            )
        tokenized_records.append(rec_copy)
        if (idx + 1) % 250 == 0 or (idx + 1) == len(selected_records):
            print(
                f"  Tokenized & verified {idx + 1}/{len(selected_records)} "
                f"(kept={len(tokenized_records)}, overlong>{FLAGS.max_seq_len}={overlong_dropped}, "
                f"render_failed={render_failed}, {time.time() - t_tok:.1f}s)"
            )

    # Perform 90/10 split by parent puzzle ID (optionally pinning a previous round's parents)
    pinned_train, pinned_val = load_pinned_parents(FLAGS.pin_split_from_data_dir)
    if FLAGS.pin_split_from_data_dir:
        print(
            f"Pinning split from {FLAGS.pin_split_from_data_dir}: {len(pinned_train)} train parents, "
            f"{len(pinned_val)} val parents (overlap {len(pinned_train & pinned_val)} -> val)."
        )
    train_recs, val_recs, split_stats = data_pipeline.split_by_parent_puzzle_id(
        tokenized_records,
        val_fraction=FLAGS.val_fraction,
        seed=FLAGS.seed,
        forced_train_parents=pinned_train,
        forced_val_parents=pinned_val,
    )
    print("\n90/10 Parent-Grouped Split Summary:")
    print(json.dumps(split_stats, indent=2))

    # ---- TRAIN-only resampling: arc2_train upsampling + rescued upsampling + Top-10K tier weights ----
    rng = np.random.default_rng(FLAGS.seed + 7)
    tier_weights = parse_tier_weights(FLAGS.tier_weights)
    resampled_train: list[dict[str, Any]] = []
    resample_stats: Counter[str] = Counter()
    for rec in train_recs:
        copies = 1
        if is_rescued_extra(rec) and FLAGS.e065_extra_upsample > 1:
            copies = int(FLAGS.e065_extra_upsample)
            resample_stats["rescued_upsampled_records"] += 1
        elif is_arc2_train(rec) and FLAGS.upsample_arc2_train > 1:
            copies = int(FLAGS.upsample_arc2_train)
            resample_stats["arc2_train_upsampled_records"] += 1
        elif is_top10k(rec) and tier_weights:
            w = float(tier_weights.get(str(rec.get("learning_tier")), 1.0))
            copies = int(np.floor(w)) + int(rng.random() < (w - np.floor(w)))
            resample_stats[f"tier_weighted_{rec.get('learning_tier')}_copies_{copies}"] += 1
        for c in range(copies):
            rc = dict(rec)
            rc["resample_copy"] = c
            resampled_train.append(rc)
        if copies == 0:
            resample_stats["dropped_by_tier_weight"] += 1
    # Shuffle so duplicates are not adjacent (the trainer re-permutes anyway)
    perm = rng.permutation(len(resampled_train))
    resampled_train = [resampled_train[i] for i in perm]
    print(
        f"Train after resampling: {len(resampled_train)} (from {len(train_recs)} unique); "
        f"val: {len(val_recs)}; resample_stats={dict(resample_stats)}"
    )
    train_recs = resampled_train

    # Write packed .npz arrays and metadata .jsonl to CNS
    gfile.MakeDirs(FLAGS.output_cns_dir)
    split_token_stats: dict[str, Any] = {}
    for split_name, split_list in (("train", train_recs), ("val", val_recs)):
        flat_ids = np.concatenate([r["input_ids"] for r in split_list]).astype(np.int32)
        flat_mask = np.concatenate([r["loss_mask"] for r in split_list]).astype(np.uint8)
        flat_weights = np.concatenate([r["loss_weights"] for r in split_list]).astype(np.uint8)
        flat_types = np.concatenate([r["token_types"] for r in split_list]).astype(np.int8)
        lengths = np.asarray([len(r["input_ids"]) for r in split_list], dtype=np.int32)
        offsets = np.zeros(len(split_list) + 1, dtype=np.int64)
        np.cumsum(lengths, out=offsets[1:])
        assert len(flat_types) == len(flat_ids), (len(flat_types), len(flat_ids))

        npz_path = os.path.join(FLAGS.output_cns_dir, f"{split_name}_tokens.npz")
        buf = io.BytesIO()
        np.savez_compressed(
            buf,
            input_ids=flat_ids,
            loss_mask=flat_mask,
            loss_weights=flat_weights,
            offsets=offsets,
            lengths=lengths,
            token_types=flat_types,
        )
        with gfile.Open(npz_path, "wb") as f:
            f.write(buf.getvalue())

        # Sidecar: per-trajectory list of PNG bytes (prompt order), aligned with the npz rows.
        # Written only when the split has at least one image so text-only datasets stay as before.
        images_list = [list(r.get("image_blobs") or []) for r in split_list]
        n_with_images = sum(1 for imgs in images_list if imgs)
        images_path = os.path.join(FLAGS.output_cns_dir, f"{split_name}_images.pkl")
        if n_with_images:
            with gfile.Open(images_path, "wb") as f:
                f.write(pickle.dumps(images_list, protocol=pickle.HIGHEST_PROTOCOL))
        elif gfile.Exists(images_path):
            gfile.Remove(images_path)

        meta_path = os.path.join(FLAGS.output_cns_dir, f"{split_name}_manifest.jsonl")
        with gfile.Open(meta_path, "w") as f:
            for i, r in enumerate(split_list):
                meta_row = {
                    k: v
                    for k, v in r.items()
                    if k not in ("input_ids", "loss_mask", "loss_weights", "token_types", "image_blobs")
                }
                meta_row["index"] = i
                f.write(json.dumps(meta_row) + "\n")
        n_image_tokens = int((flat_types != dsv41_image_tokens.TEXT).sum())
        split_token_stats[split_name] = {
            "trajectories": len(split_list),
            "total_tokens": int(len(flat_ids)),
            "supervised_tokens": int(flat_mask.sum()),
            "wic_alnum_tokens": int((flat_weights == FLAGS.wic_alnum_weight).sum()),
            "seq_len_mean": float(np.mean(lengths)) if len(lengths) else 0.0,
            "seq_len_p50": int(np.percentile(lengths, 50)) if len(lengths) else 0,
            "seq_len_p90": int(np.percentile(lengths, 90)) if len(lengths) else 0,
            "seq_len_max": int(lengths.max()) if len(lengths) else 0,
            "trajectories_with_images": n_with_images,
            "images_total": int(sum(len(imgs) for imgs in images_list)),
            "image_tokens_total": n_image_tokens,
            "by_modality": dict(Counter(str(r.get("modality", "text")) for r in split_list)),
            "by_batch": dict(Counter(str(r.get("batch")) for r in split_list)),
            "images_sidecar": images_path if n_with_images else "",
        }
        print(
            f"Saved {split_name}: {len(split_list)} trajectories, "
            f"{len(flat_ids):,} total tokens ({ int(flat_mask.sum()):,} supervised, "
            f"{int((flat_weights == FLAGS.wic_alnum_weight).sum()):,} WIC-alnum, "
            f"{n_image_tokens:,} image-span) -> {npz_path}"
            + (f" + {n_with_images} image trajectories -> {images_path}" if n_with_images else "")
        )

    tot_arr = np.asarray(total_tokens_list)
    sup_arr = np.asarray(supervised_tokens_list)
    wic_arr = np.asarray(wic_alnum_tokens_list)
    summary = {
        "model_family": FLAGS.model_family,
        "subset": FLAGS.subset,
        "max_seq_len": FLAGS.max_seq_len,
        "val_fraction": FLAGS.val_fraction,
        "seed": FLAGS.seed,
        "round2_flags": {
            "max_completion_tokens": FLAGS.max_completion_tokens,
            "max_tool_calls": FLAGS.max_tool_calls,
            "drop_rescued_or_nudged": FLAGS.drop_rescued_or_nudged,
            "top10k_require_verified_clean_sft": FLAGS.top10k_require_verified_clean_sft,
            "max_per_parent_vu": FLAGS.max_per_parent_vu,
            "max_per_task_top10k": FLAGS.max_per_task_top10k,
            "upsample_arc2_train": FLAGS.upsample_arc2_train,
            "tier_weights": tier_weights,
            "max_py_error_turns": FLAGS.max_py_error_turns,
            "max_ident_syntax_error_turns": FLAGS.max_ident_syntax_error_turns,
            "max_wasted_post_pass_turns": FLAGS.max_wasted_post_pass_turns,
            "max_post_pass_tokens": FLAGS.max_post_pass_tokens,
            "wic_alnum_weight": FLAGS.wic_alnum_weight,
            "tool_call_weight": FLAGS.tool_call_weight,
            "final_submit_weight": FLAGS.final_submit_weight,
            "max_images_per_sample": FLAGS.max_images_per_sample,
            "e065_rollouts_dir": FLAGS.e065_rollouts_dir if FLAGS.subset in E065_SUBSETS else "",
            "e065_manifest_path": FLAGS.e065_manifest_path if FLAGS.subset in E065_SUBSETS else "",
            "e065_text_part_dirs": list(FLAGS.e065_text_part_dirs),
            "e065_extra_rollouts_dir": FLAGS.e065_extra_rollouts_dir,
            "e065_extra_modality": FLAGS.e065_extra_modality,
            "e065_extra_skip_existing_modality_tasks": FLAGS.e065_extra_skip_existing_modality_tasks,
            "e065_extra_drop_meta_leaks": FLAGS.e065_extra_drop_meta_leaks,
            "e065_extra_upsample": FLAGS.e065_extra_upsample,
            "e065_extra_drop_reasons": dict(extra_drop_reasons),
            "pin_split_from_data_dir": FLAGS.pin_split_from_data_dir,
            "n_wic_alnum_tokens_total": int(wic_arr.sum()) if len(wic_arr) else 0,
        },
        "eval_tasks_indexed": len(eval_index["eval_task_ids"]),
        "known_contaminated_tasks_excluded": sorted(data_pipeline.KNOWN_CONTAMINATED_TASKS),
        "parquet_audits": all_file_stats,
        "e031_combined": {
            "total_rollouts": sum(s["rows"] for s in all_file_stats if s["file"] in E031_PARQUETS),
            "correct_count": sum(s["correct_count"] for s in all_file_stats if s["file"] in E031_PARQUETS),
            "train_pass_count": sum(s["train_pass_count"] for s in all_file_stats if s["file"] in E031_PARQUETS),
            "correct_and_train_pass": sum(
                s["correct_and_train_pass"] for s in all_file_stats if s["file"] in E031_PARQUETS
            ),
            "train_pass_but_test_wrong": sum(
                s["train_pass_but_test_wrong"] for s in all_file_stats if s["file"] in E031_PARQUETS
            ),
            "contaminated_rows": sum(
                s["contaminated_rows"] for s in all_file_stats if s["file"] in E031_PARQUETS
            ),
            "bad_turn_rows": sum(
                s["bad_turn_rows_among_verified"] for s in all_file_stats if s["file"] in E031_PARQUETS
            ),
            "verified_clean_count": len(e031_verified),
        },
        "stage_stats": stage_stats,
        "tokenization_and_length_filter": {
            "verified_before_length_filter": len(selected_records),
            "overlong_dropped": overlong_dropped,
            "render_failed": render_failed,
            "kept_after_length_filter": len(tokenized_records),
            "total_tokens": int(tot_arr.sum()) if len(tot_arr) else 0,
            "supervised_tokens": int(sup_arr.sum()) if len(sup_arr) else 0,
            "supervised_token_fraction": round(float(sup_arr.sum() / max(1, tot_arr.sum())), 4) if len(tot_arr) else 0.0,
            "seq_len_p50": int(np.percentile(tot_arr, 50)) if len(tot_arr) else 0,
            "seq_len_p90": int(np.percentile(tot_arr, 90)) if len(tot_arr) else 0,
            "seq_len_p99": int(np.percentile(tot_arr, 99)) if len(tot_arr) else 0,
            "seq_len_max": int(tot_arr.max()) if len(tot_arr) else 0,
        },
        "split_stats": split_stats,
        "train_resampling": dict(resample_stats),
        "split_token_stats": split_token_stats,
        "output_cns_dir": FLAGS.output_cns_dir,
    }

    cns_summary_path = os.path.join(FLAGS.output_cns_dir, "audit_summary.json")
    with gfile.Open(cns_summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote audit summary to {cns_summary_path}")

    if FLAGS.summary_json_path:
        with open(FLAGS.summary_json_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Wrote local audit summary to {FLAGS.summary_json_path}")


if __name__ == "__main__":
    app.run(main)
