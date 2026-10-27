"""Audits Vu's E031-E038 DeepSeek rollouts and builds the verified 90/10 SFT dataset on CNS."""

from __future__ import annotations

from collections import Counter
import json
import os
import time
from typing import Any

from absl import app
from absl import flags
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import transformers

from google3.experimental.users.arazdai.kaggle.sft import data_pipeline
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
    ["e031", "all", "top10k_v8", "all_with_top10k_v8"],
    "Which rollout files to include in the SFT dataset: 'e031', 'all', 'top10k_v8', or 'all_with_top10k_v8'.",
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


audit_parquet_file = data_pipeline.audit_rollout_file
discover_top10k_rollout_shards = data_pipeline.discover_top10k_rollout_shards



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

    if FLAGS.subset in ("top10k_v8", "all_with_top10k_v8"):
        shard_paths = discover_top10k_rollout_shards(FLAGS.top10k_rollouts_dir)
        print(f"Auditing {len(shard_paths)} Top-10K v8 rollout shards from {FLAGS.top10k_rollouts_dir}...")
        for spath in shard_paths:
            fstats, recs = audit_parquet_file(spath, eval_index)
            all_file_stats.append(fstats)
            top10k_verified.extend(recs)
        print(f"  -> Top-10K v8 verified clean trajectories: {len(top10k_verified)}")

    if FLAGS.subset == "e031":
        selected_records = e031_verified
    elif FLAGS.subset == "all":
        selected_records = all_verified
    elif FLAGS.subset == "top10k_v8":
        selected_records = top10k_verified
    else:
        selected_records = all_verified + top10k_verified

    print(
        f"\nSelected subset '{FLAGS.subset}': {len(selected_records)} verified clean trajectories "
        f"prior to tokenization & length filter."
    )

    # Load Qwen3.8-27B tokenizer and verify special token IDs
    print(f"Loading tokenizer from {FLAGS.tokenizer_path}...")
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
    total_tokens_list: list[int] = []
    supervised_tokens_list: list[int] = []

    t_tok = time.time()
    for idx, rec in enumerate(selected_records):
        segments = data_pipeline.build_segments(rec["messages"], max_turns=40)
        input_ids, loss_mask = data_pipeline.tokenize_and_verify_segments(
            segments=segments,
            tokenizer=tokenizer,
            im_start_id=im_start_id,
            im_end_id=im_end_id,
            asst_role_id=asst_role_id,
            newline_id=newline_id,
        )
        n_tok = int(len(input_ids))
        n_sup = int(loss_mask.sum())
        if n_tok > FLAGS.max_seq_len:
            overlong_dropped += 1
            continue

        total_tokens_list.append(n_tok)
        supervised_tokens_list.append(n_sup)

        rec_copy = dict(rec)
        # Drop raw messages from memory once tokenized
        del rec_copy["messages"]
        rec_copy["n_tokens"] = n_tok
        rec_copy["n_supervised_tokens"] = n_sup
        rec_copy["supervised_ratio"] = round(n_sup / max(1, n_tok), 4)
        rec_copy["input_ids"] = input_ids
        rec_copy["loss_mask"] = loss_mask
        tokenized_records.append(rec_copy)
        if (idx + 1) % 250 == 0 or (idx + 1) == len(selected_records):
            print(
                f"  Tokenized & verified {idx + 1}/{len(selected_records)} "
                f"(kept={len(tokenized_records)}, overlong>{FLAGS.max_seq_len}={overlong_dropped}, "
                f"{time.time() - t_tok:.1f}s)"
            )

    # Perform 90/10 split by parent puzzle ID
    train_recs, val_recs, split_stats = data_pipeline.split_by_parent_puzzle_id(
        tokenized_records,
        val_fraction=FLAGS.val_fraction,
        seed=FLAGS.seed,
    )
    print("\n90/10 Parent-Grouped Split Summary:")
    print(json.dumps(split_stats, indent=2))

    # Write packed .npz arrays and metadata .jsonl to CNS
    gfile.MakeDirs(FLAGS.output_cns_dir)
    for split_name, split_list in (("train", train_recs), ("val", val_recs)):
        # Sort by length descending or keep deterministic order; we keep deterministic order and save offsets
        flat_ids = np.concatenate([r["input_ids"] for r in split_list]).astype(np.int32)
        flat_mask = np.concatenate([r["loss_mask"] for r in split_list]).astype(np.uint8)
        lengths = np.asarray([len(r["input_ids"]) for r in split_list], dtype=np.int32)
        offsets = np.zeros(len(split_list) + 1, dtype=np.int64)
        np.cumsum(lengths, out=offsets[1:])

        import io
        npz_path = os.path.join(FLAGS.output_cns_dir, f"{split_name}_tokens.npz")
        buf = io.BytesIO()
        np.savez_compressed(
            buf,
            input_ids=flat_ids,
            loss_mask=flat_mask,
            offsets=offsets,
            lengths=lengths,
        )
        with gfile.Open(npz_path, "wb") as f:
            f.write(buf.getvalue())

        meta_path = os.path.join(FLAGS.output_cns_dir, f"{split_name}_manifest.jsonl")
        with gfile.Open(meta_path, "w") as f:
            for i, r in enumerate(split_list):
                meta_row = {
                    k: v
                    for k, v in r.items()
                    if k not in ("input_ids", "loss_mask")
                }
                meta_row["index"] = i
                f.write(json.dumps(meta_row) + "\n")
        print(
            f"Saved {split_name}: {len(split_list)} trajectories, "
            f"{len(flat_ids):,} total tokens ({ int(flat_mask.sum()):,} supervised) -> {npz_path}"
        )

    tot_arr = np.asarray(total_tokens_list)
    sup_arr = np.asarray(supervised_tokens_list)
    summary = {
        "subset": FLAGS.subset,
        "max_seq_len": FLAGS.max_seq_len,
        "val_fraction": FLAGS.val_fraction,
        "seed": FLAGS.seed,
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
        "tokenization_and_length_filter": {
            "verified_before_length_filter": len(selected_records),
            "overlong_dropped": overlong_dropped,
            "kept_after_length_filter": len(tokenized_records),
            "total_tokens": int(tot_arr.sum()),
            "supervised_tokens": int(sup_arr.sum()),
            "supervised_token_fraction": round(float(sup_arr.sum() / max(1, tot_arr.sum())), 4),
            "seq_len_p50": int(np.percentile(tot_arr, 50)),
            "seq_len_p90": int(np.percentile(tot_arr, 90)),
            "seq_len_p99": int(np.percentile(tot_arr, 99)),
            "seq_len_max": int(tot_arr.max()),
        },
        "split_stats": split_stats,
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
