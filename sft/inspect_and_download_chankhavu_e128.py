"""Inspects TFHub DeepSeek-V4.1-Flash-1017 and streams chankhavu's E128 checkpoint from Kaggle directly to CNS."""

from __future__ import annotations

from concurrent import futures
import hashlib
import json
import os
import time
import urllib.request
from typing import Any

from absl import app
from absl import flags
import numpy as np

from google3.experimental.users.arazdai.kaggle.sft import export_and_quantize
from google3.pyglib import gfile

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "tfhub_base_dir",
    "/tfhub/prod/tammam-ds4/deepseek-v4.1-flash-1017/1",
    "TFHub path to the unpruned base DeepSeek-V4.1-Flash-1017 checkpoint.",
)
flags.DEFINE_string(
    "local_meta_dir",
    "/usr/local/google/home/arazdai/.gemini/jetski/brain/d6923b28-7f5c-466e-a899-eb04a6ff1140/scratch/kaggle_e128_meta",
    "Local directory containing downloaded Kaggle E128 metadata, scripts, and pruning masks.",
)
flags.DEFINE_string(
    "cns_out_dir",
    "/cns/lu-d/home/arazdai/kaggle/models/dsv41_e128_2bit",
    "Target CNS directory to store chankhavu's E128 checkpoint and pruning artifacts.",
)
flags.DEFINE_string(
    "kaggle_token_path",
    "/usr/local/google/home/arazdai/.kaggle/access_token",
    "Path to Kaggle KGAT bearer access token.",
)
flags.DEFINE_enum(
    "download_tier",
    "backbone",
    ["meta_only", "backbone", "backbone_and_gptq", "all"],
    "Which large safetensors shards to stream to CNS in this invocation: "
    "'meta_only' (configs/masks/scripts), 'backbone' (+ model-00001..00002.safetensors, 9.74 GB), "
    "'backbone_and_gptq' (+ 40 gptq/layer_*.safetensors, 59.3 GB), or 'all' (+ 16 engram shards, 131.1 GB).",
)
flags.DEFINE_integer("num_workers", 8, "Parallel download streams to CNS.")

KAGGLE_BASE_URL = (
    "https://www.kaggle.com/api/v1/models/"
    "chankhavu/dsv41-flash-1017-arc-tool-e128-2bit/pyTorch/default/1/download/"
)


def inspect_tfhub(tfhub_dir: str) -> dict[str, Any]:
    """Checks the unpruned base DeepSeek-V4.1-Flash-1017 checkpoint on TFHub."""
    if not gfile.Exists(tfhub_dir):
        return {"exists": False, "path": tfhub_dir}
    entries = sorted(gfile.ListDir(tfhub_dir))
    safetensors = [e for e in entries if e.endswith(".safetensors")]
    idx_path = os.path.join(tfhub_dir, "model.safetensors.index.json")
    total_bytes = 0
    num_tensors = 0
    if gfile.Exists(idx_path):
        with gfile.Open(idx_path, "r") as f:
            idx = json.load(f)
        total_bytes = int((idx.get("metadata") or {}).get("total_size") or 0)
        num_tensors = len(idx.get("weight_map") or {})
    return {
        "exists": True,
        "path": tfhub_dir,
        "num_files": len(entries),
        "num_safetensors_shards": len(safetensors),
        "num_indexed_tensors": num_tensors,
        "total_size_gb": round(total_bytes / 1e9, 2),
        "total_size_gib": round(total_bytes / (2**30), 2),
        "has_tokenizer_json": "tokenizer.json" in entries,
        "has_config_json": "config.json" in entries,
    }


def sync_local_meta_to_cns(local_dir: str, cns_dir: str) -> list[str]:
    """Copies all metadata, scripts, and pruning masks from `local_dir` to `cns_dir`."""
    synced: list[str] = []
    if not os.path.exists(local_dir):
        return synced
    for root, _, files in os.walk(local_dir):
        for fn in sorted(files):
            src = os.path.join(root, fn)
            rel = os.path.relpath(src, local_dir)
            dst = os.path.join(cns_dir, rel)
            gfile.MakeDirs(os.path.dirname(dst))
            with open(src, "rb") as f_in:
                data = f_in.read()
            with gfile.Open(dst, "wb") as f_out:
                f_out.write(data)
            synced.append(rel)
    return synced


def stream_kaggle_file_to_cns(rel_path: str, cns_dir: str, token: str) -> dict[str, Any]:
    """Streams a single file from Kaggle directly into CNS in 16 MiB chunks without touching local disk."""
    dst = os.path.join(cns_dir, rel_path)
    gfile.MakeDirs(os.path.dirname(dst))
    url = KAGGLE_BASE_URL + rel_path
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    t0 = time.time()
    n_bytes = 0
    chunk_size = 16 << 20
    with urllib.request.urlopen(req, timeout=120) as resp:
        expected_len = int(resp.headers.get("Content-Length") or 0)
        if expected_len > 0 and gfile.Exists(dst):
            try:
                st = gfile.Stat(dst)
                if st.length == expected_len:
                    return {
                        "file": rel_path,
                        "status": "already_exists",
                        "bytes": expected_len,
                        "elapsed_sec": 0.0,
                    }
            except Exception:
                pass
        with gfile.Open(dst, "wb") as f_out:
            while True:
                buf = resp.read(chunk_size)
                if not buf:
                    break
                f_out.write(buf)
                n_bytes += len(buf)
    dt = max(1e-3, time.time() - t0)
    return {
        "file": rel_path,
        "status": "downloaded",
        "bytes": n_bytes,
        "mb_per_sec": round((n_bytes / 1e6) / dt, 1),
        "elapsed_sec": round(dt, 2),
    }


def main(argv: list[str]) -> None:
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")

    print("1. Inspecting TFHub unpruned base DeepSeek-V4.1-Flash-1017 checkpoint...")
    tfhub_info = inspect_tfhub(FLAGS.tfhub_base_dir)
    print(json.dumps(tfhub_info, indent=2))

    print(f"\n2. Syncing metadata, scripts, and pruning masks to CNS: {FLAGS.cns_out_dir}...")
    export_and_quantize.ensure_cns_quota_dir(FLAGS.cns_out_dir)
    export_and_quantize.ensure_cns_quota_dir(os.path.join(FLAGS.cns_out_dir, "gptq"))
    synced = sync_local_meta_to_cns(FLAGS.local_meta_dir, FLAGS.cns_out_dir)
    print(f"   Synced {len(synced)} files: {synced}")

    # Verify keep mask SHA256 on CNS
    keep_cns = os.path.join(FLAGS.cns_out_dir, "keep_contrib_dec_sp0_w10_e128.npy")
    if gfile.Exists(keep_cns):
        with gfile.Open(keep_cns, "rb") as f:
            keep_arr = np.load(f)
        keep_sha = hashlib.sha256(keep_arr.tobytes()).hexdigest()
        print(f"   Verified CNS keep_contrib_dec_sp0_w10_e128.npy: shape={keep_arr.shape}, tobytes_sha256={keep_sha}")

    if FLAGS.download_tier == "meta_only":
        return

    with open(FLAGS.kaggle_token_path, "r") as f:
        token = f.read().strip()

    targets: list[str] = [
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
    ]
    if FLAGS.download_tier in ("backbone_and_gptq", "all"):
        targets.extend([f"gptq/layer_{i:02d}.safetensors" for i in range(40)])
    if FLAGS.download_tier == "all":
        targets.extend([f"model-engram-{i:05d}-of-00016.safetensors" for i in range(1, 17)])

    print(f"\n3. Streaming {len(targets)} safetensors shards ({FLAGS.download_tier}) from Kaggle directly to CNS...")
    t_start = time.time()
    results: list[dict[str, Any]] = []
    with futures.ThreadPoolExecutor(max_workers=FLAGS.num_workers) as pool:
        futs = {
            pool.submit(stream_kaggle_file_to_cns, rel, FLAGS.cns_out_dir, token): rel
            for rel in targets
        }
        for fut in futures.as_completed(futs):
            rel = futs[fut]
            res = fut.result()
            results.append(res)
            print(f"   [{len(results)}/{len(targets)}] {res}")

    tot_b = sum(r["bytes"] for r in results)
    tot_dt = max(1e-3, time.time() - t_start)
    print(
        f"\nCompleted streaming {len(results)} shards ({tot_b / 1e9:.2f} GB) to {FLAGS.cns_out_dir} "
        f"in {tot_dt:.1f}s ({(tot_b / 1e6) / tot_dt:.1f} MB/s aggregate)."
    )


if __name__ == "__main__":
    app.run(main)
