"""Merges Qwen3.8-27B LoRA SFT adapters in FP32 and quantizes directly to official 128x128 block-FP8 layout."""

from __future__ import annotations

import io
import json
import os
import tempfile
import time

from absl import app
from absl import flags
import safetensors
from safetensors import torch as safetensors_torch
import torch

from google3.pyglib import gfile
from google3.tech.file.proto import types_pb2

FLAGS = flags.FLAGS

DEFAULT_CNS_ACCOUNTING_GROUP = "deepmind-resources-colossus"


def ensure_cns_quota_dir(
    cns_dir: str,
    group: str = DEFAULT_CNS_ACCOUNTING_GROUP,
) -> None:
    """Creates `cns_dir` and sets `capacity_quota_user` on CNS so bytes are billed to `group` instead of personal user quota."""
    if not cns_dir:
        return
    if not cns_dir.startswith("/cns/") or not group:
        gfile.MakeDirs(cns_dir)
        return
    stat_proto = types_pb2.StatProto(
        quota_accounting=types_pb2.QuotaAccounting(capacity_quota_user=group)
    )
    if not gfile.Exists(cns_dir):
        try:
            gfile.MakeDirs(cns_dir, stat_proto=stat_proto)
        except Exception:
            gfile.MakeDirs(cns_dir)
    try:
        gfile.SetStat(cns_dir, stat_proto=stat_proto)
    except Exception as e:
        print(
            f"Warning: could not set capacity_quota_user={group!r} on {cns_dir}: {e}",
            flush=True,
        )


flags.DEFINE_string(
    "base_bf16_path",
    "/cns/it-d/home/ads-video-creation-auto-plx/gaoxiao/models/Qwen3.8-27B",
    "Path to base Qwen3.8-27B BF16 model on CNS.",
)
flags.DEFINE_string(
    "ref_fp8_path",
    "/cns/it-d/home/ads-video-creation-auto-plx/gaoxiao/models/Qwen3.8-27B-FP8",
    "Path to reference Qwen3.8-27B-FP8 model on CNS (for exact shard layout and config.json).",
)
flags.DEFINE_string(
    "lora_adapter_path",
    "/cns/lu-d/home/arazdai/kaggle/sft/checkpoints/e031_qwen38_b200_pilot/final_adapter",
    "Path to trained LoRA adapter directory on CNS.",
)
flags.DEFINE_string(
    "full_sft_ckpt_dir",
    "",
    "Optional path to a Full-SFT distributed checkpoint directory (containing COMPLETED.json and rank_*_part_*.pt).",
)
flags.DEFINE_string(
    "output_bf16_path",
    "",
    "Optional CNS directory to save merged BF16 checkpoint.",
)
flags.DEFINE_string(
    "output_fp8_path",
    "/cns/lu-d/home/arazdai/kaggle/sft/checkpoints/e031_qwen38_b200_pilot/merged_fp8",
    "CNS directory to write quantized block-128 FP8 checkpoint ready for vLLM.",
)
flags.DEFINE_integer("block_size", 128, "Block size for 2D FP8 quantization (128x128).")


FP8_DTYPE = torch.float8_e4m3fn
FP8_MAX = float(torch.finfo(FP8_DTYPE).max)  # 448.0

_SAFETENSORS_DTYPE_MAP = {
    "BF16": torch.bfloat16,
    "F32": torch.float32,
    "F16": torch.float16,
}


def full_sft_key_to_hf_key(param_name: str) -> str:
    """Maps `FullSFTCausalLM` parameter name to HuggingFace `Qwen3_5ForConditionalGeneration` safetensors key."""
    if param_name.startswith("language_model."):
        return "model." + param_name
    return param_name


def reconstruct_full_sft_fp32_weights(
    full_sft_ckpt_dir: str,
    needed_hf_keys: set[str] | None = None,
    keep_bf16: bool = False,
) -> dict[str, torch.Tensor]:
    """Reconstructs unsharded model weights from a saved FSDP2 distributed checkpoint directory."""
    from concurrent import futures  # pylint: disable=g-import-not-at-top

    marker_path = os.path.join(full_sft_ckpt_dir, "COMPLETED.json")
    if not gfile.Exists(marker_path):
        raise FileNotFoundError(
            f"Cannot export incomplete or missing Full-SFT checkpoint (missing COMPLETED.json): {full_sft_ckpt_dir}"
        )
    with gfile.Open(marker_path, "r") as f:
        meta = json.load(f)

    world_size = int(meta["world_size"])
    num_parts = int(meta["num_parts"])
    part_hf_keys: list[list[str]] | None = meta.get("part_hf_keys")
    shards_by_hf_key: dict[str, list[torch.Tensor | None]] = {}
    out_dtype = torch.bfloat16 if keep_bf16 else torch.float32

    if needed_hf_keys is not None and part_hf_keys and len(part_hf_keys) == num_parts:
        needed_parts = [
            p_idx
            for p_idx in range(num_parts)
            if any(k in needed_hf_keys for k in part_hf_keys[p_idx])
        ]
    else:
        needed_parts = list(range(num_parts))

    def _load_rank_part(r: int, p_idx: int) -> tuple[int, dict[str, torch.Tensor]]:
        fname = f"rank_{r:02d}_part_{p_idx:02d}.pt"
        fpath = os.path.join(full_sft_ckpt_dir, fname)
        with gfile.Open(fpath, "rb") as f_in:
            raw_bytes = f_in.read()
        payload = torch.load(io.BytesIO(raw_bytes), map_location="cpu", weights_only=False)
        filtered: dict[str, torch.Tensor] = {}
        for name, tensor_shard in payload["model"].items():
            hf_key = full_sft_key_to_hf_key(name)
            if needed_hf_keys is not None and hf_key not in needed_hf_keys:
                continue
            filtered[hf_key] = tensor_shard.to(dtype=out_dtype)
        del payload
        return r, filtered

    for p_idx in needed_parts:
        with futures.ThreadPoolExecutor(max_workers=min(8, max(1, world_size))) as pool:
            futs = [pool.submit(_load_rank_part, r, p_idx) for r in range(world_size)]
            for fut in futs:
                r, filtered = fut.result()
                for hf_key, tensor_shard in filtered.items():
                    if hf_key not in shards_by_hf_key:
                        shards_by_hf_key[hf_key] = [None] * world_size
                    shards_by_hf_key[hf_key][r] = tensor_shard

    full_weights: dict[str, torch.Tensor] = {}
    for hf_key, rank_list in list(shards_by_hf_key.items()):
        if any(t is None for t in rank_list):
            raise RuntimeError(f"Missing rank shard(s) for parameter {hf_key}")
        tensors = [t for t in rank_list if t is not None]
        if world_size == 1:
            full_weights[hf_key] = tensors[0].contiguous()
        else:
            full_weights[hf_key] = torch.cat(tensors, dim=0).contiguous()
        del shards_by_hf_key[hf_key]
    return full_weights


def merge_lora_weight_fp32(
    w_base: torch.Tensor,
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
    scaling: float,
) -> torch.Tensor:
    """Computes W_merged = W_base.float() + scaling * (lora_B.float() @ lora_A.float()) in exact FP32."""
    w_fp32 = w_base.to(dtype=torch.float32)
    a_fp32 = lora_a.to(device=w_fp32.device, dtype=torch.float32)
    b_fp32 = lora_b.to(device=w_fp32.device, dtype=torch.float32)
    return w_fp32 + float(scaling) * (b_fp32 @ a_fp32)


def extract_lora_pairs(
    adapter_state_dict: dict[str, torch.Tensor],
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Maps base weight keys (e.g. 'model.language_model...q_proj.weight') to (lora_A, lora_B) tensors."""
    lora_a_map: dict[str, torch.Tensor] = {}
    lora_b_map: dict[str, torch.Tensor] = {}
    prefix = "base_model.model."
    for k, v in adapter_state_dict.items():
        clean_k = k[len(prefix) :] if k.startswith(prefix) else k
        if clean_k.endswith(".lora_A.weight"):
            base_k = clean_k[: -len(".lora_A.weight")] + ".weight"
            lora_a_map[base_k] = v
        elif clean_k.endswith(".lora_A.default.weight"):
            base_k = clean_k[: -len(".lora_A.default.weight")] + ".weight"
            lora_a_map[base_k] = v
        elif clean_k.endswith(".lora_B.weight"):
            base_k = clean_k[: -len(".lora_B.weight")] + ".weight"
            lora_b_map[base_k] = v
        elif clean_k.endswith(".lora_B.default.weight"):
            base_k = clean_k[: -len(".lora_B.default.weight")] + ".weight"
            lora_b_map[base_k] = v

    if set(lora_a_map.keys()) != set(lora_b_map.keys()):
        raise ValueError(
            f"Mismatched lora_A ({len(lora_a_map)}) and lora_B ({len(lora_b_map)}) keys!"
        )
    return {k: (lora_a_map[k], lora_b_map[k]) for k in lora_a_map}


def quantize_weight_block128(
    w: torch.Tensor,
    block_size: int = 128,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantizes a 2D weight tensor `[M, N]` into `float8_e4m3fn` and `[M//128, N//128]` float32 `weight_scale_inv`."""
    assert w.ndim == 2, f"Expected 2D weight tensor, got shape {w.shape}"
    m, n = w.shape
    assert m % block_size == 0 and n % block_size == 0, (
        f"Weight shape {(m, n)} not divisible by block_size={block_size}"
    )
    wf = w.float().view(m // block_size, block_size, n // block_size, block_size)
    amax = wf.abs().amax(dim=(1, 3))
    scale = (amax / FP8_MAX).clamp(min=1e-12)
    q = (wf / scale[:, None, :, None]).clamp(-FP8_MAX, FP8_MAX).to(FP8_DTYPE)
    return q.view(m, n).contiguous(), scale.to(torch.float32).contiguous()


import struct


def read_safetensors_header_gfile(
    path: str,
    keys: list[str] | None = None,
) -> tuple[dict[str, str], dict[str, tuple[int, ...]]]:
    """Reads only the 8-byte header size + JSON header from a safetensors file via gfile without downloading tensor data."""
    with gfile.Open(path, "rb") as f:
        header_len_bytes = f.read(8)
        if len(header_len_bytes) < 8:
            raise ValueError(f"Safetensors file too short (<8 bytes): {path}")
        (header_len,) = struct.unpack("<Q", header_len_bytes)
        header_json = json.loads(f.read(header_len).decode("utf-8"))
    ref_dtypes: dict[str, str] = {}
    ref_shapes: dict[str, tuple[int, ...]] = {}
    target_keys = keys if keys is not None else [k for k in header_json if k != "__metadata__"]
    for k in target_keys:
        if k == "__metadata__" or k not in header_json:
            continue
        info = header_json[k]
        ref_dtypes[k] = str(info["dtype"])
        ref_shapes[k] = tuple(int(d) for d in info["shape"])
    return ref_dtypes, ref_shapes


def copy_file_gfile(src: str, dst: str) -> None:
    """Copies `src` to `dst`, using multi-threaded 32MB range reads when staging large CNS files to local disk."""
    from concurrent import futures  # pylint: disable=g-import-not-at-top

    if src.startswith("/cns/") and not dst.startswith("/cns/"):
        try:
            file_len = int(gfile.Stat(src).length)
        except Exception:
            file_len = 0
        chunk_bytes = 32 * 1024 * 1024
        if file_len > 64 * 1024 * 1024:
            os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
            fd = os.open(dst, os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o644)
            try:
                os.ftruncate(fd, file_len)

                def _copy_range(offset: int) -> None:
                    to_read = min(chunk_bytes, file_len - offset)
                    with gfile.Open(src, "rb") as f_in:
                        f_in.seek(offset)
                        data = f_in.read(to_read)
                    written = 0
                    while written < len(data):
                        written += os.pwrite(fd, data[written:], offset + written)

                offsets = list(range(0, file_len, chunk_bytes))
                with futures.ThreadPoolExecutor(max_workers=min(8, len(offsets))) as pool:
                    for fut in [pool.submit(_copy_range, off) for off in offsets]:
                        fut.result()
            finally:
                os.close(fd)
            return

    with gfile.Open(src, "rb") as f_in, gfile.Open(dst, "wb") as f_out:
        while True:
            buf = f_in.read(16 * 1024 * 1024)
            if not buf:
                break
            f_out.write(buf)


def copy_cns_dir_to_local(cns_dir: str, local_dir: str) -> None:
    os.makedirs(local_dir, exist_ok=True)
    for fname in gfile.ListDir(cns_dir):
        src = os.path.join(cns_dir, fname)
        dst = os.path.join(local_dir, fname)
        if gfile.IsDirectory(src):
            copy_cns_dir_to_local(src, dst)
        else:
            copy_file_gfile(src, dst)


def main(argv: list[str]) -> None:
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")

    if not FLAGS.output_bf16_path:
        ensure_sharded_fp8_export(
            lora_adapter_path=FLAGS.lora_adapter_path,
            output_fp8_path=FLAGS.output_fp8_path,
            shard_index=0,
            num_shards=1,
            base_bf16_path=FLAGS.base_bf16_path,
            ref_fp8_path=FLAGS.ref_fp8_path,
            block_size=FLAGS.block_size,
            full_sft_ckpt_dir=FLAGS.full_sft_ckpt_dir,
        )
        return

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    t0 = time.time()

    full_sft_weights: dict[str, torch.Tensor] = {}
    lora_pairs: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    scaling = 1.0

    if FLAGS.full_sft_ckpt_dir:
        print(f"Reconstructing FP32 Full-SFT weights from {FLAGS.full_sft_ckpt_dir}...")
        full_sft_weights = reconstruct_full_sft_fp32_weights(FLAGS.full_sft_ckpt_dir)
        print(
            f"Reconstructed {len(full_sft_weights)} full-parameter FP32 tensors in {time.time() - t0:.1f}s."
        )
    else:
        with tempfile.TemporaryDirectory(prefix="lora_merge_") as tmp_lora_dir:
            print(f"Copying LoRA adapter from {FLAGS.lora_adapter_path} to {tmp_lora_dir}...")
            copy_cns_dir_to_local(FLAGS.lora_adapter_path, tmp_lora_dir)
            with open(os.path.join(tmp_lora_dir, "adapter_config.json"), "r") as f:
                adapter_cfg = json.load(f)
            lora_r = int(adapter_cfg["r"])
            lora_alpha = float(adapter_cfg["lora_alpha"])
            scaling = lora_alpha / float(lora_r)

            adapter_st_path = os.path.join(tmp_lora_dir, "adapter_model.safetensors")
            adapter_sd = safetensors_torch.load_file(adapter_st_path, device=device)
            lora_pairs = extract_lora_pairs(adapter_sd)
            print(
                f"Loaded {len(lora_pairs)} FP32 LoRA adapter pairs (r={lora_r}, alpha={lora_alpha}, "
                f"scaling={scaling:.4f}) in {time.time() - t0:.1f}s."
            )

    base_index_path = os.path.join(FLAGS.base_bf16_path, "model.safetensors.index.json")
    with gfile.Open(base_index_path, "r") as f:
        base_index = json.load(f)
    base_weight_map: dict[str, str] = base_index["weight_map"]

    ref_index_path = os.path.join(FLAGS.ref_fp8_path, "model.safetensors.index.json")
    with gfile.Open(ref_index_path, "r") as f:
        ref_index = json.load(f)

    by_shard: dict[str, list[str]] = {}
    for key, shard in ref_index["weight_map"].items():
        by_shard.setdefault(shard, []).append(key)

    gfile.MakeDirs(FLAGS.output_fp8_path)
    if FLAGS.output_bf16_path:
        gfile.MakeDirs(FLAGS.output_bf16_path)

    from concurrent import futures  # pylint: disable=g-import-not-at-top

    torch.set_num_threads(max(8, (os.cpu_count() or 16)))
    applied_lora_keys: set[str] = set()
    applied_full_sft_keys: set[str] = set()
    print(
        f"Merging in exact FP32 and quantizing directly to block-{FLAGS.block_size} FP8 "
        f"at {FLAGS.output_fp8_path}..."
    )

    def _needed_base_for_shard(keys: list[str]) -> set[str]:
        return {
            base_weight_map[k]
            for k in keys
            if not k.endswith(".weight_scale_inv")
            and k not in full_sft_weights
            and (bool(FLAGS.output_bf16_path) or k in lora_pairs)
            and k in base_weight_map
        }

    # Order shards by their required base_bf16 shard so each 3-4 GB base shard is downloaded once.
    ordered_shards = sorted(
        by_shard.items(),
        key=lambda item: (sorted(_needed_base_for_shard(item[1])), item[0]),
    )

    with (
        tempfile.TemporaryDirectory(prefix="fp8_export_cache_") as cache_dir,
        futures.ThreadPoolExecutor(max_workers=6) as io_pool,
    ):
        base_shard_futures: dict[str, futures.Future[str]] = {}
        cached_base_paths: dict[str, str] = {}

        def _download_base_shard(b_shard: str) -> str:
            dst = os.path.join(cache_dir, "base_" + b_shard)
            copy_file_gfile(os.path.join(FLAGS.base_bf16_path, b_shard), dst)
            return dst

        # Remaining usage count per base shard so we can evict from /tmp immediately after last use.
        base_remaining_uses: dict[str, int] = {}
        for _, s_keys in ordered_shards:
            for b_s in _needed_base_for_shard(s_keys):
                base_remaining_uses[b_s] = base_remaining_uses.get(b_s, 0) + 1

        # Kick off prefetch of the first 2 distinct base shards
        distinct_base_order: list[str] = []
        for _, s_keys in ordered_shards:
            for b_s in sorted(_needed_base_for_shard(s_keys)):
                if b_s not in distinct_base_order:
                    distinct_base_order.append(b_s)
        for b_s in distinct_base_order[:2]:
            base_shard_futures[b_s] = io_pool.submit(_download_base_shard, b_s)

        upload_futures: list[futures.Future[None]] = []

        for shard_idx, (shard_name, keys) in enumerate(ordered_shards, 1):
            ts = time.time()
            ref_shard_cns = os.path.join(FLAGS.ref_fp8_path, shard_name)
            needed_base_shards = _needed_base_for_shard(keys)

            # Ensure needed base shards are scheduled and prefetch the next upcoming base shard
            for b_s in sorted(needed_base_shards):
                if b_s not in base_shard_futures:
                    base_shard_futures[b_s] = io_pool.submit(_download_base_shard, b_s)
                idx_in_order = distinct_base_order.index(b_s)
                for next_bs in distinct_base_order[idx_in_order + 1 : idx_in_order + 3]:
                    if next_bs not in base_shard_futures and base_remaining_uses.get(next_bs, 0) > 0:
                        base_shard_futures[next_bs] = io_pool.submit(_download_base_shard, next_bs)

            # Fast path: if no keys in this shard are modified by LoRA or Full-SFT (and no BF16 output requested),
            # copy the reference shard directly on CNS.
            has_updates_in_shard = any(
                (k in full_sft_weights or k in lora_pairs)
                for k in keys
                if not k.endswith(".weight_scale_inv")
            )
            if not has_updates_in_shard and not FLAGS.output_bf16_path:
                copy_file_gfile(
                    ref_shard_cns, os.path.join(FLAGS.output_fp8_path, shard_name)
                )
                print(
                    f"  [{shard_idx}/{len(ordered_shards)}] {shard_name}: untouched shard copied directly ({time.time() - ts:.1f}s)",
                    flush=True,
                )
                continue

            with tempfile.TemporaryDirectory(prefix="fp8_shard_") as tmp_shard_dir:
                local_ref_shard = os.path.join(tmp_shard_dir, "ref_" + shard_name)
                copy_file_gfile(ref_shard_cns, local_ref_shard)

                local_base_shards: dict[str, str] = {}
                for b_shard in sorted(needed_base_shards):
                    if b_shard not in cached_base_paths:
                        cached_base_paths[b_shard] = base_shard_futures[b_shard].result()
                    local_base_shards[b_shard] = cached_base_paths[b_shard]

                out_tensors: dict[str, torch.Tensor] = {}
                out_bf16_tensors: dict[str, torch.Tensor] = {}
                n_fp8 = 0
                n_bf16 = 0
                n_updated = 0

                base_handles = {
                    b_s: safetensors.safe_open(b_path, framework="pt", device="cpu")
                    for b_s, b_path in local_base_shards.items()
                }
                with safetensors.safe_open(local_ref_shard, framework="pt", device="cpu") as rf:
                    ref_dtypes = {k: rf.get_slice(k).get_dtype() for k in keys}
                    ref_shapes = {k: tuple(rf.get_slice(k).get_shape()) for k in keys}

                    for k in keys:
                        if k.endswith(".weight_scale_inv"):
                            continue

                        is_modified = (k in full_sft_weights) or (k in lora_pairs)
                        if not is_modified and not FLAGS.output_bf16_path:
                            out_tensors[k] = rf.get_tensor(k)
                            if ref_dtypes[k] == "F8_E4M3":
                                scale_key = k + "_scale_inv"
                                out_tensors[scale_key] = rf.get_tensor(scale_key)
                                n_fp8 += 1
                            else:
                                n_bf16 += 1
                            continue

                        if k in full_sft_weights:
                            w_merged = full_sft_weights[k].to(device=device, dtype=torch.float32)
                            applied_full_sft_keys.add(k)
                            n_updated += 1
                        else:
                            if k in base_weight_map and base_weight_map[k] in base_handles:
                                w_base = base_handles[base_weight_map[k]].get_tensor(k)
                            else:
                                w_base = rf.get_tensor(k)

                            if k in lora_pairs:
                                lora_a, lora_b = lora_pairs[k]
                                w_merged = merge_lora_weight_fp32(
                                    w_base.to(device), lora_a, lora_b, scaling=scaling
                                )
                                applied_lora_keys.add(k)
                                n_updated += 1
                            else:
                                w_merged = w_base.to(device=device, dtype=torch.float32)

                        target_non_fp8_dtype = _SAFETENSORS_DTYPE_MAP.get(ref_dtypes[k], torch.bfloat16)
                        if FLAGS.output_bf16_path:
                            out_bf16_tensors[k] = w_merged.to(target_non_fp8_dtype).contiguous().cpu()

                        if ref_dtypes[k] == "F8_E4M3":
                            scale_key = k + "_scale_inv"
                            q, s = quantize_weight_block128(w_merged, block_size=FLAGS.block_size)
                            out_tensors[k] = q.cpu()
                            out_tensors[scale_key] = s.cpu()
                            assert tuple(out_tensors[k].shape) == ref_shapes[k]
                            assert tuple(out_tensors[scale_key].shape) == ref_shapes[scale_key]
                            n_fp8 += 1
                        else:
                            out_tensors[k] = w_merged.to(target_non_fp8_dtype).contiguous().cpu()
                            assert tuple(out_tensors[k].shape) == ref_shapes[k]
                            n_bf16 += 1

                del base_handles

                local_out_shard = os.path.join(cache_dir, "out_" + shard_name)
                safetensors_torch.save_file(out_tensors, local_out_shard, metadata={"format": "pt"})

                def _upload_and_remove(src_p: str, dst_p: str) -> None:
                    copy_file_gfile(src_p, dst_p)
                    try:
                        os.remove(src_p)
                    except OSError:
                        pass

                upload_futures.append(
                    io_pool.submit(
                        _upload_and_remove,
                        local_out_shard,
                        os.path.join(FLAGS.output_fp8_path, shard_name),
                    )
                )

                if FLAGS.output_bf16_path:
                    local_bf16_shard = os.path.join(cache_dir, "bf16_" + shard_name)
                    safetensors_torch.save_file(
                        out_bf16_tensors, local_bf16_shard, metadata={"format": "pt"}
                    )
                    upload_futures.append(
                        io_pool.submit(
                            _upload_and_remove,
                            local_bf16_shard,
                            os.path.join(FLAGS.output_bf16_path, shard_name),
                        )
                    )

                for b_shard in needed_base_shards:
                    base_remaining_uses[b_shard] -= 1
                    if base_remaining_uses[b_shard] <= 0 and b_shard in cached_base_paths:
                        try:
                            os.remove(cached_base_paths.pop(b_shard))
                        except OSError:
                            pass

                print(
                    f"  [{shard_idx}/{len(ordered_shards)}] {shard_name}: {n_fp8} FP8 + {n_bf16} non-FP8 "
                    f"({n_updated} updated in FP32, {time.time() - ts:.1f}s)",
                    flush=True,
                )
        for uf in upload_futures:
            uf.result()

    if FLAGS.full_sft_ckpt_dir:
        if len(applied_full_sft_keys) != len(full_sft_weights):
            missing = sorted(set(full_sft_weights.keys()) - applied_full_sft_keys)
            raise RuntimeError(
                f"Did not apply all Full-SFT weights! Applied {len(applied_full_sft_keys)}/{len(full_sft_weights)}, "
                f"missing sample: {missing[:5]}"
            )
    else:
        if len(applied_lora_keys) != len(lora_pairs):
            missing = sorted(set(lora_pairs.keys()) - applied_lora_keys)
            raise RuntimeError(
                f"Did not apply all LoRA adapters! Applied {len(applied_lora_keys)}/{len(lora_pairs)}, "
                f"missing sample: {missing[:5]}"
            )

    # Copy tokenizer, config.json, and index files from reference FP8 directory
    for fname in gfile.ListDir(FLAGS.ref_fp8_path):
        if fname.endswith(".safetensors"):
            continue
        src = os.path.join(FLAGS.ref_fp8_path, fname)
        if not gfile.IsDirectory(src):
            gfile.Copy(src, os.path.join(FLAGS.output_fp8_path, fname), overwrite=True)

    print(
        f"FP8 export complete ({len(applied_full_sft_keys) or len(applied_lora_keys)} weights updated in FP32 -> FP8) "
        f"in {time.time() - t0:.1f}s -> {FLAGS.output_fp8_path}"
    )


def ensure_sharded_fp8_export(
    lora_adapter_path: str = "",
    output_fp8_path: str = "",
    shard_index: int = 0,
    num_shards: int = 1,
    base_bf16_path: str = "/cns/it-d/home/ads-video-creation-auto-plx/gaoxiao/models/Qwen3.8-27B",
    ref_fp8_path: str = "/cns/it-d/home/ads-video-creation-auto-plx/gaoxiao/models/Qwen3.8-27B-FP8",
    block_size: int = 128,
    full_sft_ckpt_dir: str = "",
) -> None:
    """Distributed Borg-native FP32 LoRA merge or Full-SFT reconstruct + Block-128 FP8 export across `num_shards` workers."""
    from concurrent import futures  # pylint: disable=g-import-not-at-top

    ensure_cns_quota_dir(output_fp8_path)
    complete_marker = os.path.join(output_fp8_path, "_EXPORT_COMPLETE.json")
    if gfile.Exists(complete_marker):
        print(
            f"[fp8_export] {output_fp8_path} already exported (_EXPORT_COMPLETE.json present).",
            flush=True,
        )
        return

    t0 = time.time()
    try:
        torch.set_num_threads(max(8, (os.cpu_count() or 16)))
    except RuntimeError:
        pass
    quant_device = (
        torch.device(f"cuda:{torch.cuda.current_device()}")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )

    lora_pairs: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    scaling = 1.0
    if not full_sft_ckpt_dir:
        with tempfile.TemporaryDirectory(prefix="lora_merge_") as tmp_lora_dir:
            print(
                f"[fp8_export shard {shard_index}/{num_shards}] Loading LoRA adapter from {lora_adapter_path}...",
                flush=True,
            )
            copy_cns_dir_to_local(lora_adapter_path, tmp_lora_dir)
            with open(os.path.join(tmp_lora_dir, "adapter_config.json"), "r") as f:
                adapter_cfg = json.load(f)
            lora_r = int(adapter_cfg["r"])
            lora_alpha = float(adapter_cfg["lora_alpha"])
            scaling = lora_alpha / float(lora_r)
            adapter_st_path = os.path.join(tmp_lora_dir, "adapter_model.safetensors")
            with safetensors.safe_open(adapter_st_path, framework="pt", device="cpu") as f_st:
                adapter_sd = {k: f_st.get_tensor(k) for k in f_st.keys()}
            lora_pairs = extract_lora_pairs(adapter_sd)

    with gfile.Open(os.path.join(base_bf16_path, "model.safetensors.index.json"), "r") as f:
        base_weight_map: dict[str, str] = json.load(f)["weight_map"]
    with gfile.Open(os.path.join(ref_fp8_path, "model.safetensors.index.json"), "r") as f:
        ref_index = json.load(f)

    by_shard: dict[str, list[str]] = {}
    for key, shard in ref_index["weight_map"].items():
        by_shard.setdefault(shard, []).append(key)

    def _needed_base_for_shard(keys: list[str]) -> set[str]:
        if full_sft_ckpt_dir:
            return set()
        return {
            base_weight_map[k]
            for k in keys
            if not k.endswith(".weight_scale_inv")
            and k in lora_pairs
            and k in base_weight_map
        }

    def _natural_shard_key(shard_name: str) -> tuple[int, str]:
        if shard_name in ("model.safetensors", "outside.safetensors"):
            return (-1, shard_name)
        if shard_name.startswith("layers-"):
            try:
                return (int(shard_name.split("-")[1].split(".")[0]), shard_name)
            except ValueError:
                pass
        return (9999, shard_name)

    ordered_shards = sorted(
        by_shard.items(),
        key=lambda item: (
            sorted(_needed_base_for_shard(item[1])),
            _natural_shard_key(item[0]) if full_sft_ckpt_dir else (0, item[0]),
        ),
    )
    # Assign contiguous blocks of `ordered_shards` to each worker so each worker only downloads ~1-2 parts!
    n_total = len(ordered_shards)
    per_worker = (n_total + num_shards - 1) // max(1, num_shards)
    lo = shard_index * per_worker
    hi = min(n_total, (shard_index + 1) * per_worker)
    my_shards = ordered_shards[lo:hi]
    print(
        f"[fp8_export shard {shard_index}/{num_shards}] Exporting {len(my_shards)}/{n_total} FP8 shards...",
        flush=True,
    )

    my_full_sft_weights: dict[str, torch.Tensor] = {}
    if full_sft_ckpt_dir and my_shards:
        my_needed_hf_keys = {
            k
            for _, s_keys in my_shards
            for k in s_keys
            if not k.endswith(".weight_scale_inv")
        }
        t_rec = time.time()
        print(
            f"[fp8_export shard {shard_index}/{num_shards}] Reconstructing {len(my_needed_hf_keys)} keys from {full_sft_ckpt_dir}...",
            flush=True,
        )
        my_full_sft_weights = reconstruct_full_sft_fp32_weights(
            full_sft_ckpt_dir,
            needed_hf_keys=my_needed_hf_keys,
            keep_bf16=True,
        )
        print(
            f"[fp8_export shard {shard_index}/{num_shards}] Reconstructed {len(my_full_sft_weights)} keys in {time.time() - t_rec:.1f}s.",
            flush=True,
        )

    with (
        tempfile.TemporaryDirectory(prefix="fp8_export_cache_") as cache_dir,
        futures.ThreadPoolExecutor(max_workers=6) as io_pool,
    ):
        base_shard_futures: dict[str, futures.Future[str]] = {}
        cached_base_paths: dict[str, str] = {}

        def _download_base_shard(b_shard: str) -> str:
            dst = os.path.join(cache_dir, "base_" + b_shard)
            copy_file_gfile(os.path.join(base_bf16_path, b_shard), dst)
            return dst

        base_remaining_uses: dict[str, int] = {}
        distinct_base_order: list[str] = []
        for s_name, s_keys in my_shards:
            ref_s_cns = os.path.join(ref_fp8_path, s_name)
            dst_s_cns = os.path.join(output_fp8_path, s_name)
            if gfile.Exists(dst_s_cns) and gfile.Stat(dst_s_cns).length >= gfile.Stat(ref_s_cns).length:
                continue
            for b_s in sorted(_needed_base_for_shard(s_keys)):
                base_remaining_uses[b_s] = base_remaining_uses.get(b_s, 0) + 1
                if b_s not in distinct_base_order:
                    distinct_base_order.append(b_s)

        for b_s in distinct_base_order[:2]:
            base_shard_futures[b_s] = io_pool.submit(_download_base_shard, b_s)

        upload_futures: list[futures.Future[None]] = []

        def _upload_and_remove(src_p: str, dst_p: str) -> None:
            tmp_dst_p = f"{dst_p}.tmp.{shard_index}.{int(time.time() * 1000)}"
            copy_file_gfile(src_p, tmp_dst_p)
            gfile.Rename(tmp_dst_p, dst_p, overwrite=True)
            try:
                os.remove(src_p)
            except OSError:
                pass

        for idx_local, (shard_name, keys) in enumerate(my_shards, 1):
            ts = time.time()
            ref_shard_cns = os.path.join(ref_fp8_path, shard_name)
            dst_shard_cns = os.path.join(output_fp8_path, shard_name)
            if gfile.Exists(dst_shard_cns) and gfile.Stat(dst_shard_cns).length >= gfile.Stat(ref_shard_cns).length:
                print(
                    f"[fp8_export shard {shard_index}] [{idx_local}/{len(my_shards)}] {shard_name}: already present on CNS",
                    flush=True,
                )
                continue
            needed_base_shards = _needed_base_for_shard(keys)

            for b_s in sorted(needed_base_shards):
                if b_s not in base_shard_futures:
                    base_shard_futures[b_s] = io_pool.submit(_download_base_shard, b_s)
                idx_in_order = distinct_base_order.index(b_s)
                for next_bs in distinct_base_order[idx_in_order + 1 : idx_in_order + 3]:
                    if next_bs not in base_shard_futures and base_remaining_uses.get(next_bs, 0) > 0:
                        base_shard_futures[next_bs] = io_pool.submit(_download_base_shard, next_bs)

            non_scale_keys = [k for k in keys if not k.endswith(".weight_scale_inv")]
            has_full_sft_in_shard = bool(
                full_sft_ckpt_dir and any(k in my_full_sft_weights for k in non_scale_keys)
            )
            if not needed_base_shards and not has_full_sft_in_shard:
                gfile.Copy(ref_shard_cns, dst_shard_cns, overwrite=True)
                print(
                    f"[fp8_export shard {shard_index}] [{idx_local}/{len(my_shards)}] {shard_name}: server-side CNS copy ({time.time() - ts:.1f}s)",
                    flush=True,
                )
                continue

            all_in_full_sft = bool(
                full_sft_ckpt_dir
                and non_scale_keys
                and all(k in my_full_sft_weights for k in non_scale_keys)
            )

            with tempfile.TemporaryDirectory(prefix="fp8_shard_") as tmp_shard_dir:
                out_tensors: dict[str, torch.Tensor] = {}
                n_updated = 0

                if all_in_full_sft:
                    # Zero-download fast path: read only the ~2KB safetensors JSON header from CNS!
                    ref_dtypes, ref_shapes = read_safetensors_header_gfile(ref_shard_cns, keys=keys)
                    for k in non_scale_keys:
                        w_merged = my_full_sft_weights.pop(k).to(
                            device=quant_device, dtype=torch.float32
                        )
                        n_updated += 1
                        if ref_dtypes[k] == "F8_E4M3":
                            scale_key = k + "_scale_inv"
                            q, s = quantize_weight_block128(w_merged, block_size=block_size)
                            out_tensors[k] = q.cpu()
                            out_tensors[scale_key] = s.cpu()
                            assert tuple(out_tensors[k].shape) == ref_shapes[k]
                            assert tuple(out_tensors[scale_key].shape) == ref_shapes[scale_key]
                        else:
                            target_dtype = _SAFETENSORS_DTYPE_MAP.get(ref_dtypes[k], torch.bfloat16)
                            out_tensors[k] = w_merged.to(target_dtype).contiguous().cpu()
                        del w_merged
                else:
                    local_ref_shard = os.path.join(tmp_shard_dir, "ref_" + shard_name)
                    copy_file_gfile(ref_shard_cns, local_ref_shard)

                    local_base_shards: dict[str, str] = {}
                    for b_shard in sorted(needed_base_shards):
                        if b_shard not in cached_base_paths:
                            cached_base_paths[b_shard] = base_shard_futures[b_shard].result()
                        local_base_shards[b_shard] = cached_base_paths[b_shard]

                    base_handles = {
                        b_s: safetensors.safe_open(b_path, framework="pt", device="cpu")
                        for b_s, b_path in local_base_shards.items()
                    }
                    with safetensors.safe_open(local_ref_shard, framework="pt", device="cpu") as rf:
                        ref_dtypes = {k: rf.get_slice(k).get_dtype() for k in keys}
                        ref_shapes = {k: tuple(rf.get_slice(k).get_shape()) for k in keys}

                        for k in keys:
                            if k.endswith(".weight_scale_inv"):
                                continue
                            if k in my_full_sft_weights:
                                w_merged = my_full_sft_weights.pop(k).to(
                                    device=quant_device, dtype=torch.float32
                                )
                                n_updated += 1
                            elif k in lora_pairs:
                                w_base = base_handles[base_weight_map[k]].get_tensor(k)
                                lora_a, lora_b = lora_pairs[k]
                                w_merged = merge_lora_weight_fp32(
                                    w_base.to(device=quant_device), lora_a, lora_b, scaling=scaling
                                )
                                n_updated += 1
                            else:
                                out_tensors[k] = rf.get_tensor(k)
                                if ref_dtypes[k] == "F8_E4M3":
                                    scale_key = k + "_scale_inv"
                                    out_tensors[scale_key] = rf.get_tensor(scale_key)
                                continue

                            if ref_dtypes[k] == "F8_E4M3":
                                scale_key = k + "_scale_inv"
                                q, s = quantize_weight_block128(w_merged, block_size=block_size)
                                out_tensors[k] = q.cpu()
                                out_tensors[scale_key] = s.cpu()
                                assert tuple(out_tensors[k].shape) == ref_shapes[k]
                                assert tuple(out_tensors[scale_key].shape) == ref_shapes[scale_key]
                            else:
                                target_dtype = _SAFETENSORS_DTYPE_MAP.get(ref_dtypes[k], torch.bfloat16)
                                out_tensors[k] = w_merged.to(target_dtype).contiguous().cpu()
                            del w_merged

                    del base_handles
                local_out_shard = os.path.join(cache_dir, "out_" + shard_name)
                safetensors_torch.save_file(out_tensors, local_out_shard, metadata={"format": "pt"})
                upload_futures.append(
                    io_pool.submit(_upload_and_remove, local_out_shard, dst_shard_cns)
                )

                for b_shard in needed_base_shards:
                    base_remaining_uses[b_shard] -= 1
                    if base_remaining_uses[b_shard] <= 0 and b_shard in cached_base_paths:
                        try:
                            os.remove(cached_base_paths.pop(b_shard))
                        except OSError:
                            pass

                print(
                    f"[fp8_export shard {shard_index}] [{idx_local}/{len(my_shards)}] {shard_name}: "
                    f"{n_updated} weights updated+quantized ({time.time() - ts:.1f}s)",
                    flush=True,
                )

        for uf in upload_futures:
            uf.result()

    if shard_index == 0:
        for fname in gfile.ListDir(ref_fp8_path):
            if fname.endswith(".safetensors"):
                continue
            src = os.path.join(ref_fp8_path, fname)
            if not gfile.IsDirectory(src):
                gfile.Copy(src, os.path.join(output_fp8_path, fname), overwrite=True)

    done_marker = os.path.join(
        output_fp8_path, f"_export_shard_{shard_index:02d}_of_{num_shards:02d}.done"
    )
    with gfile.Open(done_marker, "w") as f:
        f.write(json.dumps({"shard_index": shard_index, "elapsed_s": round(time.time() - t0, 1)}))

    print(
        f"[fp8_export shard {shard_index}/{num_shards}] Finished local shards in {time.time() - t0:.1f}s; waiting for peers...",
        flush=True,
    )
    while True:
        all_done = all(
            gfile.Exists(
                os.path.join(output_fp8_path, f"_export_shard_{i:02d}_of_{num_shards:02d}.done")
            )
            for i in range(num_shards)
        )
        if all_done:
            break
        time.sleep(3.0)

    if shard_index == 0 and not gfile.Exists(complete_marker):
        with gfile.Open(complete_marker, "w") as f:
            f.write(
                json.dumps({
                    "lora_adapter_path": lora_adapter_path,
                    "full_sft_ckpt_dir": full_sft_ckpt_dir,
                    "num_shards": num_shards,
                    "elapsed_s": round(time.time() - t0, 1),
                })
            )
    print(
        f"[fp8_export shard {shard_index}/{num_shards}] All {num_shards} shards complete ({time.time() - t0:.1f}s total)!",
        flush=True,
    )


if __name__ == "__main__":
    app.run(main)

