"""Repairs legacy 2-bit DeepSeek-V4.1 E128 exports whose FP8 backbone weights lost their block scales.

The original exporter wrote the (trained) `layers.9.ffn.shared_experts.w{1,2,3}.weight` tensors as
raw `float8_e4m3fn` casts because their `.scale` tensors live in the *other* base shard, so the
exported checkpoints contain FP8 codes without any scale. This tool re-quantises those weights from
the milestone training checkpoint (`step-XXXXX-<tag>/backbone_trainable.pt`, bf16 master weights)
with the exact exporter quantiser (`quantize_fp8_block32_torch`, 2D `[out//32, in//32]` E8M0
scales like the base checkpoint) and writes a small `model-repair-00001.safetensors` shard that the
loader (`load_dequantized_dsv41_e128`) reads after the two base shards.

Usage:
  blaze run //experimental/users/arazdai/kaggle/sft:repair_dsv41_export_shared_scales -- \
    --sft_output_root=/cns/.../wu_1 --milestones=0.25,0.50,0.75,1.00
"""

from __future__ import annotations

import io
import json
import os
import struct
import tempfile
import time
from typing import Any

from absl import app
from absl import flags
from google3.pyglib import gfile
from safetensors.torch import save_file
import torch

from google3.experimental.users.arazdai.kaggle.sft import dsv41_sft_model

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "sft_output_root",
    "/cns/lu-d/home/arazdai/kaggle/sft/checkpoints_dsv41_e128/xid_305548873_dsv41_e128_moe_only_full_sft_8xb200_v4/wu_1",
    "SFT run directory containing step-XXXXX-<tag>/ and export_2bit/<tag>/.",
)
flags.DEFINE_string("milestones", "0.25,0.50,0.75,1.00", "Comma-separated epoch fractions.")
flags.DEFINE_string(
    "base_model_dir",
    "/cns/lu-d/home/arazdai/kaggle/models/dsv41_e128_2bit",
    "Base checkpoint dir (used to pick the 2D vs 1D scale layout per tensor).",
)
flags.DEFINE_bool("dry_run", False, "Only report which tensors would be repaired.")
flags.DEFINE_string(
    "report_keys",
    "",
    "If set (comma-separated param names), only print magnitude stats of these keys from the "
    "first milestone's backbone_trainable.pt and exit.",
)


def _read_safetensors_header(path: str) -> dict[str, Any]:
    with gfile.Open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
    hdr.pop("__metadata__", None)
    return hdr


def _collect_headers(model_dir: str, shards: list[str]) -> dict[str, dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for s in shards:
        p = os.path.join(model_dir, s)
        if gfile.Exists(p):
            for k, v in _read_safetensors_header(p).items():
                merged[k] = dict(v, _shard=s)
    return merged


def _unscaled_fp8_weights(hdr: dict[str, dict[str, Any]]) -> list[str]:
    return sorted(
        k
        for k, v in hdr.items()
        if k.endswith(".weight")
        and str(v.get("dtype", "")).startswith("F8")
        and (k[:-7] + ".scale") not in hdr
    )


def _find_step_dir(root: str, tag: str) -> str:
    cands = [
        e for e in gfile.ListDir(root)
        if os.path.basename(e).startswith("step-") and os.path.basename(e).endswith(f"-{tag}")
    ]
    if not cands:
        raise FileNotFoundError(f"No step-XXXXX-{tag} checkpoint dir under {root}")
    return os.path.join(root, os.path.basename(sorted(cands)[-1]))


def _repair_milestone(tag: str) -> dict[str, Any]:
    export_dir = os.path.join(FLAGS.sft_output_root, "export_2bit", tag)
    base_shards = ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]
    exp_hdr = _collect_headers(export_dir, base_shards)
    missing = _unscaled_fp8_weights(exp_hdr)
    report: dict[str, Any] = {"milestone": tag, "export_dir": export_dir, "unscaled_fp8_weights": missing}
    if not missing:
        print(f"[{tag}] no unscaled FP8 weights found; nothing to repair.", flush=True)
        return report
    print(f"[{tag}] unscaled FP8 weights: {missing}", flush=True)
    if FLAGS.dry_run:
        return report

    base_hdr = _collect_headers(FLAGS.base_model_dir, base_shards)
    step_dir = _find_step_dir(FLAGS.sft_output_root, tag)
    pt_path = os.path.join(step_dir, "backbone_trainable.pt")
    t0 = time.time()
    with gfile.Open(pt_path, "rb") as f:
        state = torch.load(io.BytesIO(f.read()), map_location="cpu")
    print(f"[{tag}] loaded {pt_path} ({len(state)} tensors) in {time.time()-t0:.1f}s", flush=True)

    out: dict[str, torch.Tensor] = {}
    details: dict[str, Any] = {}
    for k in missing:
        if k not in state:
            raise KeyError(f"{k} not found in {pt_path}; keys sample: {list(state)[:5]}")
        w = state[k].detach().float()
        scale_key = k[:-7] + ".scale"
        base_s = base_hdr.get(scale_key)
        is_2d = bool(base_s and len(base_s["shape"]) == 2 and base_s["shape"][0] == w.shape[0] // dsv41_sft_model.FP8_BLOCK_SIZE)
        w_fp8, s_e8m0 = dsv41_sft_model.quantize_fp8_block32_torch(
            w, block_size=dsv41_sft_model.FP8_BLOCK_SIZE, scale_2d=is_2d
        )
        # Round-trip check against the master weight
        w_deq = dsv41_sft_model.dequantize_fp8_block32_torch(w_fp8, s_e8m0, out_dtype=torch.float32)
        rel_err = float((w_deq - w).norm() / w.norm().clamp_min(1e-12))
        # Error of the legacy unscaled e4m3 cast that the old loader used
        legacy = w.to(torch.float8_e4m3fn).float()
        legacy_err = float((legacy - w).norm() / w.norm().clamp_min(1e-12))
        out[k] = w_fp8.contiguous().cpu()
        out[scale_key] = s_e8m0.contiguous().cpu()
        details[k] = {
            "weight_shape": list(w.shape),
            "scale_shape": list(s_e8m0.shape),
            "scale_2d": is_2d,
            "block_fp8_rel_err": rel_err,
            "legacy_unscaled_e4m3_rel_err": legacy_err,
            "weight_abs_max": float(w.abs().max()),
        }
        print(f"[{tag}] {k}: scale{list(s_e8m0.shape)} rel_err={rel_err:.4f} (legacy unscaled cast rel_err={legacy_err:.4f})", flush=True)

    repair_name = "model-repair-00001.safetensors"
    with tempfile.TemporaryDirectory(prefix="dsv41_repair_") as tmp:
        local = os.path.join(tmp, repair_name)
        save_file(out, local, metadata={"format": "pt", "repair": "shared_experts_fp8_scales"})
        dst = os.path.join(export_dir, repair_name)
        with open(local, "rb") as f_in, gfile.Open(dst, "wb") as f_out:
            f_out.write(f_in.read())
    # Verify
    rep_hdr = _read_safetensors_header(os.path.join(export_dir, repair_name))
    assert set(rep_hdr) == set(out), (sorted(rep_hdr), sorted(out))

    # Keep the HF index consistent (weights + scales now live in the repair shard)
    idx_path = os.path.join(export_dir, "model.safetensors.index.json")
    if gfile.Exists(idx_path):
        with gfile.Open(idx_path, "r") as f:
            idx = json.load(f)
        wm = idx.setdefault("weight_map", {})
        for k in out:
            wm[k] = repair_name
        with gfile.Open(idx_path, "w") as f:
            json.dump(idx, f, indent=2)

    report.update({"repair_shard": repair_name, "step_dir": step_dir, "tensors": details})
    with gfile.Open(os.path.join(export_dir, "REPAIR_COMPLETED.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"[{tag}] wrote {os.path.join(export_dir, repair_name)}", flush=True)
    return report


def main(argv: list[str]) -> None:
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")
    tags = [
        f"epoch_{float(x.strip()):.2f}".replace(".", "p")
        for x in FLAGS.milestones.split(",")
        if x.strip()
    ]
    if FLAGS.report_keys:
        step_dir = _find_step_dir(FLAGS.sft_output_root, tags[0])
        pt_path = os.path.join(step_dir, "backbone_trainable.pt")
        with gfile.Open(pt_path, "rb") as f:
            state = torch.load(io.BytesIO(f.read()), map_location="cpu")
        for k in [x.strip() for x in FLAGS.report_keys.split(",") if x.strip()]:
            w = state[k].detach().float()
            print(
                f"{k}: shape={list(w.shape)} abs_max={w.abs().max().item():.4f} "
                f"mean_abs={w.abs().mean().item():.5f} std={w.std().item():.5f} "
                f"frac_|w|>1={float((w.abs() > 1).float().mean()):.3f}",
                flush=True,
            )
        return
    for tag in tags:
        _repair_milestone(tag)


if __name__ == "__main__":
    app.run(main)
