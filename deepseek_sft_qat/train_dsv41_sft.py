"""B200 Distributed Full-SFT Trainer for Pruned DeepSeek-V4.1 (`Tool E128`) on ARC-AGI-2.

Features:
  1. De-quantizes `dsv41_e128_2bit` (`model-00001/00002.safetensors` FP8 block-32 backbone +
     `gptq/layer_00..39.safetensors` 2-bit Humming `uint2` routed experts) into BF16 / FP32 on
     `8x NVIDIA B200` (`180 GB` HBM3e per GPU), sharding the 128 routed experts per layer across
     the 8 ranks (`16` experts/layer/rank = `45.3 GB/GPU`).
  2. Trains 100% of the `5.12B` non-routed backbone parameters (`embed`, all 40 layers of
     MLA/SWA/Compressor/Indexer, mHC, 128-way router gates, shared experts, Engram `wkv/q/k`,
     RMSNorms, `head`) plus optional routed-expert LoRA (`--routed_expert_lora_r`) with FP32
     AdamW, length-bucketed batching, and token-level `loss_weights` (`WIC-alnum=10.0`,
     `final_submit=5.0`, `tool_call=2.0`).
  3. Saves resumable checkpoints and exports quantized 2-bit Humming + FP8 block-32 checkpoints
     at `0.25`, `0.50`, `0.75`, and `1.00` epoch (`--milestone_epoch_fractions=0.25,0.50,0.75,1.00`)
     for full 120-task ARC-AGI-2 evaluation with Havu's Candidate B32 harness.
"""

from __future__ import annotations

import datetime
import gc
import io
import json
import math
import os
import pickle
import random
import re
import socket
import subprocess
import sys
import threading
import time
from typing import Any

from absl import app
from absl import flags
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn

from google3.experimental.users.arazdai.kaggle.sft import dsv41_image_tokens
from google3.experimental.users.arazdai.kaggle.sft import dsv41_sft_model
from google3.experimental.users.arazdai.kaggle.sft import export_and_quantize
from google3.experimental.users.arazdai.kaggle.sft import train_full_sft
from google3.experimental.users.arazdai.kaggle.sft import train_sft
from google3.pyglib import gfile
from google3.pyglib.contrib.g3_multiprocessing import g3_multiprocessing

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "dsv41_model_dir",
    "/cns/lu-d/home/arazdai/kaggle/models/dsv41_e128_2bit",
    "CNS directory containing chankhavu's pruned DeepSeek-V4.1 Tool E128 2-bit checkpoint.",
)
flags.DEFINE_string(
    "dsv41_data_dir",
    "/cns/lu-d/home/arazdai/kaggle/sft/data/dsv41_e128_short_clean",
    "CNS directory containing train_tokens.npz and val_tokens.npz with loss_weights.",
)
flags.DEFINE_string(
    "dsv41_output_dir",
    "/cns/lu-d/home/arazdai/kaggle/sft/checkpoints_dsv41_e128/r1_full_sft",
    "CNS directory to write resumable checkpoints and exported 2-bit milestone checkpoints.",
)
flags.DEFINE_integer(
    "routed_expert_lora_r",
    32,
    "LoRA rank on the 128 routed experts per layer (default 32 = trains all 128 routed experts per layer within 82 GiB/B200 and merges + re-packs to 2-bit Humming at export; 0 = freeze routed experts or train dense if trainable_routed_experts_dense=True).",
)
flags.DEFINE_float(
    "routed_expert_lora_alpha",
    64.0,
    "LoRA alpha scaling when routed_expert_lora_r > 0.",
)
flags.DEFINE_bool(
    "trainable_routed_experts_dense",
    False,
    "If True (and routed_expert_lora_r == 0), train the full de-quantized BF16 weights (w1, w2, w3) of all 128 routed experts per layer.",
)
flags.DEFINE_bool(
    "train_moe_only",
    True,
    "If True, freeze embed, attention (MLA/SWA/Compressor/Indexer), Engram (ngrams), mHC, norms, LM head, and expert router gates, training ONLY the MoE expert layers (shared_experts + routed experts).",
)
flags.DEFINE_bool(
    "enable_qat",
    True,
    "If True, run Havu-exact Quantization-Aware Training (2-bit Humming locked-z + trainable FP32 scales & LoRA on routed experts, 2D FP8 block-32x32 STE on shared experts, and dynamic FP8/FP4 activation quantization) with FP32 master weights.",
)
flags.DEFINE_bool(
    "qat_trainable_expert_scales",
    True,
    "If True (when enable_qat=True), train the per-group 2-bit Humming scales (s) of all 128 routed experts in FP32.",
)
flags.DEFINE_bool(
    "qat_lock_gptq_codes",
    False,
    "If True, lock the 2-bit integer codes q to their initial GPTQ values and only adapt scales; if False (default), allow FP32 LoRA + scale updates to cross thresholds via STE.",
)
flags.DEFINE_float(
    "qat_kl_weight",
    0.15,
    "Nemotron QAD Forward-KL distillation weight (D_KL(P_base_teacher || P_quant_student)) added to token-weighted CE loss (0.0 disables teacher KL).",
)
flags.DEFINE_float(
    "qat_kl_temperature",
    1.0,
    "Temperature for Nemotron QAD Forward-KL logit distillation.",
)
flags.DEFINE_bool(
    "load_compact_engram",
    True,
    "If True, load the 16 compact Engram shards (model-engram-00001..00016.safetensors) and tokenizer.json into layers.{1,14}.engram.",
)
flags.DEFINE_string(
    "milestone_epoch_fractions",
    "0.25,0.50,0.75,1.00",
    "Comma-separated epoch fractions at which to run validation, save a milestone checkpoint, and export a 2-bit checkpoint.",
)
flags.DEFINE_bool(
    "use_token_loss_weights",
    True,
    "Apply token-level loss_weights (WIC-alnum=10, final_submit=5, tool_call=2) from train_tokens.npz.",
)
flags.DEFINE_bool(
    "auto_resume",
    False,
    "If True, resume from the highest-step `step-*` checkpoint in dsv41_output_dir whose trainable weight files are all present "
    "(so a Borg preemption + task restart continues instead of retraining from step 0). Trainable weights are restored exactly; "
    "the deterministic batch order is fast-forwarded; AdamW moments are re-initialized (see resume_lr_rewarmup_steps).",
)
flags.DEFINE_string(
    "resume_from_checkpoint_dir",
    "",
    "Explicit `step-XXXXX[-tag]` checkpoint directory to resume from (takes precedence over auto_resume discovery).",
)
flags.DEFINE_integer(
    "resume_lr_rewarmup_steps",
    4,
    "Linear LR re-warm-up steps right after a resume to soften the first updates of the re-initialized AdamW moments (0 disables).",
)
flags.DEFINE_string(
    "init_from_checkpoint_dir",
    "",
    "Warm start for a NEW round (e.g. epoch 2 on a new dataset): restore the trainable weights from this "
    "`step-XXXXX[-tag]` resumable checkpoint but start at global_step=0 with a fresh LR schedule, fresh AdamW "
    "moments and NO batch fast-forwarding (unlike --resume_from_checkpoint_dir, which continues the same run). "
    "Ignored when a complete checkpoint already exists in dsv41_output_dir (preemption resume takes precedence). "
    "Checkpoint tensors without a model parameter (e.g. LoRA tensors when --routed_expert_lora_r=0) are skipped.",
)
# ---- Round-2 (image checkpoint + QAT) flags ----
flags.DEFINE_float(
    "lora_learning_rate",
    0.0,
    "Peak LR for the routed-expert LoRA A/B parameters (0 = same as --learning_rate). Follows the same cosine schedule.",
)
flags.DEFINE_float(
    "scale_learning_rate",
    0.0,
    "Peak LR for the trainable 2-bit group scales `qat_scale` (0 = same as --learning_rate). At 1e-5 the relative "
    "step on s~0.028 is only 0.04%/step and many scales never cross a BF16 rounding boundary before export.",
)
flags.DEFINE_enum(
    "train_vision",
    "none",
    ["none", "aligner", "aligner+last_k", "all"],
    "Which vision-tower parameters to train when the checkpoint has one: 'none' (frozen), 'aligner' "
    "(aligner MLP + image_start/end/newline), 'aligner+last_k' (plus the last --vision_trainable_blocks ViT "
    "blocks and the final ViT norm), or 'all'.",
)
flags.DEFINE_integer("vision_trainable_blocks", 8, "Number of trailing ViT blocks trained under --train_vision=aligner+last_k.")
flags.DEFINE_float("vision_learning_rate", 0.0, "Peak LR for ViT blocks (0 = same as --learning_rate).")
flags.DEFINE_float("aligner_learning_rate", 0.0, "Peak LR for the aligner + image delimiter embeddings (0 = same as --learning_rate).")
flags.DEFINE_bool(
    "disable_vision",
    False,
    "Force the text-only path (no ViT/aligner built or loaded) even when config.json has a vision tower.",
)
flags.DEFINE_bool(
    "allow_non_qat_export",
    False,
    "Round-1 lesson: with --enable_qat=false the 2-bit export re-rounds W0+LoRA onto the frozen GPTQ grid and "
    "drops 100% of routed-expert updates (0 of 181B codes changed). Milestone exports with routed-expert "
    "adapters therefore require --enable_qat=true unless this flag is set.",
)
flags.DEFINE_string(
    "engram_model_dir",
    "",
    "Checkpoint directory holding the 16 compact Engram shards (model-engram-*.safetensors) when "
    "--dsv41_model_dir ships none (the image checkpoints reuse the E043 tables; identical "
    "`engram_compact_kept`). Empty = dsv41_model_dir, then the E043 default dir.",
)
flags.DEFINE_bool(
    "global_grad_clip",
    True,
    "Clip with the GLOBAL gradient norm (replicated params counted once + expert-parallel params summed "
    "across ranks) so every rank applies the same clip coefficient. Round 1 clipped on each rank's local "
    "norm, which scaled the replicated shared-expert weights differently per rank and let the replicas drift.",
)
flags.DEFINE_bool(
    "pilot_longest_batches_first",
    False,
    "Smoke/pilot aid: order the epoch's step batches by descending max sequence length so a --max_steps=N "
    "run exercises peak activation memory first.",
)
flags.DEFINE_integer(
    "num_nodes",
    1,
    "Number of 8-GPU nodes for DP x EP training (1 = 8x B200 EP=8; 2 = 16x B200 DP=2 x EP=8).",
)
flags.DEFINE_integer(
    "node_rank",
    0,
    "Node index in [0, num_nodes) when num_nodes > 1.",
)


def compute_milestone_steps(
    steps_per_epoch: int, total_steps: int, fractions_csv: str
) -> dict[int, str]:
    """Maps 1-indexed optimizer step numbers to milestone tags (`epoch_0p25`, `epoch_0p50`, etc.)."""
    milestones: dict[int, str] = {}
    for part in fractions_csv.split(","):
        part = part.strip()
        if not part:
            continue
        frac = float(part)
        step = max(1, min(total_steps, int(round(steps_per_epoch * frac))))
        tag = f"epoch_{frac:.2f}".replace(".", "p")
        milestones[step] = tag
    milestones[total_steps] = milestones.get(total_steps, "epoch_1p00")
    return milestones


def build_length_bucketed_step_batches(
    dataset: train_sft.PackedTrajectoryDataset,
    seed: int,
    epoch: int,
    effective_batch: int,
    world_size: int,
    grad_accum: int,
    mega_batch_multiplier: int = 8,
) -> list[list[int | None]]:
    """Builds length-bucketed, rank-load-balanced step batches for `dataset` using `train_full_sft.build_epoch_batches`."""
    n = len(dataset)
    rng = np.random.default_rng(seed + epoch * 10007)
    order = rng.permutation(n)
    valid_lengths = np.asarray(
        [int(dataset.lengths[real_i]) for real_i in dataset.valid_indices],
        dtype=np.int64,
    )
    bucket_rng = np.random.default_rng(seed + epoch * 10007 + 7919)
    return train_full_sft.build_epoch_batches(
        order=order,
        effective_batch=effective_batch,
        world_size=world_size,
        micro_batch_size=1,
        grad_accum_steps=grad_accum,
        lengths=valid_lengths,
        bucket_rng=bucket_rng,
        mega_batch_multiplier=mega_batch_multiplier,
    )


def _vision_param_trainable(
    name: str,
    train_vision: str,
    vision_trainable_blocks: int,
    n_vision_blocks: int,
) -> bool:
    """Vision-tower training policy for one parameter name (`vision.*`, `aligner.*`, `image_*`)."""
    if train_vision == "none":
        return False
    is_aligner = name.startswith("aligner.") or name.startswith("image_")
    if is_aligner:
        return True
    if not name.startswith("vision."):
        return False
    if train_vision == "all":
        return True
    if train_vision == "aligner+last_k":
        if name.startswith("vision.norm."):
            return vision_trainable_blocks > 0
        m = re.match(r"^vision\.blocks\.(\d+)\.", name)
        if m is None:
            return False  # patch embedding stays frozen
        return int(m.group(1)) >= n_vision_blocks - vision_trainable_blocks
    return False


def configure_dsv41_trainable_parameters(
    model: dsv41_sft_model.Dsv41SftModel,
    train_moe_only: bool = True,
    train_vision: str = "none",
    vision_trainable_blocks: int = 0,
) -> dict[str, int]:
    """Configures `requires_grad` across `model` when `train_moe_only=True` (freezing attention, ngrams/Engram, router gate, mHC, norms, embed, head).

    Vision-tower parameters (`vision.*`, `aligner.*`, `image_*`) follow `train_vision` (see the
    `--train_vision` flag); `gate.bias_vl` is a buffer and stays frozen.
    """
    n_vision_blocks = len(model.vision.blocks) if getattr(model, "vision", None) is not None else 0
    if train_moe_only:
        for name, p in model.named_parameters():
            is_vision_param = name.startswith(("vision.", "aligner.", "image_"))
            if is_vision_param:
                p.requires_grad_(
                    _vision_param_trainable(name, train_vision, vision_trainable_blocks, n_vision_blocks)
                )
                continue
            is_moe_expert_param = (
                ".ffn.shared_experts." in name
                or (
                    ".ffn.experts." in name
                    and (
                        model.cfg.trainable_routed_experts_dense
                        or "lora_" in name
                        or (
                            model.cfg.enable_qat
                            and model.cfg.qat_trainable_expert_scales
                            and "qat_scale" in name
                        )
                    )
                )
            )
            p.requires_grad_(is_moe_expert_param)
    elif n_vision_blocks:
        for name, p in model.named_parameters():
            if name.startswith(("vision.", "aligner.", "image_")):
                p.requires_grad_(
                    _vision_param_trainable(name, train_vision, vision_trainable_blocks, n_vision_blocks)
                )

    trainable_tensors = sum(1 for p in model.parameters() if p.requires_grad)
    frozen_tensors = sum(1 for p in model.parameters() if not p.requires_grad)
    trainable_numel = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
    frozen_numel = int(sum(p.numel() for p in model.parameters() if not p.requires_grad))
    qat_scale_params = int(
        sum(
            p.numel()
            for n, p in model.named_parameters()
            if p.requires_grad and "qat_scale" in n
        )
    )
    vision_params = int(
        sum(
            p.numel()
            for n, p in model.named_parameters()
            if p.requires_grad and n.startswith(("vision.", "aligner.", "image_"))
        )
    )
    return {
        "train_moe_only": int(train_moe_only),
        "trainable_tensors": trainable_tensors,
        "trainable_numel": trainable_numel,
        "frozen_tensors": frozen_tensors,
        "frozen_numel": frozen_numel,
        "qat_scale_params": qat_scale_params,
        "vision_trainable_numel": vision_params,
    }


def lr_tier_of(name: str) -> str:
    """Learning-rate tier of a parameter: `lora`, `scale`, `vision`, `aligner`, or `base`."""
    if "lora_" in name:
        return "lora"
    if "qat_scale" in name:
        return "scale"
    if name.startswith("vision."):
        return "vision"
    if name.startswith(("aligner.", "image_")):
        return "aligner"
    return "base"


def build_dsv41_optimizer_param_groups(
    model: dsv41_sft_model.Dsv41SftModel,
    weight_decay: float,
    lr_mults: dict[str, float] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Splits trainable parameters into decayed 2-D matrices vs non-decayed 1-D/norm/bias/mHC/sink/qat_scale params.

    `lr_mults` maps LR tiers (`lr_tier_of`) to multipliers of the scheduled base LR; each group
    carries `lr_mult` and the trainer sets `pg["lr"] = lr_now * pg["lr_mult"]` every step.
    """
    lr_mults = {k: float(v) for k, v in (lr_mults or {}).items()}
    no_decay_keywords = (
        "norm",
        "bias",
        "attn_sink",
        "hc_",
        "q_weight",
        "k_weight",
        "qat_scale",
        "image_start",
        "image_end",
        "image_newline",
    )
    buckets: dict[tuple[str, bool], list[nn.Parameter]] = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        decay = not (p.ndim < 2 or any(k in name for k in no_decay_keywords))
        buckets.setdefault((lr_tier_of(name), decay), []).append(p)

    groups: list[dict[str, Any]] = []
    stats: dict[str, int] = {
        "decay_tensors": 0,
        "decay_numel": 0,
        "no_decay_tensors": 0,
        "no_decay_numel": 0,
    }
    for (tier, decay), params in sorted(buckets.items(), key=lambda kv: (kv[0][0], not kv[0][1])):
        groups.append(
            {
                "params": params,
                "weight_decay": weight_decay if decay else 0.0,
                "lr_mult": lr_mults.get(tier, 1.0),
                "tier": tier,
            }
        )
        key = "decay" if decay else "no_decay"
        stats[f"{key}_tensors"] += len(params)
        stats[f"{key}_numel"] += int(sum(p.numel() for p in params))
        stats[f"tier_{tier}_numel"] = stats.get(f"tier_{tier}_numel", 0) + int(sum(p.numel() for p in params))
    return groups, stats


class VLPackedTrajectoryDataset(train_sft.PackedTrajectoryDataset):
    """`PackedTrajectoryDataset` plus per-token `token_types` and per-trajectory PNG image blobs.

    Reads the optional `token_types` key of `*_tokens.npz` (int8, `-1` text / `0..3` image-span
    types, see `dsv41_image_tokens`) and the optional `{split}_images.pkl` sidecar (a list aligned
    with the npz trajectories; each element is the list of encoded images in prompt order).
    Text-only datasets load exactly as before.
    """

    def __init__(self, npz_path: str, max_seq_len: int = 65536):
        super().__init__(npz_path, max_seq_len=max_seq_len)
        with gfile.Open(npz_path, "rb") as f:
            raw_bytes = f.read()
        with np.load(io.BytesIO(raw_bytes)) as data:
            self.flat_types = data["token_types"] if "token_types" in data else None
        del raw_bytes
        self.images: list[list[bytes]] | None = None
        images_path = re.sub(r"_tokens\.npz$", "_images.pkl", npz_path)
        if images_path != npz_path and gfile.Exists(images_path):
            with gfile.Open(images_path, "rb") as f:
                self.images = pickle.load(f)
            if len(self.images) != len(self.lengths):
                raise ValueError(
                    f"{images_path}: {len(self.images)} image lists for {len(self.lengths)} trajectories"
                )
        self.n_with_images = (
            sum(1 for i in self.valid_indices if self.images[i]) if self.images is not None else 0
        )

    def __getitem__(self, i: int) -> dict[str, Any]:
        item = super().__getitem__(i)
        real_idx = self.valid_indices[i]
        if self.flat_types is not None:
            start = int(self.offsets[real_idx])
            end = int(self.offsets[real_idx + 1])
            item["token_types"] = torch.from_numpy(self.flat_types[start:end].astype(np.int64))
        if self.images is not None:
            item["images"] = list(self.images[real_idx])
        return item


def prepare_vl_inputs(
    sample: dict[str, Any], device: torch.device
) -> tuple[torch.Tensor | None, list[list[dsv41_sft_model.ImageInput]] | None]:
    """Builds `(token_types [1, S], images)` for `model.forward_hidden` from a dataset sample.

    Returns `(None, None)` for text-only samples so the forward is byte-identical to round 1.
    """
    token_types = sample.get("token_types")
    blobs = sample.get("images") or []
    if token_types is None or not bool((token_types >= 0).any()):
        return None, None
    image_inputs: list[dsv41_sft_model.ImageInput] = []
    for blob in blobs:
        patches, plan = dsv41_image_tokens.preprocess_image(blob)
        image_inputs.append(
            dsv41_sft_model.ImageInput(
                patches=torch.from_numpy(patches).pin_memory() if device.type == "cuda" else torch.from_numpy(patches),
                n_vit_h=plan.n_vit_h,
                n_vit_w=plan.n_vit_w,
            )
        )
    n_slots = int((token_types == dsv41_image_tokens.IMAGE).sum().item())
    n_rows = sum(
        math.ceil(img.n_vit_h / dsv41_image_tokens.DOWNSAMPLE_RATIO)
        * math.ceil(img.n_vit_w / dsv41_image_tokens.DOWNSAMPLE_RATIO)
        for img in image_inputs
    )
    if n_slots != n_rows:
        raise ValueError(f"token_types has {n_slots} IMAGE slots but the images yield {n_rows} aligner rows")
    return token_types.unsqueeze(0).to(device, non_blocking=True), [image_inputs]


def _qat_qad_chunk_loss(
    h_s_chunk: torch.Tensor,
    h_t_chunk: torch.Tensor,
    head_w: torch.Tensor,
    y_chunk: torch.Tensor,
    w_chunk: torch.Tensor | None,
    kl_weight: float,
    kl_temp: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Computes `(total_weighted_loss_sum, unweighted_kl_sum)` for one sequence chunk."""
    logits_s = torch.nn.functional.linear(h_s_chunk, head_w).float()
    ce_tok = torch.nn.functional.cross_entropy(
        logits_s, y_chunk, ignore_index=-100, reduction="none"
    )
    valid_mask = y_chunk != -100
    if kl_weight > 0.0 and h_t_chunk.numel() > 0:
        inv_t = 1.0 / max(1e-4, kl_temp)
        with torch.no_grad():
            h_t_dev = h_t_chunk.to(
                device=h_s_chunk.device, dtype=head_w.dtype, non_blocking=True
            )
            logits_t = torch.nn.functional.linear(h_t_dev, head_w).float()
            log_p_t = torch.nn.functional.log_softmax(logits_t * inv_t, dim=-1)
            del h_t_dev, logits_t
        log_p_s = torch.nn.functional.log_softmax(logits_s * inv_t, dim=-1)
        kl_tok = (
            torch.nn.functional.kl_div(
                log_p_s, log_p_t, reduction="none", log_target=True
            ).sum(dim=-1)
            * (kl_temp * kl_temp)
        )
        kl_tok = torch.where(valid_mask, kl_tok, torch.zeros_like(kl_tok))
        comb_tok = ce_tok + kl_weight * kl_tok
        kl_sum = kl_tok.detach().sum()
    else:
        comb_tok = ce_tok
        kl_sum = torch.zeros((), dtype=torch.float32, device=h_s_chunk.device)

    if w_chunk is not None:
        w_f = torch.where(valid_mask, w_chunk.float(), torch.zeros_like(comb_tok))
        return (comb_tok * w_f).sum(), kl_sum
    return comb_tok.sum(), kl_sum


def compute_chunked_qat_qad_loss(
    student_hidden: torch.Tensor,
    teacher_hidden: torch.Tensor | None,
    lm_head: nn.Linear,
    labels: torch.Tensor,
    chunk_size: int = 2048,
    loss_weights: torch.Tensor | None = None,
    qat_kl_weight: float = 0.0,
    qat_kl_temperature: float = 1.0,
) -> tuple[torch.Tensor, int, float]:
    """Computes chunked token-weighted CE + Nemotron QAD Forward-KL loss without materializing full `[B, S, V]` logits."""
    if qat_kl_weight <= 0.0 or teacher_hidden is None:
        loss_sum, n_valid = train_sft.compute_chunked_causal_lm_loss(
            hidden_states=student_hidden,
            lm_head=lm_head,
            labels=labels,
            chunk_size=chunk_size,
            loss_weights=loss_weights,
        )
        return loss_sum, n_valid, 0.0

    shift_s = student_hidden[..., :-1, :].reshape(-1, student_hidden.size(-1))
    shift_t = teacher_hidden[..., :-1, :].reshape(-1, teacher_hidden.size(-1)).detach()
    shift_y = labels[..., 1:].reshape(-1)
    shift_w = (
        loss_weights[..., 1:].reshape(-1) if loss_weights is not None else None
    )
    valid_mask = shift_y != -100
    n_valid = int(valid_mask.sum().item())
    if n_valid == 0:
        return shift_s.sum() * 0.0, 0, 0.0

    total_loss = torch.zeros((), dtype=torch.float32, device=student_hidden.device)
    total_kl = 0.0
    n_tokens = shift_s.size(0)
    use_ckpt = torch.is_grad_enabled() and (
        shift_s.requires_grad or lm_head.weight.requires_grad
    )

    for start in range(0, n_tokens, chunk_size):
        end = min(start + chunk_size, n_tokens)
        y_c = shift_y[start:end]
        if not torch.any(y_c != -100):
            continue
        s_c = shift_s[start:end]
        t_c = shift_t[start:end]
        w_c = shift_w[start:end] if shift_w is not None else None
        if use_ckpt:
            c_loss, c_kl = torch.utils.checkpoint.checkpoint(
                _qat_qad_chunk_loss,
                s_c,
                t_c,
                lm_head.weight,
                y_c,
                w_c,
                qat_kl_weight,
                qat_kl_temperature,
                use_reentrant=False,
            )
        else:
            c_loss, c_kl = _qat_qad_chunk_loss(
                s_c,
                t_c,
                lm_head.weight,
                y_c,
                w_c,
                qat_kl_weight,
                qat_kl_temperature,
            )
        total_loss = total_loss + c_loss
        total_kl += float(c_kl.item())

    return total_loss, n_valid, total_kl


def save_dsv41_resumable_checkpoint(
    output_dir: str,
    step: int,
    epoch: int,
    step_in_epoch: int,
    model: dsv41_sft_model.Dsv41SftModel,
    optimizer: torch.optim.Optimizer,
    ema_loss: float,
    best_val_loss: float,
    metrics_history: list[dict[str, Any]],
    rank: int,
    world_size: int,
    milestone_tag: str = "",
) -> str:
    """Saves trainable backbone/shared_experts + local routed-expert LoRA/scales state_dict and trainer_state.json to CNS."""
    ckpt_name = f"step-{step:05d}" if not milestone_tag else f"step-{step:05d}-{milestone_tag}"
    ckpt_dir = os.path.join(output_dir, ckpt_name)
    if rank == 0:
        export_and_quantize.ensure_cns_quota_dir(output_dir)
        export_and_quantize.ensure_cns_quota_dir(ckpt_dir)
    if world_size > 1 and dist.is_initialized():
        dist.barrier()

    ep_ws = max(1, int(getattr(model, "ep_world_size", world_size)))
    ep_rank = int(getattr(model, "ep_rank", rank % ep_ws))
    if rank < ep_ws and (
        model.cfg.routed_expert_lora_r > 0
        or model.cfg.trainable_routed_experts_dense
        or (model.cfg.enable_qat and model.cfg.qat_trainable_expert_scales)
    ):
        expert_lora_state = {
            k: v.detach().cpu()
            for k, v in model.named_parameters()
            if v.requires_grad and ".experts." in k
        }
        buf_e = io.BytesIO()
        torch.save(expert_lora_state, buf_e)
        with gfile.Open(
            os.path.join(ckpt_dir, f"expert_lora_rank_{ep_rank:02d}.pt"), "wb"
        ) as f:
            f.write(buf_e.getvalue())

    if rank == 0:
        trainable_state = {
            k: v.detach().cpu()
            for k, v in model.named_parameters()
            if v.requires_grad and ".experts." not in k
        }
        buf_io = io.BytesIO()
        torch.save(trainable_state, buf_io)
        with gfile.Open(os.path.join(ckpt_dir, "backbone_trainable.pt"), "wb") as f:
            f.write(buf_io.getvalue())

    if world_size > 1 and dist.is_initialized():
        dist.barrier()

    if rank == 0:
        state_meta = {
            "global_step": step,
            "epoch": epoch,
            "step_in_epoch": step_in_epoch,
            "milestone_tag": milestone_tag,
            "ema_loss": ema_loss,
            "best_val_loss": best_val_loss,
            "world_size": world_size,
            "ep_world_size": ep_ws,
            "timestamp": time.time(),
        }
        with gfile.Open(os.path.join(ckpt_dir, "COMPLETED.json"), "w") as f:
            json.dump(state_meta, f, indent=2)
        with gfile.Open(os.path.join(output_dir, "metrics.jsonl"), "w") as f:
            for row in metrics_history:
                f.write(json.dumps(row) + "\n")

    if world_size > 1 and dist.is_initialized():
        dist.barrier()
    return ckpt_dir


_CKPT_DIR_RE = re.compile(r"^step-(\d{5,})(?:-(.+))?$")


def parse_checkpoint_step(ckpt_dir: str) -> tuple[int, str] | None:
    """Parses `step-00057-epoch_0p50` -> `(57, "epoch_0p50")`; returns None for non-checkpoint names."""
    m = _CKPT_DIR_RE.match(os.path.basename(ckpt_dir.rstrip("/")))
    if not m:
        return None
    return int(m.group(1)), (m.group(2) or "")


def model_has_expert_checkpoint_state(model: dsv41_sft_model.Dsv41SftModel) -> bool:
    """Mirrors the condition under which `save_dsv41_resumable_checkpoint` writes `expert_lora_rank_XX.pt`."""
    return bool(
        model.cfg.routed_expert_lora_r > 0
        or model.cfg.trainable_routed_experts_dense
        or (model.cfg.enable_qat and model.cfg.qat_trainable_expert_scales)
    )


def checkpoint_weight_files(
    ckpt_dir: str, world_size: int, has_expert_state: bool
) -> list[str]:
    files = [os.path.join(ckpt_dir, "backbone_trainable.pt")]
    if has_expert_state:
        files.extend(
            os.path.join(ckpt_dir, f"expert_lora_rank_{r:02d}.pt")
            for r in range(world_size)
        )
    return files


def find_latest_resumable_checkpoint(
    output_dir: str, world_size: int, has_expert_state: bool
) -> str | None:
    """Returns the highest-step `step-*` directory under `output_dir` that is complete.

    Complete means `COMPLETED.json` is present *and* all trainable weight files exist. The marker is
    written only after every rank's weight file has been flushed (barrier) so it is the only reliable
    completeness signal: at v12 step 57 a Borg preemption left all 9 weight files *present* but the
    backbone file truncated at 528 MB of 2.83 GB. Such marker-less directories are skipped here (they
    can still be loaded explicitly via `--resume_from_checkpoint_dir`).
    """
    if not output_dir or not gfile.Exists(output_dir):
        return None
    candidates: list[tuple[int, str]] = []
    for entry in gfile.ListDir(output_dir):
        parsed = parse_checkpoint_step(entry)
        if parsed is None:
            continue
        ckpt_dir = os.path.join(output_dir, entry)
        if not gfile.IsDirectory(ckpt_dir):
            continue
        if not gfile.Exists(os.path.join(ckpt_dir, "COMPLETED.json")):
            continue
        if all(
            gfile.Exists(p)
            for p in checkpoint_weight_files(ckpt_dir, world_size, has_expert_state)
        ):
            candidates.append((parsed[0], ckpt_dir))
    if not candidates:
        return None
    return max(candidates)[1]


def load_dsv41_resumable_checkpoint(
    ckpt_dir: str,
    model: dsv41_sft_model.Dsv41SftModel,
    rank: int,
    world_size: int,
    skip_unknown_tensors: bool = False,
) -> dict[str, Any]:
    """Restores all trainable parameters of `model` in place from `ckpt_dir` (backbone + this rank's routed-expert state).

    Returns `{"step", "milestone_tag", "loaded_tensors", "skipped_tensors", ["ema_loss", "best_val_loss"]}`; raises if
    any trainable parameter is missing from the checkpoint or has a mismatched shape. With `skip_unknown_tensors=True`,
    checkpoint tensors that have no parameter in the model (e.g. `lora_*` when the new run uses
    `--routed_expert_lora_r=0`) are skipped instead of raising.
    """
    parsed = parse_checkpoint_step(ckpt_dir)
    if parsed is None:
        marker_probe = os.path.join(ckpt_dir, "COMPLETED.json")
        if gfile.Exists(marker_probe):
            with gfile.Open(marker_probe, "r") as f:
                meta_probe = json.load(f)
            parsed = (
                int(meta_probe.get("global_step", meta_probe.get("step", 0))),
                str(meta_probe.get("milestone_tag", "")),
            )
        else:
            raise ValueError(f"Not a step-XXXXX checkpoint directory: {ckpt_dir!r}")
    params = dict(model.named_parameters())
    trainable = {k for k, p in params.items() if p.requires_grad}
    loaded: set[str] = set()
    skipped: list[str] = []

    def _load_state_file(path: str) -> None:
        with gfile.Open(path, "rb") as f:
            state = torch.load(io.BytesIO(f.read()), map_location="cpu")
        with torch.no_grad():
            for k, v in state.items():
                p = params.get(k)
                if p is None:
                    if skip_unknown_tensors:
                        skipped.append(k)
                        continue
                    raise KeyError(f"Checkpoint tensor {k!r} from {path} has no parameter in the model.")
                if tuple(p.shape) != tuple(v.shape):
                    raise ValueError(
                        f"Shape mismatch for {k!r}: checkpoint {tuple(v.shape)} vs model {tuple(p.shape)}."
                    )
                p.data.copy_(v.to(device=p.device, dtype=p.dtype))
                loaded.add(k)

    _load_state_file(os.path.join(ckpt_dir, "backbone_trainable.pt"))
    if model_has_expert_checkpoint_state(model):
        ep_ws = max(1, int(getattr(model, "ep_world_size", world_size)))
        ep_rank = int(getattr(model, "ep_rank", rank % ep_ws))
        expert_path = os.path.join(ckpt_dir, f"expert_lora_rank_{ep_rank:02d}.pt")
        if not gfile.Exists(expert_path):
            raise FileNotFoundError(
                f"Missing {expert_path}: checkpoint was saved with a different EP world size "
                f"(now ep_world_size={ep_ws}, world_size={world_size})?"
            )
        _load_state_file(expert_path)

    missing = sorted(trainable - loaded)
    if missing:
        raise RuntimeError(
            f"{len(missing)} trainable parameters are missing from {ckpt_dir} "
            f"(e.g. {missing[:3]}); refusing to resume from a partial checkpoint."
        )
    info: dict[str, Any] = {
        "step": parsed[0],
        "milestone_tag": parsed[1],
        "loaded_tensors": len(loaded),
    }
    marker = os.path.join(ckpt_dir, "COMPLETED.json")
    if gfile.Exists(marker):
        with gfile.Open(marker, "r") as f:
            meta = json.load(f)
        for key in ("ema_loss", "best_val_loss"):
            if key in meta:
                info[key] = float(meta[key])
    return info


def load_metrics_history(output_dir: str, max_step: int) -> list[dict[str, Any]]:
    """Reads `metrics.jsonl` rows with `step <= max_step` so a resumed run keeps the full loss history."""
    path = os.path.join(output_dir, "metrics.jsonl")
    if not gfile.Exists(path):
        return []
    rows: list[dict[str, Any]] = []
    with gfile.Open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if int(row.get("step", -1)) <= max_step:
                rows.append(row)
    return rows


@torch.no_grad()
def evaluate_dsv41_validation_loss(
    model: dsv41_sft_model.Dsv41SftModel,
    val_dataset: train_sft.PackedTrajectoryDataset,
    device: torch.device,
    rank: int,
    world_size: int,
    max_eval_samples: int,
    logits_chunk_size: int,
) -> dict[str, float]:
    """Evaluates teacher-forced cross-entropy loss & perplexity across `val_dataset`."""
    was_training = model.training
    model.eval()
    t0 = time.time()

    total_n = len(val_dataset)
    if max_eval_samples > 0:
        total_n = min(total_n, max_eval_samples)

    padded_n = int(math.ceil(total_n / float(world_size))) * world_size
    local_loss_sum = 0.0
    local_tok_sum = 0
    local_wic_loss_sum = 0.0
    local_wic_tok_sum = 0

    for base_idx in range(0, padded_n, world_size):
        idx = base_idx + rank
        if idx < total_n:
            sample = val_dataset[idx]
            input_ids = sample["input_ids"].unsqueeze(0).to(device, non_blocking=True)
            labels = sample["labels"].unsqueeze(0).to(device, non_blocking=True)
            loss_weights = sample.get("loss_weights")
            tt, imgs = prepare_vl_inputs(sample, device)
        else:
            s0 = val_dataset[0]
            input_ids = s0["input_ids"][:64].unsqueeze(0).to(device, non_blocking=True)
            labels = torch.full_like(input_ids, -100)
            loss_weights = None
            tt, imgs = None, None

        hidden = model.forward_hidden(input_ids, token_types=tt, images=imgs)
        loss_sum_t, n_valid = train_sft.compute_chunked_causal_lm_loss(
            hidden_states=hidden,
            lm_head=model.head,
            labels=labels,
            chunk_size=logits_chunk_size,
            loss_weights=None,
        )
        if idx < total_n and n_valid > 0:
            local_loss_sum += float(loss_sum_t.item())
            local_tok_sum += int(n_valid)
            if loss_weights is not None:
                w_dev = loss_weights.to(device, non_blocking=True)
                wic_mask = (labels != -100) & (w_dev.unsqueeze(0) >= 9.5)
                n_wic = int(wic_mask[:, 1:].sum().item())
                if n_wic > 0:
                    wic_labels = torch.where(wic_mask, labels, torch.full_like(labels, -100))
                    wic_loss_t, _ = train_sft.compute_chunked_causal_lm_loss(
                        hidden_states=hidden,
                        lm_head=model.head,
                        labels=wic_labels,
                        chunk_size=logits_chunk_size,
                        loss_weights=None,
                    )
                    local_wic_loss_sum += float(wic_loss_t.item())
                    local_wic_tok_sum += n_wic

    stats_t = torch.tensor(
        [local_loss_sum, float(local_tok_sum), local_wic_loss_sum, float(local_wic_tok_sum)],
        dtype=torch.float64,
        device=device,
    )
    if world_size > 1 and dist.is_initialized():
        dist.all_reduce(stats_t, op=dist.ReduceOp.SUM)

    global_loss_sum = float(stats_t[0].item())
    global_tok_sum = max(1, int(round(stats_t[1].item())))
    global_wic_loss = float(stats_t[2].item()) / max(1.0, float(stats_t[3].item()))
    val_loss = global_loss_sum / float(global_tok_sum)
    val_ppl = math.exp(min(val_loss, 20.0))

    if was_training:
        model.train()

    return {
        "val_loss": round(val_loss, 5),
        "val_ppl": round(val_ppl, 4),
        "val_wic_alnum_loss": round(global_wic_loss, 5),
        "val_tokens": global_tok_sum,
        "val_samples": total_n,
        "eval_time_sec": round(time.time() - t0, 2),
    }


def _worker_main(rank: int, world_size: int) -> None:
    is_distributed = world_size > 1
    num_nodes = max(1, int(FLAGS.num_nodes))
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", max(1, world_size // num_nodes)))
    local_rank = int(os.environ.get("LOCAL_RANK", rank % local_world_size))
    ep_rank = local_rank
    ep_world_size = local_world_size

    os.environ.setdefault("HOME", "/tmp")
    os.environ["PYTORCH_KERNEL_CACHE_PATH"] = f"/tmp/pytorch_kernel_cache_dsv41_sft_rank_{local_rank}"
    os.makedirs(os.environ["PYTORCH_KERNEL_CACHE_PATH"], exist_ok=True)

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        backend = "nccl"
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    else:
        device = torch.device("cpu")
        backend = "gloo"

    if is_distributed and not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "::1")
        os.environ.setdefault("MASTER_PORT", "29509")
        os.environ.setdefault("NCCL_SOCKET_FAMILY", "AF_INET6")
        os.environ["TRITON_CACHE_DIR"] = f"/tmp/triton_cache_dsv41_sft_rank_{local_rank}"
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = f"/tmp/inductor_cache_dsv41_sft_rank_{local_rank}"
        init_method = os.environ.get("TORCH_DIST_INIT_METHOD", "")
        if not init_method:
            store_path = os.environ.get(
                "TORCH_DIST_INIT_FILE", "/tmp/dsv41_sft_c10d_filestore"
            )
            init_method = f"file://{store_path}"
        dist.init_process_group(
            backend=backend,
            init_method=init_method,
            rank=rank,
            world_size=world_size,
            timeout=datetime.timedelta(minutes=90),
            device_id=device if device.type == "cuda" else None,
        )

    ep_group: dist.ProcessGroup | None = None
    dp_group: dist.ProcessGroup | None = None
    if is_distributed and world_size > ep_world_size:
        n_dp = world_size // ep_world_size
        for n_idx in range(n_dp):
            ep_ranks = list(range(n_idx * ep_world_size, (n_idx + 1) * ep_world_size))
            grp = dist.new_group(ranks=ep_ranks)
            if rank in ep_ranks:
                ep_group = grp
        for r_idx in range(ep_world_size):
            dp_ranks = [n_idx * ep_world_size + r_idx for n_idx in range(n_dp)]
            grp = dist.new_group(ranks=dp_ranks)
            if rank in dp_ranks:
                dp_group = grp

    # Seed by ep_rank so data-parallel replicas of the same expert shard (rank r and rank 8+r)
    # initialize identical LoRA adapters (lora_A) before step 0.
    seed = int(FLAGS.seed)
    random.seed(seed + ep_rank)
    np.random.seed(seed + ep_rank)
    torch.manual_seed(seed + ep_rank)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed + ep_rank)

    if rank == 0:
        export_and_quantize.ensure_cns_quota_dir(FLAGS.dsv41_output_dir)
    if is_distributed:
        dist.barrier()

    # 1. Load model configuration & build differentiable Pruned DeepSeek-V4.1 E128 model
    cfg_path = os.path.join(FLAGS.dsv41_model_dir, "config.json")
    if gfile.Exists(cfg_path):
        with gfile.Open(cfg_path, "r") as f:
            cfg = dsv41_sft_model.Dsv41E128Config.from_json_dict(json.load(f))
    else:
        cfg = dsv41_sft_model.Dsv41E128Config()
    cfg.max_seq_len = max(cfg.max_seq_len, int(FLAGS.max_seq_len))
    cfg.routed_expert_lora_r = int(FLAGS.routed_expert_lora_r)
    cfg.routed_expert_lora_alpha = float(FLAGS.routed_expert_lora_alpha)
    cfg.trainable_routed_experts_dense = bool(
        FLAGS.trainable_routed_experts_dense and cfg.routed_expert_lora_r == 0
    )
    cfg.enable_qat = bool(FLAGS.enable_qat)
    cfg.qat_trainable_expert_scales = bool(FLAGS.qat_trainable_expert_scales)
    cfg.qat_lock_gptq_codes = bool(FLAGS.qat_lock_gptq_codes)
    cfg.load_compact_engram = bool(FLAGS.load_compact_engram)
    if bool(FLAGS.disable_vision):
        cfg.vision_n_layers = 0

    routed_adapters_trained = cfg.routed_expert_lora_r > 0 or cfg.trainable_routed_experts_dense
    exports_enabled = bool(str(FLAGS.milestone_epoch_fractions).strip())
    if (
        exports_enabled
        and routed_adapters_trained
        and not cfg.enable_qat
        and not bool(FLAGS.allow_non_qat_export)
    ):
        raise ValueError(
            "Refusing to train routed-expert adapters with --enable_qat=false while milestone 2-bit exports "
            "are on: the non-QAT export re-rounds W0+LoRA onto the frozen GPTQ grid and discards the adapters "
            "entirely (round 1: 0 of 181,193,932,800 codes changed). Use --enable_qat=true "
            "--qat_trainable_expert_scales=true, or pass --allow_non_qat_export=true knowingly."
        )

    if rank == 0:
        print(
            f"[Rank 0] Building Dsv41SftModel on {device} (n_layers={cfg.n_layers}, dim={cfg.dim}, "
            f"n_routed_experts={cfg.n_routed_experts}, world_size={world_size}, ep_world_size={ep_world_size}, "
            f"routed_expert_lora_r={cfg.routed_expert_lora_r}, "
            f"trainable_routed_experts_dense={cfg.trainable_routed_experts_dense}, "
            f"train_moe_only={FLAGS.train_moe_only}, enable_qat={cfg.enable_qat}, "
            f"qat_trainable_expert_scales={cfg.qat_trainable_expert_scales}, "
            f"qat_kl_weight={FLAGS.qat_kl_weight}, load_compact_engram={cfg.load_compact_engram}, "
            f"vision_n_layers={cfg.vision_n_layers}, train_vision={FLAGS.train_vision}/"
            f"{FLAGS.vision_trainable_blocks})...",
            flush=True,
        )

    t_build = time.time()
    with torch.device(device):
        model = dsv41_sft_model.Dsv41SftModel(
            cfg=cfg,
            ep_rank=ep_rank,
            ep_world_size=ep_world_size,
            gradient_checkpointing=bool(FLAGS.gradient_checkpointing),
            grad_ckpt_stride=int(FLAGS.grad_ckpt_stride),
            ep_replicated_input=False,
            ep_group=ep_group,
        )
    if rank == 0:
        print(
            f"[Rank 0] Allocated Dsv41SftModel on {device} in {time.time() - t_build:.2f}s",
            flush=True,
        )

    t_load = time.time()
    load_stats = dsv41_sft_model.load_dequantized_dsv41_e128(
        model_dir=FLAGS.dsv41_model_dir,
        model=model,
        device=device,
        engram_model_dir=(str(FLAGS.engram_model_dir) or None),
    )
    if float(FLAGS.qat_kl_weight) > 0.0:
        model.snapshot_teacher_weights()
    if rank == 0:
        print(
            f"[Rank 0] De-quantized E128 checkpoint in {time.time() - t_load:.1f}s: {load_stats}",
            flush=True,
        )

    freeze_stats = configure_dsv41_trainable_parameters(
        model,
        train_moe_only=bool(FLAGS.train_moe_only),
        train_vision=str(FLAGS.train_vision),
        vision_trainable_blocks=int(FLAGS.vision_trainable_blocks),
    )
    if dp_group is not None:
        with torch.no_grad():
            for p in model.parameters():
                if p.requires_grad:
                    dist.broadcast(p.data, src=ep_rank, group=dp_group)

    base_lr = float(FLAGS.learning_rate)
    lr_mults = {
        tier: (float(v) / base_lr if float(v) > 0.0 and base_lr > 0.0 else 1.0)
        for tier, v in (
            ("lora", FLAGS.lora_learning_rate),
            ("scale", FLAGS.scale_learning_rate),
            ("vision", FLAGS.vision_learning_rate),
            ("aligner", FLAGS.aligner_learning_rate),
        )
    }
    param_groups, param_stats = build_dsv41_optimizer_param_groups(
        model, weight_decay=float(FLAGS.weight_decay), lr_mults=lr_mults
    )
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=float(FLAGS.learning_rate),
        betas=(float(FLAGS.adam_beta1), float(FLAGS.adam_beta2)),
        eps=1e-8,
        fused=(device.type == "cuda"),
    )

    if rank == 0:
        print(
            f"[Rank 0] Trainable parameter policy: {freeze_stats} | Optimizer groups: {param_stats} | "
            f"lr_mults={lr_mults}",
            flush=True,
        )

    # 1b. Optional resume (trainable weights only; AdamW moments restart). Rank 0 decides which
    #     checkpoint to use and broadcasts it so all ranks restore the same step.
    resume_dir = str(FLAGS.resume_from_checkpoint_dir or "")
    if not resume_dir and bool(FLAGS.auto_resume) and rank == 0:
        resume_dir = (
            find_latest_resumable_checkpoint(
                output_dir=FLAGS.dsv41_output_dir,
                world_size=ep_world_size,
                has_expert_state=model_has_expert_checkpoint_state(model),
            )
            or ""
        )
    if is_distributed:
        resume_obj = [resume_dir]
        dist.broadcast_object_list(resume_obj, src=0)
        resume_dir = str(resume_obj[0] or "")
    resume_step = 0
    resume_info: dict[str, Any] = {}
    warm_start_info: dict[str, Any] = {}
    if resume_dir:
        t_res = time.time()
        resume_info = load_dsv41_resumable_checkpoint(
            ckpt_dir=resume_dir, model=model, rank=rank, world_size=world_size
        )
        resume_step = int(resume_info["step"])
        if rank == 0:
            print(
                f"[Rank 0] Resumed trainable weights from {resume_dir} (step {resume_step}) "
                f"in {time.time() - t_res:.1f}s: {resume_info}. AdamW moments are re-initialized; "
                f"LR re-warm-up over {int(FLAGS.resume_lr_rewarmup_steps)} step(s).",
                flush=True,
            )
    elif str(FLAGS.init_from_checkpoint_dir or ""):
        # Warm start of a NEW round: weights only; global_step stays 0 (fresh cosine schedule, fresh AdamW,
        # no batch fast-forwarding), so the new dataset is consumed from its first batch.
        t_res = time.time()
        warm_start_info = load_dsv41_resumable_checkpoint(
            ckpt_dir=str(FLAGS.init_from_checkpoint_dir),
            model=model,
            rank=rank,
            world_size=world_size,
            skip_unknown_tensors=True,
        )
        warm_start_info["init_from_checkpoint_dir"] = str(FLAGS.init_from_checkpoint_dir)
        if dp_group is not None:
            with torch.no_grad():
                for p in model.parameters():
                    if p.requires_grad:
                        dist.broadcast(p.data, src=ep_rank, group=dp_group)
        if rank == 0:
            print(
                f"[Rank 0] WARM START: loaded trainable weights from {FLAGS.init_from_checkpoint_dir} "
                f"(source step {warm_start_info.get('step')}, {warm_start_info.get('loaded_tensors')} tensors, "
                f"{len(warm_start_info.get('skipped_tensors') or [])} skipped) in {time.time() - t_res:.1f}s; "
                "training starts at global_step=0 with a fresh LR schedule and fresh AdamW moments.",
                flush=True,
            )
    elif bool(FLAGS.auto_resume) and rank == 0:
        print(
            f"[Rank 0] auto_resume: no complete step-* checkpoint under {FLAGS.dsv41_output_dir}; starting from step 0.",
            flush=True,
        )

    # 2. Load tokenized dataset with `loss_weights`
    train_npz = os.path.join(FLAGS.dsv41_data_dir, "train_tokens.npz")
    val_npz = os.path.join(FLAGS.dsv41_data_dir, "val_tokens.npz")
    train_dataset = VLPackedTrajectoryDataset(
        train_npz, max_seq_len=int(FLAGS.max_seq_len)
    )
    val_dataset = VLPackedTrajectoryDataset(
        val_npz, max_seq_len=int(FLAGS.max_seq_len)
    )
    if (train_dataset.n_with_images or val_dataset.n_with_images) and model.vision is None:
        raise ValueError(
            f"Dataset {FLAGS.dsv41_data_dir} contains image trajectories "
            f"(train={train_dataset.n_with_images}, val={val_dataset.n_with_images}) but the model has no "
            "vision tower (text-only checkpoint or --disable_vision=true)."
        )
    if rank == 0 and (train_dataset.n_with_images or val_dataset.n_with_images):
        print(
            f"[Rank 0] Image trajectories: train {train_dataset.n_with_images}/{len(train_dataset)}, "
            f"val {val_dataset.n_with_images}/{len(val_dataset)}",
            flush=True,
        )

    grad_accum = max(1, int(FLAGS.grad_accum_steps))
    effective_batch = world_size * grad_accum
    steps_per_epoch = int(math.ceil(len(train_dataset) / float(effective_batch)))
    total_steps = (
        int(FLAGS.max_steps)
        if int(FLAGS.max_steps) > 0
        else steps_per_epoch * int(FLAGS.num_epochs)
    )
    warmup_steps = max(1, int(round(total_steps * float(FLAGS.warmup_ratio))))
    milestones = compute_milestone_steps(
        steps_per_epoch=steps_per_epoch,
        total_steps=total_steps,
        fractions_csv=FLAGS.milestone_epoch_fractions,
    )

    if rank == 0:
        print(
            f"[Rank 0] Dataset: train={len(train_dataset)}, val={len(val_dataset)}, "
            f"effective_batch={effective_batch} (world_size={world_size} [dp={world_size // ep_world_size} x ep={ep_world_size}] x grad_accum={grad_accum}), "
            f"steps_per_epoch={steps_per_epoch}, total_steps={total_steps}, "
            f"milestones={milestones}",
            flush=True,
        )

    # 3. Initial validation loss before training (Step 0), or of the restored weights when resuming
    val_metrics_0 = evaluate_dsv41_validation_loss(
        model=model,
        val_dataset=val_dataset,
        device=device,
        rank=rank,
        world_size=world_size,
        max_eval_samples=int(FLAGS.max_eval_samples),
        logits_chunk_size=int(FLAGS.logits_chunk_size),
    )
    best_val_loss = val_metrics_0["val_loss"]
    ema_loss = val_metrics_0["val_loss"]
    metrics_history: list[dict[str, Any]] = [{"step": 0, **val_metrics_0}]
    if resume_step > 0:
        ema_loss = float(resume_info.get("ema_loss", ema_loss))
        best_val_loss = min(best_val_loss, float(resume_info.get("best_val_loss", best_val_loss)))
        prior_rows = load_metrics_history(FLAGS.dsv41_output_dir, max_step=resume_step)
        if prior_rows:
            metrics_history = prior_rows
        metrics_history.append(
            {"step": resume_step, "resumed_from": resume_dir, **val_metrics_0}
        )
    if rank == 0:
        label = "Initial Validation" if resume_step == 0 else f"Resume Validation (restored step {resume_step})"
        print(
            f"[Step {resume_step:05d} / {total_steps:05d}] {label}: {val_metrics_0}",
            flush=True,
        )
    if not math.isfinite(val_metrics_0["val_loss"]) or val_metrics_0["val_loss"] > 3.0:
        raise RuntimeError(
            f"Validation loss of the loaded weights is unexpectedly high ({val_metrics_0['val_loss']:.4f} > 3.0); "
            f"aborting before training to prevent corrupted checkpoint export. stats={load_stats}"
        )

    def _export_milestone(m_tag: str, step: int, row: dict[str, Any], ckpt_dir: str) -> None:
        export_2bit_dir = os.path.join(FLAGS.dsv41_output_dir, "export_2bit", m_tag)
        t_exp = time.time()
        exp_info = dsv41_sft_model.export_dsv41_e128_2bit_checkpoint(
            model=model,
            base_model_dir=FLAGS.dsv41_model_dir,
            export_cns_dir=export_2bit_dir,
            rank=rank,
            world_size=world_size,
        )
        if rank == 0:
            delta_keys = [k for k in exp_info if k.startswith(("routed_", "backbone_"))]
            export_delta = {k: exp_info[k] for k in delta_keys}
            marker_path = os.path.join(export_2bit_dir, "EXPORT_COMPLETED.json")
            with gfile.Open(marker_path, "w") as mf:
                json.dump(
                    {
                        "milestone_tag": m_tag,
                        "step": step,
                        "epoch": row["epoch"],
                        "val_loss": row.get("val_loss"),
                        "val_wic_alnum_loss": row.get("val_wic_alnum_loss"),
                        "export_dir": export_2bit_dir,
                        "elapsed_sec": round(time.time() - t_exp, 2),
                        "export_delta": export_delta,
                        "enable_qat": bool(cfg.enable_qat),
                    },
                    mf,
                    indent=2,
                )
            print(
                f"[Milestone {m_tag} @ Step {step:05d}] Saved checkpoint {ckpt_dir} "
                f"and exported 2-bit checkpoint to {export_2bit_dir} in {time.time() - t_exp:.1f}s: {exp_info}",
                flush=True,
            )
            print(
                f"[Milestone {m_tag}] EXPORT DELTA vs base: routed scales changed "
                f"{exp_info.get('routed_scales_changed', 0):,}/{exp_info.get('routed_scales_total', 0):,} "
                f"({100.0 * exp_info.get('routed_scales_changed_frac', 0.0):.2f}%), routed 2-bit codes flipped "
                f"{exp_info.get('routed_codes_changed', 0):,}/{exp_info.get('routed_codes_total', 0):,} "
                f"({100.0 * exp_info.get('routed_codes_changed_frac', 0.0):.4f}%), backbone trainable elems changed "
                f"{exp_info.get('backbone_changed_elems', 0):,}/{exp_info.get('backbone_trainable_elems', 0):,} "
                f"({100.0 * exp_info.get('backbone_changed_frac', 0.0):.2f}%)",
                flush=True,
            )
            if (
                exp_info.get("routed_scales_total", 0) > 0
                and exp_info.get("routed_scales_changed", 0) == 0
                and exp_info.get("routed_codes_changed", 0) == 0
            ):
                print(
                    f"[Milestone {m_tag}] WARNING: exported routed experts are byte-identical to the base "
                    "checkpoint (0 scales / 0 codes changed) -- the SFT did not reach the 2-bit weights.",
                    flush=True,
                )

    # 3b. A preemption can kill the run between a milestone's checkpoint save and its 2-bit export
    #     (v12 step 57): if we resumed exactly at such a milestone and its export marker is missing,
    #     export now so the polling evaluation jobs can start.
    if resume_step > 0 and resume_step in milestones:
        m_tag_res = milestones[resume_step]
        marker_res = os.path.join(
            FLAGS.dsv41_output_dir, "export_2bit", m_tag_res, "EXPORT_COMPLETED.json"
        )
        if not gfile.Exists(marker_res):
            if rank == 0:
                print(
                    f"[Rank 0] Milestone {m_tag_res} export is missing for restored step {resume_step}; exporting now.",
                    flush=True,
                )
            model.eval()
            _export_milestone(
                m_tag_res,
                resume_step,
                {
                    "epoch": round(float(resume_step) / float(steps_per_epoch), 4),
                    **val_metrics_0,
                },
                resume_dir,
            )

    # 4. Main SFT training loop
    model.train()
    global_step = 0
    t_train_start = time.time()
    rewarmup_steps = max(0, int(FLAGS.resume_lr_rewarmup_steps)) if resume_step > 0 else 0

    for epoch in range(int(FLAGS.num_epochs)):
        if FLAGS.length_bucketed_batching:
            step_batches = build_length_bucketed_step_batches(
                dataset=train_dataset,
                seed=seed,
                epoch=epoch,
                effective_batch=effective_batch,
                world_size=world_size,
                grad_accum=grad_accum,
                mega_batch_multiplier=int(FLAGS.bucket_mega_batch_multiplier),
            )
        else:
            g = torch.Generator()
            g.manual_seed(seed + epoch)
            perm = torch.randperm(len(train_dataset), generator=g).tolist()
            step_batches = [
                perm[i : i + effective_batch]
                for i in range(0, len(perm), effective_batch)
            ]
        if bool(FLAGS.pilot_longest_batches_first):
            # Smoke/pilot runs: put the batches with the longest sequences first so the few
            # steps executed exercise peak activation memory (the VRAM gate), not a random sample.
            step_batches = sorted(
                step_batches,
                key=lambda b: -max((int(train_dataset.lengths[i]) for i in b if i is not None), default=0),
            )
            if rank == 0:
                top = [max(int(train_dataset.lengths[i]) for i in b if i is not None) for b in step_batches[:3]]
                print(f"[Rank 0] pilot_longest_batches_first: first step batch max lengths = {top}", flush=True)

        for step_in_epoch, batch_indices in enumerate(step_batches):
            if global_step >= total_steps:
                break
            if global_step < resume_step:
                # Fast-forward the deterministic (seeded) batch order consumed before the resume point.
                global_step += 1
                continue

            t_step = time.time()
            lr_now = train_sft.get_cosine_lr(
                step=global_step,
                total_steps=total_steps,
                peak_lr=float(FLAGS.learning_rate),
                warmup_steps=warmup_steps,
                min_lr_ratio=float(FLAGS.min_lr_ratio),
            )
            if rewarmup_steps > 0 and (global_step - resume_step) < rewarmup_steps:
                lr_now *= float(global_step - resume_step + 1) / float(rewarmup_steps)
            for pg in optimizer.param_groups:
                pg["lr"] = lr_now * float(pg.get("lr_mult", 1.0))

            optimizer.zero_grad(set_to_none=True)

            # Pad partial batch to effective_batch with None slots so collectives match
            padded_slots: list[int | None] = list(batch_indices)
            while len(padded_slots) < effective_batch:
                padded_slots.append(None)
            fallback_idx = next((idx for idx in padded_slots if idx is not None), 0)

            step_max_seq = max(
                (int(train_dataset.lengths[idx]) for idx in padded_slots if idx is not None),
                default=0,
            )
            if device.type == "cuda" and step_max_seq >= 16384:
                torch.cuda.empty_cache()

            # Pre-compute global supervised token count across all ranks in this optimizer step
            local_sup_tokens = 0
            for m in range(grad_accum):
                slot = m * world_size + rank
                item_idx = padded_slots[slot]
                if item_idx is not None:
                    local_sup_tokens += int(
                        train_dataset[item_idx]["supervised_tokens"]
                    )
            sup_tok_tensor = torch.tensor(
                [local_sup_tokens], dtype=torch.float64, device=device
            )
            if is_distributed:
                dist.all_reduce(sup_tok_tensor, op=dist.ReduceOp.SUM)
            global_sup_tokens = max(1.0, float(sup_tok_tensor.item()))

            local_loss_accum = 0.0
            local_kl_accum = 0.0
            local_max_seq = 0
            kl_w = float(FLAGS.qat_kl_weight)
            kl_temp = float(FLAGS.qat_kl_temperature)

            for m in range(grad_accum):
                slot = m * world_size + rank
                item_idx = padded_slots[slot]
                is_real = item_idx is not None
                sample = train_dataset[item_idx if item_idx is not None else fallback_idx]
                vl_types: torch.Tensor | None = None
                vl_images: list[list[dsv41_sft_model.ImageInput]] | None = None
                if is_real:
                    input_ids = (
                        sample["input_ids"].unsqueeze(0).to(device, non_blocking=True)
                    )
                    labels = (
                        sample["labels"].unsqueeze(0).to(device, non_blocking=True)
                    )
                    loss_w = (
                        sample["loss_weights"].unsqueeze(0).to(device, non_blocking=True)
                        if (FLAGS.use_token_loss_weights and "loss_weights" in sample)
                        else None
                    )
                    vl_types, vl_images = prepare_vl_inputs(sample, device)
                else:
                    input_ids = (
                        sample["input_ids"][:64]
                        .unsqueeze(0)
                        .to(device, non_blocking=True)
                    )
                    labels = torch.full_like(input_ids, -100)
                    loss_w = None

                local_max_seq = max(local_max_seq, int(input_ids.size(1)))
                teacher_hidden: torch.Tensor | None = None
                if kl_w > 0.0:
                    with torch.no_grad():
                        model.set_teacher_mode(True)
                        teacher_hidden = model.forward_hidden(
                            input_ids, token_types=vl_types, images=vl_images
                        ).detach()
                        model.set_teacher_mode(False)

                hidden = model.forward_hidden(input_ids, token_types=vl_types, images=vl_images)
                loss_sum_t, n_valid, kl_sum_val = compute_chunked_qat_qad_loss(
                    student_hidden=hidden,
                    teacher_hidden=teacher_hidden,
                    lm_head=model.head,
                    labels=labels,
                    chunk_size=int(FLAGS.logits_chunk_size),
                    loss_weights=loss_w,
                    qat_kl_weight=kl_w,
                    qat_kl_temperature=kl_temp,
                )
                del teacher_hidden
                if not is_real or n_valid == 0:
                    # Touch hidden (and head if trainable) with 0.0 multiplier so EP collectives run in backward
                    scaled_loss = hidden.sum() * 0.0
                    if model.head.weight.requires_grad:
                        scaled_loss = scaled_loss + model.head.weight.sum() * 0.0
                else:
                    scaled_loss = loss_sum_t * (
                        float(world_size) / global_sup_tokens
                    )
                    local_loss_accum += float(loss_sum_t.detach().item())
                    local_kl_accum += float(kl_sum_val)

                scaled_loss.backward()

            # All-reduce gradients across ranks for replicated parameters; scale local expert grads.
            # Replicated params may have no grad on some ranks (vision tower when this rank's
            # micro-batches were text-only) -- zero-fill so all ranks issue identical collectives.
            if is_distributed:
                for name, p in model.named_parameters():
                    if not p.requires_grad:
                        continue
                    if ".experts." not in name:
                        if p.grad is None:
                            p.grad = torch.zeros_like(p)
                        dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
                    else:
                        if p.grad is None:
                            p.grad = torch.zeros_like(p)
                        if dp_group is not None:
                            dist.all_reduce(
                                p.grad, op=dist.ReduceOp.SUM, group=dp_group
                            )
                        p.grad.div_(float(world_size))

            trainable_params = [p for p in model.parameters() if p.requires_grad]
            if is_distributed and bool(FLAGS.global_grad_clip):
                # Global norm: replicated grads are identical on all ranks after the AVG all-reduce
                # (count once); expert-parallel grads are shard-local across the 8 EP ranks in ep_group.
                sq_rep = torch.zeros((), dtype=torch.float32, device=device)
                sq_exp = torch.zeros((), dtype=torch.float32, device=device)
                for name, p in model.named_parameters():
                    if not p.requires_grad or p.grad is None:
                        continue
                    g_sq = p.grad.detach().float().pow(2).sum()
                    if ".experts." in name:
                        sq_exp += g_sq
                    else:
                        sq_rep += g_sq
                dist.all_reduce(sq_exp, op=dist.ReduceOp.SUM, group=ep_group)
                total_norm_t = torch.sqrt(sq_rep + sq_exp)
                max_norm = float(FLAGS.max_grad_norm)
                if max_norm > 0.0:
                    clip_coef = (max_norm / (total_norm_t + 1e-6)).clamp(max=1.0)
                    for p in trainable_params:
                        if p.grad is not None:
                            p.grad.detach().mul_(clip_coef)
                grad_norm = float(total_norm_t.item())
            else:
                grad_norm = float(
                    torch.nn.utils.clip_grad_norm_(
                        trainable_params,
                        max_norm=float(FLAGS.max_grad_norm),
                    ).item()
                )
            optimizer.step()
            global_step += 1

            loss_stat = torch.tensor(
                [local_loss_accum, float(local_max_seq), local_kl_accum],
                dtype=torch.float64,
                device=device,
            )
            if is_distributed:
                dist.all_reduce(loss_stat, op=dist.ReduceOp.SUM)
            step_loss = float(loss_stat[0].item()) / global_sup_tokens
            step_kl = float(loss_stat[2].item()) / global_sup_tokens
            if global_step == 1 and resume_step == 0:
                ema_loss = step_loss
            else:
                ema_loss = 0.9 * ema_loss + 0.1 * step_loss
            dt_step = time.time() - t_step
            peak_gb = (
                torch.cuda.max_memory_allocated(device) / (1024.0**3)
                if device.type == "cuda"
                else 0.0
            )

            row: dict[str, Any] = {
                "step": global_step,
                "epoch": round(float(global_step) / float(steps_per_epoch), 4),
                "train_loss": round(step_loss, 5),
                "qad_kl": round(step_kl, 5),
                "ema_loss": round(ema_loss, 5),
                "grad_norm": round(grad_norm, 4),
                "lr": lr_now,
                "sup_tokens": int(global_sup_tokens),
                "step_time_sec": round(dt_step, 2),
                "peak_vram_gib": round(peak_gb, 2),
            }

            is_milestone = global_step in milestones
            should_eval = (
                is_milestone
                or (int(FLAGS.eval_every_steps) > 0 and global_step % int(FLAGS.eval_every_steps) == 0)
                or global_step == total_steps
            )
            if should_eval:
                val_res = evaluate_dsv41_validation_loss(
                    model=model,
                    val_dataset=val_dataset,
                    device=device,
                    rank=rank,
                    world_size=world_size,
                    max_eval_samples=int(FLAGS.max_eval_samples),
                    logits_chunk_size=int(FLAGS.logits_chunk_size),
                )
                row.update(val_res)
                best_val_loss = min(best_val_loss, val_res["val_loss"])

            metrics_history.append(row)
            if rank == 0 and (
                global_step % int(FLAGS.log_every_steps) == 0 or should_eval
            ):
                elapsed = time.time() - t_train_start
                steps_this_run = max(1, global_step - resume_step)
                eta_min = (
                    (elapsed / steps_this_run) * (total_steps - global_step) / 60.0
                )
                print(
                    f"[Step {global_step:05d}/{total_steps:05d} | ep={row['epoch']:.2f}] "
                    f"loss={step_loss:.4f} kl={step_kl:.4f} (ema={ema_loss:.4f}) grad={grad_norm:.3f} "
                    f"lr={lr_now:.2e} tok={int(global_sup_tokens)} "
                    f"vram={peak_gb:.1f}GiB dt={dt_step:.1f}s ETA={eta_min:.1f}m"
                    + (
                        f" | val_loss={row['val_loss']:.4f} wic_loss={row['val_wic_alnum_loss']:.4f}"
                        if "val_loss" in row
                        else ""
                    ),
                    flush=True,
                )

            should_save = (
                is_milestone
                or (int(FLAGS.save_every_steps) > 0 and global_step % int(FLAGS.save_every_steps) == 0)
                or global_step == total_steps
            )
            if should_save:
                m_tag = milestones.get(global_step, "")
                ckpt_dir = save_dsv41_resumable_checkpoint(
                    output_dir=FLAGS.dsv41_output_dir,
                    step=global_step,
                    epoch=epoch,
                    step_in_epoch=step_in_epoch,
                    model=model,
                    optimizer=optimizer,
                    ema_loss=ema_loss,
                    best_val_loss=best_val_loss,
                    metrics_history=metrics_history,
                    rank=rank,
                    world_size=world_size,
                    milestone_tag=m_tag,
                )
                if is_milestone:
                    _export_milestone(m_tag, global_step, row, ckpt_dir)

    if is_distributed and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def _serialize_dsv41_flags_for_children() -> list[str]:
    flag_names = [
        "dsv41_model_dir",
        "dsv41_data_dir",
        "dsv41_output_dir",
        "routed_expert_lora_r",
        "routed_expert_lora_alpha",
        "trainable_routed_experts_dense",
        "train_moe_only",
        "enable_qat",
        "qat_trainable_expert_scales",
        "qat_lock_gptq_codes",
        "qat_kl_weight",
        "qat_kl_temperature",
        "load_compact_engram",
        "milestone_epoch_fractions",
        "use_token_loss_weights",
        "learning_rate",
        "min_lr_ratio",
        "weight_decay",
        "adam_beta1",
        "adam_beta2",
        "warmup_ratio",
        "max_grad_norm",
        "num_epochs",
        "max_steps",
        "micro_batch_size",
        "grad_accum_steps",
        "max_seq_len",
        "logits_chunk_size",
        "eval_every_steps",
        "max_eval_samples",
        "save_every_steps",
        "log_every_steps",
        "seed",
        "grad_ckpt_stride",
        "use_multi_gpu",
        "length_bucketed_batching",
        "bucket_mega_batch_multiplier",
        "auto_resume",
        "resume_from_checkpoint_dir",
        "resume_lr_rewarmup_steps",
        "init_from_checkpoint_dir",
        # Round-2 flags (must reach every worker subprocess).
        "lora_learning_rate",
        "scale_learning_rate",
        "train_vision",
        "vision_trainable_blocks",
        "vision_learning_rate",
        "aligner_learning_rate",
        "disable_vision",
        "allow_non_qat_export",
        "engram_model_dir",
        "global_grad_clip",
        "pilot_longest_batches_first",
        "num_nodes",
        "node_rank",
    ]
    args: list[str] = [
        "--undefok=cfs_d_cid_enable,cfs_d_resource_user_override,cfs_default_traffic_class,cfs_max_write_bytes_inflight",
        "--cfs_d_cid_enable=true",
        "--cfs_d_resource_user_override=deepmind-resources-colossus",
        "--cfs_default_traffic_class=AF1",
        "--cfs_max_write_bytes_inflight=33554432",
    ]
    for name in flag_names:
        if name not in FLAGS:
            continue
        val = FLAGS[name].value
        if isinstance(val, bool):
            args.append(f"--{name}={'true' if val else 'false'}")
        else:
            args.append(f"--{name}={val}")
    return args


def _resolve_local_ipv6_or_host() -> str:
    try:
        for res in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET6):
            cand = str(res[4][0]).split("%")[0]
            if cand and cand != "::1" and not cand.startswith("fe80"):
                return cand
    except Exception:  # pylint: disable=broad-exception-caught
        pass
    return socket.getfqdn()


def _pick_free_port() -> int:
    with socket.socket(socket.AF_INET6, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return int(s.getsockname()[1])


def main(argv: list[str]) -> None:
    if len(argv) > 1:
        raise app.UsageError(f"Unknown command-line arguments: {argv[1:]}")

    env_rank = os.environ.get("RANK")
    env_world_size = os.environ.get("WORLD_SIZE")
    if env_rank is not None and env_world_size is not None:
        _worker_main(rank=int(env_rank), world_size=int(env_world_size))
        return

    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
    num_nodes = max(1, int(FLAGS.num_nodes))
    node_rank = int(FLAGS.node_rank)
    if FLAGS.use_multi_gpu and (num_gpus > 1 or num_nodes > 1):
        exe_path = sys.argv[0]
        child_args = _serialize_dsv41_flags_for_children()
        total_world_size = num_nodes * num_gpus
        hb_stop = threading.Event()
        hb_thread: threading.Thread | None = None

        if num_nodes > 1:
            rendezvous_path = os.path.join(
                FLAGS.dsv41_output_dir, "dist_rendezvous.json"
            )
            if node_rank == 0:
                export_and_quantize.ensure_cns_quota_dir(FLAGS.dsv41_output_dir)
                master_addr = _resolve_local_ipv6_or_host()
                master_port = str(_pick_free_port())

                def _write_rendezvous() -> None:
                    tmp_path = f"{rendezvous_path}.tmp.{os.getpid()}"
                    payload = {
                        "master_addr": master_addr,
                        "master_port": int(master_port),
                        "num_nodes": num_nodes,
                        "timestamp": time.time(),
                    }
                    with gfile.Open(tmp_path, "w") as f:
                        json.dump(payload, f)
                    gfile.Rename(tmp_path, rendezvous_path, overwrite=True)

                _write_rendezvous()

                def _heartbeat_loop() -> None:
                    while not hb_stop.wait(10.0):
                        try:
                            _write_rendezvous()
                        except Exception:  # pylint: disable=broad-exception-caught
                            pass

                hb_thread = threading.Thread(target=_heartbeat_loop, daemon=True)
                hb_thread.start()
            else:
                print(
                    f"[Node {node_rank}/{num_nodes}] Waiting for fresh master rendezvous at {rendezvous_path}...",
                    flush=True,
                )
                master_addr = ""
                master_port = ""
                while True:
                    try:
                        if gfile.Exists(rendezvous_path):
                            with gfile.Open(rendezvous_path, "r") as f:
                                info = json.load(f)
                            age = time.time() - float(info.get("timestamp", 0.0))
                            if age < 35.0 and info.get("master_addr") and info.get("master_port"):
                                master_addr = str(info["master_addr"])
                                master_port = str(info["master_port"])
                                break
                    except Exception:  # pylint: disable=broad-exception-caught
                        pass
                    time.sleep(2.0)

            host_fmt = f"[{master_addr}]" if ":" in master_addr else master_addr
            init_method = f"tcp://{host_fmt}:{master_port}"
            store_file = ""
            print(
                f"[Node {node_rank}/{num_nodes}] Launching {num_gpus} local workers "
                f"(global ranks {node_rank * num_gpus}..{node_rank * num_gpus + num_gpus - 1} of {total_world_size}, "
                f"init_method={init_method})...",
                flush=True,
            )
        else:
            master_addr = "::1"
            master_port = os.environ.get("MASTER_PORT", "29509")
            store_file = f"/tmp/dsv41_sft_c10d_filestore_{os.getpid()}_{int(time.time())}"
            if os.path.exists(store_file):
                os.remove(store_file)
            init_method = f"file://{store_file}"
            print(
                f"Launching {num_gpus} DeepSeek-V4.1 E128 SFT worker subprocesses via {exe_path} "
                f"(init_method={init_method})...",
                flush=True,
            )

        procs: list[subprocess.Popen[bytes]] = []
        for r in range(num_gpus):
            global_r = node_rank * num_gpus + r
            child_env = os.environ.copy()
            child_env["RANK"] = str(global_r)
            child_env["LOCAL_RANK"] = str(r)
            child_env["LOCAL_WORLD_SIZE"] = str(num_gpus)
            child_env["WORLD_SIZE"] = str(total_world_size)
            child_env["MASTER_ADDR"] = master_addr
            child_env["MASTER_PORT"] = master_port
            child_env["NCCL_SOCKET_FAMILY"] = "AF_INET6"
            child_env["TORCH_DIST_INIT_METHOD"] = init_method
            if store_file:
                child_env["TORCH_DIST_INIT_FILE"] = store_file
            procs.append(subprocess.Popen([exe_path, *child_args], env=child_env))

        exit_codes: list[int | None] = [None] * len(procs)
        failed_rank: int | None = None
        while any(p.poll() is None for p in procs):
            for r, p in enumerate(procs):
                rc = p.poll()
                if rc is not None:
                    exit_codes[r] = rc
                    if rc != 0:
                        failed_rank = r
                        break
            if failed_rank is not None:
                print(
                    f"Worker local_rank {failed_rank} (global rank {node_rank * num_gpus + failed_rank}) "
                    f"exited with non-zero code {exit_codes[failed_rank]}; terminating remaining workers.",
                    flush=True,
                )
                for other in procs:
                    if other.poll() is None:
                        other.terminate()
                break
            time.sleep(0.5)

        hb_stop.set()
        for r, p in enumerate(procs):
            exit_codes[r] = p.wait()

        if any(rc != 0 for rc in exit_codes):
            raise RuntimeError(
                f"DeepSeek-V4.1 SFT worker subprocesses failed with exit codes: {exit_codes}"
            )
    else:
        _worker_main(rank=0, world_size=1)


if __name__ == "__main__":
    g3_multiprocessing.handle_main(main)

