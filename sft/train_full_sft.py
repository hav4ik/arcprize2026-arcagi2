"""B200 Full-Parameter FSDP2 SFT Trainer for Qwen3.8-27B on ARC-AGI-2 Verified Multi-Turn Trajectories.

Key features:
  1. Starts from original BF16 base Qwen3.8-27B checkpoint; excludes visual tower; trains 100% of text-model
     parameters (embed_tokens, all 64 full/linear attention & MLP decoder layers, all RMSNorm / Qwen3_5RMSNormGated
     weights, DeltaNet A_log / dt_bias / conv1d / projections, and untied lm_head).
  2. Uses PyTorch FSDP2 (`fully_shard`) across 8 B200 GPUs with `MixedPrecisionPolicy(param_dtype=torch.bfloat16,
     reduce_dtype=torch.float32)`: forward/backward compute runs in BF16 while sharded master weights, reduced
     gradients, and sharded AdamW optimizer states (`exp_avg`, `exp_avg_sq`) are kept in FP32 (`1/8` per GPU,
     never replicating full Adam states across GPUs).
  3. Validates custom DeltaNet backward across every newly trainable parameter (`in_proj_b`, `in_proj_a`,
     `A_log`, `dt_bias`, `conv1d`, `norm`, `in_proj_qkv`, `in_proj_z`, `out_proj`) before training and
     automatically falls back to the verified PyTorch recurrence if needed.
  4. Normalizes loss and gradients across all ranks by the exact global count of supervised assistant tokens
     in the batch and handles the final partial batch without dropping trajectories or deadlocking collectives.
  5. Preemption-safe resumable distributed checkpoints every 10 steps (retaining latest 2 completed + milestones
     every 50 steps + final checkpoint), saving model shards, optimizer shards, scheduler, step, RNG, and
     sampler position, with an atomic `COMPLETED.json` marker published only after all shards finish.
"""

from __future__ import annotations

from concurrent import futures
import gc
import io
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
from typing import Any

from absl import app
from absl import flags
import numpy as np
import torch
import torch.distributed as dist
from torch.distributed.fsdp import fully_shard
from torch.distributed.fsdp import MixedPrecisionPolicy
from torch.distributed.tensor import DTensor
from torch.distributed.tensor import init_device_mesh
import torch.nn as nn
import torch.nn.functional as F
import transformers

from google3.experimental.users.arazdai.kaggle.sft import export_and_quantize
from google3.experimental.users.arazdai.kaggle.sft import fast_deltanet
from google3.experimental.users.arazdai.kaggle.sft import train_sft
from google3.pyglib import gfile
from google3.pyglib.contrib.g3_multiprocessing import g3_multiprocessing

FLAGS = flags.FLAGS

# Reuse shared flags from train_sft and set Full-SFT defaults
FLAGS.set_default(
    "output_dir",
    "/cns/it-d/home/arazdai/kaggle/sft/checkpoints_full_sft/e031_qwen38_b200_full_sft",
)
FLAGS.set_default("learning_rate", 5e-6)
FLAGS.set_default("min_lr_ratio", 0.1)
FLAGS.set_default("weight_decay", 0.01)
FLAGS.set_default("warmup_ratio", 0.05)
FLAGS.set_default("max_grad_norm", 1.0)
FLAGS.set_default("num_epochs", 1)
FLAGS.set_default("max_steps", -1)
FLAGS.set_default("grad_accum_steps", 1)
FLAGS.set_default("max_seq_len", 32768)
FLAGS.set_default("logits_chunk_size", 2048)
FLAGS.set_default("eval_every_steps", 25)
FLAGS.set_default("max_eval_samples", 64)
FLAGS.set_default("save_every_steps", 25)
FLAGS.set_default("log_every_steps", 1)
FLAGS.set_default("seed", 20260405)
FLAGS.set_default("gradient_checkpointing", True)
FLAGS.set_default("grad_ckpt_stride", 1)
FLAGS.set_default("use_fast_deltanet", True)
FLAGS.set_default("use_torch_compile_norms", True)
FLAGS.set_default("attn_implementation", "flash_attention_2")

flags.DEFINE_float("adam_beta1", 0.9, "AdamW beta1.")
flags.DEFINE_float("adam_beta2", 0.95, "AdamW beta2.")
flags.DEFINE_integer("micro_batch_size", 1, "Per-rank microbatch size in trajectories.")
flags.DEFINE_bool(
    "save_optimizer_state",
    False,
    "Whether to save sharded FP32 AdamW optimizer states in checkpoints (False = save model weights only, 5x smaller).",
)
flags.DEFINE_string(
    "save_weights_dtype",
    "bfloat16",
    "Dtype for saved model weights in checkpoints ('bfloat16' = 53.8 GB total, 'float32' = 107.6 GB total).",
)
flags.DEFINE_integer(
    "keep_last_n_checkpoints",
    1,
    "Number of latest completed non-milestone checkpoints to retain (1 = only keep latest).",
)
flags.DEFINE_integer(
    "milestone_every_steps",
    0,
    "Retain milestone checkpoints every N optimizer steps permanently (0 = disabled, only keep latest).",
)
flags.DEFINE_integer(
    "export_fp8_every_steps",
    50,
    "Collaboratively export Block-128 FP8 checkpoint every N optimizer steps (and final) for full ARC-AGI-2 eval.",
)
flags.DEFINE_integer("num_ckpt_parts_per_rank", 8, "Number of shard files written per rank.")
flags.DEFINE_bool(
    "use_multi_gpu",
    True,
    "Automatically spawn 1 FSDP2 worker process per visible GPU when WORLD_SIZE is not set.",
)
flags.DEFINE_bool(
    "smoke_test_mode",
    False,
    "Run pre-launch smoke test verification (DeltaNet grad check, param update check, save/reload/resume test, partial batch test).",
)
# ---- Round-2 flags ----
flags.DEFINE_bool(
    "length_bucketed_batching",
    False,
    "Group similar-length trajectories into each optimizer step and balance tokens across ranks (kills the straggler wait; "
    "same per-step token-mean loss, no numerics change).",
)
flags.DEFINE_integer(
    "bucket_mega_batch_multiplier",
    8,
    "Length-bucketing sorts within mega-batches of `effective_batch * multiplier` random trajectories, then shuffles steps.",
)
flags.DEFINE_bool(
    "no_decay_norm_bias_deltanet",
    False,
    "Exclude 1-D params, biases, norms, and DeltaNet A_log / dt_bias / conv1d from AdamW weight decay.",
)
flags.DEFINE_float(
    "save_every_epoch_fraction",
    0.0,
    "If > 0, overrides --save_every_steps with round(steps_per_epoch * fraction).",
)
flags.DEFINE_float(
    "export_fp8_every_epoch_fraction",
    0.0,
    "If > 0, overrides --export_fp8_every_steps with round(steps_per_epoch * fraction).",
)
flags.DEFINE_float(
    "milestone_every_epoch_fraction",
    0.0,
    "If > 0, overrides --milestone_every_steps with round(steps_per_epoch * fraction).",
)


def compute_chunked_causal_lm_loss_functional(
    hidden_states: torch.Tensor,
    lm_head_weight: torch.Tensor,
    lm_head_bias: torch.Tensor | None,
    labels: torch.Tensor,
    chunk_size: int = 4096,
) -> tuple[torch.Tensor, int]:
    """Computes exact next-token cross-entropy loss in sequence chunks using already-unsharded lm_head_weight.

    Always touches both `hidden_states` and `lm_head_weight` in the returned `total_loss_sum` tensor
    even when `num_valid == 0` (e.g. on a padded dummy rank in a final partial batch), ensuring
    all FSDP2 backward hooks fire in lockstep across all ranks with exact 0.0 gradient on dummy ranks.
    """
    shift_hidden = hidden_states[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    seq_len = shift_hidden.shape[1]

    valid_mask = shift_labels != -100
    num_valid = int(valid_mask.sum().item())
    if num_valid == 0:
        # Keep autograd graph connected through both hidden_states and lm_head_weight with exact 0.0 weight
        dummy_proj = F.linear(shift_hidden[:, :1, :], lm_head_weight, lm_head_bias)
        return dummy_proj.float().sum() * 0.0, 0

    vocab_size = lm_head_weight.shape[0]
    total_loss_sum = torch.zeros((), device=hidden_states.device, dtype=torch.float32)
    for start in range(0, seq_len, chunk_size):
        end = min(start + chunk_size, seq_len)
        chunk_labels = shift_labels[:, start:end].reshape(-1)
        if not torch.any(chunk_labels != -100):
            continue
        chunk_h = shift_hidden[:, start:end, :].to(lm_head_weight.dtype)
        chunk_logits = F.linear(chunk_h, lm_head_weight, lm_head_bias).float().reshape(-1, vocab_size)
        chunk_loss = F.cross_entropy(
            chunk_logits,
            chunk_labels,
            ignore_index=-100,
            reduction="sum",
        )
        total_loss_sum = total_loss_sum + chunk_loss

    return total_loss_sum, num_valid


class FullSFTCausalLM(nn.Module):
    """Wraps Qwen3.8-27B text language_model and lm_head into a single root FSDP2 module.

    Because `lm_head` is owned directly by the root FSDP2 group (`FullSFTCausalLM`) rather than
    sharded as a standalone child module, `self.lm_head.weight` is all-gathered ONCE per forward
    and reduce-scattered ONCE per backward, avoiding per-chunk collectives across variable-length sequences.
    """

    def __init__(self, qwen_model: nn.Module):
        super().__init__()
        if hasattr(qwen_model, "model") and hasattr(qwen_model.model, "visual"):
            del qwen_model.model.visual
        self.language_model = qwen_model.model.language_model
        self.lm_head = qwen_model.lm_head

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        chunk_size: int = 4096,
    ) -> tuple[torch.Tensor, int]:
        outputs = self.language_model(input_ids=input_ids, use_cache=False)
        hidden_states = outputs.last_hidden_state
        return compute_chunked_causal_lm_loss_functional(
            hidden_states=hidden_states,
            lm_head_weight=self.lm_head.weight,
            lm_head_bias=self.lm_head.bias,
            labels=labels,
            chunk_size=chunk_size,
        )


def apply_fsdp2_full_sharding(
    model: FullSFTCausalLM,
    device: torch.device,
    world_size: int,
    param_dtype: torch.dtype = torch.bfloat16,
    reduce_dtype: torch.dtype = torch.float32,
) -> FullSFTCausalLM:
    """Applies FSDP2 `fully_shard` bottom-up, keeping sharded master params in FP32 and compute in BF16."""
    for p in model.parameters():
        p.requires_grad_(True)

    if world_size <= 1 and not dist.is_initialized():
        # Single-process non-distributed mode: keep master weights in FP32
        model.to(device=device, dtype=torch.float32)
        return model

    mesh_device_type = device.type if device.type == "cuda" else "cpu"
    dp_mesh = init_device_mesh(mesh_device_type, (world_size,))
    mp_policy = MixedPrecisionPolicy(
        param_dtype=param_dtype,
        reduce_dtype=reduce_dtype,
    )

    # Shard each of the 64 decoder layers one by one so peak GPU memory during init stays < 18 GB
    for layer in model.language_model.layers:
        layer.to(device=device, dtype=torch.float32)
        fully_shard(
            layer,
            mesh=dp_mesh,
            mp_policy=mp_policy,
            reshard_after_forward=True,
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Move remaining root parameters (embed_tokens, final norm, lm_head) to device in FP32 and shard root
    model.language_model.embed_tokens.to(device=device, dtype=torch.float32)
    model.language_model.norm.to(device=device, dtype=torch.float32)
    model.language_model.rotary_emb.to(device=device)
    model.lm_head.to(device=device, dtype=torch.float32)

    fully_shard(
        model,
        mesh=dp_mesh,
        mp_policy=mp_policy,
        reshard_after_forward=True,
    )
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return model


def _to_local_tensor(t: torch.Tensor) -> torch.Tensor:
    if isinstance(t, DTensor):
        return t.to_local()
    return t


@torch.no_grad()
def clip_fsdp2_grad_norm_(
    parameters: list[nn.Parameter],
    max_norm: float,
    device: torch.device,
) -> float:
    """Computes exact global L2 gradient norm across FSDP2 `Shard(0)` parameters with a single scalar all-reduce."""
    local_grads = [
        _to_local_tensor(p.grad)
        for p in parameters
        if p.grad is not None and _to_local_tensor(p.grad).numel() > 0
    ]
    if not local_grads:
        local_sq_norm = torch.zeros((), device=device, dtype=torch.float64)
    else:
        norms = [torch.linalg.vector_norm(g.detach().float(), 2.0) for g in local_grads]
        local_sq_norm = torch.sum(torch.stack([n.to(torch.float64) ** 2 for n in norms]))

    if dist.is_initialized() and dist.get_world_size() > 1:
        dist.all_reduce(local_sq_norm, op=dist.ReduceOp.SUM)

    total_norm = float(torch.sqrt(local_sq_norm).item())
    if max_norm > 0.0 and math.isfinite(total_norm) and total_norm > max_norm:
        scale = float(max_norm / (total_norm + 1e-6))
        for g in local_grads:
            g.mul_(scale)
    return total_norm


def build_epoch_batches(
    order: np.ndarray,
    effective_batch: int,
    world_size: int,
    micro_batch_size: int,
    grad_accum_steps: int,
    lengths: np.ndarray | None = None,
    bucket_rng: np.random.Generator | None = None,
    mega_batch_multiplier: int = 8,
) -> list[list[int | None]]:
    """Partitions `order` into steps of size `effective_batch`, padding the final partial batch with `None`.

    When `lengths` is given, trajectories are length-bucketed: each mega-batch of
    `effective_batch * mega_batch_multiplier` random trajectories is sorted by length and cut into
    steps of similar length; the step order is then shuffled with `bucket_rng` (deterministic given
    `order` + rng state, so checkpoint resume rebuilds the identical schedule). Within a step, the
    trajectories are assigned to ranks greedily (longest first, to the rank with the fewest tokens)
    so all ranks finish their `grad_accum_steps` micro-batches at about the same time. Slot layout
    is preserved: rank r consumes `batch_slots[r::world_size]`.
    """
    assert effective_batch == world_size * micro_batch_size * grad_accum_steps
    batches: list[list[int | None]] = []
    n = len(order)
    if lengths is None:
        for start in range(0, n, effective_batch):
            chunk = [int(x) for x in order[start : start + effective_batch]]
            if len(chunk) < effective_batch:
                chunk.extend([None] * (effective_batch - len(chunk)))
            batches.append(chunk)
        return batches

    slots_per_rank = micro_batch_size * grad_accum_steps
    mega = effective_batch * max(1, mega_batch_multiplier)
    full_steps: list[list[int]] = []
    partial_step: list[int] | None = None
    for start in range(0, n, mega):
        chunk = [int(x) for x in order[start : start + mega]]
        chunk.sort(key=lambda i: (-int(lengths[i]), i))
        for j in range(0, len(chunk), effective_batch):
            group = chunk[j : j + effective_batch]
            if len(group) == effective_batch:
                full_steps.append(group)
            else:
                assert partial_step is None, "Only the final step may be partial"
                partial_step = group
    if bucket_rng is not None and len(full_steps) > 1:
        perm = bucket_rng.permutation(len(full_steps))
        full_steps = [full_steps[i] for i in perm]
    if partial_step is not None:
        full_steps.append(partial_step)

    for group in full_steps:
        # Greedy longest-processing-time assignment across ranks, respecting slots_per_rank
        group_sorted = sorted(group, key=lambda i: (-int(lengths[i]), i))
        rank_items: list[list[int]] = [[] for _ in range(world_size)]
        rank_tokens = [0] * world_size
        for idx in group_sorted:
            candidates = [r for r in range(world_size) if len(rank_items[r]) < slots_per_rank]
            r_best = min(candidates, key=lambda r: (rank_tokens[r], r))
            rank_items[r_best].append(idx)
            rank_tokens[r_best] += int(lengths[idx])
        batch_slots: list[int | None] = [None] * effective_batch
        for r in range(world_size):
            for j, idx in enumerate(rank_items[r]):
                batch_slots[r + world_size * j] = idx
        batches.append(batch_slots)
    return batches


def compute_step_global_supervised_tokens(
    batch_slots: list[int | None],
    train_dataset: train_sft.PackedTrajectoryDataset,
) -> int:
    """Computes exact total supervised assistant tokens across all active trajectories in the global batch."""
    total_valid = 0
    for idx in batch_slots:
        if idx is not None:
            sample = train_dataset[int(idx)]
            # Next-token shift: labels[:, 1:] != -100
            n_valid = int((sample["labels"][1:] != -100).sum().item())
            total_valid += n_valid
    return total_valid


def _write_bytes_to_gfile(dst_path: str, payload: bytes) -> int:
    with gfile.Open(dst_path, "wb") as f_out:
        view = memoryview(payload)
        chunk_bytes = 64 * 1024 * 1024
        for offset in range(0, len(payload), chunk_bytes):
            f_out.write(view[offset : offset + chunk_bytes])
    return len(payload)


def _serialize_and_write_gfile(dst_path: str, obj: Any) -> int:
    buf = io.BytesIO()
    torch.save(obj, buf, _use_new_zipfile_serialization=False)
    payload = buf.getvalue()
    del buf
    return _write_bytes_to_gfile(dst_path, payload)


def _read_bytes_from_gfile(src_path: str) -> bytes:
    chunks: list[bytes] = []
    with gfile.Open(src_path, "rb") as f_in:
        while True:
            buf = f_in.read(64 * 1024 * 1024)
            if not buf:
                break
            chunks.append(buf)
    return b"".join(chunks)


def save_distributed_checkpoint(
    model: FullSFTCausalLM,
    optimizer: torch.optim.Optimizer,
    ckpt_root_dir: str,
    ckpt_name: str,
    rank: int,
    world_size: int,
    step: int,
    epoch: int,
    batch_idx_in_epoch: int,
    order: np.ndarray,
    scheduler_meta: dict[str, Any],
    rng: np.random.Generator,
    extra_meta: dict[str, Any] | None = None,
    num_parts: int = 8,
    keep_last_n: int = 1,
    milestone_every_steps: int = 0,
    save_optimizer_state: bool = False,
    save_weights_dtype: str = "bfloat16",
) -> str:
    """Saves a resumable distributed checkpoint across all ranks and writes COMPLETED.json only after all shards finish."""
    ckpt_dir = os.path.join(ckpt_root_dir, ckpt_name)
    if rank == 0:
        export_and_quantize.ensure_cns_quota_dir(ckpt_root_dir)
        if keep_last_n == 1 and milestone_every_steps <= 0:
            _prune_old_checkpoints(
                ckpt_root_dir=ckpt_root_dir,
                keep_last_n=0,
                milestone_every_steps=0,
            )
        export_and_quantize.ensure_cns_quota_dir(ckpt_dir)
    if world_size > 1 and dist.is_initialized():
        dist.barrier()

    t0 = time.time()
    named_params = list(model.named_parameters())
    num_parts = max(1, min(num_parts, len(named_params)))
    per_part = math.ceil(len(named_params) / float(num_parts))
    target_dtype = (
        torch.bfloat16 if save_weights_dtype.lower() in ("bfloat16", "bf16") else torch.float32
    )

    # Partition parameters into contiguous layer-ordered blocks across `num_parts`
    part_objects: list[tuple[str, Any]] = []
    part_hf_keys: list[list[str]] = []
    for part_idx in range(num_parts):
        lo = part_idx * per_part
        hi = min(len(named_params), (part_idx + 1) * per_part)
        part_params = named_params[lo:hi]
        part_hf_keys.append(
            [export_and_quantize.full_sft_key_to_hf_key(name) for name, _ in part_params]
        )
        model_part: dict[str, torch.Tensor] = {}
        opt_part: dict[str, dict[str, torch.Tensor]] = {}
        for name, p in part_params:
            model_part[name] = _to_local_tensor(p).detach().to(dtype=target_dtype).cpu()
            if save_optimizer_state and p in optimizer.state:
                st = optimizer.state[p]
                opt_entry: dict[str, torch.Tensor] = {}
                for k, v in st.items():
                    if isinstance(v, torch.Tensor):
                        opt_entry[k] = _to_local_tensor(v).detach().cpu()
                opt_part[name] = opt_entry

        fname = f"rank_{rank:02d}_part_{part_idx:02d}.pt"
        part_objects.append((os.path.join(ckpt_dir, fname), {"model": model_part, "optimizer": opt_part}))

    # Save per-rank RNG, scheduler, optimizer param_groups, and sampler state
    cuda_rng = (
        torch.cuda.get_rng_state(rank)
        if torch.cuda.is_available() and next(model.parameters()).is_cuda
        else None
    )
    rank_state = {
        "rank": rank,
        "world_size": world_size,
        "step": step,
        "epoch": epoch,
        "batch_idx_in_epoch": batch_idx_in_epoch,
        "order": order.astype(np.int64).tolist(),
        "scheduler": scheduler_meta,
        "optimizer_param_groups": [
            {k: v for k, v in pg.items() if k != "params"}
            for pg in optimizer.param_groups
        ],
        "rng": {
            "python_random_state": random.getstate(),
            "numpy_rng_bit_generator_state": rng.bit_generator.state,
            "numpy_global_random_state": np.random.get_state(),
            "torch_cpu_rng_state": torch.get_rng_state(),
            "torch_cuda_rng_state": cuda_rng,
        },
    }
    state_fname = f"rank_{rank:02d}_state.pt"
    part_objects.append((os.path.join(ckpt_dir, state_fname), rank_state))

    with futures.ThreadPoolExecutor(max_workers=min(4, len(part_objects))) as pool:
        futs = [
            pool.submit(_serialize_and_write_gfile, dst_path, obj)
            for dst_path, obj in part_objects
        ]
        total_rank_bytes = sum(f.result() for f in futs)
    del part_objects
    gc.collect()

    if world_size > 1 and dist.is_initialized():
        dist.barrier()

    if rank == 0:
        expected_files: list[str] = []
        for r in range(world_size):
            for p_idx in range(num_parts):
                expected_files.append(f"rank_{r:02d}_part_{p_idx:02d}.pt")
            expected_files.append(f"rank_{r:02d}_state.pt")

        for fname in expected_files:
            fpath = os.path.join(ckpt_dir, fname)
            if not gfile.Exists(fpath):
                raise RuntimeError(f"Checkpoint shard missing before completion marker: {fpath}")

        completion_doc = {
            "completed": True,
            "ckpt_name": ckpt_name,
            "step": step,
            "epoch": epoch,
            "batch_idx_in_epoch": batch_idx_in_epoch,
            "world_size": world_size,
            "num_parts": num_parts,
            "part_hf_keys": part_hf_keys,
            "save_optimizer_state": save_optimizer_state,
            "save_weights_dtype": save_weights_dtype,
            "shard_files": expected_files,
            "rank0_bytes": total_rank_bytes,
            "elapsed_sec": round(time.time() - t0, 2),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            **(extra_meta or {}),
        }
        completed_path = os.path.join(ckpt_dir, "COMPLETED.json")
        with gfile.Open(completed_path, "w") as f:
            json.dump(completion_doc, f, indent=2)

        effective_keep_n = (
            0
            if (ckpt_name == "final" and keep_last_n == 1 and milestone_every_steps <= 0)
            else keep_last_n
        )
        _prune_old_checkpoints(
            ckpt_root_dir=ckpt_root_dir,
            keep_last_n=effective_keep_n,
            milestone_every_steps=milestone_every_steps,
        )

    if world_size > 1 and dist.is_initialized():
        dist.barrier()
    return ckpt_dir


def find_latest_completed_checkpoint(
    ckpt_root_dir: str,
    expected_world_size: int,
) -> tuple[str, dict[str, Any]] | None:
    """Scans `ckpt_root_dir` for the highest-step checkpoint with a valid `COMPLETED.json` and all shards present."""
    if not gfile.Exists(ckpt_root_dir):
        return None

    candidates: list[tuple[int, str, dict[str, Any]]] = []
    for entry in sorted(gfile.ListDir(ckpt_root_dir)):
        sub_dir = os.path.join(ckpt_root_dir, entry)
        if not gfile.IsDirectory(sub_dir):
            continue
        marker_path = os.path.join(sub_dir, "COMPLETED.json")
        if not gfile.Exists(marker_path):
            print(f"Ignoring incomplete checkpoint directory (missing COMPLETED.json): {sub_dir}")
            continue
        try:
            with gfile.Open(marker_path, "r") as f:
                meta = json.load(f)
            if not meta.get("completed", False):
                continue
            if int(meta.get("world_size", -1)) != expected_world_size:
                print(
                    f"Skipping checkpoint {sub_dir}: world_size={meta.get('world_size')} != {expected_world_size}"
                )
                continue
            shard_files = meta.get("shard_files", [])
            existing_set = set(gfile.ListDir(sub_dir))
            if not all(sf in existing_set for sf in shard_files):
                print(f"Ignoring checkpoint {sub_dir}: missing one or more shard files.")
                continue
            step = int(meta.get("step", -1))
            if step > 0:
                candidates.append((step, sub_dir, meta))
        except Exception as e:
            print(f"Warning: could not parse COMPLETED.json in {sub_dir} ({e}); ignoring.")

    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0], reverse=True)
    _, best_dir, best_meta = candidates[0]
    return best_dir, best_meta


def _prune_old_checkpoints(
    ckpt_root_dir: str,
    keep_last_n: int = 1,
    milestone_every_steps: int = 0,
) -> None:
    """Retains the latest `keep_last_n` completed `step-*` checkpoints, milestones (if >0), and `final`."""
    if not gfile.Exists(ckpt_root_dir):
        return

    completed_step_dirs: list[tuple[int, str]] = []
    for entry in sorted(gfile.ListDir(ckpt_root_dir)):
        if not entry.startswith("step-"):
            continue
        sub_dir = os.path.join(ckpt_root_dir, entry)
        marker_path = os.path.join(sub_dir, "COMPLETED.json")
        if not gfile.Exists(marker_path):
            continue
        try:
            with gfile.Open(marker_path, "r") as f:
                meta = json.load(f)
            step = int(meta.get("step", -1))
            if step > 0:
                completed_step_dirs.append((step, sub_dir))
        except Exception:
            continue

    completed_step_dirs.sort(key=lambda x: x[0])
    if keep_last_n > 0 and len(completed_step_dirs) <= keep_last_n:
        return

    protected_dirs = (
        {d for _, d in completed_step_dirs[-keep_last_n:]} if keep_last_n > 0 else set()
    )
    for step, d in completed_step_dirs:
        if milestone_every_steps > 0 and step % milestone_every_steps == 0:
            protected_dirs.add(d)

    for step, d in completed_step_dirs:
        if d not in protected_dirs:
            try:
                gfile.DeleteRecursively(d)
                print(f"Pruned old non-milestone checkpoint: {d} (step {step})")
            except Exception as e:
                print(f"Warning: failed to prune {d}: {e}")


def _read_and_deserialize_gfile(src_path: str) -> Any:
    raw_bytes = _read_bytes_from_gfile(src_path)
    return torch.load(io.BytesIO(raw_bytes), map_location="cpu", weights_only=False)


def load_distributed_checkpoint(
    model: FullSFTCausalLM,
    optimizer: torch.optim.Optimizer,
    ckpt_dir: str,
    meta: dict[str, Any],
    rank: int,
    device: torch.device,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """Restores sharded model weights, sharded FP32 optimizer states, scheduler, RNG, and sampler position."""
    num_parts = int(meta["num_parts"])
    part_paths = [
        os.path.join(ckpt_dir, f"rank_{rank:02d}_part_{p_idx:02d}.pt")
        for p_idx in range(num_parts)
    ]
    state_path = os.path.join(ckpt_dir, f"rank_{rank:02d}_state.pt")

    with futures.ThreadPoolExecutor(max_workers=len(part_paths) + 1) as pool:
        part_futs = [pool.submit(_read_and_deserialize_gfile, p) for p in part_paths]
        state_fut = pool.submit(_read_and_deserialize_gfile, state_path)
        part_data_list = [f.result() for f in part_futs]
        rank_state = state_fut.result()

    param_map = dict(model.named_parameters())
    with torch.no_grad():
        for part_data in part_data_list:
            for name, saved_local in part_data["model"].items():
                if name not in param_map:
                    raise KeyError(f"Parameter {name} missing from model during checkpoint load")
                p = param_map[name]
                # Direct CPU -> device copy with dtype cast; avoids a per-tensor GPU temporary that the
                # caching allocator would keep reserved (this OOM'd the resumed run of xid 304951546).
                _to_local_tensor(p).copy_(saved_local, non_blocking=False)

            for name, saved_opt_entry in part_data["optimizer"].items():
                p = param_map[name]
                restored_st: dict[str, Any] = {}
                for k, v in saved_opt_entry.items():
                    if k == "step":
                        restored_st[k] = v.to(dtype=torch.float32)
                    elif isinstance(p, DTensor):
                        local_v = v.to(device=device, dtype=torch.float32)
                        restored_st[k] = DTensor.from_local(
                            local_v,
                            device_mesh=p.device_mesh,
                            placements=p.placements,
                            run_check=False,
                        )
                    else:
                        restored_st[k] = v.to(device=device, dtype=torch.float32)
                optimizer.state[p] = restored_st
            part_data["model"].clear()
    del part_data_list
    gc.collect()
    if torch.cuda.is_available() and device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()

    for pg, saved_pg in zip(optimizer.param_groups, rank_state.get("optimizer_param_groups", [])):
        for k, v in saved_pg.items():
            if k != "params":
                pg[k] = v

    rng_dict = rank_state["rng"]
    random.setstate(rng_dict["python_random_state"])
    rng.bit_generator.state = rng_dict["numpy_rng_bit_generator_state"]
    np.random.set_state(rng_dict["numpy_global_random_state"])
    torch.set_rng_state(rng_dict["torch_cpu_rng_state"])
    if (
        torch.cuda.is_available()
        and device.type == "cuda"
        and rng_dict.get("torch_cuda_rng_state") is not None
    ):
        torch.cuda.set_rng_state(rng_dict["torch_cuda_rng_state"], device=device)

    return {
        "step": int(rank_state["step"]),
        "epoch": int(rank_state["epoch"]),
        "batch_idx_in_epoch": int(rank_state["batch_idx_in_epoch"]),
        "order": np.asarray(rank_state["order"], dtype=np.int64),
        "scheduler": rank_state["scheduler"],
    }


@torch.no_grad()
def evaluate_validation_loss_fsdp2(
    model: FullSFTCausalLM,
    val_dataset: train_sft.PackedTrajectoryDataset,
    device: torch.device,
    rank: int = 0,
    world_size: int = 1,
    max_samples: int = 64,
    chunk_size: int = 4096,
) -> dict[str, float]:
    """Computes assistant-only token NLL and perplexity on the fixed 64 validation trajectories across all FSDP2 ranks."""
    model.eval()
    n_eval = len(val_dataset) if max_samples <= 0 else min(len(val_dataset), max_samples)
    # Pad total slots to a multiple of world_size so every FSDP2 rank runs the exact same number of forward passes
    num_iters = math.ceil(n_eval / float(world_size))

    t0 = time.time()
    local_nll = 0.0
    local_tokens = 0
    for step_idx in range(num_iters):
        sample_idx = step_idx * world_size + rank
        if sample_idx < n_eval:
            sample = val_dataset[sample_idx]
            input_ids = sample["input_ids"].unsqueeze(0).to(device)
            labels = sample["labels"].unsqueeze(0).to(device)
            is_real = True
        else:
            # Dummy padded slot so FSDP2 all-gathers stay synchronized across all ranks
            dummy_ids = val_dataset[0]["input_ids"][:64].unsqueeze(0).to(device)
            dummy_labels = torch.full_like(dummy_ids, -100)
            input_ids = dummy_ids
            labels = dummy_labels
            is_real = False

        loss_sum, n_valid = model(
            input_ids=input_ids,
            labels=labels,
            chunk_size=chunk_size,
        )
        if is_real:
            local_nll += float(loss_sum.item())
            local_tokens += n_valid

    if world_size > 1 and dist.is_initialized():
        stats_tensor = torch.tensor(
            [local_nll, float(local_tokens)], dtype=torch.float64, device=device
        )
        dist.all_reduce(stats_tensor, op=dist.ReduceOp.SUM)
        total_nll = float(stats_tensor[0].item())
        total_tokens = int(round(float(stats_tensor[1].item())))
    else:
        total_nll = local_nll
        total_tokens = local_tokens

    mean_nll = total_nll / max(1, total_tokens)
    ppl = math.exp(min(mean_nll, 20.0))
    model.train()
    return {
        "val_loss": round(mean_nll, 5),
        "val_perplexity": round(ppl, 4),
        "val_supervised_tokens": total_tokens,
        "val_samples": n_eval,
        "val_elapsed_sec": round(time.time() - t0, 2),
    }


def categorize_parameter_groups(model: nn.Module) -> dict[str, list[tuple[str, nn.Parameter]]]:
    """Groups model parameters by architectural component to verify all intended groups update."""
    groups: dict[str, list[tuple[str, nn.Parameter]]] = {
        "embed_tokens": [],
        "lm_head": [],
        "deltanet_newly_trainable": [],  # A_log, dt_bias, conv1d, in_proj_a, in_proj_b
        "deltanet_projections": [],  # in_proj_qkv, in_proj_z, out_proj
        "full_attn": [],
        "mlp": [],
        "norms": [],
    }
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "embed_tokens" in name:
            groups["embed_tokens"].append((name, p))
        elif "lm_head" in name:
            groups["lm_head"].append((name, p))
        elif "norm" in name:
            groups["norms"].append((name, p))
        elif any(k in name for k in ("A_log", "dt_bias", "conv1d", "in_proj_a", "in_proj_b")):
            groups["deltanet_newly_trainable"].append((name, p))
        elif any(k in name for k in ("in_proj_qkv", "in_proj_z", "linear_attn.out_proj")):
            groups["deltanet_projections"].append((name, p))
        elif "self_attn" in name:
            groups["full_attn"].append((name, p))
        elif "mlp" in name:
            groups["mlp"].append((name, p))
        else:
            groups["deltanet_projections"].append((name, p))
    return groups


@torch.no_grad()
def snapshot_group_local_clones(
    groups: dict[str, list[tuple[str, nn.Parameter]]],
) -> dict[str, list[torch.Tensor]]:
    """Takes a CPU FP32 snapshot of the first 2 parameters in each architectural group to verify updates."""
    snap: dict[str, list[torch.Tensor]] = {}
    for gname, items in groups.items():
        snap[gname] = [
            _to_local_tensor(p).detach().float().cpu().clone() for _, p in items[:2]
        ]
    return snap


@torch.no_grad()
def compute_group_update_norms(
    groups: dict[str, list[tuple[str, nn.Parameter]]],
    initial_snap: dict[str, list[torch.Tensor]],
    device: torch.device,
) -> dict[str, float]:
    """Computes global L2 norm of parameter updates `(w_current - w_init)` for each architectural group."""
    sq_diffs: list[float] = []
    keys = list(groups.keys())
    for gname in keys:
        s = 0.0
        for (_, p), init_t in zip(groups[gname][:2], initial_snap[gname]):
            cur_t = _to_local_tensor(p).detach().float().cpu()
            diff = cur_t - init_t
            s += float((diff * diff).sum().item())
        sq_diffs.append(s)

    t_sq = torch.tensor(sq_diffs, dtype=torch.float64, device=device)
    if dist.is_initialized() and dist.get_world_size() > 1:
        dist.all_reduce(t_sq, op=dist.ReduceOp.SUM)

    return {
        gname: float(math.sqrt(max(0.0, float(t_sq[idx].item()))))
        for idx, gname in enumerate(keys)
    }


def _train_worker(rank: int, world_size: int) -> None:
    """Per-rank FSDP2 training worker."""
    is_main = rank == 0
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")
        backend = "nccl"
    else:
        device = torch.device("cpu")
        backend = "gloo"

    if world_size > 1 and not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "::1")
        os.environ.setdefault("MASTER_PORT", "29509")
        os.environ.setdefault("NCCL_SOCKET_FAMILY", "AF_INET6")
        os.environ["TRITON_CACHE_DIR"] = f"/tmp/triton_cache_full_sft_rank_{rank}"
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = f"/tmp/inductor_cache_full_sft_rank_{rank}"
        store_path = os.environ.get(
            "TORCH_DIST_INIT_FILE", "/tmp/full_sft_c10d_filestore"
        )
        dist.init_process_group(
            backend=backend,
            init_method=f"file://{store_path}",
            rank=rank,
            world_size=world_size,
            device_id=device if device.type == "cuda" else None,
        )

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)
    torch.manual_seed(FLAGS.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(FLAGS.seed)

    if is_main:
        export_and_quantize.ensure_cns_quota_dir(FLAGS.output_dir)

    metrics_log_path = os.path.join(FLAGS.output_dir, "metrics.jsonl")

    def log_event(event: dict[str, Any]) -> None:
        if not is_main:
            return
        event["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        line = json.dumps(event)
        print(line, flush=True)
        with gfile.Open(metrics_log_path, "a") as f:
            f.write(line + "\n")

    # 1. Validate custom DeltaNet backward across all 9 parameters (including newly trainable params)
    if FLAGS.use_fast_deltanet:
        if is_main:
            print("Validating custom DeltaNet backward across all 9 Qwen3_5GatedDeltaNet parameters...")
        deltanet_report = fast_deltanet.validate_deltanet_backward_all_params(
            device=device,
            dtype=torch.bfloat16,
        )
        all_deltanet_ok = all(bool(v.get("passed", False)) for v in deltanet_report.values())
        if is_main:
            log_event(
                {
                    "phase": "deltanet_backward_validation",
                    "all_passed": all_deltanet_ok,
                    "using_triton_recurrence": fast_deltanet._USE_TRITON_RECURRENCE,  # pylint: disable=protected-access
                    "params": deltanet_report,
                }
            )
        fast_deltanet.patch_qwen35_fast_kernels()

    if is_main:
        print(f"Starting Qwen3.8-27B Full-Parameter FSDP2 SFT: world_size={world_size}, device={device}")
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(rank)
            vram_gb = torch.cuda.get_device_properties(rank).total_memory / (1024**3)
            print(f"Detected GPU [{rank}/{world_size}]: {gpu_name} ({vram_gb:.1f} GB VRAM)")

    # 2. Load prepared train and validation datasets (exact same split & masks as LoRA)
    train_npz = os.path.join(FLAGS.data_dir, "train_tokens.npz")
    val_npz = os.path.join(FLAGS.data_dir, "val_tokens.npz")
    train_dataset = train_sft.PackedTrajectoryDataset(train_npz, max_seq_len=FLAGS.max_seq_len)
    val_dataset = train_sft.PackedTrajectoryDataset(val_npz, max_seq_len=FLAGS.max_seq_len)
    if is_main:
        print(
            f"Train samples: {len(train_dataset)} (skipped overlong>{FLAGS.max_seq_len}: {train_dataset.skipped_overlong})"
        )
        print(
            f"Val samples:   {len(val_dataset)} (skipped overlong>{FLAGS.max_seq_len}: {val_dataset.skipped_overlong})"
        )

    # 3. Collaboratively stage original BF16 base model from CNS into local /tmp RAM disk (if on /cns),
    # load on CPU, remove visual tower, shard across 8 GPUs with FSDP2, and immediately free /tmp!
    t_load = time.time()
    local_model_path = FLAGS.model_path
    staged_tmp_dir: str | None = None
    if FLAGS.model_path.startswith("/cns/"):
        staged_tmp_dir = "/tmp/qwen38_bf16_staged"
        if is_main:
            if os.path.exists(staged_tmp_dir):
                shutil.rmtree(staged_tmp_dir, ignore_errors=True)
            os.makedirs(staged_tmp_dir, exist_ok=True)
        if world_size > 1 and dist.is_initialized():
            dist.barrier()

        all_files = sorted(
            f
            for f in gfile.ListDir(FLAGS.model_path)
            if not gfile.IsDirectory(os.path.join(FLAGS.model_path, f))
        )
        my_files = all_files[rank::world_size]
        t_stage = time.time()
        if is_main:
            print(
                f"Collaboratively staging {len(all_files)} files from {FLAGS.model_path} -> {staged_tmp_dir} "
                f"across {world_size} ranks ({len(my_files)} files on rank 0)...",
                flush=True,
            )
        for idx_f, fname in enumerate(my_files, 1):
            t_f = time.time()
            src_f = os.path.join(FLAGS.model_path, fname)
            dst_f = os.path.join(staged_tmp_dir, fname)
            export_and_quantize.copy_file_gfile(src_f, dst_f)
            if fname.endswith(".safetensors"):
                print(
                    f"  [stage rank {rank}/{world_size}] [{idx_f}/{len(my_files)}] staged {fname} in {time.time() - t_f:.1f}s",
                    flush=True,
                )
        if world_size > 1 and dist.is_initialized():
            dist.barrier()
        if is_main:
            print(
                f"Collaboratively staged {FLAGS.model_path} to {staged_tmp_dir} in {time.time() - t_stage:.1f}s.",
                flush=True,
            )
        local_model_path = staged_tmp_dir

    config = transformers.AutoConfig.from_pretrained(local_model_path, trust_remote_code=True)
    config.text_config.use_cache = False

    attn_impl = FLAGS.attn_implementation
    try:
        qwen_base = transformers.Qwen3_5ForConditionalGeneration.from_pretrained(
            local_model_path,
            config=config,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            attn_implementation=attn_impl,
        )
    except Exception as e:
        if is_main:
            print(f"attn_implementation={attn_impl} unavailable ({e}), falling back to 'sdpa'.")
        attn_impl = "sdpa"
        qwen_base = transformers.Qwen3_5ForConditionalGeneration.from_pretrained(
            local_model_path,
            config=config,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            attn_implementation=attn_impl,
        )

    has_visual_before = hasattr(qwen_base, "model") and hasattr(qwen_base.model, "visual")
    if FLAGS.gradient_checkpointing:
        qwen_base.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        qwen_base.enable_input_require_grads()
        if FLAGS.grad_ckpt_stride > 1:
            layers = qwen_base.model.language_model.layers
            for idx, layer in enumerate(layers):
                layer.gradient_checkpointing = (idx % FLAGS.grad_ckpt_stride == 0)

    full_model = FullSFTCausalLM(qwen_base)
    del qwen_base
    has_visual_after = any("visual" in name for name, _ in full_model.named_parameters())
    assert not has_visual_after, "Visual tower parameters must be excluded from FullSFTCausalLM!"

    if FLAGS.use_fast_deltanet:
        n_patched = fast_deltanet.patch_qwen35_fast_kernels(full_model)
        if is_main:
            print(f"Patched {n_patched} Qwen3_5GatedDeltaNet layers with fast Triton kernel.")

    if FLAGS.use_torch_compile_norms and torch.cuda.is_available():
        os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "1"
        torch._inductor.config.compile_threads = 1  # pylint: disable=protected-access
        from transformers.models.qwen3_5 import modeling_qwen3_5

        modeling_qwen3_5.Qwen3_5RMSNorm.forward = torch.compile(
            modeling_qwen3_5.Qwen3_5RMSNorm.forward, dynamic=True
        )
        modeling_qwen3_5.Qwen3_5RMSNormGated.forward = torch.compile(
            modeling_qwen3_5.Qwen3_5RMSNormGated.forward, dynamic=True
        )

    total_logical_params = sum(p.numel() for p in full_model.parameters())
    full_model = apply_fsdp2_full_sharding(
        model=full_model,
        device=device,
        world_size=world_size,
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
    )
    if staged_tmp_dir is not None:
        if world_size > 1 and dist.is_initialized():
            dist.barrier()
        if is_main:
            shutil.rmtree(staged_tmp_dir, ignore_errors=True)
            print(f"Freed temporary RAM disk staging directory {staged_tmp_dir}.", flush=True)
        if world_size > 1 and dist.is_initialized():
            dist.barrier()
    gc.collect()

    local_shard_params = sum(_to_local_tensor(p).numel() for p in full_model.parameters())
    frozen_params = sum(p.numel() for p in full_model.parameters() if not p.requires_grad)
    assert frozen_params == 0, f"Expected 0 frozen text parameters, found {frozen_params}"

    param_groups_by_arch = categorize_parameter_groups(full_model)
    initial_arch_snap = snapshot_group_local_clones(param_groups_by_arch)

    if is_main:
        init_vram_gb = (
            torch.cuda.memory_allocated(device) / (1024**3) if torch.cuda.is_available() else 0.0
        )
        print(
            f"Loaded & FSDP2-sharded model in {time.time() - t_load:.1f}s "
            f"(visual_removed={has_visual_before and not has_visual_after}, "
            f"total_params={total_logical_params:,}, local_shard_params={local_shard_params:,}, "
            f"init_vram_gb={init_vram_gb:.2f} GB)"
        )
        log_event(
            {
                "phase": "model_init",
                "total_text_params": total_logical_params,
                "local_shard_params_per_rank": local_shard_params,
                "frozen_text_params": frozen_params,
                "visual_excluded": not has_visual_after,
                "param_groups_counts": {k: len(v) for k, v in param_groups_by_arch.items()},
                "init_vram_gb": round(init_vram_gb, 2),
                "attn_implementation": attn_impl,
            }
        )

    trainable_params = [p for p in full_model.parameters() if p.requires_grad]
    if FLAGS.no_decay_norm_bias_deltanet:
        no_decay_markers = ("A_log", "dt_bias", "conv1d", "norm", "bias")
        decay_params: list[nn.Parameter] = []
        no_decay_params: list[nn.Parameter] = []
        for name, p in full_model.named_parameters():
            if not p.requires_grad:
                continue
            if p.ndim < 2 or any(m in name for m in no_decay_markers):
                no_decay_params.append(p)
            else:
                decay_params.append(p)
        optimizer_param_groups: list[dict[str, Any]] = [
            {"params": decay_params, "weight_decay": FLAGS.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]
        if is_main:
            print(
                f"AdamW param groups: decay={len(decay_params)} tensors "
                f"({sum(p.numel() for p in decay_params):,} params), "
                f"no_decay={len(no_decay_params)} tensors ({sum(p.numel() for p in no_decay_params):,} params)"
            )
    else:
        optimizer_param_groups = [{"params": trainable_params, "weight_decay": FLAGS.weight_decay}]
    optimizer = torch.optim.AdamW(
        optimizer_param_groups,
        lr=FLAGS.learning_rate,
        betas=(FLAGS.adam_beta1, FLAGS.adam_beta2),
        weight_decay=FLAGS.weight_decay,
        foreach=False,
    )

    effective_batch = world_size * FLAGS.micro_batch_size * FLAGS.grad_accum_steps
    # Include the final partial batch via math.ceil
    steps_per_epoch = math.ceil(len(train_dataset) / float(effective_batch))
    total_steps = (
        FLAGS.max_steps
        if FLAGS.max_steps > 0
        else steps_per_epoch * FLAGS.num_epochs
    )
    # Schedule LR over full 1-epoch horizon (192 steps) even during short smoke tests unless overridden
    lr_horizon_steps = steps_per_epoch * FLAGS.num_epochs
    warmup_steps = int(round(lr_horizon_steps * FLAGS.warmup_ratio))

    # Checkpoint / FP8-export / milestone cadence, optionally expressed as a fraction of an epoch
    def _cadence(flag_steps: int, frac: float) -> int:
        if frac > 0:
            return max(1, int(round(steps_per_epoch * frac)))
        return int(flag_steps)

    save_every_steps = _cadence(FLAGS.save_every_steps, FLAGS.save_every_epoch_fraction)

    def _multiple_of_save(flag_steps: int, frac: float) -> int:
        # Exports / milestones only happen at save steps, so snap them to integer multiples of the save cadence.
        if frac > 0 and FLAGS.save_every_epoch_fraction > 0:
            return save_every_steps * max(1, int(round(frac / FLAGS.save_every_epoch_fraction)))
        return _cadence(flag_steps, frac)

    export_fp8_every_steps = _multiple_of_save(FLAGS.export_fp8_every_steps, FLAGS.export_fp8_every_epoch_fraction)
    milestone_every_steps = _multiple_of_save(FLAGS.milestone_every_steps, FLAGS.milestone_every_epoch_fraction)
    train_lengths = np.asarray(
        [int(train_dataset.lengths[i]) for i in train_dataset.valid_indices], dtype=np.int64
    )

    if is_main:
        print(
            f"Training plan: {len(train_dataset)} samples/epoch, world_size={world_size}, "
            f"micro_batch_size={FLAGS.micro_batch_size}, grad_accum_steps={FLAGS.grad_accum_steps}, "
            f"effective_batch={effective_batch}, steps_per_epoch={steps_per_epoch} (includes final partial batch), "
            f"total_steps={total_steps}, lr_horizon_steps={lr_horizon_steps}, warmup_steps={warmup_steps}, "
            f"save_optimizer_state={FLAGS.save_optimizer_state}, save_weights_dtype={FLAGS.save_weights_dtype}, "
            f"keep_last_n_checkpoints={FLAGS.keep_last_n_checkpoints}, milestone_every_steps={milestone_every_steps}, "
            f"save_every_steps={save_every_steps}, export_fp8_every_steps={export_fp8_every_steps}, "
            f"length_bucketed_batching={FLAGS.length_bucketed_batching}"
        )
        log_event(
            {
                "phase": "training_plan",
                "train_samples": len(train_dataset),
                "effective_batch": effective_batch,
                "steps_per_epoch": steps_per_epoch,
                "total_steps": total_steps,
                "warmup_steps": warmup_steps,
                "save_every_steps": save_every_steps,
                "export_fp8_every_steps": export_fp8_every_steps,
                "milestone_every_steps": milestone_every_steps,
                "length_bucketed_batching": FLAGS.length_bucketed_batching,
                "no_decay_norm_bias_deltanet": FLAGS.no_decay_norm_bias_deltanet,
                "train_tokens_per_epoch": int(train_lengths.sum()),
                "train_seq_len_mean": float(train_lengths.mean()) if len(train_lengths) else 0.0,
            }
        )

    ckpt_root_dir = os.path.join(FLAGS.output_dir, "checkpoints")
    rng = np.random.default_rng(FLAGS.seed)

    # 4. Check for existing completed distributed checkpoint to resume from
    resumed_info = find_latest_completed_checkpoint(ckpt_root_dir, expected_world_size=world_size)
    if resumed_info is not None:
        resume_dir, resume_meta = resumed_info
        if is_main:
            print(f"Resuming from latest completed checkpoint: {resume_dir} (step {resume_meta['step']})...")
        restored = load_distributed_checkpoint(
            model=full_model,
            optimizer=optimizer,
            ckpt_dir=resume_dir,
            meta=resume_meta,
            rank=rank,
            device=device,
            rng=rng,
        )
        global_step = int(restored["step"])
        start_epoch = int(restored["epoch"])
        resumed_batch_idx = int(restored["batch_idx_in_epoch"])
        resumed_order = restored["order"]
        if is_main:
            log_event(
                {
                    "phase": "resume",
                    "resumed_from": resume_dir,
                    "step": global_step,
                    "epoch": start_epoch,
                    "batch_idx_in_epoch": resumed_batch_idx,
                }
            )
    else:
        global_step = 0
        start_epoch = 0
        resumed_batch_idx = 0
        resumed_order = None

    if global_step == 0:
        if is_main:
            print("Evaluating baseline (step 0) validation loss on fixed 64 validation trajectories...")
        val0 = evaluate_validation_loss_fsdp2(
            model=full_model,
            val_dataset=val_dataset,
            device=device,
            rank=rank,
            world_size=world_size,
            max_samples=FLAGS.max_eval_samples,
            chunk_size=FLAGS.logits_chunk_size,
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        log_event({"step": 0, "phase": "eval", **val0})

    # HF models are constructed in eval mode; on a fresh start the step-0 validation above happens to flip the
    # model to train mode, but on resume it is skipped — leaving gradient checkpointing inactive (it requires
    # `self.training`) and OOMing the first 32k-token forward (xid 304951546 restart, xid 305093497).
    full_model.train()
    optimizer.zero_grad(set_to_none=True)
    step_times: list[float] = []

    for epoch in range(start_epoch, FLAGS.num_epochs if FLAGS.max_steps <= 0 else 999):
        if epoch == start_epoch and resumed_order is not None:
            order = resumed_order
            first_batch_idx = resumed_batch_idx
            resumed_order = None
        else:
            order = rng.permutation(len(train_dataset))
            first_batch_idx = 0

        epoch_batches = build_epoch_batches(
            order=order,
            effective_batch=effective_batch,
            world_size=world_size,
            micro_batch_size=FLAGS.micro_batch_size,
            grad_accum_steps=FLAGS.grad_accum_steps,
            lengths=train_lengths if FLAGS.length_bucketed_batching else None,
            bucket_rng=(
                np.random.default_rng(int(FLAGS.seed) * 1_000_003 + int(epoch))
                if FLAGS.length_bucketed_batching
                else None
            ),
            mega_batch_multiplier=FLAGS.bucket_mega_batch_multiplier,
        )

        # In smoke_test_mode, also force the last step to exercise the final partial batch!
        if FLAGS.smoke_test_mode and FLAGS.max_steps > 0 and len(epoch_batches) > FLAGS.max_steps:
            epoch_batches[FLAGS.max_steps - 1] = epoch_batches[-1]

        for batch_idx in range(first_batch_idx, len(epoch_batches)):
            if global_step >= total_steps:
                break

            step_t0 = time.time()
            batch_slots = epoch_batches[batch_idx]
            active_trajectories = sum(1 for x in batch_slots if x is not None)
            is_partial_batch = active_trajectories < effective_batch

            # Exact global supervised assistant tokens across all ranks & microbatches in this step
            global_supervised_tokens = compute_step_global_supervised_tokens(
                batch_slots=batch_slots,
                train_dataset=train_dataset,
            )
            rank_slots = batch_slots[rank::world_size]

            local_loss_sum = 0.0
            local_valid_tokens = 0

            for sample_idx in rank_slots:
                if sample_idx is not None:
                    sample = train_dataset[int(sample_idx)]
                    input_ids = sample["input_ids"].unsqueeze(0).to(device)
                    labels = sample["labels"].unsqueeze(0).to(device)
                    is_real_sample = True
                else:
                    # Padded slot in final partial batch: run short masked dummy sequence
                    dummy_ids = train_dataset[0]["input_ids"][:64].unsqueeze(0).to(device)
                    dummy_labels = torch.full_like(dummy_ids, -100)
                    input_ids = dummy_ids
                    labels = dummy_labels
                    is_real_sample = False

                seq_len_cur = int(input_ids.shape[1])
                loss_sum, n_valid = full_model(
                    input_ids=input_ids,
                    labels=labels,
                    chunk_size=FLAGS.logits_chunk_size,
                )

                if is_real_sample and n_valid > 0 and global_supervised_tokens > 0:
                    # FSDP2 averages gradients by 1 / world_size during reduce-scatter, so scaling
                    # local loss by world_size / global_supervised_tokens yields exact global token mean!
                    scaled_loss = loss_sum * (float(world_size) / float(global_supervised_tokens))
                    local_loss_sum += float(loss_sum.item())
                    local_valid_tokens += n_valid
                else:
                    scaled_loss = loss_sum * 0.0

                scaled_loss.backward()
                del input_ids, labels, loss_sum, scaled_loss
                if seq_len_cur >= 24576 and torch.cuda.is_available():
                    torch.cuda.empty_cache()

            if world_size > 1 and dist.is_initialized():
                metrics_tensor = torch.tensor(
                    [local_loss_sum, float(local_valid_tokens)],
                    dtype=torch.float64,
                    device=device,
                )
                dist.all_reduce(metrics_tensor, op=dist.ReduceOp.SUM)
                step_loss_sum = float(metrics_tensor[0].item())
                step_tokens = int(round(float(metrics_tensor[1].item())))
            else:
                step_loss_sum = local_loss_sum
                step_tokens = local_valid_tokens

            assert step_tokens == global_supervised_tokens, (
                f"Token normalization mismatch: reduced {step_tokens} != expected {global_supervised_tokens}"
            )

            lr = train_sft.get_cosine_lr(
                step=global_step,
                total_steps=lr_horizon_steps,
                peak_lr=FLAGS.learning_rate,
                warmup_steps=warmup_steps,
                min_lr_ratio=FLAGS.min_lr_ratio,
            )
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            grad_norm = clip_fsdp2_grad_norm_(
                parameters=trainable_params,
                max_norm=FLAGS.max_grad_norm,
                device=device,
            )
            if not math.isfinite(grad_norm) or not math.isfinite(step_loss_sum):
                raise RuntimeError(
                    f"Non-finite loss or gradient at step {global_step + 1}: "
                    f"loss_sum={step_loss_sum}, grad_norm={grad_norm}"
                )

            optimizer.step()

            # Verify on step 1 that optimizer states are FP32 DTensors of sharded size (not replicated)
            if global_step == 0:
                sample_p = trainable_params[0]
                st0 = optimizer.state[sample_p]
                exp_avg_local = _to_local_tensor(st0["exp_avg"])
                exp_avg_sq_local = _to_local_tensor(st0["exp_avg_sq"])
                assert exp_avg_local.dtype == torch.float32, (
                    f"Expected FP32 exp_avg, got {exp_avg_local.dtype}"
                )
                assert exp_avg_sq_local.dtype == torch.float32, (
                    f"Expected FP32 exp_avg_sq, got {exp_avg_sq_local.dtype}"
                )
                assert exp_avg_local.numel() == _to_local_tensor(sample_p).numel(), (
                    "Optimizer state numel does not match local parameter shard!"
                )

            optimizer.zero_grad(set_to_none=True)
            global_step += 1

            step_loss = step_loss_sum / max(1, step_tokens)
            step_elapsed = time.time() - step_t0
            step_times.append(step_elapsed)

            if torch.cuda.is_available():
                alloc_gb = torch.cuda.memory_allocated(device) / (1024**3)
                max_mem_gb = torch.cuda.max_memory_allocated(device) / (1024**3)
                reserved_gb = torch.cuda.max_memory_reserved(device) / (1024**3)
                total_vram_gb = torch.cuda.get_device_properties(device).total_memory / (1024**3)
                headroom_gb = total_vram_gb - max_mem_gb
            else:
                alloc_gb = max_mem_gb = reserved_gb = total_vram_gb = headroom_gb = 0.0

            if global_step % FLAGS.log_every_steps == 0:
                warm_times = step_times[1:] if len(step_times) > 1 else step_times
                avg_step_sec = sum(warm_times) / len(warm_times)
                eta_min = (max(0, lr_horizon_steps - global_step) * avg_step_sec) / 60.0
                log_event(
                    {
                        "step": global_step,
                        "total_steps": total_steps,
                        "lr_horizon_steps": lr_horizon_steps,
                        "epoch": epoch,
                        "phase": "train",
                        "train_loss": round(step_loss, 5),
                        "train_ppl": round(math.exp(min(step_loss, 20.0)), 4),
                        "lr": round(lr, 9),
                        "grad_norm": round(grad_norm, 4),
                        "supervised_tokens": step_tokens,
                        "active_trajectories": active_trajectories,
                        "is_partial_batch": is_partial_batch,
                        "step_sec": round(step_elapsed, 2),
                        "warm_avg_step_sec": round(avg_step_sec, 2),
                        "estimated_full_epoch_min": round((lr_horizon_steps * avg_step_sec) / 60.0, 1),
                        "eta_remaining_min": round(eta_min, 1),
                        "tok_per_sec": round(step_tokens / max(0.01, step_elapsed), 1),
                        "alloc_vram_gb": round(alloc_gb, 2),
                        "max_vram_gb": round(max_mem_gb, 2),
                        "reserved_vram_gb": round(reserved_gb, 2),
                        "vram_headroom_gb": round(headroom_gb, 2),
                        "world_size": world_size,
                    }
                )

            if FLAGS.eval_every_steps > 0 and global_step % FLAGS.eval_every_steps == 0:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                val_metrics = evaluate_validation_loss_fsdp2(
                    model=full_model,
                    val_dataset=val_dataset,
                    device=device,
                    rank=rank,
                    world_size=world_size,
                    max_samples=FLAGS.max_eval_samples,
                    chunk_size=FLAGS.logits_chunk_size,
                )
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                log_event({"step": global_step, "phase": "eval", **val_metrics})

            is_last_step = global_step >= total_steps
            if save_every_steps > 0 and global_step % save_every_steps == 0 and not is_last_step:
                ckpt_name = f"step-{global_step:05d}"
                if is_main:
                    print(
                        f"Saving distributed checkpoint {ckpt_name} at step {global_step} "
                        f"(save_optimizer_state={FLAGS.save_optimizer_state}, dtype={FLAGS.save_weights_dtype})..."
                    )
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                saved_dir = save_distributed_checkpoint(
                    model=full_model,
                    optimizer=optimizer,
                    ckpt_root_dir=ckpt_root_dir,
                    ckpt_name=ckpt_name,
                    rank=rank,
                    world_size=world_size,
                    step=global_step,
                    epoch=epoch,
                    batch_idx_in_epoch=batch_idx + 1,
                    order=order,
                    scheduler_meta={
                        "step": global_step,
                        "lr_horizon_steps": lr_horizon_steps,
                        "warmup_steps": warmup_steps,
                        "peak_lr": FLAGS.learning_rate,
                        "min_lr_ratio": FLAGS.min_lr_ratio,
                        "current_lr": lr,
                    },
                    rng=rng,
                    extra_meta={"train_loss": round(step_loss, 5), "grad_norm": round(grad_norm, 4)},
                    num_parts=FLAGS.num_ckpt_parts_per_rank,
                    keep_last_n=FLAGS.keep_last_n_checkpoints,
                    milestone_every_steps=milestone_every_steps,
                    save_optimizer_state=FLAGS.save_optimizer_state,
                    save_weights_dtype=FLAGS.save_weights_dtype,
                )
                log_event({"step": global_step, "phase": "checkpoint_saved", "ckpt_dir": saved_dir})

                if (
                    export_fp8_every_steps > 0
                    and global_step % export_fp8_every_steps == 0
                ):
                    fp8_root = os.path.join(FLAGS.output_dir, "fp8_checkpoints")
                    fp8_dir = os.path.join(fp8_root, ckpt_name)
                    if is_main:
                        export_and_quantize.ensure_cns_quota_dir(fp8_root)
                        export_and_quantize.ensure_cns_quota_dir(fp8_dir)
                    if world_size > 1 and dist.is_initialized():
                        dist.barrier()
                    export_and_quantize.ensure_sharded_fp8_export(
                        output_fp8_path=fp8_dir,
                        shard_index=rank,
                        num_shards=world_size,
                        full_sft_ckpt_dir=saved_dir,
                    )
                    if world_size > 1 and dist.is_initialized():
                        dist.barrier()
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    log_event(
                        {
                            "step": global_step,
                            "phase": "fp8_checkpoint_exported",
                            "fp8_dir": fp8_dir,
                        }
                    )

                # During smoke_test_mode, test save -> incomplete checkpoint ignore -> reload -> resume!
                if FLAGS.smoke_test_mode and global_step == FLAGS.save_every_steps:
                    if is_main:
                        incomplete_dir = os.path.join(ckpt_root_dir, f"step-{global_step + 1:05d}")
                        gfile.MakeDirs(incomplete_dir)
                        with gfile.Open(os.path.join(incomplete_dir, "partial_shard.tmp"), "w") as f:
                            f.write("incomplete")
                    if world_size > 1 and dist.is_initialized():
                        dist.barrier()

                    found = find_latest_completed_checkpoint(ckpt_root_dir, expected_world_size=world_size)
                    assert found is not None, "Failed to locate completed checkpoint during smoke test!"
                    found_dir, found_meta = found
                    assert int(found_meta["step"]) == global_step, (
                        f"Expected resume step {global_step} (ignoring incomplete {global_step + 1}), "
                        f"got {found_meta['step']}"
                    )
                    pre_reload_snap = snapshot_group_local_clones(param_groups_by_arch)
                    restored_smoke = load_distributed_checkpoint(
                        model=full_model,
                        optimizer=optimizer,
                        ckpt_dir=found_dir,
                        meta=found_meta,
                        rank=rank,
                        device=device,
                        rng=rng,
                    )
                    post_reload_diff = compute_group_update_norms(
                        param_groups_by_arch, pre_reload_snap, device
                    )
                    max_reload_diff = max(post_reload_diff.values())
                    tol = 0.05 if FLAGS.save_weights_dtype.lower() in ("bfloat16", "bf16") else 0.0
                    assert max_reload_diff <= tol, (
                        f"Reloaded weights differed by {max_reload_diff} > {tol}!"
                    )
                    if is_main:
                        gfile.DeleteRecursively(os.path.join(ckpt_root_dir, f"step-{global_step + 1:05d}"))
                        log_event(
                            {
                                "step": global_step,
                                "phase": "smoke_test_save_reload_verified",
                                "reloaded_from": found_dir,
                                "ignored_incomplete_step": global_step + 1,
                                "restored_batch_idx": restored_smoke["batch_idx_in_epoch"],
                                "max_reload_weight_diff": max_reload_diff,
                            }
                        )
                    if world_size > 1 and dist.is_initialized():
                        dist.barrier()

        if global_step >= total_steps:
            break

    # Verify that all intended parameter groups updated during training
    update_norms = compute_group_update_norms(param_groups_by_arch, initial_arch_snap, device)
    all_groups_updated = all(val > 0.0 for val in update_norms.values())
    if is_main:
        log_event(
            {
                "step": global_step,
                "phase": "parameter_update_verification",
                "all_groups_updated": all_groups_updated,
                "group_update_l2_norms": {k: round(v, 8) for k, v in update_norms.items()},
            }
        )
    assert all_groups_updated, f"Some parameter groups did not update: {update_norms}"

    if is_main:
        print("Running final validation evaluation on fixed 64 validation trajectories...")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    final_val = evaluate_validation_loss_fsdp2(
        model=full_model,
        val_dataset=val_dataset,
        device=device,
        rank=rank,
        world_size=world_size,
        max_samples=FLAGS.max_eval_samples,
        chunk_size=FLAGS.logits_chunk_size,
    )
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    log_event({"step": global_step, "phase": "final_eval", **final_val})

    if is_main:
        print(
            f"Saving final distributed checkpoint to {ckpt_root_dir}/final "
            f"(save_optimizer_state={FLAGS.save_optimizer_state}, dtype={FLAGS.save_weights_dtype})..."
        )
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    final_ckpt_dir = save_distributed_checkpoint(
        model=full_model,
        optimizer=optimizer,
        ckpt_root_dir=ckpt_root_dir,
        ckpt_name="final",
        rank=rank,
        world_size=world_size,
        step=global_step,
        epoch=FLAGS.num_epochs,
        batch_idx_in_epoch=0,
        order=np.arange(len(train_dataset), dtype=np.int64),
        scheduler_meta={
            "step": global_step,
            "lr_horizon_steps": lr_horizon_steps,
            "warmup_steps": warmup_steps,
            "peak_lr": FLAGS.learning_rate,
            "min_lr_ratio": FLAGS.min_lr_ratio,
        },
        rng=rng,
        extra_meta=final_val,
        num_parts=FLAGS.num_ckpt_parts_per_rank,
        keep_last_n=FLAGS.keep_last_n_checkpoints,
        milestone_every_steps=milestone_every_steps,
        save_optimizer_state=FLAGS.save_optimizer_state,
        save_weights_dtype=FLAGS.save_weights_dtype,
    )
    log_event({"step": global_step, "phase": "final_checkpoint_saved", "ckpt_dir": final_ckpt_dir})

    if export_fp8_every_steps > 0:
        fp8_root = os.path.join(FLAGS.output_dir, "fp8_checkpoints")
        fp8_final_dir = os.path.join(fp8_root, "final")
        if is_main:
            export_and_quantize.ensure_cns_quota_dir(fp8_root)
            export_and_quantize.ensure_cns_quota_dir(fp8_final_dir)
        if world_size > 1 and dist.is_initialized():
            dist.barrier()
        export_and_quantize.ensure_sharded_fp8_export(
            output_fp8_path=fp8_final_dir,
            shard_index=rank,
            num_shards=world_size,
            full_sft_ckpt_dir=final_ckpt_dir,
        )
        if world_size > 1 and dist.is_initialized():
            dist.barrier()
        log_event(
            {
                "step": global_step,
                "phase": "fp8_checkpoint_exported",
                "fp8_dir": fp8_final_dir,
            }
        )

    if is_main:
        print("Full-parameter FSDP2 SFT complete.")

    if world_size > 1 and dist.is_initialized():
        dist.destroy_process_group()


def _serialize_flags_for_children() -> list[str]:
    """Serializes explicitly parsed Full-SFT CLI flags to pass to child rank subprocesses."""
    flag_names = [
        "model_path",
        "data_dir",
        "output_dir",
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
        "save_optimizer_state",
        "save_weights_dtype",
        "keep_last_n_checkpoints",
        "milestone_every_steps",
        "export_fp8_every_steps",
        "num_ckpt_parts_per_rank",
        "log_every_steps",
        "seed",
        "gradient_checkpointing",
        "grad_ckpt_stride",
        "use_fast_deltanet",
        "use_torch_compile_norms",
        "attn_implementation",
        "use_multi_gpu",
        "smoke_test_mode",
        "length_bucketed_batching",
        "bucket_mega_batch_multiplier",
        "no_decay_norm_bias_deltanet",
        "save_every_epoch_fraction",
        "export_fp8_every_epoch_fraction",
        "milestone_every_epoch_fraction",
    ]
    args: list[str] = [
        "--undefok=cfs_d_cid_enable,cfs_d_resource_user_override,cfs_default_traffic_class,cfs_max_write_bytes_inflight",
        "--cfs_d_cid_enable=true",
        "--cfs_d_resource_user_override=deepmind-resources-colossus",
        "--cfs_default_traffic_class=AF1",
        "--cfs_max_write_bytes_inflight=33554432",
    ]
    for name in flag_names:
        val = FLAGS[name].value
        if isinstance(val, bool):
            args.append(f"--{name}={'true' if val else 'false'}")
        else:
            args.append(f"--{name}={val}")
    return args


def main(argv: list[str]) -> None:
    if len(argv) > 1:
        raise app.UsageError(f"Unknown command-line arguments: {argv[1:]}")

    env_rank = os.environ.get("RANK")
    env_world_size = os.environ.get("WORLD_SIZE")
    if env_rank is not None and env_world_size is not None:
        _train_worker(rank=int(env_rank), world_size=int(env_world_size))
        return

    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
    if FLAGS.use_multi_gpu and num_gpus > 1:
        exe_path = sys.argv[0]
        child_args = _serialize_flags_for_children()
        master_port = os.environ.get("MASTER_PORT", "29509")
        store_file = f"/tmp/full_sft_c10d_filestore_{os.getpid()}_{int(time.time())}"
        if os.path.exists(store_file):
            os.remove(store_file)
        print(
            f"Launching {num_gpus} FSDP2 worker subprocesses via {exe_path} "
            f"(init_method=file://{store_file})...",
            flush=True,
        )
        procs: list[subprocess.Popen[bytes]] = []
        for r in range(num_gpus):
            child_env = os.environ.copy()
            child_env["RANK"] = str(r)
            child_env["LOCAL_RANK"] = str(r)
            child_env["WORLD_SIZE"] = str(num_gpus)
            child_env["MASTER_ADDR"] = "::1"
            child_env["MASTER_PORT"] = master_port
            child_env["NCCL_SOCKET_FAMILY"] = "AF_INET6"
            child_env["TORCH_DIST_INIT_FILE"] = store_file
            procs.append(subprocess.Popen([exe_path, *child_args], env=child_env))

        exit_codes: list[int] = []
        for r, p in enumerate(procs):
            rc = p.wait()
            exit_codes.append(rc)
            if rc != 0:
                print(f"Worker rank {r} exited with non-zero code {rc}; terminating remaining workers.")
                for other in procs:
                    if other.poll() is None:
                        other.terminate()
                break

        if any(rc != 0 for rc in exit_codes):
            raise RuntimeError(f"FSDP2 worker subprocesses failed with exit codes: {exit_codes}")
    else:
        _train_worker(rank=0, world_size=1)


if __name__ == "__main__":
    g3_multiprocessing.handle_main(main)
