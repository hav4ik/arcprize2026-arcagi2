"""B200 LoRA SFT Trainer for Qwen3.8-27B on ARC-AGI-2 Verified Multi-Turn Trajectories.

Supports all B200 efficiency optimizations:
  1. Fused Triton forward + backward Gated DeltaNet inter-chunk recurrence + vectorized cuBLAS triangular solve
     (`--use_fast_deltanet=True`), replacing the 63-step + 300-step Python loops across all 48 linear_attn layers.
  2. FlashAttention-2 (`--attn_implementation=flash_attention_2` with fallback to `sdpa`) for the 16 full_attn layers.
  3. Selective Gradient Checkpointing (`--grad_ckpt_stride=2`): checkpoints every K-th decoder layer on 180GB B200.
  4. Multi-GPU DistributedDataParallel (DDP) / multi-B200 data parallelism (`--use_multi_gpu_ddp=True`):
     automatically spawns 1 worker per visible B200 GPU and synchronizes LoRA gradients via NCCL all-reduce.
  5. Sequence-chunked LM head + cross-entropy (`--logits_chunk_size=4096`) + TF32 matmul precision.
"""

from __future__ import annotations

import io
import json
import math
import os
import shutil
import sys
import tempfile
import threading
import time
from typing import Any

from absl import app
from absl import flags
import numpy as np
import peft
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.utils.data import Dataset
import transformers

from google3.experimental.users.arazdai.kaggle.sft import fast_deltanet
from google3.pyglib import gfile

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "model_path",
    "/cns/it-d/home/ads-video-creation-auto-plx/gaoxiao/models/Qwen3.8-27B",
    "Path to base Qwen3.8-27B BF16 checkpoint on CNS.",
)
flags.DEFINE_string(
    "data_dir",
    "/cns/lu-d/home/arazdai/kaggle/sft/data/e031_pilot",
    "CNS directory containing train_tokens.npz and val_tokens.npz.",
)
flags.DEFINE_string(
    "output_dir",
    "/cns/lu-d/home/arazdai/kaggle/sft/checkpoints/e031_qwen38_b200_pilot",
    "CNS directory to write LoRA checkpoints and training metrics.",
)
flags.DEFINE_integer("lora_r", 64, "LoRA rank.")
flags.DEFINE_integer("lora_alpha", 128, "LoRA alpha scaling.")
flags.DEFINE_float("lora_dropout", 0.0, "LoRA dropout.")
flags.DEFINE_string(
    "lora_target_modules",
    "q_proj,k_proj,v_proj,o_proj,in_proj_qkv,in_proj_z,out_proj,gate_proj,up_proj,down_proj",
    "Comma-separated linear layer names in Qwen3.8-27B language_model to attach LoRA adapters.",
)
flags.DEFINE_float("learning_rate", 1e-4, "Peak learning rate for AdamW.")
flags.DEFINE_float("min_lr_ratio", 0.1, "Minimum learning rate ratio at end of cosine schedule.")
flags.DEFINE_float("weight_decay", 0.01, "Weight decay for AdamW.")
flags.DEFINE_float("warmup_ratio", 0.05, "Fraction of total steps used for linear warmup.")
flags.DEFINE_float("max_grad_norm", 1.0, "Gradient clipping max L2 norm.")
flags.DEFINE_integer("num_epochs", 1, "Number of training epochs.")
flags.DEFINE_integer("max_steps", -1, "If > 0, overrides num_epochs and stops after max_steps optimizer steps.")
flags.DEFINE_integer("grad_accum_steps", 8, "Global number of micro-steps per optimizer step across all GPUs.")
flags.DEFINE_integer(
    "max_seq_len",
    32768,
    "Maximum sequence length per training sample (samples exceeding this are skipped).",
)
flags.DEFINE_integer(
    "logits_chunk_size",
    4096,
    "Chunk size along sequence dimension when computing LM head + cross-entropy.",
)
flags.DEFINE_integer("eval_every_steps", 25, "Run validation loss evaluation every N optimizer steps.")
flags.DEFINE_integer("max_eval_samples", 64, "Maximum number of validation trajectories to evaluate per check (0 = all).")
flags.DEFINE_integer("save_every_steps", 10, "Save LoRA adapter checkpoint every N optimizer steps.")
flags.DEFINE_integer("log_every_steps", 1, "Log training metrics every N optimizer steps.")
flags.DEFINE_integer("seed", 20260405, "Random seed.")
flags.DEFINE_bool("gradient_checkpointing", True, "Enable activation gradient checkpointing.")
flags.DEFINE_integer(
    "grad_ckpt_stride",
    1,
    "Selective gradient checkpointing stride (1 = checkpoint all 64 layers; 2 = checkpoint every 2nd layer to save 50% recompute time on 180GB B200).",
)
flags.DEFINE_bool(
    "use_fast_deltanet",
    True,
    "Replace pure-PyTorch Gated DeltaNet loops with fused Triton forward/backward kernels + cuBLAS solve_triangular.",
)
flags.DEFINE_bool(
    "use_torch_compile_norms",
    True,
    "Apply torch.compile(dynamic=True) to RMSNorm / Qwen3_5RMSNormGated modules.",
)
flags.DEFINE_string(
    "attn_implementation",
    "flash_attention_2",
    "Attention implementation ('flash_attention_2' with automatic fallback to 'sdpa').",
)
flags.DEFINE_bool(
    "use_multi_gpu_ddp",
    True,
    "Automatically spawn 1 DDP worker per visible GPU when torch.cuda.device_count() > 1.",
)
flags.DEFINE_string(
    "init_adapter_path",
    "",
    "Optional CNS path to an existing LoRA adapter directory (e.g. Epoch 1 final_adapter) to initialize from when starting a new run (resumed_step == 0).",
)
flags.DEFINE_float(
    "loss_weight_gamma",
    0.0,
    "Exponent gamma >= 0 for per-example loss reweighting w_i = clip((loss_i / ema_loss)**gamma, min_w, max_w), normalized within each step so mean weight = 1.0 (0.0 = uniform).",
)
flags.DEFINE_float(
    "loss_weight_min",
    0.25,
    "Minimum unnormalized per-example weight when loss_weight_gamma > 0.",
)
flags.DEFINE_float(
    "loss_weight_max",
    3.0,
    "Maximum unnormalized per-example weight when loss_weight_gamma > 0.",
)


class PackedTrajectoryDataset(Dataset):
    """Loads packed `input_ids`, `loss_mask`, and optional `loss_weights` arrays from CNS `.npz` files."""

    def __init__(self, npz_path: str, max_seq_len: int = 65536):
        with gfile.Open(npz_path, "rb") as f:
            raw_bytes = f.read()
        with np.load(io.BytesIO(raw_bytes)) as data:
            self.flat_ids = data["input_ids"]
            self.flat_mask = data["loss_mask"]
            self.flat_weights = data["loss_weights"] if "loss_weights" in data else None
            self.offsets = data["offsets"]
            self.lengths = data["lengths"]

        self.valid_indices: list[int] = []
        self.skipped_overlong = 0
        for idx, length in enumerate(self.lengths):
            if int(length) <= max_seq_len:
                self.valid_indices.append(idx)
            else:
                self.skipped_overlong += 1

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, i: int) -> dict[str, Any]:
        real_idx = self.valid_indices[i]
        start = int(self.offsets[real_idx])
        end = int(self.offsets[real_idx + 1])
        ids = torch.from_numpy(self.flat_ids[start:end].astype(np.int64))
        mask = torch.from_numpy(self.flat_mask[start:end].astype(np.int64))
        labels = torch.where(mask == 1, ids, torch.full_like(ids, -100))
        item: dict[str, Any] = {
            "input_ids": ids,
            "labels": labels,
            "seq_len": int(end - start),
            "supervised_tokens": int(mask.sum().item()),
        }
        if self.flat_weights is not None:
            item["loss_weights"] = torch.from_numpy(
                self.flat_weights[start:end].astype(np.float32)
            )
        return item


def compute_chunked_causal_lm_loss(
    hidden_states: torch.Tensor,
    lm_head: torch.nn.Module,
    labels: torch.Tensor,
    chunk_size: int = 4096,
    loss_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, int]:
    """Computes exact next-token cross-entropy loss in sequence chunks without materializing full [B, S, V] logits."""
    shift_hidden = hidden_states[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    shift_weights = (
        loss_weights[:, 1:].contiguous() if loss_weights is not None else None
    )
    seq_len = shift_hidden.shape[1]

    valid_mask = shift_labels != -100
    num_valid = int(valid_mask.sum().item())
    if num_valid == 0:
        return hidden_states.sum() * 0.0, 0

    total_loss_sum = torch.zeros((), device=hidden_states.device, dtype=torch.float32)
    out_features = (
        lm_head.out_features
        if hasattr(lm_head, "out_features")
        else lm_head.weight.shape[0]
    )
    for start in range(0, seq_len, chunk_size):
        end = min(start + chunk_size, seq_len)
        chunk_labels = shift_labels[:, start:end].reshape(-1)
        if not torch.any(chunk_labels != -100):
            continue
        chunk_h = shift_hidden[:, start:end, :]
        chunk_logits = lm_head(chunk_h).float().reshape(-1, out_features)
        if shift_weights is not None:
            chunk_w = shift_weights[:, start:end].reshape(-1).float()
            per_tok = F.cross_entropy(
                chunk_logits,
                chunk_labels,
                ignore_index=-100,
                reduction="none",
            )
            valid_w = torch.where(chunk_labels != -100, chunk_w, torch.zeros_like(chunk_w))
            w_mean = valid_w.sum() / torch.clamp((chunk_labels != -100).float().sum(), min=1.0)
            norm_w = valid_w / torch.clamp(w_mean, min=1e-6)
            chunk_loss = (per_tok * norm_w).sum()
        else:
            chunk_loss = F.cross_entropy(
                chunk_logits,
                chunk_labels,
                ignore_index=-100,
                reduction="sum",
            )
        total_loss_sum = total_loss_sum + chunk_loss

    return total_loss_sum, num_valid


def get_cosine_lr(
    step: int,
    total_steps: int,
    peak_lr: float,
    warmup_steps: int,
    min_lr_ratio: float = 0.1,
) -> float:
    """Computes linear warmup followed by cosine decay learning rate."""
    if step < warmup_steps and warmup_steps > 0:
        return peak_lr * float(step + 1) / float(warmup_steps)
    if total_steps <= warmup_steps:
        return peak_lr
    progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    progress = min(max(progress, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    min_lr = peak_lr * min_lr_ratio
    return min_lr + (peak_lr - min_lr) * cosine


def copy_local_dir_to_cns(local_dir: str, cns_dir: str) -> None:
    """Recursively copies files from a local directory to CNS using gfile, writing trainer_state.json last for atomicity."""
    gfile.MakeDirs(cns_dir)
    deferred_state_files: list[tuple[str, str]] = []
    for root, dirs, files in os.walk(local_dir):
        rel = os.path.relpath(root, local_dir)
        target_root = cns_dir if rel == "." else os.path.join(cns_dir, rel)
        gfile.MakeDirs(target_root)
        for d in dirs:
            gfile.MakeDirs(os.path.join(target_root, d))
        for fname in sorted(files):
            src_path = os.path.join(root, fname)
            dst_path = os.path.join(target_root, fname)
            if fname == "trainer_state.json":
                deferred_state_files.append((src_path, dst_path))
                continue
            with open(src_path, "rb") as f_in, gfile.Open(dst_path, "wb") as f_out:
                while True:
                    buf = f_in.read(16 * 1024 * 1024)
                    if not buf:
                        break
                    f_out.write(buf)
    for src_path, dst_path in deferred_state_files:
        with open(src_path, "rb") as f_in, gfile.Open(dst_path, "wb") as f_out:
            f_out.write(f_in.read())


_UPLOAD_THREAD: threading.Thread | None = None


def save_lora_checkpoint(
    model: torch.nn.Module,
    tokenizer: Any,
    cns_ckpt_dir: str,
    step_meta: dict[str, Any],
    optimizer: torch.optim.Optimizer | None = None,
    async_upload: bool = True,
) -> None:
    """Saves PEFT LoRA adapter, optional optimizer state, and training metadata to local scratch then copies atomically to CNS."""
    global _UPLOAD_THREAD
    if _UPLOAD_THREAD is not None and _UPLOAD_THREAD.is_alive():
        _UPLOAD_THREAD.join()

    tmp_dir = tempfile.mkdtemp(prefix="sft_lora_ckpt_")
    model.save_pretrained(tmp_dir)
    if tokenizer is not None:
        tokenizer.save_pretrained(tmp_dir)
    if optimizer is not None:
        torch.save(optimizer.state_dict(), os.path.join(tmp_dir, "optimizer.pt"))
    with open(os.path.join(tmp_dir, "trainer_state.json"), "w") as f:
        json.dump(step_meta, f, indent=2)

    def _upload_and_cleanup() -> None:
        try:
            copy_local_dir_to_cns(tmp_dir, cns_ckpt_dir)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    if async_upload:
        _UPLOAD_THREAD = threading.Thread(target=_upload_and_cleanup, daemon=False)
        _UPLOAD_THREAD.start()
    else:
        _upload_and_cleanup()


def maybe_resume_from_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    output_dir: str,
    device: torch.device,
) -> int:
    """Checks for `checkpoint-latest/trainer_state.json` on CNS and restores LoRA weights + optimizer if present."""
    latest_dir = os.path.join(output_dir, "checkpoint-latest")
    state_path = os.path.join(latest_dir, "trainer_state.json")
    if not gfile.Exists(state_path):
        return 0

    try:
        with gfile.Open(state_path, "r") as f:
            state = json.load(f)
        resumed_step = int(state.get("step", 0))
        if resumed_step <= 0:
            return 0

        with tempfile.TemporaryDirectory(prefix="sft_lora_resume_") as tmp_dir:
            for fname in gfile.ListDir(latest_dir):
                src = os.path.join(latest_dir, fname)
                dst = os.path.join(tmp_dir, fname)
                if not gfile.IsDirectory(src):
                    with gfile.Open(src, "rb") as f_in, open(dst, "wb") as f_out:
                        while True:
                            buf = f_in.read(16 * 1024 * 1024)
                            if not buf:
                                break
                            f_out.write(buf)

            adapter_safetensors = os.path.join(tmp_dir, "adapter_model.safetensors")
            adapter_bin = os.path.join(tmp_dir, "adapter_model.bin")
            if os.path.exists(adapter_safetensors):
                from safetensors.torch import load_file
                peft_weights = load_file(adapter_safetensors, device="cpu")
                peft.set_peft_model_state_dict(model, peft_weights)
                del peft_weights
            elif os.path.exists(adapter_bin):
                peft_weights = torch.load(adapter_bin, map_location="cpu")
                peft.set_peft_model_state_dict(model, peft_weights)
                del peft_weights

            opt_path = os.path.join(tmp_dir, "optimizer.pt")
            if os.path.exists(opt_path):
                opt_state = torch.load(opt_path, map_location="cpu")
                optimizer.load_state_dict(opt_state)
                del opt_state

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"Resumed LoRA adapter and optimizer state from {latest_dir} at step {resumed_step}.")
        return resumed_step
    except Exception as e:
        print(f"Warning: failed to resume from {latest_dir} ({e}); starting from step 0.")
        return 0


def compute_example_loss_weight(
    sample_loss: float,
    ema_loss: float,
    gamma: float,
    min_weight: float = 0.25,
    max_weight: float = 3.0,
) -> float:
    """Computes a focal/power loss weight for a trajectory based on its per-token NLL relative to ema_loss."""
    if gamma <= 0.0:
        return 1.0
    ref_loss = max(float(ema_loss), 1e-4)
    ratio = max(float(sample_loss), 0.0) / ref_loss
    raw_weight = ratio ** float(gamma)
    return float(max(min_weight, min(max_weight, raw_weight)))


def load_initial_lora_adapter(model: torch.nn.Module, adapter_dir: str) -> None:
    """Loads initial PEFT LoRA adapter weights from a CNS or local directory (without optimizer state)."""
    if not adapter_dir:
        return
    if not gfile.Exists(adapter_dir):
        raise FileNotFoundError(f"Initial LoRA adapter directory does not exist: {adapter_dir}")

    with tempfile.TemporaryDirectory(prefix="sft_lora_init_") as tmp_dir:
        for fname in ["adapter_model.safetensors", "adapter_model.bin", "adapter_config.json"]:
            src = os.path.join(adapter_dir, fname)
            if gfile.Exists(src):
                dst = os.path.join(tmp_dir, fname)
                with gfile.Open(src, "rb") as f_in, open(dst, "wb") as f_out:
                    while True:
                        buf = f_in.read(16 * 1024 * 1024)
                        if not buf:
                            break
                        f_out.write(buf)

        adapter_safetensors = os.path.join(tmp_dir, "adapter_model.safetensors")
        adapter_bin = os.path.join(tmp_dir, "adapter_model.bin")
        if os.path.exists(adapter_safetensors):
            from safetensors.torch import load_file
            peft_weights = load_file(adapter_safetensors, device="cpu")
            peft.set_peft_model_state_dict(model, peft_weights)
            del peft_weights
        elif os.path.exists(adapter_bin):
            peft_weights = torch.load(adapter_bin, map_location="cpu")
            peft.set_peft_model_state_dict(model, peft_weights)
            del peft_weights
        else:
            raise FileNotFoundError(
                f"Neither adapter_model.safetensors nor adapter_model.bin found in {adapter_dir}"
            )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"Loaded initial LoRA adapter weights from {adapter_dir}.")


@torch.no_grad()
def evaluate_validation_loss(
    model: torch.nn.Module,
    val_dataset: PackedTrajectoryDataset,
    device: torch.device,
    rank: int = 0,
    world_size: int = 1,
    max_samples: int = 64,
    chunk_size: int = 4096,
) -> dict[str, float]:
    """Computes assistant-only token NLL and perplexity on the held-out validation split (sharded across GPUs)."""
    model.eval()
    n_eval = len(val_dataset) if max_samples <= 0 else min(len(val_dataset), max_samples)

    base_qwen = model.base_model.model
    lang_model = base_qwen.model.language_model
    lm_head = base_qwen.lm_head

    t0 = time.time()
    local_nll = 0.0
    local_tokens = 0
    for i in range(rank, n_eval, world_size):
        sample = val_dataset[i]
        input_ids = sample["input_ids"].unsqueeze(0).to(device)
        labels = sample["labels"].unsqueeze(0).to(device)

        outputs = lang_model(input_ids=input_ids, use_cache=False)
        hidden_states = outputs.last_hidden_state
        loss_sum, n_valid = compute_chunked_causal_lm_loss(
            hidden_states=hidden_states,
            lm_head=lm_head,
            labels=labels,
            chunk_size=chunk_size,
        )
        local_nll += float(loss_sum.item())
        local_tokens += n_valid

    if world_size > 1 and dist.is_initialized():
        stats_tensor = torch.tensor([local_nll, float(local_tokens)], dtype=torch.float64, device=device)
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


def _train_worker(rank: int, world_size: int, flag_dict: dict[str, Any] | None = None) -> None:
    """Per-GPU training worker (runs standalone when world_size == 1 or under mp.spawn when world_size > 1)."""
    if not FLAGS.is_parsed():
        FLAGS([sys.argv[0]])
    if flag_dict is not None:
        for k, v in flag_dict.items():
            if k in FLAGS:
                setattr(FLAGS, k, v)

    if world_size > 1:
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29505")
        os.environ["TRITON_CACHE_DIR"] = f"/tmp/triton_cache_rank_{rank}"
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = f"/tmp/inductor_cache_rank_{rank}"
        torch.cuda.set_device(rank)
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

    is_main = rank == 0
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    torch.manual_seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    if FLAGS.use_fast_deltanet:
        fast_deltanet.patch_qwen35_fast_kernels()
        if is_main:
            print("Enabled fast Triton Gated DeltaNet forward/backward kernels + cuBLAS solve_triangular.")

    if is_main:
        print(f"Starting Qwen3.8-27B SFT trainer: world_size={world_size}, primary_device={device}")
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(rank)
            vram_gb = torch.cuda.get_device_properties(rank).total_memory / (1024**3)
            print(f"Detected GPU [{rank}/{world_size}]: {gpu_name} ({vram_gb:.1f} GB VRAM)")

    train_npz = os.path.join(FLAGS.data_dir, "train_tokens.npz")
    val_npz = os.path.join(FLAGS.data_dir, "val_tokens.npz")
    train_dataset = PackedTrajectoryDataset(train_npz, max_seq_len=FLAGS.max_seq_len)
    val_dataset = PackedTrajectoryDataset(val_npz, max_seq_len=FLAGS.max_seq_len)
    if is_main:
        print(
            f"Train samples: {len(train_dataset)} (skipped overlong>{FLAGS.max_seq_len}: {train_dataset.skipped_overlong})"
        )
        print(
            f"Val samples:   {len(val_dataset)} (skipped overlong>{FLAGS.max_seq_len}: {val_dataset.skipped_overlong})"
        )

    t_load = time.time()
    tokenizer = transformers.AutoTokenizer.from_pretrained(FLAGS.model_path, trust_remote_code=True) if is_main else None
    config = transformers.AutoConfig.from_pretrained(FLAGS.model_path, trust_remote_code=True)
    config.text_config.use_cache = False

    attn_impl = FLAGS.attn_implementation
    try:
        model = transformers.Qwen3_5ForConditionalGeneration.from_pretrained(
            FLAGS.model_path,
            config=config,
            torch_dtype=torch.bfloat16,
            device_map={"": device},
            attn_implementation=attn_impl,
        )
    except Exception as e:
        if is_main:
            print(f"attn_implementation={attn_impl} unavailable ({e}), falling back to 'sdpa'.")
        attn_impl = "sdpa"
        model = transformers.Qwen3_5ForConditionalGeneration.from_pretrained(
            FLAGS.model_path,
            config=config,
            torch_dtype=torch.bfloat16,
            device_map={"": device},
            attn_implementation=attn_impl,
        )

    if is_main:
        print(f"Loaded base model in {time.time() - t_load:.1f}s (attn_implementation={attn_impl})")

    if hasattr(model, "model") and hasattr(model.model, "visual"):
        del model.model.visual
        torch.cuda.empty_cache()

    if FLAGS.use_fast_deltanet:
        n_patched = fast_deltanet.patch_qwen35_fast_kernels(model)
        if is_main:
            print(f"Patched {n_patched} Qwen3_5GatedDeltaNet layers with fast Triton kernel.")

    if FLAGS.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        model.enable_input_require_grads()
        if FLAGS.grad_ckpt_stride > 1:
            layers = model.model.language_model.layers
            n_ckpt = 0
            for idx, layer in enumerate(layers):
                layer.gradient_checkpointing = (idx % FLAGS.grad_ckpt_stride == 0)
                if layer.gradient_checkpointing:
                    n_ckpt += 1
            if is_main:
                print(
                    f"Selective gradient checkpointing enabled (stride={FLAGS.grad_ckpt_stride}): "
                    f"{n_ckpt}/{len(layers)} decoder layers checkpointed."
                )

    target_modules = [m.strip() for m in FLAGS.lora_target_modules.split(",") if m.strip()]
    lora_config = peft.LoraConfig(
        r=FLAGS.lora_r,
        lora_alpha=FLAGS.lora_alpha,
        lora_dropout=FLAGS.lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = peft.get_peft_model(model, lora_config)
    if is_main:
        model.print_trainable_parameters()

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
        if is_main:
            print("Enabled torch.compile(dynamic=True, compile_threads=1) on Qwen3_5RMSNorm and Qwen3_5RMSNormGated.")

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=FLAGS.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=FLAGS.weight_decay,
        fused=torch.cuda.is_available(),
    )

    local_grad_accum = max(1, FLAGS.grad_accum_steps // world_size)
    effective_batch = local_grad_accum * world_size
    steps_per_epoch = len(train_dataset) // effective_batch
    total_steps = (
        FLAGS.max_steps
        if FLAGS.max_steps > 0
        else steps_per_epoch * FLAGS.num_epochs
    )
    warmup_steps = int(round(total_steps * FLAGS.warmup_ratio))
    if is_main:
        print(
            f"Training plan: {len(train_dataset)} samples/epoch, world_size={world_size}, "
            f"local_grad_accum={local_grad_accum}, effective_batch={effective_batch}, "
            f"steps_per_epoch={steps_per_epoch}, total_steps={total_steps}, warmup_steps={warmup_steps}, "
            f"loss_weight_gamma={FLAGS.loss_weight_gamma} [{FLAGS.loss_weight_min}, {FLAGS.loss_weight_max}]"
        )
        gfile.MakeDirs(FLAGS.output_dir)

    metrics_log_path = os.path.join(FLAGS.output_dir, "metrics.jsonl")

    def log_event(event: dict[str, Any]) -> None:
        if not is_main:
            return
        event["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        line = json.dumps(event)
        print(line, flush=True)
        with gfile.Open(metrics_log_path, "a") as f:
            f.write(line + "\n")

    resumed_step = maybe_resume_from_checkpoint(
        model=model,
        optimizer=optimizer,
        output_dir=FLAGS.output_dir,
        device=device,
    )

    ema_loss = 0.7627
    if resumed_step == 0:
        if FLAGS.init_adapter_path:
            load_initial_lora_adapter(model, FLAGS.init_adapter_path)
        if is_main:
            print("Evaluating baseline (step 0) validation loss...")
        val0 = evaluate_validation_loss(
            model=model,
            val_dataset=val_dataset,
            device=device,
            rank=rank,
            world_size=world_size,
            max_samples=FLAGS.max_eval_samples,
            chunk_size=FLAGS.logits_chunk_size,
        )
        ema_loss = float(val0["val_loss"])
        log_event(
            {
                "step": 0,
                "phase": "eval",
                "init_adapter_path": FLAGS.init_adapter_path,
                **val0,
            }
        )

    rng = np.random.default_rng(FLAGS.seed)
    global_step = 0
    step_t0 = time.time()

    base_qwen = model.base_model.model
    lang_model = base_qwen.model.language_model
    lm_head = base_qwen.lm_head

    model.train()
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(FLAGS.num_epochs if FLAGS.max_steps <= 0 else 999):
        # Identical permutation across all ranks, then slice by rank
        order = rng.permutation(len(train_dataset))
        usable_len = (len(order) // effective_batch) * effective_batch
        order = order[:usable_len]

        for batch_start in range(0, usable_len, effective_batch):
            if global_step < resumed_step:
                global_step += 1
                continue

            batch_indices = order[batch_start : batch_start + effective_batch]
            rank_indices = batch_indices[rank::world_size]

            local_loss_sum = 0.0
            local_tokens = 0
            local_weight_sum = 0.0
            local_weighted_sample_loss_sum = 0.0
            local_valid_samples = 0
            local_min_w = float("inf")
            local_max_w = 0.0
            local_min_sample_loss = float("inf")
            local_max_sample_loss = 0.0

            for sample_idx in rank_indices:
                sample = train_dataset[int(sample_idx)]
                input_ids = sample["input_ids"].unsqueeze(0).to(device)
                labels = sample["labels"].unsqueeze(0).to(device)

                outputs = lang_model(input_ids=input_ids, use_cache=False)
                hidden_states = outputs.last_hidden_state
                loss_sum, n_valid = compute_chunked_causal_lm_loss(
                    hidden_states=hidden_states,
                    lm_head=lm_head,
                    labels=labels,
                    chunk_size=FLAGS.logits_chunk_size,
                )
                if n_valid > 0:
                    sample_loss = float(loss_sum.item()) / float(n_valid)
                    w_raw = compute_example_loss_weight(
                        sample_loss=sample_loss,
                        ema_loss=ema_loss,
                        gamma=FLAGS.loss_weight_gamma,
                        min_weight=FLAGS.loss_weight_min,
                        max_weight=FLAGS.loss_weight_max,
                    )
                    scaled_loss = w_raw * (loss_sum / float(n_valid)) / float(effective_batch)
                    scaled_loss.backward()
                    local_loss_sum += float(loss_sum.item())
                    local_tokens += n_valid
                    local_weight_sum += w_raw
                    local_weighted_sample_loss_sum += w_raw * sample_loss
                    local_valid_samples += 1
                    local_min_w = min(local_min_w, w_raw)
                    local_max_w = max(local_max_w, w_raw)
                    local_min_sample_loss = min(local_min_sample_loss, sample_loss)
                    local_max_sample_loss = max(local_max_sample_loss, sample_loss)

            if world_size > 1 and dist.is_initialized():
                # All-reduce gradients across all B200 ranks (already divided by effective_batch)
                flat_grads = [
                    p.grad if p.grad is not None else torch.zeros_like(p)
                    for p in trainable_params
                ]
                coalesced = torch._utils._flatten_dense_tensors(flat_grads)
                dist.all_reduce(coalesced, op=dist.ReduceOp.SUM)
                for p, g_sync in zip(
                    trainable_params,
                    torch._utils._unflatten_dense_tensors(coalesced, flat_grads),
                ):
                    p.grad = g_sync

                metrics_tensor = torch.tensor(
                    [
                        local_loss_sum,
                        float(local_tokens),
                        local_weight_sum,
                        local_weighted_sample_loss_sum,
                        float(local_valid_samples),
                    ],
                    dtype=torch.float64,
                    device=device,
                )
                dist.all_reduce(metrics_tensor, op=dist.ReduceOp.SUM)
                step_loss_sum = float(metrics_tensor[0].item())
                step_tokens = int(round(float(metrics_tensor[1].item())))
                step_weight_sum = float(metrics_tensor[2].item())
                step_weighted_sample_loss_sum = float(metrics_tensor[3].item())
                step_valid_samples = int(round(float(metrics_tensor[4].item())))
            else:
                step_loss_sum = local_loss_sum
                step_tokens = local_tokens
                step_weight_sum = local_weight_sum
                step_weighted_sample_loss_sum = local_weighted_sample_loss_sum
                step_valid_samples = local_valid_samples

            mean_w = (
                step_weight_sum / float(step_valid_samples)
                if step_valid_samples > 0 and step_weight_sum > 0.0
                else 1.0
            )
            if FLAGS.loss_weight_gamma > 0.0 and mean_w > 0.0:
                inv_mean_w = 1.0 / mean_w
                for p in trainable_params:
                    if p.grad is not None:
                        p.grad.mul_(inv_mean_w)

            lr = get_cosine_lr(
                step=global_step,
                total_steps=total_steps,
                peak_lr=FLAGS.learning_rate,
                warmup_steps=warmup_steps,
                min_lr_ratio=FLAGS.min_lr_ratio,
            )
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            grad_norm = float(
                torch.nn.utils.clip_grad_norm_(trainable_params, FLAGS.max_grad_norm).item()
            )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

            step_loss = step_loss_sum / max(1, step_tokens)
            weighted_sample_loss = (
                step_weighted_sample_loss_sum / max(1e-8, step_weight_sum)
                if step_weight_sum > 0.0
                else step_loss
            )
            ema_loss = 0.95 * ema_loss + 0.05 * step_loss
            step_elapsed = time.time() - step_t0
            max_mem_gb = (
                torch.cuda.max_memory_allocated(device) / (1024**3)
                if torch.cuda.is_available()
                else 0.0
            )

            if global_step % FLAGS.log_every_steps == 0:
                norm_min_w = (local_min_w / mean_w) if local_valid_samples > 0 else 1.0
                norm_max_w = (local_max_w / mean_w) if local_valid_samples > 0 else 1.0
                log_event(
                    {
                        "step": global_step,
                        "total_steps": total_steps,
                        "epoch": epoch,
                        "phase": "train",
                        "train_loss": round(step_loss, 5),
                        "weighted_sample_loss": round(weighted_sample_loss, 5),
                        "ema_loss": round(ema_loss, 5),
                        "norm_weight_min": round(norm_min_w, 3),
                        "norm_weight_max": round(norm_max_w, 3),
                        "sample_loss_min": round(
                            local_min_sample_loss if local_valid_samples > 0 else 0.0, 4
                        ),
                        "sample_loss_max": round(
                            local_max_sample_loss if local_valid_samples > 0 else 0.0, 4
                        ),
                        "train_ppl": round(math.exp(min(step_loss, 20.0)), 4),
                        "lr": round(lr, 8),
                        "grad_norm": round(grad_norm, 4),
                        "supervised_tokens": step_tokens,
                        "step_sec": round(step_elapsed, 2),
                        "tok_per_sec": round(step_tokens / max(0.01, step_elapsed), 1),
                        "max_vram_gb": round(max_mem_gb, 2),
                        "world_size": world_size,
                    }
                )

            step_t0 = time.time()

            if FLAGS.eval_every_steps > 0 and global_step % FLAGS.eval_every_steps == 0:
                val_metrics = evaluate_validation_loss(
                    model=model,
                    val_dataset=val_dataset,
                    device=device,
                    rank=rank,
                    world_size=world_size,
                    max_samples=FLAGS.max_eval_samples,
                    chunk_size=FLAGS.logits_chunk_size,
                )
                log_event({"step": global_step, "phase": "eval", **val_metrics})
                step_t0 = time.time()

            if FLAGS.save_every_steps > 0 and global_step % FLAGS.save_every_steps == 0:
                if is_main:
                    latest_dir = os.path.join(FLAGS.output_dir, "checkpoint-latest")
                    print(f"Saving preemption checkpoint to {latest_dir} (step {global_step})...")
                    save_lora_checkpoint(
                        model=model,
                        tokenizer=None,
                        cns_ckpt_dir=latest_dir,
                        step_meta={"step": global_step, "epoch": epoch, "train_loss": step_loss},
                        optimizer=optimizer,
                    )
                    if global_step % 50 == 0:
                        ckpt_dir = os.path.join(FLAGS.output_dir, f"checkpoint-{global_step}")
                        print(f"Saving milestone LoRA checkpoint to {ckpt_dir}...")
                        save_lora_checkpoint(
                            model=model,
                            tokenizer=tokenizer,
                            cns_ckpt_dir=ckpt_dir,
                            step_meta={"step": global_step, "epoch": epoch, "train_loss": step_loss},
                        )
                if world_size > 1 and dist.is_initialized():
                    dist.barrier()
                step_t0 = time.time()

            if global_step >= total_steps:
                break
        if global_step >= total_steps:
            break

    if is_main:
        print("Running final validation evaluation...")
    final_val = evaluate_validation_loss(
        model=model,
        val_dataset=val_dataset,
        device=device,
        rank=rank,
        world_size=world_size,
        max_samples=FLAGS.max_eval_samples,
        chunk_size=FLAGS.logits_chunk_size,
    )
    log_event({"step": global_step, "phase": "final_eval", **final_val})

    if is_main:
        final_ckpt_dir = os.path.join(FLAGS.output_dir, "final_adapter")
        print(f"Saving final LoRA adapter to {final_ckpt_dir}...")
        save_lora_checkpoint(
            model=model,
            tokenizer=tokenizer,
            cns_ckpt_dir=final_ckpt_dir,
            step_meta={"step": global_step, **final_val},
        )
        print("Training complete.")

    if world_size > 1 and dist.is_initialized():
        dist.destroy_process_group()


def main(argv: list[str]) -> None:
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")

    flag_dict = {
        name: FLAGS[name].value
        for name in [
            "model_path",
            "data_dir",
            "output_dir",
            "init_adapter_path",
            "lora_r",
            "lora_alpha",
            "lora_dropout",
            "lora_target_modules",
            "learning_rate",
            "min_lr_ratio",
            "weight_decay",
            "warmup_ratio",
            "max_grad_norm",
            "loss_weight_gamma",
            "loss_weight_min",
            "loss_weight_max",
            "num_epochs",
            "max_steps",
            "grad_accum_steps",
            "max_seq_len",
            "logits_chunk_size",
            "eval_every_steps",
            "max_eval_samples",
            "save_every_steps",
            "log_every_steps",
            "seed",
            "gradient_checkpointing",
            "grad_ckpt_stride",
            "use_fast_deltanet",
            "use_torch_compile_norms",
            "attn_implementation",
            "use_multi_gpu_ddp",
        ]
    }

    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
    if FLAGS.use_multi_gpu_ddp and num_gpus > 1:
        print(f"Forking {num_gpus} DDP worker processes across {num_gpus} B200 GPUs...")
        mp.start_processes(
            _train_worker,
            args=(num_gpus, flag_dict),
            nprocs=num_gpus,
            join=True,
            start_method="fork",
        )
    else:
        _train_worker(0, 1, flag_dict)


if __name__ == "__main__":
    app.run(main)
