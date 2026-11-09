"""Qwen-compatible ``starter.py`` pipeline for DiffusionGemma ARC TTT.

The public process topology remains owned by :mod:`starter`: independent
trainer processes publish task LoRAs into bounded staging tiers and persistent
vLLM services consume them asynchronously.  This module supplies only the
DiffusionGemma-specific trainer and service implementations.

Training uses the exact ARC31 SFT objective, a frozen Transformer-Engine FP8
base, and BF16 task LoRAs.  Serving uses vLLM's native DiffusionGemma runner,
online FP8 quantization, and the same BF16 LoRA through a PEFT compatibility
export.  Self-conditioning LoRA is deliberately frozen because the native
vLLM implementation does not currently expose those three projections to its
LoRA manager.
"""

from __future__ import annotations

import argparse
import bz2
from collections import deque
import gc
import importlib.metadata
import json
import math
import os
from pathlib import Path
import pickle
from queue import Empty
import shutil
import time
from typing import Any, Iterable


DIFFUSION_GEMMA_ARCHITECTURE = "DiffusionGemmaForBlockDiffusion"
DIFFUSION_GEMMA_PLUGIN = "arc_diffusion_gemma"
DEFAULT_RUNTIME_PARENT = "/dev/shm"
# Inclusive lower encoder-token bounds for frozen-FP8 LoRA on a 22 GiB L4.
# Each row is ``(bound, decoder %, encoder %, input offload)``.
# A 2,030-token task OOMed at every tested policy through 75%/75%; 90%/90%
# completed at 20,018 MiB.  At 2,904 tokens 90%/90% completed too close to the
# device limit (21,603 MiB), while 3,865 tokens at 90%/100% used 19,373 MiB.
# The 6,562-token 100%/100% gate completed at 21,798 MiB, so the long tier
# offloads encoder checkpoint inputs before that marginal range.  This is the
# DiffusionGemma analogue of QWEN3_4B_R128_AUTO_GRADIENT_CHECKPOINTING_TIERS,
# but the dual encoder/decoder stack needs independent percentages.
DIFFUSION_GEMMA_L4_R64_AUTO_CHECKPOINTING_TIERS = (
    (0, 90.0, 90.0, False),
    (2560, 90.0, 100.0, False),
    (4096, 100.0, 100.0, False),
    (5632, 100.0, 100.0, True),
)

# Rank 32 is the long-context production profile.  It is the largest tested
# rank that completed a real 11,858-token input / 12,114-token physical row on
# one L4 (full dual-stack checkpointing, encoder input offload, 20,586 MiB with
# the opt-in TE-grouped expert backend; the loop backend used 21,007 MiB).
# The 80% short tier completed all eight rows of the 1,855--2,030-token gate;
# 50% OOMed and 75% ran within roughly 0.5 GiB of the device limit.  Keep the
# larger-rank profile more conservative because rank 64 failed the near-12k
# gate even with every layer checkpointed.
DIFFUSION_GEMMA_L4_R32_AUTO_CHECKPOINTING_TIERS = (
    (0, 80.0, 80.0, False),
    (2560, 90.0, 100.0, False),
    (4096, 100.0, 100.0, False),
    (5632, 100.0, 100.0, True),
)

# Opt-in calibration for the 48-expert step-2800 model and the fused
# cutlass-ada-triton training path. Bounds use *physical* prompt+canvas width,
# unlike the historical 64-expert tables above. The short tier completed 128
# real updates at rank 64; the medium and 12K tiers have shorter memory gates.
# Keep legacy defaults unchanged until a broader task/accuracy run is done.
DIFFUSION_GEMMA_L4_EXP48_AUTO_CHECKPOINTING_TIERS = (
    (0, 80.0, 80.0, False),
    (3457, 90.0, 100.0, False),
    (8193, 100.0, 100.0, True),
)
DIFFUSION_GEMMA_L4_EXP48_R128_AUTO_CHECKPOINTING_TIERS = (
    (0, 90.0, 90.0, False),
    (3457, 90.0, 100.0, False),
    (4609, 100.0, 100.0, True),
)
DIFFUSION_GEMMA_CHECKPOINTING_PROFILES = ("legacy", "l4-exp48")

# Pair-FSDP needs one contiguous 364 MiB expert all-gather during checkpoint
# replay. At the 2,904-token gate, full encoder checkpointing alone left only
# 256--358 MiB physically free on an L4 (allocator variants and LoRA chunk
# sizes did not change the failure). Offload the saved encoder checkpoint
# inputs one tier earlier for the paired topology so that the all-gather has a
# stable margin. This is separate from the unsharded one-GPU calibration.
DIFFUSION_GEMMA_PAIR_FSDP_INPUT_OFFLOAD_MIN_TOKENS = 2560


def _diffusion_gemma_auto_checkpointing_tiers(
    lora_rank: int,
    profile: str = "legacy",
) -> tuple[tuple[int, float, float, bool], ...]:
    lora_rank = int(lora_rank)
    if lora_rank <= 0:
        raise ValueError("lora_rank must be positive")
    if profile not in DIFFUSION_GEMMA_CHECKPOINTING_PROFILES:
        raise ValueError(f"unknown DiffusionGemma checkpointing profile: {profile!r}")
    if profile == "l4-exp48":
        if lora_rank > 128:
            raise ValueError("l4-exp48 checkpointing supports ranks through 128")
        if lora_rank > 64:
            return DIFFUSION_GEMMA_L4_EXP48_R128_AUTO_CHECKPOINTING_TIERS
        return DIFFUSION_GEMMA_L4_EXP48_AUTO_CHECKPOINTING_TIERS
    if lora_rank <= 32:
        return DIFFUSION_GEMMA_L4_R32_AUTO_CHECKPOINTING_TIERS
    return DIFFUSION_GEMMA_L4_R64_AUTO_CHECKPOINTING_TIERS


def select_diffusion_gemma_auto_checkpointing(
    max_train_tokens: int,
    lora_rank: int = 64,
    *,
    profile: str = "legacy",
) -> tuple[float, float]:
    """Choose decoder/encoder GC percentages for an L4 task LoRA."""
    max_train_tokens = int(max_train_tokens)
    if max_train_tokens < 0:
        raise ValueError("max_train_tokens must be non-negative")
    tiers = _diffusion_gemma_auto_checkpointing_tiers(lora_rank, profile)
    decoder, encoder = tiers[0][1:3]
    for lower_bound, tier_decoder, tier_encoder, _input_offload in tiers:
        if max_train_tokens < lower_bound:
            break
        decoder, encoder = tier_decoder, tier_encoder
    return float(decoder), float(encoder)


def select_diffusion_gemma_checkpoint_input_offload(
    max_train_tokens: int,
    lora_rank: int = 64,
    *,
    profile: str = "legacy",
) -> bool:
    """Select pinned-CPU storage for long encoder checkpoint inputs."""
    max_train_tokens = int(max_train_tokens)
    if max_train_tokens < 0:
        raise ValueError("max_train_tokens must be non-negative")
    tiers = _diffusion_gemma_auto_checkpointing_tiers(lora_rank, profile)
    selected = bool(tiers[0][3])
    for lower_bound, _decoder, _encoder, input_offload in tiers:
        if max_train_tokens < lower_bound:
            break
        selected = bool(input_offload)
    return selected


def resolve_diffusion_gemma_checkpoint_input_offload(
    max_train_tokens: int,
    lora_rank: int = 64,
    *,
    pair_fsdp: bool = False,
    force: bool = False,
    profile: str = "legacy",
) -> bool:
    """Resolve automatic and explicitly forced encoder-input offload."""
    return bool(
        force
        or select_diffusion_gemma_checkpoint_input_offload(
            max_train_tokens, lora_rank, profile=profile
        )
        or (
            pair_fsdp
            and int(max_train_tokens)
            >= DIFFUSION_GEMMA_PAIR_FSDP_INPUT_OFFLOAD_MIN_TOKENS
        )
    )


def resolve_diffusion_gemma_task_checkpointing(
    max_train_tokens: int,
    decoder: str | float,
    encoder: str | float,
    lora_rank: int = 64,
    *,
    profile: str = "legacy",
) -> tuple[float, float]:
    """Resolve independent explicit or automatic task checkpoint policies."""
    auto_decoder, auto_encoder = select_diffusion_gemma_auto_checkpointing(
        max_train_tokens, lora_rank, profile=profile
    )
    return (
        auto_decoder if decoder == "auto" else float(decoder),
        auto_encoder if encoder == "auto" else float(encoder),
    )


def validate_diffusion_gemma_checkpointing_profile(
    profile: str,
    config: dict[str, Any],
    lora_rank: int,
    options: dict[str, Any],
) -> None:
    """Reject reuse of an experimental memory table on a different workload."""
    _diffusion_gemma_auto_checkpointing_tiers(lora_rank, profile)
    if profile == "legacy":
        return
    text_config = config.get("text_config") or {}
    geometry = {
        "num_experts": 48,
        "num_hidden_layers": 30,
        "hidden_size": 2816,
        "moe_intermediate_size": 704,
    }
    if any(text_config.get(key) != value for key, value in geometry.items()):
        raise ValueError("l4-exp48 checkpointing requires the 48-expert/30-layer ARC model")
    if options.get("diffusion_gemma_pair_fsdp", False) or int(
        options.get("diffusion_gemma_expert_parallel_size", 1)
    ) != 1:
        raise ValueError("l4-exp48 checkpointing is for independent one-GPU trainers")
    if options.get("diffusion_gemma_lora_experts_backend", "loop") != "cutlass-ada-triton":
        raise ValueError("l4-exp48 checkpointing requires cutlass-ada-triton experts")
    if options.get("diffusion_gemma_optimizer", "muon8bit") != "muon8bit" or not options.get(
        "diffusion_gemma_optimizer_state_offload", True
    ):
        raise ValueError("l4-exp48 checkpointing requires Muon8bit optimizer-state offload")
    if int(options.get("max_seq_length", 12288)) > 12288:
        raise ValueError("l4-exp48 checkpointing has only been screened up to 12,288 tokens")


def configure_diffusion_gemma_rank_memory(
    lora_rank: int, *, profile: str = "legacy"
) -> dict[str, Any]:
    """Apply opt-in rank-128 working-buffer limits before importing REAM.

    An explicit EXP48 profile owns these three switches for larger ranks;
    retaining the rank-32 fast settings can OOM even with full outer GC.
    The caller also uses the returned Muon chunk limit. Model values and the
    optimizer algorithm remain unchanged; only recomputation/update batching
    differ. Lower ranks and legacy workloads are untouched.
    """
    _diffusion_gemma_auto_checkpointing_tiers(lora_rank, profile)
    if profile != "l4-exp48" or lora_rank <= 64:
        return {}
    settings = {
        "REAM_DIFFUSION_GEMMA_GROUPED_TOKEN_CHUNK_SIZE": "8192",
        "REAM_DIFFUSION_GEMMA_CHECKPOINT_GROUPED_TILES": "1",
        "REAM_DIFFUSION_GEMMA_GROUPED_LORA_PAIR_RECOMPUTE": "1",
    }
    previous = {key: os.environ.get(key) for key in settings}
    os.environ.update(settings)
    return {"environment": settings, "previous_environment": previous,
            "muon_ns_chunk_size": 1}


def diffusion_gemma_checkpointing_token_count(
    prompt_tokens: int, physical_tokens: int, *, profile: str = "legacy"
) -> int:
    """Keep legacy prompt-width semantics; EXP48 includes the response canvas."""
    if profile not in DIFFUSION_GEMMA_CHECKPOINTING_PROFILES:
        raise ValueError(f"unknown DiffusionGemma checkpointing profile: {profile!r}")
    return int(physical_tokens if profile == "l4-exp48" else prompt_tokens)


def _read_config(model_path: str | os.PathLike[str]) -> dict[str, Any]:
    config_path = Path(model_path).expanduser().resolve() / "config.json"
    if not config_path.is_file():
        return {}
    payload = json.loads(config_path.read_text())
    if not isinstance(payload, dict):
        raise TypeError(f"model config must be a JSON object: {config_path}")
    return payload


def model_declares_diffusion_gemma(model_path: str | os.PathLike[str]) -> bool:
    """Detect the native DiffusionGemma family without importing Transformers."""
    config = _read_config(model_path)
    architectures = {str(value) for value in config.get("architectures", ())}
    model_types = {
        str(config.get("model_type", "")),
        str((config.get("text_config") or {}).get("model_type", "")),
    }
    return bool(
        DIFFUSION_GEMMA_ARCHITECTURE in architectures
        or "DiffusionGemmaForConditionalGeneration" in architectures
        or "diffusion_gemma" in model_types
        or "diffusion_gemma_text" in model_types
    )


def preflight_diffusion_gemma_plugin() -> str:
    """Validate and allowlist the native vLLM plugin before process spawn."""
    try:
        entries = importlib.metadata.entry_points(group="vllm.general_plugins")
    except TypeError:  # Python 3.10 importlib.metadata compatibility.
        entries = importlib.metadata.entry_points().get(
            "vllm.general_plugins", ()
        )
    matching = [entry for entry in entries if entry.name == DIFFUSION_GEMMA_PLUGIN]
    if len(matching) != 1:
        project_root = Path(__file__).resolve().parent / "vllm_diffusion_gemma_plugin"
        raise RuntimeError(
            "DiffusionGemma vLLM inference requires the repository plugin. "
            "Install it offline before starter.py with "
            f"`python -m pip install --no-build-isolation --no-deps -e "
            f"{project_root}`."
        )
    allowed = [value for value in os.getenv("VLLM_PLUGINS", "").split(",") if value]
    if allowed and DIFFUSION_GEMMA_PLUGIN not in allowed:
        raise RuntimeError(
            "VLLM_PLUGINS excludes required plugin "
            f"{DIFFUSION_GEMMA_PLUGIN!r}: {allowed}"
        )
    if not allowed:
        os.environ["VLLM_PLUGINS"] = DIFFUSION_GEMMA_PLUGIN
    return matching[0].value


def prepare_diffusion_gemma_runtime_model(
    base_model_path: str | os.PathLike[str],
    delta_checkpoint: str | os.PathLike[str] | None,
    runtime_model_dir: str | os.PathLike[str] | None,
    *,
    work_dir: str | os.PathLike[str] | None = None,
    reconstruct_workers: int = 1,
) -> tuple[str, dict[str, Any] | None, bool]:
    """Return a complete HF model path, reconstructing an exact delta once.

    The returned boolean identifies a directory created by this call and thus
    safe for ``starter.py`` to remove after every spawned process exits.
    """
    base = Path(base_model_path).expanduser().resolve()
    if not base.is_dir():
        raise FileNotFoundError(f"DiffusionGemma base model not found: {base}")
    if not model_declares_diffusion_gemma(base):
        raise ValueError(f"model is not DiffusionGemma: {base}")
    if not delta_checkpoint:
        return str(base), None, False

    delta = Path(delta_checkpoint).expanduser().resolve()
    if not delta.is_file() and not delta.is_dir():
        raise FileNotFoundError(f"DiffusionGemma delta checkpoint not found: {delta}")
    auto_created = runtime_model_dir is None
    if runtime_model_dir:
        destination = Path(runtime_model_dir).expanduser().resolve()
    else:
        destination = Path(DEFAULT_RUNTIME_PARENT) / (
            f"arc_diffusion_gemma_runtime_{os.getpid()}_{time.time_ns()}"
        )
    if destination.exists():
        if any(destination.iterdir()):
            raise FileExistsError(
                "DiffusionGemma runtime reconstruction directory must be absent "
                f"or empty: {destination}"
            )
        destination.rmdir()
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)

    from diffusion_gemma_delta_checkpoint import reconstruct_delta_checkpoint

    metadata = reconstruct_delta_checkpoint(
        delta,
        base,
        destination,
        work_dir=work_dir,
        workers=reconstruct_workers,
    )
    runtime_path = Path(metadata["checkpoint"]).resolve()
    if runtime_path != destination:
        raise RuntimeError(
            f"delta reconstruction returned {runtime_path}, expected {destination}"
        )
    if not model_declares_diffusion_gemma(runtime_path):
        raise RuntimeError(
            "reconstructed checkpoint is missing a DiffusionGemma config: "
            f"{runtime_path}"
        )
    if not any(runtime_path.glob("*.safetensors")):
        raise RuntimeError(
            f"reconstructed checkpoint has no safetensors: {runtime_path}"
        )
    metadata = dict(metadata)
    metadata["runtime_model_path"] = str(runtime_path)
    return str(runtime_path), metadata, auto_created


def remove_created_runtime_model(path: str | os.PathLike[str]) -> None:
    """Remove only the uniquely named runtime directory created above."""
    runtime = Path(path).expanduser().resolve()
    parent = Path(DEFAULT_RUNTIME_PARENT).resolve()
    if runtime.parent != parent or not runtime.name.startswith(
        "arc_diffusion_gemma_runtime_"
    ):
        raise ValueError(f"refusing to remove non-runtime model path: {runtime}")
    shutil.rmtree(runtime, ignore_errors=True)


class _RowsDataset:
    def __init__(self, rows: Iterable[dict[str, Any]]):
        self.rows = list(rows)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.rows[int(index)]


def _load_training_model(
    model_path: str,
    *,
    rank: int,
    alpha: float,
    ream_root: str,
    dynamic_backward: bool,
    split_accumulator: bool,
    token_chunk_size: int,
    lora_experts_backend: str,
    decoder_checkpointing_percentage: str | float,
    encoder_checkpointing_percentage: str | float,
    kernel_backend: str,
    encoder_attention_backend: str,
    decoder_attention_backend: str,
    expert_parallel_group: Any | None = None,
):
    """Load one persistent frozen-FP8 trainer model on its visible L4."""
    from pre_sft_diffusion_gemma import (
        _import_ream,
        _import_unsloth_runtime,
        _load_model,
        install_decoder_checkpointing,
        install_encoder_checkpointing,
    )

    (
        UnslothTrainingArguments,
        UnslothTrainer,
        AutoTokenizer,
        model_class,
        _TrainerCallback,
    ) = _import_unsloth_runtime()
    from diffusion_gemma_text_only_runtime import (
        install_diffusion_gemma_text_only_runtime,
    )
    from diffusion_gemma_training_kernels import (
        install_diffusion_gemma_decoder_attention,
        install_diffusion_gemma_encoder_attention,
        install_diffusion_gemma_training_kernels,
    )

    install_diffusion_gemma_text_only_runtime()
    (
        _DiffusionCalibration,
        _load_adapter,
        prepare_fp8_lora,
        _save_adapter,
        _trainable_parameters,
    ) = _import_ream(ream_root)
    model = _load_model(model_class, model_path)
    conversion = prepare_fp8_lora(
        model,
        rank=int(rank),
        alpha=float(alpha),
        device="cuda",
        dynamic_backward=bool(dynamic_backward),
        split_accumulator=bool(split_accumulator),
        token_chunk_size=int(token_chunk_size),
        experts_backend=str(lora_experts_backend),
        expert_parallel_group=expert_parallel_group,
    )
    from diffusion_gemma_ttt_lora import freeze_unserved_self_conditioning_lora

    self_conditioning = freeze_unserved_self_conditioning_lora(model)
    kernels = install_diffusion_gemma_training_kernels(model, kernel_backend)
    encoder_attention = install_diffusion_gemma_encoder_attention(
        model, encoder_attention_backend
    )
    decoder_attention = install_diffusion_gemma_decoder_attention(
        model, decoder_attention_backend
    )
    # Install every wrapper that an automatic per-task tier may activate.
    # Reconfiguring a wrapper is a cheap boolean toggle; the model and FP8
    # weights stay resident across tasks.
    installed_decoder_percentage = (
        100.0
        if decoder_checkpointing_percentage == "auto"
        else float(decoder_checkpointing_percentage)
    )
    installed_encoder_percentage = (
        100.0
        if encoder_checkpointing_percentage == "auto"
        else float(encoder_checkpointing_percentage)
    )
    decoder_checkpoints = install_decoder_checkpointing(
        model,
        installed_decoder_percentage,
        preserve_rng_state=False,
    )
    encoder_checkpoints = install_encoder_checkpointing(
        model,
        installed_encoder_percentage,
        preserve_rng_state=False,
        selective_save=False,
        input_offload=False,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=False,
    )
    model.train()
    return {
        "model": model,
        "UnslothTrainingArguments": UnslothTrainingArguments,
        "UnslothTrainer": UnslothTrainer,
        "AutoTokenizer": AutoTokenizer,
        "tokenizer": tokenizer,
        "conversion": conversion,
        "self_conditioning": self_conditioning,
        "kernels": kernels,
        "encoder_attention": encoder_attention,
        "decoder_attention": decoder_attention,
        "decoder_checkpoints": decoder_checkpoints,
        "encoder_checkpoints": encoder_checkpoints,
        "decoder_checkpointing_policy": decoder_checkpointing_percentage,
        "encoder_checkpointing_policy": encoder_checkpointing_percentage,
        "configure_decoder_checkpointing": install_decoder_checkpointing,
        "configure_encoder_checkpointing": install_encoder_checkpointing,
    }


def _make_task_trainer(
    runtime: dict[str, Any],
    rows: list[dict[str, Any]],
    *,
    task_seed: int,
    model_path: str,
    max_seq_length: int,
    steps: int,
    learning_rate: float,
    optimizer: str,
    optimizer_state_offload: bool,
    optimizer_state_offload_pin_memory: bool,
    max_grad_norm: float,
    arc_output_vocab_size: int,
    muon_adjust_lr: str,
    muon_momentum: float,
    muon_nesterov: bool,
    muon_ns_chunk_size: int,
    muon_min_8bit_size: int,
    self_conditioning_seed: int | None = None,
    pair_physical_tokens: int | None = None,
):
    from diffusion_gemma_arc_data import CANVAS_LENGTH, DiffusionArcCollator
    from pre_sft_diffusion_gemma import make_trainer_class

    model = runtime["model"]
    collator = DiffusionArcCollator(
        seed=int(task_seed),
        self_conditioning_probability=0.5,
        canvas_length=CANVAS_LENGTH,
    )
    if self_conditioning_seed is not None:
        collator = _PairSelfConditioningCollator(
            collator,
            seed=int(self_conditioning_seed),
            probability=0.5,
            physical_tokens=pair_physical_tokens,
        )
    arguments = runtime["UnslothTrainingArguments"](
        output_dir=f"/dev/shm/arc_diffusion_gemma_ttt_trainer_{os.getpid()}",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        max_steps=int(steps),
        learning_rate=float(learning_rate),
        warmup_steps=0,
        lr_scheduler_type="cosine",
        optim="adamw_torch",
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        weight_decay=0.0,
        max_grad_norm=float(max_grad_norm),
        seed=int(task_seed),
        data_seed=int(task_seed),
        report_to="none",
        save_strategy="no",
        eval_strategy="no",
        logging_strategy="steps",
        logging_steps=max(1, int(steps) // 4),
        logging_first_step=False,
        disable_tqdm=True,
        fp16=False,
        bf16=False,
        tf32=True,
        gradient_checkpointing=False,
        remove_unused_columns=False,
        dataloader_num_workers=0,
        dataloader_drop_last=False,
        packing=False,
        padding_free=False,
        max_length=int(max_seq_length),
        dataset_kwargs={"skip_prepare_dataset": True},
    )
    arc_options = argparse.Namespace(
        arc_output_vocab_size=int(arc_output_vocab_size),
        data_mode="nvarc",
        diffusion_loss_mask="full-canvas",
        encoder_loss_weight=1.0,
        gradient_diagnostics_max_events=0,
        gradient_diagnostics_steps=0,
        gradient_diagnostics_threshold=0.0,
        learning_rate=float(learning_rate),
        microbatch_timing_steps=0,
        muon_adjust_lr=str(muon_adjust_lr),
        muon_min_8bit_size=int(muon_min_8bit_size),
        muon_momentum=float(muon_momentum),
        muon_nesterov=bool(muon_nesterov),
        muon_ns_chunk_size=int(muon_ns_chunk_size),
        muon_state_offload=bool(optimizer_state_offload),
        optimizer=str(optimizer),
        optimizer_state_offload=(
            bool(optimizer_state_offload) if optimizer == "adamw_torch" else False
        ),
        optimizer_state_offload_pin_memory=bool(
            optimizer_state_offload_pin_memory
        ),
        phase_timing_steps=0,
        self_conditioning="artifact",
        # Batch size is one. "selected" is exactly equivalent to "full" for
        # rows selected for self-conditioning and skips the first pass for the
        # other rows.
        self_conditioning_first_pass="selected",
        training_mode="lora",
    )
    trainer_class = make_trainer_class(runtime["UnslothTrainer"])
    return trainer_class(
        model=model,
        args=arguments,
        train_dataset=_RowsDataset(rows),
        data_collator=collator,
        processing_class=runtime["tokenizer"],
        arc_options=arc_options,
    )


class _PairSelfConditioningCollator:
    """Keep pair-FSDP tensor shapes/control flow identical across task lanes."""

    def __init__(
        self,
        collator: Any,
        *,
        seed: int,
        probability: float,
        physical_tokens: int | None = None,
    ):
        self.collator = collator
        self.probability = float(probability)
        self.physical_tokens = (
            None if physical_tokens is None else int(physical_tokens)
        )
        if self.physical_tokens is not None and self.physical_tokens <= 0:
            raise ValueError("pair physical token count must be positive")
        self.generator = __import__("torch").Generator(device="cpu")
        self.generator.manual_seed(int(seed))

    def __call__(self, rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
        torch = __import__("torch")
        batch = self.collator(rows)
        mask = batch["self_conditioning_mask"]
        synchronized = torch.rand(
            mask.shape,
            generator=self.generator,
            dtype=torch.float32,
        ) < self.probability
        batch["self_conditioning_mask"] = synchronized
        if self.physical_tokens is not None:
            from diffusion_gemma_arc_data import PAD_ID

            current = int(batch["input_ids"].shape[1])
            if current > self.physical_tokens:
                raise RuntimeError(
                    "paired DiffusionGemma batch exceeds its synchronized "
                    f"physical length: {current} > {self.physical_tokens}"
                )
            padding = self.physical_tokens - current
            if padding:
                pad = torch.nn.functional.pad
                batch["input_ids"] = pad(
                    batch["input_ids"], (0, padding), value=int(PAD_ID)
                )
                batch["attention_mask"] = pad(
                    batch["attention_mask"], (0, padding), value=0
                )
                batch["position_ids"] = pad(
                    batch["position_ids"], (0, padding), value=0
                )
        return batch


def _maximum_row_physical_tokens(rows: Iterable[dict[str, Any]]) -> int:
    """Return the width after DiffusionArcCollator joins prompt and canvas."""
    maximum = max(
        (
            len(row["input_ids"]) + len(row["target_ids"])
            for row in rows
        ),
        default=0,
    )
    if maximum <= 0:
        raise ValueError("DiffusionGemma TTT rows must not be empty")
    return int(maximum)


def _initialize_pair_fsdp(rank: int, port: int) -> Any:
    """Create the private two-trainer NCCL group without enabling Trainer DDP."""
    from datetime import timedelta

    torch = __import__("torch")
    if torch.distributed.is_initialized():
        raise RuntimeError("pair-FSDP process group was initialized twice")
    torch.cuda.set_device(0)
    torch.distributed.init_process_group(
        backend="nccl",
        init_method=f"tcp://127.0.0.1:{int(port)}",
        rank=int(rank),
        world_size=2,
        timeout=timedelta(minutes=30),
    )
    return torch.distributed.group.WORLD


def _initialize_expert_parallel(rank: int, world_size: int, port: int) -> Any:
    """Create a private activation-dispatch EP group for independent tasks."""
    from datetime import timedelta

    torch = __import__("torch")
    if torch.distributed.is_initialized():
        raise RuntimeError("expert-parallel process group was initialized twice")
    if int(world_size) not in {2, 3}:
        raise ValueError(
            f"DiffusionGemma EP currently supports 2 or 3 ranks, got {world_size}"
        )
    torch.cuda.set_device(0)
    torch.distributed.init_process_group(
        backend="nccl",
        init_method=f"tcp://127.0.0.1:{int(port)}",
        rank=int(rank),
        world_size=int(world_size),
        timeout=timedelta(minutes=30),
    )
    return torch.distributed.group.WORLD


def _collective_task_seeds(seed: int, world_size: int, device: Any) -> tuple[int, ...]:
    torch = __import__("torch")
    local = torch.tensor([int(seed)], dtype=torch.int64, device=device)
    gathered = torch.empty(int(world_size), dtype=torch.int64, device=device)
    torch.distributed.all_gather_into_tensor(gathered, local)
    return tuple(int(value) for value in gathered.cpu().tolist())


def _configure_expert_parallel_route_envelope(
    model: Any, physical_tokens: int
) -> int:
    configured = 0
    for module in model.modules():
        setter = getattr(module, "set_synchronized_physical_tokens", None)
        if callable(setter) and bool(getattr(module, "is_expert_parallel", False)):
            setter(int(physical_tokens))
            configured += 1
    if configured == 0:
        raise RuntimeError("no expert-parallel layers accepted the route envelope")
    return configured


def _pair_task_state(has_task: bool, device: Any) -> int:
    """Return how many pair ranks received tasks at this queue boundary."""
    torch = __import__("torch")
    value = torch.tensor(
        [int(bool(has_task))],
        dtype=torch.int32,
        device=device,
    )
    torch.distributed.all_reduce(value, op=torch.distributed.ReduceOp.SUM)
    return int(value.item())


def _pair_maximum(value: int, device: Any) -> int:
    torch = __import__("torch")
    maximum = torch.tensor([int(value)], dtype=torch.int64, device=device)
    torch.distributed.all_reduce(maximum, op=torch.distributed.ReduceOp.MAX)
    return int(maximum.item())


def _acquire_pending_slot(
    semaphore: Any,
    *,
    abort_event: Any,
    end_time: float,
    poll_seconds: float = 0.25,
) -> bool:
    while time.time() < float(end_time):
        if abort_event is not None and abort_event.is_set():
            return False
        if semaphore.acquire(timeout=float(poll_seconds)):
            return True
    return False


def _publish_task_adapter(
    model: Any,
    *,
    rank: int,
    task_key: str,
    adapter_id: int,
    run_id: str,
    training_seconds: float,
    training_loss: float | None,
    vllm_pending_adapter_slots: Any,
    vllm_service_outstanding: Any,
    vllm_service_load_lock: Any,
    vllm_job_queues: Any,
    vllm_adapter_root: str,
    vllm_adapter_roots: Any,
    vllm_adapter_staging_slots: Any,
    vllm_service_active: Any,
    vllm_service_ready: Any,
    abort_event: Any,
    end_time: float,
    expert_state: dict[tuple[int, str], Any] | None = None,
) -> dict[str, Any]:
    from arc_vllm_adapter import (
        acquire_adapter_staging_root,
        make_adapter_name,
        remove_job_adapter_artifact,
    )
    from arc_vllm_router import (
        NoReadyVllmServiceError,
        release_service_task,
        reserve_least_loaded_service,
    )
    from diffusion_gemma_ttt_lora import export_vllm_peft_adapter

    wait_started = time.perf_counter()
    if not _acquire_pending_slot(
        vllm_pending_adapter_slots,
        abort_event=abort_event,
        end_time=end_time,
    ):
        raise TimeoutError(
            f"stopped waiting for pending DiffusionGemma adapter slot: {task_key}"
        )
    pending_wait = time.perf_counter() - wait_started
    service_index = None
    while service_index is None and time.time() < float(end_time):
        if abort_event is not None and abort_event.is_set():
            break
        try:
            service_index, service_loads = reserve_least_loaded_service(
                vllm_service_outstanding,
                vllm_service_load_lock,
                preferred_service_index=rank % len(vllm_job_queues),
                service_active=vllm_service_active,
                service_ready=vllm_service_ready,
            )
        except NoReadyVllmServiceError:
            time.sleep(0.25)
    if service_index is None:
        vllm_pending_adapter_slots.release()
        raise TimeoutError(f"no ready DiffusionGemma vLLM service for {task_key}")

    staging_index = None
    publish_root = vllm_adapter_root
    try:
        if vllm_adapter_staging_slots:
            staging_index, publish_root = acquire_adapter_staging_root(
                vllm_adapter_roots,
                vllm_adapter_staging_slots,
                abort_event=abort_event,
                end_time=end_time,
            )
        adapter_name = make_adapter_name(run_id, adapter_id, task_key)
        final_path = Path(publish_root) / adapter_name
        temporary_path = Path(publish_root) / (
            f".{adapter_name}.tmp-{os.getpid()}-{time.time_ns()}"
        )
        export_started = time.perf_counter()
        manifest = export_vllm_peft_adapter(
            model,
            temporary_path,
            metadata={
                "task_key": task_key,
                "trainer_rank": rank,
                "training_seconds": training_seconds,
                "training_loss": training_loss,
            },
            expert_state=expert_state,
        )
        os.replace(temporary_path, final_path)
        export_seconds = time.perf_counter() - export_started
        job = {
            "kind": "adapter",
            "task_key": task_key,
            "adapter_id": int(adapter_id),
            "adapter_name": adapter_name,
            "adapter_path": str(final_path),
            "adapter_root": str(publish_root),
            "adapter_staging_index": staging_index,
            "published_at": time.time(),
            "trainer_rank": int(rank),
            "training_seconds": float(training_seconds),
            "training_loss": training_loss,
            "export_seconds": export_seconds,
            "tensor_bytes": int(manifest["bytes"]),
            "service_index": int(service_index),
            "service_loads_after_reservation": list(service_loads),
        }
        vllm_job_queues[service_index].put(job)
        return job
    except BaseException:
        if "temporary_path" in locals() and temporary_path.exists():
            shutil.rmtree(temporary_path, ignore_errors=True)
        if "job" in locals() and Path(job["adapter_path"]).exists():
            remove_job_adapter_artifact(
                job,
                vllm_adapter_root,
                vllm_adapter_staging_slots,
            )
        elif "final_path" in locals() and final_path.exists():
            shutil.rmtree(final_path, ignore_errors=True)
            if staging_index is not None:
                vllm_adapter_staging_slots[staging_index].release()
        elif staging_index is not None:
            vllm_adapter_staging_slots[staging_index].release()
        release_service_task(
            service_index,
            vllm_service_outstanding,
            vllm_service_load_lock,
        )
        vllm_pending_adapter_slots.release()
        raise


def diffusion_gemma_trainer_worker(
    rank: int,
    task_queue: Any,
    end_time: float,
    *,
    keep_training_state: bool = False,
    vllm_job_queues: Any,
    vllm_pending_adapter_slots: Any,
    vllm_service_outstanding: Any,
    vllm_service_load_lock: Any,
    vllm_adapter_root: str,
    vllm_adapter_roots: Any,
    vllm_adapter_staging_slots: Any,
    vllm_run_id: str,
    vllm_adapter_ids: dict[str, int],
    vllm_service_active: Any,
    vllm_service_ready: Any,
    vllm_published_adapter_count: Any = None,
    vllm_ramp_ready_event: Any = None,
    vllm_ramp_after_adapters: int = 0,
    vllm_ramp_service_limit: int | None = None,
    abort_event: Any = None,
    **options: Any,
) -> int | None:
    """Persistent one-GPU DiffusionGemma trainer matching ``arc_solver.worker``."""
    if keep_training_state:
        raise ValueError(
            "DiffusionGemma always resets a fresh task LoRA; "
            "--keep-training-state is unsupported"
        )
    if not options.get("vllm_final_inference", False):
        raise ValueError("DiffusionGemma TTT currently requires final vLLM inference")

    from arc_loader import ArcDataset
    from run_config import resolve_challenge_path
    from diffusion_gemma_ttt_data import (
        build_ttt_conversation_rows,
        stable_task_seed,
    )
    from diffusion_gemma_ttt_lora import (
        gather_expert_parallel_slot_state,
        install_expert_parallel_gradient_clipping,
        reset_task_adapter,
    )
    from pre_sft_diffusion_gemma import load_arc_vocab_contract

    model_path = str(options["diffusion_gemma_model_path"])
    vocab = load_arc_vocab_contract(model_path)
    lora_rank = int(options.get("lora_r", 16))
    checkpointing_profile = os.getenv(
        "ARC_DIFFUSION_GEMMA_CHECKPOINTING_PROFILE", "legacy"
    ).strip().lower()
    validate_diffusion_gemma_checkpointing_profile(
        checkpointing_profile, _read_config(model_path), lora_rank, options
    )
    if lora_rank > 64 and checkpointing_profile != "l4-exp48":
        raise ValueError(
            "DiffusionGemma ranks above 64 require the opt-in "
            "ARC_DIFFUSION_GEMMA_CHECKPOINTING_PROFILE=l4-exp48 profile"
        )
    rank_memory = configure_diffusion_gemma_rank_memory(
        lora_rank, profile=checkpointing_profile
    )
    muon_ns_chunk_size = int(rank_memory.get(
        "muon_ns_chunk_size", options.get("muon_ns_chunk_size", 1)
    ))
    pair_fsdp = bool(options.get("diffusion_gemma_pair_fsdp", False))
    expert_parallel_size = int(
        options.get("diffusion_gemma_expert_parallel_size", 1)
    )
    expert_parallel = expert_parallel_size > 1
    if pair_fsdp and expert_parallel:
        raise ValueError(
            "DiffusionGemma pair-FSDP and expert parallelism are mutually exclusive"
        )
    pair_group = None
    expert_parallel_group = None
    pair_round = 0
    if pair_fsdp:
        if rank not in {0, 1}:
            raise ValueError(f"pair-FSDP trainer rank must be 0 or 1, got {rank}")
        pair_group = _initialize_pair_fsdp(
            rank,
            int(options["diffusion_gemma_pair_fsdp_port"]),
        )
        # The original 32,768-routed-row tile exceeded the L4 backward peak
        # after FSDP all-gathers. 8,192 crossed the measured rank-16 memory
        # gate. Respect an explicit experiment override, but make the audited
        # low-memory value the paired default before expert modules are built.
        os.environ.setdefault(
            "REAM_DIFFUSION_GEMMA_GROUPED_TOKEN_CHUNK_SIZE", "8192"
        )
    elif expert_parallel:
        if not 0 <= rank < expert_parallel_size:
            raise ValueError(
                f"EP trainer rank must be in [0, {expert_parallel_size}), got {rank}"
            )
        expert_parallel_group = _initialize_expert_parallel(
            rank,
            expert_parallel_size,
            int(options["diffusion_gemma_expert_parallel_port"]),
        )
        os.environ.setdefault(
            "REAM_DIFFUSION_GEMMA_EP_ROUTE_CHUNK_SIZE", "32768"
        )
    runtime = _load_training_model(
        model_path,
        rank=lora_rank,
        alpha=float(options.get("lora_alpha", lora_rank)),
        ream_root=str(options["diffusion_gemma_ream_root"]),
        dynamic_backward=bool(options.get("fp8_dynamic_backward", True)),
        split_accumulator=bool(
            options.get("fp8_dynamic_backward_split_accumulator", False)
        ),
        token_chunk_size=int(
            options.get("diffusion_gemma_lora_token_chunk_size", 1024)
        ),
        lora_experts_backend=str(
            options.get("diffusion_gemma_lora_experts_backend", "loop")
        ),
        decoder_checkpointing_percentage=options.get(
            "diffusion_gemma_decoder_checkpointing_percentage", "auto"
        ),
        encoder_checkpointing_percentage=options.get(
            "diffusion_gemma_encoder_checkpointing_percentage", "auto"
        ),
        kernel_backend=str(
            options.get("diffusion_gemma_kernel_backend", "liger-rms")
        ),
        encoder_attention_backend=str(
            options.get("diffusion_gemma_encoder_attention_backend", "hybrid-flex")
        ),
        decoder_attention_backend=str(
            options.get(
                "diffusion_gemma_decoder_attention_backend", "flash-attention-2"
            )
        ),
        expert_parallel_group=expert_parallel_group,
    )
    model = runtime["model"]
    pair_fsdp_stats = None
    if pair_fsdp:
        from ream.diffusion_gemma_expert_fsdp import (
            shard_frozen_expert_bases_fsdp,
        )

        model, pair_fsdp_stats = shard_frozen_expert_bases_fsdp(
            model,
            process_group=pair_group,
            device=__import__("torch").device("cuda", 0),
        )
        runtime["model"] = model
        gc.collect()
        __import__("torch").cuda.empty_cache()
    print(
        f"[Rank {rank}] DiffusionGemma frozen-FP8 trainer ready: "
        f"rank={lora_rank}, conversion={runtime['conversion']}, "
        f"self_conditioning={runtime['self_conditioning']}, "
        f"decoder_gc={len(runtime['decoder_checkpoints'])}, "
        f"encoder_gc={len(runtime['encoder_checkpoints'])}, "
        f"rank_memory={rank_memory}, "
        f"pair_fsdp={pair_fsdp_stats}, "
        f"expert_parallel={runtime['conversion'].get('expert_parallel_stats')}",
        flush=True,
    )
    rerun_mode = os.getenv("KAGGLE_IS_COMPETITION_RERUN")
    challenge = ArcDataset.from_file(resolve_challenge_path(rerun_mode))
    adaptive_service_index = None
    while time.time() < float(end_time):
        if abort_event is not None and abort_event.is_set():
            break
        task_key = task_queue.get()
        if pair_fsdp or expert_parallel:
            pair_tasks = _pair_task_state(
                task_key is not None,
                __import__("torch").device("cuda", 0),
            )
            expected_task_lanes = 2 if pair_fsdp else expert_parallel_size
            if pair_tasks not in {0, expected_task_lanes}:
                raise RuntimeError(
                    "DiffusionGemma synchronized task queue split a task group "
                    f"across a stop marker; expected {expected_task_lanes} lanes"
                )
        else:
            pair_tasks = int(task_key is not None)
        if pair_tasks == 0:
            break
        task_seed = stable_task_seed(task_key, int(options.get("seed", 42)))
        expert_parallel_task_seeds = (
            _collective_task_seeds(
                task_seed,
                expert_parallel_size,
                __import__("torch").device("cuda", 0),
            )
            if expert_parallel
            else None
        )
        reset_stats = reset_task_adapter(
            model,
            task_seed,
            expert_parallel_task_seeds=expert_parallel_task_seeds,
        )
        rows = build_ttt_conversation_rows(
            task_key,
            challenge.queries[task_key],
            instruction_token_ids=vocab["instruction_token_ids"],
            color_augmentations=int(
                options.get("diffusion_gemma_color_augmentations", 16)
            ),
            max_seq_length=int(options.get("max_seq_length", 12288)),
            seed=int(options.get("seed", 42)),
        )
        # Preserve the calibrated one-GPU policy's historical prompt-width
        # metric. Pair FSDP separately needs the complete collated width to
        # make routed-expert collective counts identical across ranks.
        max_train_tokens = max(len(row["input_ids"]) for row in rows)
        physical_train_tokens = _maximum_row_physical_tokens(rows)
        pair_max_train_tokens = (
            _pair_maximum(
                physical_train_tokens,
                __import__("torch").device("cuda", 0),
            )
            if pair_fsdp or expert_parallel
            else max_train_tokens
        )
        expert_parallel_layers = (
            _configure_expert_parallel_route_envelope(
                model, pair_max_train_tokens
            )
            if expert_parallel
            else 0
        )
        policy_train_tokens = diffusion_gemma_checkpointing_token_count(
            pair_max_train_tokens,
            physical_train_tokens,
            profile=checkpointing_profile,
        )
        decoder_gc, encoder_gc = resolve_diffusion_gemma_task_checkpointing(
            policy_train_tokens,
            runtime["decoder_checkpointing_policy"],
            runtime["encoder_checkpointing_policy"],
            lora_rank,
            profile=checkpointing_profile,
        )
        encoder_input_offload = resolve_diffusion_gemma_checkpoint_input_offload(
            policy_train_tokens,
            lora_rank,
            pair_fsdp=pair_fsdp,
            force=bool(options.get("attention_checkpoint_input_offload", False)),
            profile=checkpointing_profile,
        )
        decoder_indices = runtime["configure_decoder_checkpointing"](
            model,
            decoder_gc,
            preserve_rng_state=False,
            input_offload=bool(pair_fsdp and encoder_input_offload),
        )
        encoder_indices = runtime["configure_encoder_checkpointing"](
            model,
            encoder_gc,
            preserve_rng_state=False,
            selective_save=False,
            input_offload=encoder_input_offload,
        )
        print(
            f"[Rank {rank}] DiffusionGemma TTT policy for {task_key}: "
            f"max_train_tokens={max_train_tokens}, "
            f"physical_train_tokens={physical_train_tokens}, "
            f"pair_max_train_tokens={pair_max_train_tokens}, "
            f"checkpointing_profile={checkpointing_profile}, "
            f"policy_train_tokens={policy_train_tokens}, "
            f"expert_parallel_layers={expert_parallel_layers}, "
            f"decoder_gc={decoder_gc:g}%/{len(decoder_indices)}, "
            f"encoder_gc={encoder_gc:g}%/{len(encoder_indices)}, "
            f"decoder_checkpoint_input_offload="
            f"{bool(pair_fsdp and encoder_input_offload)}, "
            f"encoder_checkpoint_input_offload={encoder_input_offload}",
            flush=True,
        )
        expected_steps = int(options.get("diffusion_gemma_ttt_steps", 128))
        if len(rows) != expected_steps:
            raise RuntimeError(
                f"DiffusionGemma TTT rows/steps mismatch for {task_key}: "
                f"rows={len(rows)}, steps={expected_steps}"
            )
        __import__("torch").cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        trainer = _make_task_trainer(
            runtime,
            rows,
            task_seed=task_seed,
            model_path=model_path,
            max_seq_length=int(options.get("max_seq_length", 12288)),
            steps=expected_steps,
            learning_rate=float(
                options.get("diffusion_gemma_learning_rate", 5e-5)
            ),
            optimizer=str(options.get("diffusion_gemma_optimizer", "muon8bit")),
            optimizer_state_offload=bool(
                options.get("diffusion_gemma_optimizer_state_offload", True)
            ),
            optimizer_state_offload_pin_memory=bool(
                options.get(
                    "diffusion_gemma_optimizer_state_offload_pin_memory", True
                )
            ),
            max_grad_norm=float(options.get("max_grad_norm", 1.0)),
            arc_output_vocab_size=int(vocab["arc_output_vocab_size"]),
            muon_adjust_lr=str(options.get("muon_adjust_lr", "rms_norm")),
            muon_momentum=float(options.get("muon_momentum", 0.95)),
            muon_nesterov=bool(options.get("muon_nesterov", True)),
            muon_ns_chunk_size=muon_ns_chunk_size,
            muon_min_8bit_size=int(options.get("muon_min_8bit_size", 4096)),
            self_conditioning_seed=(
                int(options.get("seed", 42)) + 1_000_003 * pair_round
                if pair_fsdp or expert_parallel
                else None
            ),
            pair_physical_tokens=(
                pair_max_train_tokens if pair_fsdp else None
            ),
        )
        if expert_parallel:
            install_expert_parallel_gradient_clipping(
                trainer,
                model,
                local_slot=rank,
                process_group=expert_parallel_group,
            )
        if pair_fsdp:
            accelerator = trainer.accelerator
            print(
                f"[Rank {rank}] pair-FSDP Trainer boundary for {task_key}: "
                f"distributed_type={accelerator.state.distributed_type}, "
                f"num_processes={accelerator.num_processes}, "
                f"process_index={accelerator.process_index}, "
                f"is_fsdp_enabled={trainer.is_fsdp_enabled}",
                flush=True,
            )
        output = trainer.train()
        training_seconds = time.perf_counter() - started
        training_loss = float(getattr(output, "training_loss", math.nan))
        print(
            f"[Rank {rank}] DiffusionGemma TTT complete for {task_key}: "
            f"steps={getattr(output, 'global_step', expected_steps)}, "
            f"loss={training_loss:.6g}, wall={training_seconds:.3f}s, "
            f"reset={reset_stats}, peak={__import__('torch').cuda.max_memory_allocated() / 1024**2:.0f}MiB",
            flush=True,
        )
        if pair_fsdp or expert_parallel:
            __import__("torch").distributed.barrier(
                group=(pair_group if pair_fsdp else expert_parallel_group),
                device_ids=[0],
            )
            pair_round += 1
        expert_state = (
            gather_expert_parallel_slot_state(
                model,
                slot=rank,
                process_group=expert_parallel_group,
            )
            if expert_parallel
            else None
        )
        job = _publish_task_adapter(
            model,
            rank=rank,
            task_key=task_key,
            adapter_id=int(vllm_adapter_ids[task_key]),
            run_id=vllm_run_id,
            training_seconds=training_seconds,
            training_loss=training_loss,
            vllm_pending_adapter_slots=vllm_pending_adapter_slots,
            vllm_service_outstanding=vllm_service_outstanding,
            vllm_service_load_lock=vllm_service_load_lock,
            vllm_job_queues=vllm_job_queues,
            vllm_adapter_root=vllm_adapter_root,
            vllm_adapter_roots=vllm_adapter_roots,
            vllm_adapter_staging_slots=vllm_adapter_staging_slots,
            vllm_service_active=vllm_service_active,
            vllm_service_ready=vllm_service_ready,
            abort_event=abort_event,
            end_time=end_time,
            expert_state=expert_state,
        )
        print(
            f"[Rank {rank}] published {job['adapter_name']} "
            f"({job['tensor_bytes'] / 1024**2:.1f}MiB) to vLLM service "
            f"{job['service_index']}; continuing without waiting for inference",
            flush=True,
        )
        # Release optimizer and dataloader state while retaining the frozen FP8
        # base and the just-exported LoRA values until the next deterministic
        # reset.
        del trainer, output, rows, expert_state
        gc.collect()
        __import__("torch").cuda.empty_cache()

        if int(vllm_ramp_after_adapters) > 0:
            from arc_vllm_router import record_adapter_publication_and_maybe_activate

            (
                _published,
                adaptive_service_index,
                _loads,
                _active,
            ) = record_adapter_publication_and_maybe_activate(
                vllm_published_adapter_count,
                int(vllm_ramp_after_adapters),
                vllm_service_outstanding,
                vllm_service_active,
                vllm_service_load_lock,
                vllm_ramp_ready_event,
                service_limit=vllm_ramp_service_limit,
            )
            if adaptive_service_index is not None:
                break
    del model, runtime
    gc.collect()
    __import__("torch").cuda.empty_cache()
    if (pair_fsdp or expert_parallel) and __import__("torch").distributed.is_initialized():
        __import__("torch").distributed.destroy_process_group()
    return adaptive_service_index


def _load_vllm_engine(
    model_path: str,
    options: dict[str, Any],
    startup_lock: Any,
    *,
    log_prefix: str,
):
    """Construct native online-FP8 DiffusionGemma with BF16 LoRA slots."""
    from arc_vllm_diffusion_gemma import register

    # Register in this frontend even if vLLM was imported before entry-point
    # loading. EngineCore independently loads the allowlisted entry point.
    register()
    from vllm import LLM, SamplingParams

    lock_started = time.perf_counter()
    acquired = startup_lock.acquire(
        timeout=float(options.get("vllm_startup_timeout_seconds", 1200.0))
    )
    if not acquired:
        raise TimeoutError(f"{log_prefix} timed out waiting for startup lock")
    try:
        engine = LLM(
            model=model_path,
            tokenizer=model_path,
            trust_remote_code=False,
            dtype="bfloat16",
            quantization="fp8",
            max_model_len=int(options.get("max_seq_length", 12288)),
            gpu_memory_utilization=float(
                options.get("vllm_gpu_memory_utilization", 0.89)
            ),
            max_num_seqs=min(int(options.get("vllm_max_num_seqs", 8)), 8),
            max_num_batched_tokens=max(
                int(options.get("max_seq_length", 12288)),
                int(options.get("vllm_max_num_batched_tokens", 16384)),
            ),
            enable_prefix_caching=False,
            enable_chunked_prefill=False,
            enforce_eager=bool(options.get("vllm_enforce_eager", False)),
            enable_lora=True,
            max_loras=int(options.get("vllm_max_loras", 2)),
            max_cpu_loras=int(options.get("vllm_max_loras", 2)),
            max_lora_rank=int(options.get("lora_r", 16)),
            lora_dtype="bfloat16",
            disable_log_stats=True,
        )
    finally:
        startup_lock.release()
    params = SamplingParams(
        max_tokens=int(options.get("diffusion_gemma_max_output_tokens", 1024)),
        detokenize=False,
    )
    print(
        f"{log_prefix} native online-FP8 engine ready; startup-lock+load="
        f"{time.perf_counter() - lock_started:.3f}s",
        flush=True,
    )
    return engine, params


def _shutdown_vllm_engine(engine: Any) -> None:
    core = getattr(getattr(engine, "llm_engine", None), "engine_core", None)
    shutdown = getattr(core, "shutdown", None)
    if callable(shutdown):
        try:
            shutdown(timeout=10.0)
        except TypeError:
            shutdown()


def _write_task_outputs(
    task_key: str,
    views: list[Any],
    outputs: list[Any],
    *,
    output_dir: str,
) -> dict[str, Any]:
    """Invert every valid view and serialize vote-compatible ARC records."""
    import numpy as np
    from arc_loader import ArcDataset, is_valid_solution
    from diffusion_gemma_ttt_data import parse_generated_grid

    grouped: dict[str, list[dict[str, Any]]] = {}
    invalid: dict[str, int] = {}
    if len(views) != len(outputs):
        raise RuntimeError(
            f"DiffusionGemma vLLM output count mismatch: {len(outputs)}/{len(views)}"
        )
    for view, request_output in zip(views, outputs):
        samples = getattr(request_output, "outputs", None)
        if not samples:
            invalid[view.split_key] = invalid.get(view.split_key, 0) + 1
            continue
        token_ids = tuple(int(value) for value in samples[0].token_ids)
        grid = parse_generated_grid(token_ids)
        if grid is None:
            invalid[view.split_key] = invalid.get(view.split_key, 0) + 1
            continue
        restored = ArcDataset.invert_mod(grid, view.view_key)
        if not is_valid_solution(restored):
            invalid[view.split_key] = invalid.get(view.split_key, 0) + 1
            continue
        # One record per view intentionally retains duplicate solutions. The
        # standard KGMon selector therefore becomes a consensus vote; no fake
        # autoregressive NLL is mixed into the diffusion score.
        grouped.setdefault(view.split_key, []).append(
            {
                "beam_score": 0.0,
                "score_aug": [0.0],
                "score_aug_count": 1,
                "score_aug_seed": None,
                "solution": np.asarray(restored, dtype=np.int64),
            }
        )
    written = 0
    for split_key, records in grouped.items():
        path = Path(output_dir) / f"{split_key}.diffusion_gemma"
        with bz2.BZ2File(path, "w") as handle:
            pickle.dump(records, handle)
        written += 1
    return {
        "task_key": task_key,
        "views": len(views),
        "valid": sum(map(len, grouped.values())),
        "invalid": sum(invalid.values()),
        "written_splits": written,
        "records_per_split": {key: len(value) for key, value in grouped.items()},
    }


def _rebalance_pending_jobs(
    *,
    pending_jobs: deque[dict[str, Any]],
    service_index: int,
    initial_service_count: int,
    all_job_queues: Any,
    service_outstanding: Any,
    service_load_lock: Any,
    rebalance_count: Any,
    rebalance_complete_event: Any,
) -> dict[str, Any]:
    from arc_vllm_router import rebalance_pending_service_task

    post_indices = tuple(range(initial_service_count, len(all_job_queues)))
    retained: deque[dict[str, Any]] = deque()
    migrated: dict[int, int] = {}
    original = len(pending_jobs)
    while pending_jobs:
        job = pending_jobs.popleft()
        target, _loads = rebalance_pending_service_task(
            service_index,
            (service_index, *post_indices),
            service_outstanding,
            service_load_lock,
        )
        if target == service_index:
            retained.append(job)
        else:
            moved = dict(job)
            moved["service_index"] = target
            moved["post_training_rebalanced"] = True
            all_job_queues[target].put(moved)
            migrated[target] = migrated.get(target, 0) + 1
    pending_jobs.extend(retained)
    for target in post_indices:
        all_job_queues[target].put(
            {
                "kind": "post_training_rebalance_done",
                "source_service_index": service_index,
                "service_index": target,
            }
        )
    with rebalance_count.get_lock():
        rebalance_count.value += 1
        completed = int(rebalance_count.value)
        if completed == initial_service_count:
            rebalance_complete_event.set()
    return {
        "original": original,
        "retained": len(retained),
        "migrated": migrated,
        "completed": completed,
    }


def run_diffusion_gemma_vllm_worker(
    *,
    job_queue: Any,
    rollout_response_queues: Any,
    pending_adapter_slots: Any,
    service_index: int,
    service_outstanding: Any,
    service_load_lock: Any,
    startup_lock: Any,
    trainer_count: int,
    end_time: float,
    model_path: str,
    adapter_root: str,
    options: dict[str, Any],
    abort_event: Any = None,
    all_job_queues: Any = None,
    service_active: Any = None,
    service_ready: Any = None,
    initial_service_count: int = 1,
    post_training_scale: bool = False,
    post_training_rebalance_count: Any = None,
    post_training_rebalance_event: Any = None,
    adapter_staging_slots: Any = None,
    **_unused: Any,
) -> None:
    """Consume streamed DiffusionGemma adapters using native vLLM."""
    del rollout_response_queues
    from arc_loader import ArcDataset
    from arc_vllm_adapter import remove_job_adapter_artifact
    from arc_vllm_router import release_service_task
    from diffusion_gemma_ttt_data import build_inference_views
    from pre_sft_diffusion_gemma import load_arc_vocab_contract
    from run_config import resolve_challenge_path
    from vllm.lora.request import LoRARequest

    log_prefix = f"[DiffusionGemma vLLM {service_index}]"
    if service_ready is not None:
        with service_load_lock:
            service_ready[service_index] = 0
    engine = None
    try:
        engine, sampling_params = _load_vllm_engine(
            model_path, options, startup_lock, log_prefix=log_prefix
        )
        if service_ready is not None:
            with service_load_lock:
                service_ready[service_index] = 1
        rerun_mode = os.getenv("KAGGLE_IS_COMPETITION_RERUN")
        challenge = ArcDataset.from_file(resolve_challenge_path(rerun_mode))
        vocab = load_arc_vocab_contract(model_path)
        output_dir = os.getenv("ARC_INFERENCE_OUTPUT_DIR", "/kaggle/inference_outputs")
        os.makedirs(output_dir, exist_ok=True)
        pending: deque[dict[str, Any]] = deque()
        trainers_done = 0
        rebalance_sources: set[int] = set()
        rebalanced = not post_training_scale or service_index >= initial_service_count

        def receive(message: dict[str, Any]) -> None:
            nonlocal trainers_done
            kind = message.get("kind")
            if kind == "adapter":
                if int(message.get("service_index", service_index)) != service_index:
                    raise RuntimeError(f"{log_prefix} received misrouted adapter")
                pending.append(message)
            elif kind == "trainer_done":
                trainers_done += 1
            elif kind == "post_training_rebalance_done":
                rebalance_sources.add(int(message["source_service_index"]))
            else:
                raise RuntimeError(f"{log_prefix} unknown message: {message}")

        def drain() -> None:
            while True:
                try:
                    receive(job_queue.get_nowait())
                except Empty:
                    return

        def upstream_complete() -> bool:
            if trainers_done < trainer_count:
                return False
            if not post_training_scale or service_index < initial_service_count:
                return True
            return len(rebalance_sources) >= initial_service_count

        while time.time() < float(end_time):
            if abort_event is not None and abort_event.is_set():
                break
            drain()
            if (
                not rebalanced
                and service_index < initial_service_count
                and trainers_done >= trainer_count
            ):
                summary = _rebalance_pending_jobs(
                    pending_jobs=pending,
                    service_index=service_index,
                    initial_service_count=initial_service_count,
                    all_job_queues=all_job_queues,
                    service_outstanding=service_outstanding,
                    service_load_lock=service_load_lock,
                    rebalance_count=post_training_rebalance_count,
                    rebalance_complete_event=post_training_rebalance_event,
                )
                rebalanced = True
                print(f"{log_prefix} post-training rebalance: {summary}", flush=True)
            if not pending:
                if upstream_complete():
                    break
                try:
                    receive(job_queue.get(timeout=0.25))
                except Empty:
                    continue
                continue

            job = pending.popleft()
            request = LoRARequest(
                lora_name=job["adapter_name"],
                lora_int_id=int(job["adapter_id"]),
                lora_path=job["adapter_path"],
            )
            loaded = engine.llm_engine.add_lora(request)
            loaded_ids = set(engine.llm_engine.list_loras())
            if not loaded and int(job["adapter_id"]) not in loaded_ids:
                raise RuntimeError(
                    f"{log_prefix} failed to load {job['adapter_name']}"
                )
            # It is no longer queued; retain the staged files until inference
            # and unload complete, but let a trainer publish the next adapter.
            pending_adapter_slots.release()
            task_key = str(job["task_key"])
            views = build_inference_views(
                task_key,
                challenge.queries[task_key],
                instruction_token_ids=vocab["instruction_token_ids"],
                views=int(options.get("diffusion_gemma_inference_views", 8)),
                max_prompt_tokens=(
                    int(options.get("max_seq_length", 12288))
                    - int(options.get("diffusion_gemma_max_output_tokens", 1024))
                ),
                seed=int(options.get("seed", 42)),
            )
            prompts = [
                {"prompt_token_ids": list(view.prompt_token_ids)} for view in views
            ]
            started = time.perf_counter()
            outputs = engine.generate(
                prompts,
                sampling_params,
                lora_request=request,
                use_tqdm=False,
            )
            summary = _write_task_outputs(
                task_key,
                views,
                outputs,
                output_dir=output_dir,
            )
            removed = engine.llm_engine.remove_lora(int(job["adapter_id"]))
            if not removed and int(job["adapter_id"]) in set(
                engine.llm_engine.list_loras()
            ):
                raise RuntimeError(
                    f"{log_prefix} failed to unload {job['adapter_name']}"
                )
            remove_job_adapter_artifact(
                job,
                adapter_root,
                adapter_staging_slots,
            )
            loads = release_service_task(
                service_index,
                service_outstanding,
                service_load_lock,
            )
            print(
                f"{log_prefix} completed {task_key} in "
                f"{time.perf_counter() - started:.3f}s: {summary}; "
                f"outstanding={list(loads)}",
                flush=True,
            )
        # Do not silently strand published work at a deadline.
        drain()
        while pending:
            job = pending.popleft()
            pending_adapter_slots.release()
            remove_job_adapter_artifact(job, adapter_root, adapter_staging_slots)
            release_service_task(
                service_index, service_outstanding, service_load_lock
            )
    finally:
        if service_ready is not None:
            with service_load_lock:
                service_ready[service_index] = 0
        if engine is not None:
            _shutdown_vllm_engine(engine)


__all__ = [
    "DIFFUSION_GEMMA_ARCHITECTURE",
    "DIFFUSION_GEMMA_PLUGIN",
    "diffusion_gemma_trainer_worker",
    "model_declares_diffusion_gemma",
    "preflight_diffusion_gemma_plugin",
    "prepare_diffusion_gemma_runtime_model",
    "remove_created_runtime_model",
    "run_diffusion_gemma_vllm_worker",
]
