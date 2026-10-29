import hashlib
import math
import os
import re


DEFAULT_DATA_ROOT = "/kaggle/input/competitions/arc-prize-2026-arc-agi-2"
EVALUATION_CHALLENGE_FILENAME = "arc-agi_evaluation_challenges.json"
TEST_CHALLENGE_FILENAME = "arc-agi_test_challenges.json"
EVALUATION_SOLUTION_FILENAME = "arc-agi_evaluation_solutions.json"
EVALUATION_CHALLENGE_PATH = (
    f"{DEFAULT_DATA_ROOT}/{EVALUATION_CHALLENGE_FILENAME}"
)
TEST_CHALLENGE_PATH = (
    f"{DEFAULT_DATA_ROOT}/{TEST_CHALLENGE_FILENAME}"
)
PRIMARY_MODEL_PATH = (
    "/kaggle/input/qwen3_4b_grids15_sft139/transformers/bfloat16/1"
)
FALLBACK_MODEL_PATH = (
    "/kaggle/input/models/sorokin/qwen3_4b_grids15_sft139/"
    "transformers/bfloat16/1"
)
# Every checkpointing tuple is ``(inclusive_lower_token_bound, percentage)``.
# The selected percentage remains active until the next tuple.  Lower bounds
# make a row such as ``(4096, 50.0)`` read directly as
# ``4096 <= effective_tokens < next_bound``.
AUTO_GRADIENT_CHECKPOINTING_TIERS = (
    (0, 0.0),
    (2048, 30.0),
    (4096, 50.0),
    (5120, 55.0),
    (6144, 60.0),
    (7168, 65.0),
    (8192, 75.0),
    # GRPO can process several prompt/completion sequences in one physical
    # micro-batch, so its effective workload can exceed the model's 8,192-token
    # per-sequence context. After the SFT/GRPO model-reload boundary, reserve a
    # 75% tier for the measured 9,048-token micro-batch, increase gradually
    # above 10,240 tokens, and use full checkpointing from 16,384 onward.
    (10240, 80.0),
    (15360, 85.0),
    (16384, 100.0),
)
QWEN3_4B_R128_AUTO_GRADIENT_CHECKPOINTING_TIERS = (
    # BF16 Qwen3-4B, rank <= 128, physical batch one, on a 22 GiB L4.
    # 2026-09-12 retune: reduce only bands with measured headroom; retain
    # the tight 4--8K settings and unvalidated terminal bands. New bounds
    # use 512-token steps; repeated percentages make each band explicit.
    # Reduced bands were screened at/above their upper bounds using real
    # filled ARC rows. See docs/QWEN3_R128_GC_RETUNE_20260912.md for evidence
    # and runtime limits; this is not a global 21 GiB memory guarantee.
    (0, 0.0),
    (512, 0.0),
    (1024, 0.0),
    (1536, 0.0),
    (2048, 0.0),
    (2560, 0.0),
    (3072, 10.0),
    (3584, 20.0),
    (4096, 30.0),
    (4608, 30.0),
    (5120, 30.0),
    (5632, 30.0),
    (6144, 40.0),
    (6656, 40.0),
    (7168, 50.0),
    (7680, 55.0),
    (8192, 55.0),
    # Keep this legacy safety boundary: rounding down would increase GC
    # in an unchanged band; rounding up would extend the tight 55% band.
    (8448, 65.0),
    (8704, 65.0),
    (9216, 70.0),
    (9728, 70.0),
    (10240, 70.0),
    (10752, 70.0),
    (11264, 75.0),
    (11776, 75.0),
    (12288, 75.0),
    (12800, 80.0),
    (13312, 80.0),
    (13824, 80.0),
    (14336, 80.0),
    (14848, 80.0),
    (15360, 85.0),
    (15872, 85.0),
    (16384, 100.0),
)
QWEN35_9B_AUTO_GRADIENT_CHECKPOINTING_TIERS = (
    # Qwen3.5-9B retains substantially larger hybrid GDN activations than the
    # calibrated Qwen3-4B model. These tiers target a 22 GiB L4 with BF16
    # rank-128 LoRA and physical batch one. The 7,464-token gate OOMed at 65%
    # and 75%, while 90% completed at 20,841 MiB; use full checkpointing for
    # the unmeasured final 512 tokens of the 8,192-token ARC range.
    (0, 0.0),
    (1280, 60.0),
    (2048, 75.0),
    (3072, 80.0),
    (4096, 85.0),
    (5120, 90.0),
    (7680, 100.0),
)
DEFAULT_GRADIENT_CHECKPOINTING_PROFILE = "qwen3_4b"
QWEN3_4B_R128_GRADIENT_CHECKPOINTING_PROFILE = "qwen3_4b_r128"
QWEN35_9B_GRADIENT_CHECKPOINTING_PROFILE = "qwen3_5_9b"
AUTO_GRADIENT_CHECKPOINTING_PROFILES = {
    DEFAULT_GRADIENT_CHECKPOINTING_PROFILE: AUTO_GRADIENT_CHECKPOINTING_TIERS,
    QWEN3_4B_R128_GRADIENT_CHECKPOINTING_PROFILE: (
        QWEN3_4B_R128_AUTO_GRADIENT_CHECKPOINTING_TIERS
    ),
    QWEN35_9B_GRADIENT_CHECKPOINTING_PROFILE: (
        QWEN35_9B_AUTO_GRADIENT_CHECKPOINTING_TIERS
    ),
}
GRADIENT_CHECKPOINTING_MODES = ("standard", "unsloth")
TASK_ORDER_MODES = ("input", "train-pairs", "train-pairs-desc")
LORA_METHODS = ("rslora", "lora", "nora", "nora-init", "bimi")
# vLLM's max_lora_rank is a buffer capacity, not the adapter's actual rank.
VLLM_LORA_RANK_CAPACITIES = (1, 8, 16, 32, 64, 128, 256, 320, 512)
VLLM_KV_CACHE_DTYPES = ("auto", "fp8", "fp8_e4m3", "fp8_e5m2")


def normalize_vllm_kv_cache_dtype(value):
    """Validate target KV precision independently of weights and task LoRAs."""
    dtype = str(value).strip().lower()
    if dtype not in VLLM_KV_CACHE_DTYPES:
        raise ValueError(
            f"vLLM KV cache dtype must be one of {VLLM_KV_CACHE_DTYPES}; got {value!r}"
        )
    return dtype


def normalize_vllm_search_limit_seconds(value):
    """Validate a finite positive wall-time limit for one threshold DFS group."""
    message = "vLLM search limit must be a finite positive number of seconds"
    try:
        seconds = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(message) from exc
    if not math.isfinite(seconds) or seconds <= 0.0:
        raise ValueError(message)
    return seconds


def normalize_vllm_kv_resident_fraction(value):
    """Accept a fixed per-task root budget or the opt-in ``auto`` policy."""
    if isinstance(value, str) and value.strip().lower() == "auto":
        return "auto"
    message = "KV-resident fraction must be 'auto' or a number in (0, 1]"
    try:
        fraction = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(message) from exc
    # This also rejects NaN and infinities.
    if not 0.0 < fraction <= 1.0:
        raise ValueError(message)
    return fraction


def resolve_vllm_kv_resident_fraction(value, *, max_loras):
    """Resolve against configured per-service slots, not transient occupancy.

    Auto keeps the one-adapter 0.95 budget, or shares 0.90 across multiple
    configured adapters. This sizes decode groups; it does not partition
    vLLM's cache or reserve space for generated tokens and rescoring.
    """
    fraction = normalize_vllm_kv_resident_fraction(value)
    if fraction != "auto":
        return fraction
    message = "auto KV-resident fraction requires a positive integer max_loras"
    try:
        count = int(max_loras)
        if count < 1 or count != float(max_loras):
            raise ValueError(message)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(message) from exc
    return 0.95 if count == 1 else 0.90 / count


def normalize_lora_init(value):
    """Keep initialization independent of LoRA's scaling/parameterization."""
    value = str(value).strip().lower()
    if value in {"default", "pissa"}:
        return value
    if re.fullmatch(r"pissa_niter_[0-9]+", value):
        return f"pissa_niter_{int(value.rsplit('_', 1)[1])}"
    raise ValueError(
        "--lora-init must be default, pissa, or pissa_niter_N "
        "with a nonnegative integer N"
    )


def is_pissa_init(value):
    return isinstance(value, str) and (
        value == "pissa" or re.fullmatch(r"pissa_niter_[0-9]+", value) is not None
    )


def normalize_pissa_base_mode(value):
    value = str(value).strip().lower()
    if value not in {"original", "shared-residual"}:
        raise ValueError("--pissa-base-mode must be original or shared-residual")
    return value


def lora_export_rank(rank, initialization="default", pissa_base_mode="original"):
    """PiSSA's update relative to the original base needs two rank-r factors."""
    rank = int(rank)
    if rank <= 0:
        raise ValueError("LoRA rank must be positive")
    mode = normalize_pissa_base_mode(pissa_base_mode)
    pissa = is_pissa_init(normalize_lora_init(initialization))
    if mode == "shared-residual" and not pissa:
        raise ValueError("shared-residual serving requires PiSSA initialization")
    return rank * (2 if pissa and mode == "original" else 1)


def lora_engine_max_rank(rank, initialization="default", pissa_base_mode="original"):
    """Round exported rank up for vLLM without changing adapter rank/scaling."""
    initialization = normalize_lora_init(initialization)
    exported = lora_export_rank(rank, initialization, pissa_base_mode)
    # Retain the existing PiSSA minimum reservation for small adapters.
    required = max(32, exported) if is_pissa_init(initialization) else exported
    for capacity in VLLM_LORA_RANK_CAPACITIES:
        if required <= capacity:
            return capacity
    raise ValueError(
        f"LoRA export rank {exported} exceeds the supported vLLM serving rank of 512"
    )


def validate_pissa_options(
    method, initialization, *, model_family="qwen", fp8_frozen_linear=False,
    quantization="none", task_rl=False, deepspeed=False, distributed=False,
    keep_training_state=False, inference_backend="vllm",
):
    """Fail before launching workers for unvalidated mutated-base workflows."""
    configured = resolve_lora_method(method, initialization)
    if not is_pissa_init(configured["init_lora_weights"]):
        return
    incompatible = []
    if model_family != "qwen":
        incompatible.append(f"model family {model_family}")
    if fp8_frozen_linear or str(quantization).lower() not in {"none", ""}:
        incompatible.append("FP8/quantized base weights")
    if task_rl:
        incompatible.append("task-time GRPO")
    if deepspeed:
        incompatible.append("DeepSpeed")
    if distributed:
        incompatible.append("distributed training")
    if keep_training_state:
        incompatible.append("--keep-training-state")
    if inference_backend == "sglang":
        incompatible.append("SGLang serving")
    if incompatible:
        raise ValueError(
            "PiSSA currently supports independent unquantized Qwen SFT "
            "with native/vLLM inference; incompatible with " + ", ".join(incompatible)
        )


def resolve_lora_method(method, initialization="default"):
    """Map the public task-adapter method to PEFT's orthogonal controls."""
    normalized = str(method).strip().lower()
    if normalized not in LORA_METHODS:
        supported = ", ".join(LORA_METHODS)
        raise ValueError(
            f"Unknown LoRA method {method!r}; expected one of: {supported}"
        )
    initialization = normalize_lora_init(initialization)
    if initialization != "default" and normalized not in {"lora", "rslora"}:
        raise ValueError("PiSSA initialization requires --lora-method lora or rslora")
    return {
        "method": normalized,
        "use_rslora": normalized == "rslora",
        "use_nora": (
            True
            if normalized == "nora"
            else "init"
            if normalized == "nora-init"
            else False
        ),
        "init_lora_weights": (
            initialization if initialization != "default"
            else "bimi" if normalized == "bimi" else True
        ),
    }


def lora_effective_scaling(method, rank, alpha):
    """Return PEFT's effective adapter multiplier for logging/tests."""
    rank = int(rank)
    alpha = float(alpha)
    if rank <= 0 or alpha <= 0:
        raise ValueError(
            f"LoRA rank and alpha must be positive; got r={rank}, alpha={alpha}"
        )
    configured = resolve_lora_method(method)
    denominator = rank**0.5 if configured["use_rslora"] else rank
    return alpha / denominator


def normalize_gradient_checkpointing_mode(value):
    """Normalize the public checkpoint-input storage policy."""
    mode = str(value).strip().lower()
    if mode not in GRADIENT_CHECKPOINTING_MODES:
        supported = ", ".join(GRADIENT_CHECKPOINTING_MODES)
        raise ValueError(
            f"Unknown gradient-checkpointing mode {value!r}; expected {supported}"
        )
    return mode


def resolve_gradient_checkpointing_value(enabled, mode):
    """Return the value expected by Unsloth's checkpointing APIs."""
    mode = normalize_gradient_checkpointing_mode(mode)
    if not enabled:
        return False
    return "unsloth" if mode == "unsloth" else True


def stable_task_seed(task_key, seed=0, namespace="arc", modulus=1024**2):
    """Derive a reproducible task-local seed without Python's salted hash."""
    if modulus <= 0:
        raise ValueError(f"modulus must be positive, got {modulus}")
    payload = f"{namespace}:{int(seed)}:{task_key}".encode("utf-8")
    value = int.from_bytes(
        hashlib.sha256(payload).digest()[:8],
        byteorder="big",
    )
    return value % modulus


def resolve_data_root():
    """Return the common ARC JSON directory, preserving the Kaggle default."""
    return os.path.expanduser(os.getenv("ARC_DATA_ROOT", DEFAULT_DATA_ROOT))


def resolve_data_path(filename):
    """Resolve one ARC input filename below the configured common root."""
    return os.path.join(resolve_data_root(), filename)


def resolve_data_challenge_path(rerun_mode):
    """Return the evaluation or hidden-test challenge under the common root."""
    filename = (
        TEST_CHALLENGE_FILENAME
        if rerun_mode
        else EVALUATION_CHALLENGE_FILENAME
    )
    return resolve_data_path(filename)


def resolve_challenge_path(rerun_mode):
    """Return an explicit challenge override or the common-root default."""
    override = os.getenv("ARC_CHALLENGE_PATH")
    if override:
        return override
    return resolve_data_challenge_path(rerun_mode)


def resolve_evaluation_solution_path():
    """Return an explicit evaluation-label path or the common-root default."""
    return os.getenv(
        "ARC_SOLUTION_PATH",
        resolve_data_path(EVALUATION_SOLUTION_FILENAME),
    )


def resolve_model_path():
    """Return the explicit ARC checkpoint or the first available default."""
    override = os.getenv("ARC_MODEL_PATH")
    if override:
        return override
    if os.path.exists(PRIMARY_MODEL_PATH):
        return PRIMARY_MODEL_PATH
    return FALLBACK_MODEL_PATH


def resolve_train_prediction_output_dir(inference_output_dir):
    """Return the isolated store for leave-one-out train predictions."""
    return os.getenv(
        "ARC_TRAIN_INFERENCE_OUTPUT_DIR",
        os.path.join(inference_output_dir, "train_predictions"),
    )


def select_auto_gradient_checkpointing_percentage(
    max_train_tokens,
    *,
    workload_multiplier=1,
    profile=DEFAULT_GRADIENT_CHECKPOINTING_PROFILE,
):
    """Choose the calibrated 22 GiB L4 tier for an activation workload."""
    max_train_tokens = int(max_train_tokens)
    workload_multiplier = int(workload_multiplier)
    if max_train_tokens < 0:
        raise ValueError("max_train_tokens must be non-negative")
    if workload_multiplier < 1:
        raise ValueError("workload_multiplier must be positive")
    try:
        tiers = AUTO_GRADIENT_CHECKPOINTING_PROFILES[str(profile)]
    except KeyError as exc:
        raise ValueError(
            f"Unknown gradient-checkpointing profile: {profile!r}"
        ) from exc
    effective_tokens = max_train_tokens * workload_multiplier
    selected_percentage = tiers[0][1]
    for token_lower_bound, percentage in tiers:
        if effective_tokens < token_lower_bound:
            break
        selected_percentage = percentage
    return selected_percentage


def resolve_task_gradient_checkpointing_profile(
    profile,
    *,
    lora_r=256,
    fp8_frozen_linear=False,
):
    """Select a task-SFT memory calibration without weakening GRPO."""
    if (
        profile == DEFAULT_GRADIENT_CHECKPOINTING_PROFILE
        and int(lora_r) <= 128
        and not fp8_frozen_linear
    ):
        return QWEN3_4B_R128_GRADIENT_CHECKPOINTING_PROFILE
    return profile


def select_task_auto_gradient_checkpointing_percentage(
    max_train_tokens,
    *,
    profile=DEFAULT_GRADIENT_CHECKPOINTING_PROFILE,
    lora_r=256,
    fp8_frozen_linear=False,
):
    """Choose task-SFT GC, including measured FP8 transient workspace."""
    profile = resolve_task_gradient_checkpointing_profile(
        profile,
        lora_r=lora_r,
        fp8_frozen_linear=fp8_frozen_linear,
    )
    percentage = select_auto_gradient_checkpointing_percentage(
        max_train_tokens,
        profile=profile,
    )
    if (
        profile == DEFAULT_GRADIENT_CHECKPOINTING_PROFILE
        and percentage > 85.0
    ):
        # The 100% terminal tier protects GRPO workloads at/above 16,384
        # effective tokens.  Ordinary rank-256 SFT retains the established
        # 85% ceiling at that exact context boundary.
        percentage = 85.0
    if (
        fp8_frozen_linear
        and profile == QWEN35_9B_GRADIENT_CHECKPOINTING_PROFILE
    ):
        # On Qwen3.5-9B/L4, TE dynamic dgrad needs enough workspace to
        # materialize/dequantize projections that do not use the fused LoRA
        # kernels. The 1,136-token gate OOMed at 0% and 25%; 50% completed at
        # 18,516 MiB. Longer existing tiers are already >= 60%.
        percentage = max(50.0, percentage)
    return percentage


def select_auto_gradient_checkpointing_profile(model_config):
    """Select a model-aware L4 activation-memory calibration profile."""
    model_type = str(getattr(model_config, "model_type", ""))
    hidden_size = int(getattr(model_config, "hidden_size", 0) or 0)
    num_hidden_layers = int(
        getattr(model_config, "num_hidden_layers", 0) or 0
    )
    if (
        model_type == "qwen3_5_text"
        and hidden_size >= 4096
        and num_hidden_layers >= 32
    ):
        return QWEN35_9B_GRADIENT_CHECKPOINTING_PROFILE
    return DEFAULT_GRADIENT_CHECKPOINTING_PROFILE


def configure_task_peft_params_for_model(peft_params, model_config):
    """Preserve ARC's embedding/head training contract on new-model wrappers.

    Unsloth routes ``qwen3_5_text`` through its generic FastBaseModel PEFT
    implementation.  FastLlama supplies ``finetune_vision_layers=False`` on
    that route, which makes an explicit target list pass through a family
    regex.  The regex covers decoder-layer projections but silently drops the
    top-level ``lm_head`` and ``embed_tokens`` names.  FastBaseModel also does
    not perform FastLlama's conversion of ``embed_tokens`` into a full PEFT
    ``modules_to_save`` module.

    This workflow needs both top-level modules to survive that initial family
    scoping. It passes ``lm_head`` onward as a target and explicitly saves the
    tiny 16-token ARC embedding. Current Unsloth subsequently redirects the
    head to ``modules_to_save`` as well because LoRA on either vocabulary
    module is otherwise silently inactive; the vLLM exporter converts both
    full deltas back to native LoRA form. Setting the vision-family flag true
    only disables the inappropriate text-model scoping; the explicit target
    list still prevents any unrelated module from matching.
    """
    configured = dict(peft_params)
    if str(getattr(model_config, "model_type", "")) != "qwen3_5_text":
        return configured

    target_modules = list(configured.get("target_modules") or ())
    train_embed_tokens = "embed_tokens" in target_modules
    if train_embed_tokens:
        target_modules = [
            module for module in target_modules if module != "embed_tokens"
        ]
        modules_to_save = list(configured.get("modules_to_save") or ())
        if "embed_tokens" not in modules_to_save:
            modules_to_save.append("embed_tokens")
        configured["modules_to_save"] = modules_to_save

    configured["target_modules"] = target_modules
    configured["finetune_vision_layers"] = True
    return configured


def resolve_torch_compile_training_options(enabled):
    """Return Trainer compile arguments without implicit compile activation.

    Transformers treats either a non-None ``torch_compile_backend`` or
    ``torch_compile_mode`` as a request to enable ``torch.compile``.  Keep all
    three fields consistent so a default-off ARC run remains an ordinary PEFT
    module instead of being silently wrapped in ``OptimizedModule``.
    """
    enabled = bool(enabled)
    return {
        "torch_compile": enabled,
        "torch_compile_backend": "inductor" if enabled else None,
        "torch_compile_mode": "reduce-overhead" if enabled else None,
    }


def select_grpo_auto_gradient_checkpointing_percentage(
    sequence_tokens,
    physical_micro_batch,
    *,
    profile=DEFAULT_GRADIENT_CHECKPOINTING_PROFILE,
):
    """Choose an L4-safe GRPO tier including persistent optimizer state."""
    percentage = select_auto_gradient_checkpointing_percentage(
        sequence_tokens,
        workload_multiplier=physical_micro_batch,
        profile=profile,
    )
    effective_tokens = int(sequence_tokens) * int(physical_micro_batch)
    # A 1,867-token sequence at physical micro-batch two (3,734 effective
    # tokens) OOMed in the first SwiGLU forward with only 30% of Qwen layers
    # checkpointed. The same exact task completed two NorMuon updates at 50%
    # with a 15,352 MiB peak, but a later multi-task sweep still exhausted the
    # L4 at this boundary. Cross-task CUDA cleanup now removes known retention
    # risks; 60% supplies four additional checkpointed layers as a narrow
    # safety margin until that sweep is repeated. Keep the measured 875 x 4 =
    # 3,500-token short path at 30%.
    if 3584 < effective_tokens <= 4096:
        return max(60.0, percentage)
    # The ordinary SFT calibration assigns 50% to this band. GRPO also keeps
    # policy/reference log-probability tensors and materializes optimizer state
    # after update one. On a 22 GiB L4, the measured 4,524-token workload was
    # short by 74 MiB at 50%, even with quantized FlashAdamW. Two additional
    # checkpointed Qwen layers (20/36 instead of 18/36) provide a narrow safety
    # margin without slowing ordinary SFT tasks in the same token band.
    if 4096 < effective_tokens <= 5120:
        return max(55.0, percentage)
    return percentage


def select_rl_micro_batch_size(
    sequence_tokens,
    requested_micro_batch,
    *,
    token_limit=8192,
):
    """Reduce a GRPO micro-batch to its largest safe positive divisor."""
    sequence_tokens = int(sequence_tokens)
    requested_micro_batch = int(requested_micro_batch)
    token_limit = int(token_limit)
    if sequence_tokens < 1:
        raise ValueError("sequence_tokens must be positive")
    if requested_micro_batch < 1:
        raise ValueError("requested_micro_batch must be positive")
    if token_limit < 1:
        raise ValueError("token_limit must be positive")
    max_safe_batch = max(1, token_limit // sequence_tokens)
    for active_micro_batch in range(
        min(requested_micro_batch, max_safe_batch),
        0,
        -1,
    ):
        # The full generation group is known to be divisible by the requested
        # micro-batch. Keeping the active value as one of its divisors avoids
        # creating a partial accumulation group after automatic reduction.
        if requested_micro_batch % active_micro_batch == 0:
            return active_micro_batch
    raise AssertionError("Every positive integer is divisible by one")


def select_rl_micro_batch_token_limit(optimizer_name):
    """Return the L4 GRPO activation budget for optimizer-state residency."""
    optimizer_name = str(optimizer_name)
    if optimizer_name in {"paged_adamw_8bit", "paged_adamw_32bit"}:
        # CUDA managed optimizer pages can migrate away before the next model
        # forward, leaving enough room for the measured 4,524-token batch.
        return 8192
    # Fully resident state is materialized after update one. Restrict the next
    # forward to one sequence for the 2,262-token task that otherwise OOMs in
    # a 74 MiB SwiGLU allocation at physical micro-batch two.
    return 4096


def parse_list(value):
    return [item.strip() for item in value.split(",") if item.strip()]


def unique_in_order(values):
    return list(dict.fromkeys(values))


def select_task_keys(keys, *, task_keys=None):
    """Resolve task selection once for all pipelines.

    Precedence: explicit ``task_keys`` CSV (e.g. a CLI argument), then
    ``ARC_TASK_KEY``, then zero-based ``ARC_TASK_INDEX``, then all keys.
    Indices address the original JSON insertion order, never sorted keys.
    Requested order is retained and duplicates are removed. An explicit
    empty CSV selects nothing, rather than falling back to environment/all
    tasks; ``None`` means there is no explicit override.
    """
    input_keys = list(keys)

    explicit_keys = task_keys is not None
    selector_name = "--task-keys" if explicit_keys else "ARC_TASK_KEY"
    exact_keys = parse_list(task_keys if explicit_keys else os.getenv("ARC_TASK_KEY", ""))
    if explicit_keys or exact_keys:
        missing_keys = [key for key in exact_keys if key not in input_keys]
        if missing_keys:
            raise ValueError(
                f"{selector_name} contains keys not present in the challenge file: "
                f"{missing_keys}"
            )
        return unique_in_order(exact_keys)

    raw_index_value = os.getenv("ARC_TASK_INDEX", "")
    if not raw_index_value.strip():
        return input_keys

    raw_indices = parse_list(raw_index_value)
    if not raw_indices:
        raise ValueError(
            "ARC_TASK_INDEX must contain at least one integer when set"
        )
    try:
        task_indices = [int(index) for index in raw_indices]
    except ValueError as error:
        raise ValueError(
            "ARC_TASK_INDEX must be a comma-separated list of integers, "
            "for example: 0,1,5"
        ) from error

    invalid_indices = [
        index for index in task_indices
        if not 0 <= index < len(input_keys)
    ]
    if invalid_indices:
        raise IndexError(
            f"ARC_TASK_INDEX contains indices outside the valid range "
            f"0..{len(input_keys) - 1}: {invalid_indices}"
        )

    return unique_in_order(input_keys[index] for index in task_indices)


def order_task_keys(keys, data, mode="input"):
    """Order selected ARC keys without changing selection semantics.

    ``train-pairs`` sorts ascending; ``train-pairs-desc`` sorts descending.
    Both are stable, so equal counts retain the prior input/selection order.
    These modes reorder tasks, not the demonstration pairs inside each task.
    """
    ordered_keys = list(keys)
    mode = str(mode).strip().lower()
    if mode == "input":
        return ordered_keys
    if mode not in TASK_ORDER_MODES:
        supported = ", ".join(TASK_ORDER_MODES)
        raise ValueError(
            f"Unknown ARC task order {mode!r}; expected one of: {supported}"
        )

    pair_counts = {}
    for key in ordered_keys:
        try:
            train_pairs = data[key]["train"]
        except (KeyError, TypeError) as error:
            raise ValueError(
                f"ARC task {key!r} has no valid 'train' examples"
            ) from error
        try:
            pair_counts[key] = len(train_pairs)
        except TypeError as error:
            raise ValueError(
                f"ARC task {key!r} has non-sized 'train' examples"
            ) from error
    return sorted(
        ordered_keys, key=pair_counts.__getitem__, reverse=mode == "train-pairs-desc"
    )
