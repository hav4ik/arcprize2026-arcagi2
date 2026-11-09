"""Optional routed-MoE backends for DiffusionGemma BF16 full training."""

from __future__ import annotations

import importlib.metadata
import os
from typing import Any

import torch
import torch.nn.functional as F

from sonicmoe_runtime import (
    SONICMOE_UPSTREAM_COMMIT,
    bootstrap_sonicmoe_runtime,
)


EXPERT_IMPLEMENTATIONS = (
    "grouped_mm",
    "arc_compiled_grouped_mm",
    "arc_quack_moe",
    "arc_sonicmoe",
    "arc_sonicmoe_fp8",
)
_ARC_COMPILED_GROUPED_MM = None
_ARC_SONICMOE = None
_ARC_SONICMOE_VERSION = None
_ARC_SONICMOE_COMMIT = None
_ARC_SONICMOE_PACK_SHA256 = None
_ARC_SONICMOE_PACKED_M_KEY_FIX = None

_CUTLASS_DSL_FRONTEND_DISTRIBUTION = "nvidia-cutlass-dsl"
_CUTLASS_DSL_SHARED_DISTRIBUTIONS = (
    "nvidia-cutlass-dsl-libs-base",
    "nvidia-cutlass-dsl-libs-core",
)
_QUACK_MINIMUM_CUTLASS_DSL_VERSION = "4.7"
_QUACK_ASYNC_COMPILE_START_ENV = "QUACK_ASYNC_COMPILE_START"


def cutlass_dsl_runtime_identity(
    *,
    cuda_version: str | None = None,
    version_reader: Any = None,
) -> dict[str, Any]:
    """Describe the tightly coupled CUTLASS DSL frontend/native packages.

    CUTLASS DSL distributes its Python frontend and native MLIR compiler across
    separate wheels. Pip can leave an older CUDA-specific wheel installed when
    an offline setup upgrades only the frontend. That combination imports but
    fails on the first QuACK compilation, so inspect distribution metadata
    without importing CUTLASS or initializing a compiler.
    """

    if cuda_version is None:
        cuda_version = torch.version.cuda
    cuda_version = str(cuda_version or "")
    cuda_major = cuda_version.split(".", 1)[0]
    if not cuda_major.isdigit():
        raise RuntimeError(
            "arc_sonicmoe could not determine the CUDA major version from "
            f"torch.version.cuda={cuda_version!r}"
        )
    cuda_distribution = f"nvidia-cutlass-dsl-libs-cu{cuda_major}"
    distributions = (
        _CUTLASS_DSL_FRONTEND_DISTRIBUTION,
        *_CUTLASS_DSL_SHARED_DISTRIBUTIONS,
        cuda_distribution,
    )
    if version_reader is None:
        version_reader = importlib.metadata.version
    versions: dict[str, str | None] = {}
    for distribution in distributions:
        try:
            versions[distribution] = str(version_reader(distribution))
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = None
    return {
        "cuda_version": cuda_version,
        "cuda_extra": f"cu{cuda_major}",
        "cuda_distribution": cuda_distribution,
        "versions": versions,
    }


def validate_quack_cutlass_dsl_runtime(
    *,
    cuda_version: str | None = None,
    version_reader: Any = None,
) -> dict[str, Any]:
    """Fail before model/data setup when QuACK's CUTLASS wheels are mixed."""

    from packaging.version import InvalidVersion, Version

    identity = cutlass_dsl_runtime_identity(
        cuda_version=cuda_version,
        version_reader=version_reader,
    )
    versions = identity["versions"]
    frontend_version = versions[_CUTLASS_DSL_FRONTEND_DISTRIBUTION]
    missing = [name for name, version in versions.items() if version is None]
    mismatched = (
        []
        if frontend_version is None
        else [
            name
            for name, version in versions.items()
            if version is not None and version != frontend_version
        ]
    )
    too_old = False
    if frontend_version is not None:
        try:
            too_old = Version(frontend_version) < Version(
                _QUACK_MINIMUM_CUTLASS_DSL_VERSION
            )
        except InvalidVersion:
            too_old = True
    if not missing and not mismatched and not too_old:
        return identity

    target_version = frontend_version or "4.7.1"
    observed = ", ".join(
        f"{name}={version or 'missing'}" for name, version in versions.items()
    )
    reason = []
    if missing:
        reason.append(f"missing {', '.join(missing)}")
    if mismatched:
        reason.append("version mismatch in " + ", ".join(mismatched))
    if too_old:
        reason.append(
            f"QuACK requires CUTLASS DSL >= {_QUACK_MINIMUM_CUTLASS_DSL_VERSION}"
        )
    raise RuntimeError(
        "arc_sonicmoe requires one coherent CUTLASS DSL frontend/native "
        f"version for CUDA {identity['cuda_version']}; {'; '.join(reason)}. "
        f"Found: {observed}. Repair the offline environment before training "
        "with: uv pip install --reinstall "
        f"'nvidia-cutlass-dsl[{identity['cuda_extra']}]=={target_version}'. "
        "A mixed install can import successfully but fail during the first "
        "QuACK compile with '<Pass-Options-Parser>: no such option "
        "enable-pyir'."
    )


def configure_quack_async_compile_runtime() -> str:
    """Use QuACK's fork-safe documented worker mode unless explicitly set.

    The forkserver sidecar in current QuACK preloads CUTLASS DSL. On the tested
    CUDA 13.3/CUTLASS DSL 4.7.1 stack its children segfault in
    ``cuGetProcAddress_v2``. QuACK documents ``spawn`` as the sidecar opt-out;
    it changes only cold compiler-worker startup, not generated kernels or the
    steady training path. Preserve an explicit user choice for diagnostics.
    """

    os.environ.setdefault(_QUACK_ASYNC_COMPILE_START_ENV, "spawn")
    return os.environ[_QUACK_ASYNC_COMPILE_START_ENV]


def _bucketize_sonicmoe_m(m: int) -> int:
    """Match SonicMoE's broad 8x token-count autotuning buckets."""

    if m <= 1:
        return 1 << 3
    exponent = (m - 1).bit_length()
    level = (exponent - 1) // 3
    return 1 << (3 * (level + 1))


def _arc_sonicmoe_bucketized_autotune_key(
    tuner: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> tuple[Any, ...]:
    """Key routed GEMMs by token-count ranges, including transposed dW inputs.

    SonicMoE main buckets every tensor's leading dimension.  That covers its
    forward GEMMs, but the two expert weight-gradient GEMMs receive ``x.T`` and
    ``dout.T``: their packed token count is the trailing dimension of a
    column-major 2-D view.  Leaving it exact launches a multi-second QuACK
    autotuning sweep for nearly every packed sequence length.

    Only that narrow transpose layout gets trailing-dimension bucketing.  Normal
    row-major GEMM N/K dimensions remain exact, so structurally different GEMMs
    cannot alias merely because their sizes fall in the same broad range.
    """

    all_args = {**dict(zip(tuner.arg_names, args)), **kwargs}
    named_args = {
        name: value
        for name, value in all_args.items()
        if name in tuner.arg_names
    }
    key = [str(named_args[name]) for name in tuner.keys if name in named_args]
    for argument in named_args.values():
        if not isinstance(argument, torch.Tensor):
            continue
        shape = list(argument.shape)
        strides = argument.stride()
        if shape:
            shape[0] = _bucketize_sonicmoe_m(shape[0])
        if (
            len(shape) == 2
            and len(strides) == 2
            and strides[0] == 1
            and strides[1] > 1
        ):
            shape[1] = _bucketize_sonicmoe_m(shape[1])
        key.append(str(tuple(shape)))
        key.append(str([stride if stride in {0, 1} else 2 for stride in strides]))
        key.append(str(argument.dtype))
    return tuple(key)


def _install_arc_sonicmoe_packed_m_key_fix(sonicmoe: Any) -> bool:
    """Install the packed-length cache-key fix without requiring a custom wheel."""

    functional = getattr(sonicmoe, "functional", None)
    if functional is None:
        return False
    try:
        import quack.autotuner as quack_autotuner
    except Exception:  # pragma: no cover - runtime dependency
        return False

    # Upstream main's wrapper resolves this helper through its module globals on
    # every call, so replacing it fixes that path too.  The additional wrapper
    # below also supports the fixed PyPI release that predates M bucketing.
    if hasattr(functional, "_make_bucketized_key"):
        functional._make_bucketized_key = _arc_sonicmoe_bucketized_autotune_key

    autotuner_class = quack_autotuner.Autotuner
    current_call = autotuner_class.__call__
    if getattr(current_call, "_arc_packed_m_key_fix", False):
        return True

    original_call = current_call

    @torch.compiler.disable
    def call_with_packed_m_buckets(self, *args, **kwargs):
        if len(self.configs) > 1:
            bucket_key = _arc_sonicmoe_bucketized_autotune_key(
                self, args, kwargs
            )
            if bucket_key in self.cache:
                config = self.cache[bucket_key]
                self.best_config = config
                self.nargs = dict(zip(self.arg_names, args))
                result = self.fn(*args, **kwargs, **config.all_kwargs())
                self.nargs = None
                return result

        result = original_call(self, *args, **kwargs)
        if len(self.configs) > 1 and hasattr(self, "best_config"):
            self.cache[
                _arc_sonicmoe_bucketized_autotune_key(self, args, kwargs)
            ] = self.best_config
        return result

    call_with_packed_m_buckets._arc_packed_m_key_fix = True
    autotuner_class.__call__ = call_with_packed_m_buckets
    return True


def _arc_grouped_mm_tensor_forward(
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
    gate_up_proj: torch.Tensor,
    down_proj: torch.Tensor,
) -> torch.Tensor:
    """Transformers' grouped expert math as one dynamic Inductor graph.

    Keeping this function tensor-only lets every routed-expert module share one
    compiled graph.  It preserves the reference sort, FP32 router weighting,
    deterministic reshape-and-sum reduction, and BF16 grouped GEMMs.  Inductor
    can then fuse the otherwise launch-heavy tanh-GELU, multiply, masking, and
    permutation epilogue around the two grouped GEMMs.
    """

    num_tokens, hidden_dim = hidden_states.shape
    num_top_k = top_k_index.shape[-1]
    num_experts = gate_up_proj.shape[0]
    sample_weights = top_k_weights.reshape(-1)
    expert_ids = top_k_index.reshape(-1)

    expert_ids_g, permutation = torch.sort(expert_ids)
    selected_hidden_states = hidden_states[permutation // num_top_k]
    sorted_sample_weights = sample_weights[permutation]

    # Match Transformers' CUDA grouped_mm dispatch. histc is graph-safe while
    # bincount is not, and int32 offsets are the native grouped_mm contract.
    tokens_per_expert = torch.histc(
        expert_ids_g.int(),
        bins=num_experts,
        min=0,
        max=num_experts - 1,
    )
    offsets = torch.cumsum(tokens_per_expert, dim=0, dtype=torch.int32)

    # Expert-parallel sentinels are not emitted by this single-GPU workload,
    # but retaining the reference pre/post masks keeps the function safe if a
    # zero-weight sentinel is ever passed by the model interface.
    sentinel_mask = (expert_ids_g >= num_experts).unsqueeze(-1)
    selected_hidden_states = selected_hidden_states.masked_fill(sentinel_mask, 0)

    projected = F.grouped_mm(
        selected_hidden_states,
        gate_up_proj.transpose(-2, -1),
        offs=offsets,
    )
    gate, up = projected.chunk(2, dim=-1)
    activated = F.gelu(gate, approximate="tanh") * up
    routed_output = F.grouped_mm(
        activated,
        down_proj.transpose(-2, -1),
        offs=offsets,
    )
    weighted_output = routed_output * sorted_sample_weights.unsqueeze(-1)
    weighted_output = weighted_output.masked_fill(sentinel_mask, 0)

    inverse_permutation = torch.empty_like(permutation)
    inverse_permutation[permutation] = torch.arange(
        permutation.numel(), device=permutation.device
    )
    return (
        weighted_output[inverse_permutation]
        .view(num_tokens, num_top_k, hidden_dim)
        .sum(dim=1)
        .to(hidden_states.dtype)
    )


def _load_arc_compiled_grouped_mm():
    """Create the shared full-graph MoE callable only when selected."""

    global _ARC_COMPILED_GROUPED_MM
    if _ARC_COMPILED_GROUPED_MM is None:
        if not hasattr(F, "grouped_mm"):
            raise RuntimeError(
                "arc_compiled_grouped_mm requires torch.nn.functional.grouped_mm"
            )
        _ARC_COMPILED_GROUPED_MM = torch.compile(
            _arc_grouped_mm_tensor_forward,
            fullgraph=True,
            dynamic=True,
            mode="default",
        )
    return _ARC_COMPILED_GROUPED_MM


def _arc_compiled_grouped_mm_experts_forward(
    self: torch.nn.Module,
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
) -> torch.Tensor:
    """Run the shared compiled BF16 grouped-MoE graph."""

    if not self.has_gate or not self.is_concatenated:
        raise ValueError(
            "arc_compiled_grouped_mm requires concatenated gated experts"
        )
    if self.has_bias:
        raise ValueError("arc_compiled_grouped_mm does not support expert biases")
    if self.is_transposed:
        raise ValueError(
            "arc_compiled_grouped_mm requires the native non-transposed layout"
        )
    if hidden_states.dtype != torch.bfloat16:
        raise ValueError("arc_compiled_grouped_mm requires BF16 activations")
    if self.gate_up_proj.dtype != torch.bfloat16:
        raise ValueError("arc_compiled_grouped_mm requires BF16 expert weights")
    if self.down_proj.dtype != torch.bfloat16:
        raise ValueError("arc_compiled_grouped_mm requires BF16 expert weights")
    return _load_arc_compiled_grouped_mm()(
        hidden_states,
        top_k_index,
        top_k_weights,
        self.gate_up_proj,
        self.down_proj,
    )


def _load_arc_quack_ops():
    """Load the current QuACK GEMMs lazily.

    QuACK is an optional Kaggle wheel.  Keeping the import behind the selected
    backend preserves the reference ``grouped_mm`` environment and avoids
    importing CuTeDSL during argument parsing or CPU tests.
    """
    try:
        from quack.gemm_interface import gemm, gemm_dgated, gemm_gated
    except Exception as error:  # pragma: no cover - runtime dependency
        raise RuntimeError(
            "arc_quack_moe requires a compatible quack-kernels and "
            "NVIDIA CUTLASS DSL installation"
        ) from error
    return gemm, gemm_gated, gemm_dgated


class _ArcQuackMoE(torch.autograd.Function):
    """QuACK grouped GEGLU experts with an explicit, deterministic backward.

    The implementation follows Transformers' routing order exactly, but uses
    QuACK's SM120 varlen GEMMs.  ``gemm_dgated`` fuses the down-projection
    input gradient, GEGLU derivative, routing-score multiplication, routing
    dot-product reduction, and activation reconstruction.  This avoids saving
    the routed expert output solely to form the router gradient.
    """

    @staticmethod
    def forward(
        ctx,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
        gate_up_proj: torch.Tensor,
        down_proj: torch.Tensor,
    ) -> torch.Tensor:
        gemm, gemm_gated, _ = _load_arc_quack_ops()
        num_tokens, hidden_dim = hidden_states.shape
        num_top_k = top_k_index.shape[-1]
        num_experts = gate_up_proj.shape[0]
        if hidden_states.dtype != torch.bfloat16:
            raise ValueError("arc_quack_moe currently requires BF16 activations")
        if gate_up_proj.dtype != torch.bfloat16 or down_proj.dtype != torch.bfloat16:
            raise ValueError("arc_quack_moe currently requires BF16 expert weights")

        expert_ids_g, permutation = torch.sort(top_k_index.reshape(-1))
        token_rows = torch.div(
            permutation, num_top_k, rounding_mode="floor"
        ).to(torch.int32)
        histogram_input = expert_ids_g.int()
        tokens_per_expert = torch.histc(
            histogram_input,
            bins=num_experts,
            min=0,
            max=num_experts - 1,
        ).to(torch.int32)
        offsets = torch.cat(
            (
                torch.zeros(1, dtype=torch.int32, device=hidden_states.device),
                torch.cumsum(tokens_per_expert, dim=0, dtype=torch.int32),
            )
        )
        sorted_scores = top_k_weights.reshape(-1)[permutation].float()

        with torch.amp.autocast("cuda", enabled=False):
            preactivation, activation = gemm_gated(
                hidden_states,
                gate_up_proj.transpose(1, 2),
                activation="geglu",
                cu_seqlens_m=offsets,
                A_idx=token_rows,
                concat_layout=("B",),
            )
            routed_output = gemm(
                activation,
                down_proj.transpose(1, 2),
                cu_seqlens_m=offsets,
            )

        inverse_permutation = torch.empty_like(permutation)
        inverse_permutation[permutation] = torch.arange(
            permutation.numel(), device=permutation.device
        )
        output = (
            (routed_output.float() * sorted_scores.unsqueeze(-1))[
                inverse_permutation
            ]
            .view(num_tokens, num_top_k, hidden_dim)
            .sum(dim=1)
            .to(hidden_states.dtype)
        )

        ctx.num_tokens = num_tokens
        ctx.num_top_k = num_top_k
        ctx.save_for_backward(
            hidden_states,
            gate_up_proj,
            down_proj,
            sorted_scores,
            token_rows,
            offsets,
            inverse_permutation,
            preactivation,
        )
        return output

    @staticmethod
    def backward(ctx, output_gradient: torch.Tensor):
        gemm, _, gemm_dgated = _load_arc_quack_ops()
        (
            hidden_states,
            gate_up_proj,
            down_proj,
            sorted_scores,
            token_rows,
            offsets,
            inverse_permutation,
            preactivation,
        ) = ctx.saved_tensors
        output_gradient = output_gradient.contiguous()

        with torch.amp.autocast("cuda", enabled=False):
            (
                preactivation_gradient,
                activation,
                sorted_score_gradient,
            ) = gemm_dgated(
                output_gradient,
                down_proj,
                preactivation,
                activation="geglu",
                colvec_scale=sorted_scores,
                colvec_reduce=True,
                cu_seqlens_m=offsets,
                A_idx=token_rows,
                dynamic_scheduler=False,
            )
            routed_hidden_gradient = gemm(
                preactivation_gradient,
                gate_up_proj,
                cu_seqlens_m=offsets,
                concat_layout=("B",),
                dynamic_scheduler=False,
            )
            down_proj_gradient = gemm(
                output_gradient.transpose(0, 1),
                activation,
                cu_seqlens_k=offsets,
                A_idx=token_rows,
                dynamic_scheduler=False,
            )
            # QuACK's varlen-K gather applies to the left operand.  Materialize
            # this one routed hidden-state view so dW1 can retain the direct
            # concatenated output layout expected by the model Parameter.
            sorted_hidden_states = hidden_states[token_rows]
            gate_up_proj_gradient = gemm(
                preactivation_gradient.transpose(0, 1),
                sorted_hidden_states,
                cu_seqlens_k=offsets,
                concat_layout=("out",),
                dynamic_scheduler=False,
            )

        hidden_gradient = (
            routed_hidden_gradient[inverse_permutation]
            .view(ctx.num_tokens, ctx.num_top_k, hidden_states.shape[-1])
            .sum(dim=1)
            .to(hidden_states.dtype)
        )
        top_k_weights_gradient = sorted_score_gradient[
            inverse_permutation
        ].view(ctx.num_tokens, ctx.num_top_k)
        return (
            hidden_gradient,
            None,
            top_k_weights_gradient,
            gate_up_proj_gradient,
            down_proj_gradient,
        )


def _arc_quack_moe_experts_forward(
    self: torch.nn.Module,
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
) -> torch.Tensor:
    """Run the ARC QuACK backend for DiffusionGemma's weight layout."""
    if not self.has_gate or not self.is_concatenated:
        raise ValueError("arc_quack_moe requires concatenated gated experts")
    if self.has_bias:
        raise ValueError("arc_quack_moe does not support expert biases")
    if self.is_transposed:
        raise ValueError("arc_quack_moe requires the native non-transposed layout")
    return _ArcQuackMoE.apply(
        hidden_states,
        top_k_index,
        top_k_weights,
        self.gate_up_proj,
        self.down_proj,
    )


def _load_arc_sonicmoe_kernel():
    """Load the audited upstream-main SonicMoE pack lazily.

    The older ``kernels-community/sonic-moe`` snapshot fails long-sequence
    backward on this workload.  Upstream main moved to QuACK 0.6.4 and fixes
    that SM120 failure, but still reports the old release version.  The pinned
    repository wheel identifies the exact tested commit and is extracted
    automatically, so do not silently fall back to the stale release.
    """
    global _ARC_SONICMOE, _ARC_SONICMOE_VERSION, _ARC_SONICMOE_COMMIT
    global _ARC_SONICMOE_PACK_SHA256
    global _ARC_SONICMOE_PACKED_M_KEY_FIX
    if _ARC_SONICMOE is None:
        validate_quack_cutlass_dsl_runtime()
        configure_quack_async_compile_runtime()
        try:
            pack = bootstrap_sonicmoe_runtime()
            import sonicmoe
            from packaging.version import Version
        except Exception as error:  # pragma: no cover - runtime dependency
            raise RuntimeError(
                "arc_sonicmoe requires the bundled pinned SonicMoE pack and "
                "quack-kernels>=0.6.4; remove the manual sonic-moe PYTHONPATH "
                "entry and keep third_party/sonic_moe in this checkout"
            ) from error
        installed_version = str(getattr(sonicmoe, "__version__", "0"))
        if Version(installed_version) < Version("0.1.2.post1"):
            raise RuntimeError(
                "arc_sonicmoe requires sonic-moe>=0.1.2.post1; "
                f"found {installed_version}"
            )
        installed_commit = getattr(sonicmoe, "__arc_upstream_commit__", None)
        if installed_commit != SONICMOE_UPSTREAM_COMMIT:
            raise RuntimeError(
                "arc_sonicmoe loaded the wrong SonicMoE source: expected "
                f"{SONICMOE_UPSTREAM_COMMIT}, found {installed_commit!r}"
            )
        _ARC_SONICMOE = sonicmoe
        _ARC_SONICMOE_VERSION = installed_version
        _ARC_SONICMOE_COMMIT = installed_commit
        _ARC_SONICMOE_PACK_SHA256 = pack.sha256
        _ARC_SONICMOE_PACKED_M_KEY_FIX = (
            _install_arc_sonicmoe_packed_m_key_fix(sonicmoe)
        )
    return _ARC_SONICMOE


def _arc_sonicmoe_experts_forward(
    self: torch.nn.Module,
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
) -> torch.Tensor:
    """Run the official SonicMoE kernel with DiffusionGemma's tanh GEGLU."""
    if hasattr(self, "_arc_bf16_expert_grad_accum") and torch.is_grad_enabled():
        from diffusion_gemma_bf16_fusions import fused_sonic_experts

        return fused_sonic_experts(self, hidden_states, top_k_index, top_k_weights)
    if hasattr(self, "_arc_fp8_qat_scaling"):
        from diffusion_gemma_fp8_qat import qat_experts_forward

        return qat_experts_forward(
            self, hidden_states, top_k_index, top_k_weights,
            _arc_sonicmoe_with_weights,
        )
    return _arc_sonicmoe_with_weights(
        self, hidden_states, top_k_index, top_k_weights,
        self.gate_up_proj, self.down_proj,
    )


def _arc_sonicmoe_with_weights(
    self, hidden_states, top_k_index, top_k_weights, gate_up_weight, down_weight,
):
    """Explicit weights allow QAT without replacing registered Parameters."""
    num_tokens = hidden_states.size(0)
    num_top_k = top_k_index.size(-1)
    token_indices = (
        torch.arange(num_tokens, device=hidden_states.device)
        .unsqueeze(1)
        .expand(-1, num_top_k)
        .reshape(-1)
        .int()
    )
    # Keep the reference grouped_mm path's FP32 routing weights.  The upstream
    # wrapper narrows these to the activation dtype before SonicMoE widens them
    # again, which is unnecessary for DiffusionGemma and changes router grads.
    router_scores = top_k_weights.reshape(-1).float()
    expert_ids = top_k_index.reshape(-1).int()
    sonicmoe = _load_arc_sonicmoe_kernel()
    if not self.has_gate:
        raise ValueError("arc_sonicmoe requires gated experts")
    permutation = (2, 1, 0) if self.is_transposed else (1, 2, 0)
    gate_up_bias = self.gate_up_proj_bias if self.has_bias else None
    down_bias = self.down_proj_bias if self.has_bias else None
    output, _ = sonicmoe.moe_general_routing_inputs(
        hidden_states,
        router_scores,
        token_indices,
        expert_ids,
        gate_up_weight.permute(*permutation),
        gate_up_bias,
        down_weight.permute(*permutation),
        down_bias,
        E=self.num_experts,
        activation_type=sonicmoe.enums.ActivationType.GEGLU,
        is_inference_mode_enabled=not torch.is_grad_enabled(),
        concat_layout=self.is_concatenated,
        stream_id=torch.cuda.current_stream(hidden_states.device).cuda_stream,
    )
    return output


def install_diffusion_gemma_experts_backend(
    model: torch.nn.Module,
    implementation: str,
) -> dict[str, Any]:
    """Select grouped_mm or register an opt-in ARC expert backend."""
    implementation = str(implementation).lower()
    if implementation not in EXPERT_IMPLEMENTATIONS:
        raise ValueError(
            f"unsupported experts implementation {implementation!r}; "
            f"expected one of {EXPERT_IMPLEMENTATIONS}"
        )
    parameter_ids_before = {id(parameter) for parameter in model.parameters()}
    fp8_details = {}
    if implementation == "arc_sonicmoe_fp8":
        from diffusion_gemma_sonic_moe_fp8 import sonic_fp8_experts_forward
        from quack_fp8_runtime import bootstrap_quack_sm120_fp8

        text_config = getattr(model.config, "text_config", model.config)
        if getattr(text_config, "hidden_activation", None) != "gelu_pytorch_tanh":
            raise ValueError("arc_sonicmoe_fp8 requires DiffusionGemma tanh-GeGLU experts")
        if torch.cuda.get_device_capability() != (12, 0):
            raise RuntimeError("arc_sonicmoe_fp8 is validated only on SM120")
        fp8_details["quack_fp8_runtime"] = bootstrap_quack_sm120_fp8()
    if implementation in {
        "arc_compiled_grouped_mm",
        "arc_quack_moe",
        "arc_sonicmoe",
        "arc_sonicmoe_fp8",
    }:
        try:
            from transformers.integrations.moe import ALL_EXPERTS_FUNCTIONS
        except Exception as error:  # pragma: no cover - runtime dependency
            raise RuntimeError(
                f"{implementation} requires a Transformers build with the "
                "experts interface"
            ) from error
        forward_functions = {
            "arc_compiled_grouped_mm": (
                _arc_compiled_grouped_mm_experts_forward
            ),
            "arc_quack_moe": _arc_quack_moe_experts_forward,
            "arc_sonicmoe": _arc_sonicmoe_experts_forward,
        }
        if implementation == "arc_sonicmoe_fp8":
            forward_functions[implementation] = sonic_fp8_experts_forward
        experts_forward = forward_functions[implementation]
        ALL_EXPERTS_FUNCTIONS.register(implementation, experts_forward)
    sonicmoe_details = {}
    if implementation in {"arc_sonicmoe", "arc_sonicmoe_fp8"}:
        _load_arc_sonicmoe_kernel()
        sonicmoe_details = {
            "sonicmoe_version": _ARC_SONICMOE_VERSION,
            "sonicmoe_upstream_commit": _ARC_SONICMOE_COMMIT,
            "sonicmoe_pack_sha256": _ARC_SONICMOE_PACK_SHA256,
            "packed_weight_gradient_m_key_fix": bool(
                _ARC_SONICMOE_PACKED_M_KEY_FIX
            ),
        }
        if implementation == "arc_sonicmoe_fp8":
            expert_parameters = {
                id(p): p for module in model.modules()
                if module.__class__.__name__ == "DiffusionGemmaTextExperts"
                for p in (module.gate_up_proj, module.down_proj)
            }
            if not expert_parameters:
                raise RuntimeError("FP8 SonicMoE found no DiffusionGemma experts")
            fp8_details.update({
                "mode": "fp8-e4m3-forward-input-gradient-weight-gradient",
                "covered_numel": sum(p.numel() for p in expert_parameters.values()),
                "covered_parameter_tensors": len(expert_parameters),
                "master_and_gradient_dtype": "torch.bfloat16",
                "weight_scaling": "per-expert-matrix-fp32-current",
                "activation_scaling": "per-tensor-fp32-current",
                "persistent_quantized_weight_cache": False,
            })
            sonicmoe_details["fp8_training"] = fp8_details
    model.set_experts_implementation(implementation)
    parameter_ids_after = {id(parameter) for parameter in model.parameters()}
    if parameter_ids_after != parameter_ids_before:
        raise RuntimeError("experts backend selection changed model Parameters")
    return {
        "implementation": implementation,
        "activation": (
            "geglu-tanh-compiled"
            if implementation == "arc_compiled_grouped_mm"
            else (
                "geglu-tanh-explicit"
                if implementation in {"arc_quack_moe", "arc_sonicmoe", "arc_sonicmoe_fp8"}
                else "transformers-default"
            )
        ),
        "parameter_identity_preserved": True,
        **sonicmoe_details,
    }
