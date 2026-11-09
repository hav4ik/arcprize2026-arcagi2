"""Out-of-tree vLLM backend for ARC intermittent prompt attention.

The extension is loaded only when ``--bidirectional-prompt-attention`` is
enabled.  It keeps vLLM's ordinary paged FlashAttention decode path, but
replaces prefill segments that contain ARC prompt queries with exact
rectangular paged-FlashAttention calls.  The resulting K/V values are written
to vLLM's normal cache and can be reused by later causal answer tokens.
"""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass
import importlib
from typing import Any

import torch

from arc_bidirectional_attention import (
    ArcAttentionSegment,
    intersect_attention_segments,
)


ARC_VLLM_SEGMENTS_EXTRA_KEY = "arc_bidirectional_attention_segments"


@dataclass(frozen=True)
class ArcVllmRequestPlan:
    request_index: int
    query_offset: int
    context_length: int
    sequence_length: int
    segments: tuple[ArcAttentionSegment, ...]


@dataclass(frozen=True)
class ArcVllmForwardPlan:
    requests: tuple[ArcVllmRequestPlan, ...]
    num_actual_tokens: int

    @property
    def has_prompt_queries(self) -> bool:
        return any(
            segment.is_prompt
            for request in self.requests
            for segment in request.segments
        )


_CURRENT_PLAN: ContextVar[ArcVllmForwardPlan | None] = ContextVar(
    "arc_vllm_bidirectional_forward_plan",
    default=None,
)


def build_vllm_forward_plan(model_runner: Any) -> ArcVllmForwardPlan | None:
    """Build one CPU run table from vLLM's scheduled request state."""
    input_batch = model_runner.input_batch
    num_reqs = int(input_batch.num_reqs)
    if num_reqs <= 0:
        return None
    query_starts = model_runner.query_start_loc.np[: num_reqs + 1]
    plans: list[ArcVllmRequestPlan] = []
    saw_metadata = False
    missing_metadata = False
    for request_index, request_id in enumerate(input_batch.req_ids[:num_reqs]):
        request_state = model_runner.requests[request_id]
        sampling_params = request_state.sampling_params
        extra_args = (
            getattr(sampling_params, "extra_args", None)
            if sampling_params is not None
            else None
        )
        serialized = (
            extra_args.get(ARC_VLLM_SEGMENTS_EXTRA_KEY)
            if extra_args is not None
            else None
        )
        if serialized is None:
            missing_metadata = True
            continue
        saw_metadata = True
        query_offset = int(query_starts[request_index])
        query_stop = int(query_starts[request_index + 1])
        query_length = query_stop - query_offset
        if query_length <= 0:
            raise RuntimeError("vLLM scheduled an empty ARC attention request")
        context_length = int(
            input_batch.num_computed_tokens_cpu[request_index]
        )
        sequence_length = context_length + query_length
        plans.append(
            ArcVllmRequestPlan(
                request_index=request_index,
                query_offset=query_offset,
                context_length=context_length,
                sequence_length=sequence_length,
                segments=intersect_attention_segments(
                    serialized,
                    query_start=context_length,
                    query_end=sequence_length,
                ),
            )
        )

    # Dummy profile/warmup requests carry no application metadata and must use
    # the stock backend.  A mixed real batch is a wiring bug and cannot safely
    # fall back because it would create causal K/V for only some ARC requests.
    if not saw_metadata:
        return None
    if missing_metadata or len(plans) != num_reqs:
        raise RuntimeError(
            "ARC vLLM attention metadata is missing from part of a real batch"
        )
    return ArcVllmForwardPlan(
        requests=tuple(plans),
        num_actual_tokens=int(query_starts[-1]),
    )


def _register_backend() -> None:
    from vllm import ModelRegistry
    from vllm.v1.attention.backends.registry import (
        AttentionBackendEnum,
        register_backend,
    )

    register_backend(
        AttentionBackendEnum.CUSTOM,
        f"{__name__}.ArcBidirectionalFlashAttentionBackend",
    )
    ModelRegistry.register_model(
        "Qwen3ForCausalLM",
        f"{__name__}:ArcBidirectionalQwen3ForCausalLM",
    )


def register_vllm_bidirectional_support() -> None:
    """Register the custom backend/model in the calling vLLM process."""
    _register_backend()


# Imports below deliberately stay after the small metadata helpers.  vLLM
# resolves this module in every worker process via ``worker_cls``.
from vllm.model_executor.models.qwen3 import (  # noqa: E402
    Qwen3ForCausalLM,
)
from vllm.utils.torch_utils import (  # noqa: E402
    canonicalize_singleton_dim_strides,
    is_quantized_kv_cache,
)
from vllm.v1.attention.backends.fa_utils import (  # noqa: E402
    flash_attn_varlen_func,
)
from vllm.v1.attention.backends.flash_attn import (  # noqa: E402
    FlashAttentionBackend,
    FlashAttentionImpl,
)
from vllm.v1.worker.gpu_model_runner import GPUModelRunner  # noqa: E402
from vllm.v1.worker.gpu_worker import Worker  # noqa: E402


class ArcBidirectionalQwen3ForCausalLM(Qwen3ForCausalLM):
    """Qwen3 wrapper that exposes one runner-built plan to every layer."""

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: Any | None = None,
        inputs_embeds: torch.Tensor | None = None,
        arc_attention_plan: ArcVllmForwardPlan | None = None,
        **kwargs: Any,
    ) -> Any:
        if kwargs:
            raise TypeError(
                "Unexpected kwargs for ARC bidirectional Qwen3: "
                f"{sorted(kwargs)}"
            )
        token = _CURRENT_PLAN.set(arc_attention_plan)
        try:
            return super().forward(
                input_ids,
                positions,
                intermediate_tensors,
                inputs_embeds,
            )
        finally:
            _CURRENT_PLAN.reset(token)


class ArcBidirectionalFlashAttentionImpl(FlashAttentionImpl):
    """Paged FlashAttention with exact intermittent prompt-prefill runs."""

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: Any,
        output: torch.Tensor,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        plan = _CURRENT_PLAN.get()
        if plan is None or not plan.has_prompt_queries:
            return super().forward(
                layer,
                query,
                key,
                value,
                kv_cache,
                attn_metadata,
                output,
                output_scale,
                output_block_scale,
            )
        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError(
                "ARC bidirectional attention does not support fused output "
                "quantization"
            )
        if attn_metadata is None:
            raise RuntimeError("ARC bidirectional attention received no metadata")
        if self.dcp_world_size != 1:
            raise RuntimeError(
                "ARC bidirectional vLLM attention currently requires DCP=1"
            )
        if is_quantized_kv_cache(self.kv_cache_dtype):
            raise RuntimeError(
                "ARC bidirectional vLLM attention requires BF16/FP16 KV cache"
            )
        if self.alibi_slopes is not None or self.sliding_window != (-1, -1):
            raise RuntimeError(
                "ARC bidirectional vLLM attention supports global RoPE "
                "attention only"
            )
        if plan.num_actual_tokens != int(attn_metadata.num_actual_tokens):
            raise RuntimeError(
                "ARC vLLM plan/token mismatch: "
                f"plan={plan.num_actual_tokens}, "
                f"metadata={attn_metadata.num_actual_tokens}"
            )

        if kv_cache.shape[-1] == 2 * self.head_size:
            # Newer vLLM: logical [blocks, kv_heads, block, 2 * dim],
            # transposed to paged-FA [blocks, block, kv_heads, dim].
            key_cache, value_cache = kv_cache.transpose(1, 2).split(
                self.head_size,
                dim=-1,
            )
        elif kv_cache.ndim == 5 and kv_cache.shape[1] == 2:
            # Earlier vLLM: [blocks, 2, block, kv_heads, dim].
            key_cache, value_cache = kv_cache.unbind(1)
        else:
            raise RuntimeError(
                "Unsupported vLLM KV-cache layout for ARC bidirectional "
                f"attention: shape={tuple(kv_cache.shape)}"
            )
        key_cache = canonicalize_singleton_dim_strides(key_cache)
        value_cache = canonicalize_singleton_dim_strides(value_cache)
        output[: plan.num_actual_tokens].zero_()
        device = query.device
        for request in plan.requests:
            for segment in request.segments:
                local_start = segment.start - request.context_length
                local_end = segment.end - request.context_length
                query_start = request.query_offset + local_start
                query_end = request.query_offset + local_end
                query_length = query_end - query_start
                key_length = segment.end
                if query_length <= 0:
                    raise RuntimeError("invalid ARC vLLM attention segment")
                cu_seqlens_q = torch.tensor(
                    [0, query_length],
                    device=device,
                    dtype=torch.int32,
                )
                seqused_k = torch.tensor(
                    [key_length],
                    device=device,
                    dtype=torch.int32,
                )
                descale_shape = (1, self.num_kv_heads)
                q_descale = (
                    layer._q_scale.expand(descale_shape)
                    if self.supports_quant_query_input
                    else None
                )
                flash_attn_varlen_func(
                    q=query[query_start:query_end],
                    k=key_cache,
                    v=value_cache,
                    out=output[query_start:query_end],
                    cu_seqlens_q=cu_seqlens_q,
                    max_seqlen_q=query_length,
                    seqused_k=seqused_k,
                    max_seqlen_k=key_length,
                    softmax_scale=self.scale,
                    causal=not segment.is_prompt,
                    alibi_slopes=None,
                    window_size=[-1, -1],
                    block_table=attn_metadata.block_table[
                        request.request_index : request.request_index + 1
                    ],
                    softcap=self.logits_soft_cap,
                    scheduler_metadata=None,
                    fa_version=self.vllm_flash_attn_version,
                    q_descale=q_descale,
                    k_descale=layer._k_scale.expand(descale_shape),
                    v_descale=layer._v_scale.expand(descale_shape),
                    num_splits=(
                        1 if self.batch_invariant_enabled else 0
                    ),
                    s_aux=self.sinks,
                )
        return output


class ArcBidirectionalFlashAttentionBackend(FlashAttentionBackend):
    @staticmethod
    def get_name() -> str:
        # Attention.__init__ indexes AttentionBackendEnum with this value.
        # Third-party backends therefore report the registry slot rather than
        # a private display name.
        return "CUSTOM"

    @staticmethod
    def get_impl_cls() -> type[ArcBidirectionalFlashAttentionImpl]:
        return ArcBidirectionalFlashAttentionImpl


class ArcBidirectionalGPUModelRunner(GPUModelRunner):
    """Pass per-request role runs from SamplingParams to the Qwen3 model."""

    def _init_model_kwargs(self) -> dict[str, Any]:
        kwargs = super()._init_model_kwargs()
        plan = build_vllm_forward_plan(self)
        if plan is not None:
            kwargs["arc_attention_plan"] = plan
        return kwargs


class ArcBidirectionalWorker(Worker):
    """Select the metadata-aware V1 runner without modifying vLLM files."""

    def init_device(self) -> None:
        if self.use_v2_model_runner:
            raise RuntimeError(
                "ARC bidirectional prompt attention requires vLLM's V1 GPU "
                "model runner; set VLLM_USE_V2_MODEL_RUNNER=0"
            )
        import vllm.v1.worker.gpu_model_runner as runner_module

        kernel_config = self.vllm_config.kernel_config
        if not getattr(kernel_config, "enable_cutedsl_warmup", True):
            # vLLM 0.23 currently calls this CuteDSL router warmup
            # unconditionally on SM90+, even for a dense Qwen model.  Honor
            # its own config flag so an optional quack/CUTLASS-DSL mismatch
            # cannot prevent an otherwise valid engine from starting.
            warmup_module = importlib.import_module(
                "vllm.model_executor.warmup.kernel_warmup"
            )
            warmup_module._warmup_ll_bf16_router_gemm = lambda: None

        original_runner = runner_module.GPUModelRunner
        runner_module.GPUModelRunner = ArcBidirectionalGPUModelRunner
        try:
            super().init_device()
        finally:
            runner_module.GPUModelRunner = original_runner


# Importing the worker module must reproduce registration in spawned workers.
_register_backend()


__all__ = [
    "ARC_VLLM_SEGMENTS_EXTRA_KEY",
    "ArcBidirectionalFlashAttentionBackend",
    "ArcBidirectionalFlashAttentionImpl",
    "ArcBidirectionalGPUModelRunner",
    "ArcBidirectionalQwen3ForCausalLM",
    "ArcBidirectionalWorker",
    "ArcVllmForwardPlan",
    "ArcVllmRequestPlan",
    "build_vllm_forward_plan",
    "register_vllm_bidirectional_support",
]
