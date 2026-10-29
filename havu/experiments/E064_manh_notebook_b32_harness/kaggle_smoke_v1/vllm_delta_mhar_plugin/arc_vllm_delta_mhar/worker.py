"""Small vLLM worker compatibility shim for the Kaggle wheel set."""

from __future__ import annotations

import importlib

from vllm.v1.worker.gpu_worker import Worker


class ArcDeltaMHARWorker(Worker):
    """Honor vLLM's switch for the optional CuteDSL router warmup.

    vLLM 0.23 calls the low-latency BF16 router warmup unconditionally on
    SM90+ even for dense Qwen models.  The Kaggle CUTLASS-DSL wrapper can be
    newer than that call site, and then engine startup fails before our model
    runs.  Delta-MH does not use that MoE/router kernel, so skipping only this
    optional warmup is both safe and narrower than patching installed vLLM.
    """

    def init_device(self) -> None:
        kernel_config = self.vllm_config.kernel_config
        if not getattr(kernel_config, "enable_cutedsl_warmup", True):
            warmup_module = importlib.import_module(
                "vllm.model_executor.warmup.kernel_warmup"
            )
            warmup_module._warmup_ll_bf16_router_gemm = lambda: None
        super().init_device()


__all__ = ["ArcDeltaMHARWorker"]
