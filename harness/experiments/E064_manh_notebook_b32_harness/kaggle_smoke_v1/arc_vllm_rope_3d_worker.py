"""Offline vLLM V1 worker for ordinary or Delta-MH ARC 3D-RoPE Qwen3."""

import importlib
import inspect
import os

import torch

from vllm.v1.worker.gpu_model_runner import GPUModelRunner
from vllm.v1.worker.gpu_worker import Worker

from arc_vllm_rope_3d import install_vllm_rope_3d, runner_positions, refresh_position_buffer
from pre_sft_rope_3d import CONFIG_KEY


class ArcRope3DModelRunner(GPUModelRunner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Keep address/strides stable across profile, capture and real replay.
        self.arc_rope_positions = torch.zeros(
            self.max_num_tokens, 3, dtype=torch.long, device=self.device
        )
        self.arc_rope_position_cache_enabled = os.environ.get("ARC_ROPE3D_POSITION_CACHE", "1") != "0"
        self._arc_parent_accepts_num_reqs = (
            "num_reqs" in inspect.signature(super()._init_model_kwargs).parameters
        )

    def load_model(self, *args, **kwargs):
        result = super().load_model(*args, **kwargs)
        config = getattr(self.model_config.hf_config, CONFIG_KEY)
        # Graph wrappers hold this same runnable; setting wrapper.forward
        # instead would be bypassed by their __call__ during capture.
        install_vllm_rope_3d(self.get_model(), config)
        return result

    def _init_model_kwargs(self, num_reqs=None):
        parent = super()._init_model_kwargs
        # vLLM 0.23 has no num_reqs argument; newer V1 runners pass 0 for
        # profiling. Do not catch an arbitrary TypeError from inside vLLM.
        if self._arc_parent_accepts_num_reqs:
            kwargs = parent(num_reqs=num_reqs)
        else:
            kwargs = parent()
        xyz = runner_positions(self, num_reqs=num_reqs)
        kwargs["arc_rope_positions"] = refresh_position_buffer(self.arc_rope_positions, xyz)
        return kwargs


class ArcRope3DWorker(Worker):
    def init_device(self):
        if self.use_v2_model_runner:
            raise RuntimeError("ARC 3D RoPE requires VLLM_USE_V2_MODEL_RUNNER=0")
        if self.vllm_config.parallel_config.pipeline_parallel_size != 1:
            raise RuntimeError("ARC 3D RoPE currently requires PP=1")
        if getattr(self.vllm_config.parallel_config, "use_ubatching", False):
            raise RuntimeError("ARC 3D RoPE does not yet support vLLM microbatch wrappers")
        if not getattr(self.vllm_config.kernel_config, "enable_cutedsl_warmup", True):
            warmup = importlib.import_module("vllm.model_executor.warmup.kernel_warmup")
            warmup._warmup_ll_bf16_router_gemm = lambda: None
        runner = importlib.import_module("vllm.v1.worker.gpu_model_runner")
        original = runner.GPUModelRunner
        runner.GPUModelRunner = ArcRope3DModelRunner
        try:
            super().init_device()
        finally:
            runner.GPUModelRunner = original
