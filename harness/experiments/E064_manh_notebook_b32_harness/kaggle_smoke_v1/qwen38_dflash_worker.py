"""Opt-in draft-quantization compatibility, optionally with the shard guard."""

import os

from vllm.v1.worker.gpu_worker import Worker


class QuantizedDFlashWorker(Worker):
    def __init__(self, *args, **kwargs):
        from qwen38_dflash_quantization import install
        install()
        cap = int(os.environ.get("ARC_FASTSAFETENSORS_MAX_GPU_SHARD_BYTES", "0"))
        if cap:
            from qwen38_fastsafetensors import install_guard
            install_guard(cap)
        super().__init__(*args, **kwargs)
