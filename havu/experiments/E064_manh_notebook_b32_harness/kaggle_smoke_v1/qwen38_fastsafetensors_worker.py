"""Native vLLM CUDA worker with an opt-in load-time iterator guard only."""

import os

from vllm.v1.worker.gpu_worker import Worker

from qwen38_fastsafetensors import install_guard


class GuardedFastSafetensorsWorker(Worker):
    def __init__(self, *args, **kwargs):
        install_guard(int(os.environ["ARC_FASTSAFETENSORS_MAX_GPU_SHARD_BYTES"]))
        super().__init__(*args, **kwargs)
