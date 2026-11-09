"""DFlash2 worker hook, reproduced automatically in every spawned engine."""
from vllm.v1.worker.gpu_worker import Worker


class ArcDFlashWorker(Worker):
    def __init__(self, *args, **kwargs):
        from arc_vllm_dflash_kv import install
        install()
        super().__init__(*args, **kwargs)
