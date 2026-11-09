"""E056 worker extension (test servers only): E051's KV-byte parity methods (e051_layout / e051_snapshot / e051_poison)
plus two read-only probes, called through the dev-mode `POST /collective_rpc` endpoint:

- e056_indexer_state(): the E054 indexer module as the worker loaded it (file, sha256, _INDEXER_SKIP flags) and how
  many compiled variants each Triton kernel has per device. Under candidate B the grid-stride kernel must have
  compiled variants and the old kernel none; with DSV41_INDEXER_SKIP=0 the reverse.
- e056_loaded_files(): sha256 of the offload modules and the indexer as loaded in the worker (proves the workers run
  the venv copy's overlay files).

Loaded with `--worker-extension-cls e056_ext.E056Ext` (PYTHONPATH: this dir + E051's gpu/). It adds no serving
behaviour.
"""
import hashlib
import importlib

from e051_ext import E051Ext

KERNELS = ("_mxfp4_indexer_logits_kernel", "_mxfp4_indexer_logits_stride_kernel", "_mxfp4_indexer_q_decode_kernel",
           "_map_candidate_topk_kernel")
MODULES = ("vllm.model_executor.kernels.attention.dsa.dsv41_indexer",
           "vllm.distributed.kv_transfer.kv_connector.v1.offloading.config",
           "vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler",
           "vllm.v1.kv_offload.cpu.manager", "vllm.v1.kv_offload.cpu.spec")


def _compiled(fn):
    while fn is not None and not hasattr(fn, "device_caches"):
        fn = getattr(fn, "fn", None)  # autotuner / heuristics wrappers
    if fn is None:
        return None
    return {str(dev): len(entry[0]) for dev, entry in list(fn.device_caches.items())}


class E056Ext(E051Ext):
    def e056_indexer_state(self):
        m = importlib.import_module(MODULES[0])
        return {"file": m.__file__, "sha256": hashlib.sha256(open(m.__file__, "rb").read()).hexdigest(),
                "skip": getattr(m, "_INDEXER_SKIP", None),
                "skip_candidates": getattr(m, "_INDEXER_SKIP_CANDIDATES", None),
                "target_programs": getattr(m, "_LOGITS_TARGET_PROGRAMS", None),
                "compiled_variants": {k: _compiled(getattr(m, k, None)) for k in KERNELS}}

    def e056_loaded_files(self):
        out = {}
        for name in MODULES:
            m = importlib.import_module(name)
            out[name] = [m.__file__, hashlib.sha256(open(m.__file__, "rb").read()).hexdigest()]
        return out
