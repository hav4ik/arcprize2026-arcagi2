"""Reproduce the E042 startup failures against the unpatched fork source."""
import sys, traceback
sys.path.insert(0, sys.argv[1])  # tests dir
from dsv41_fixture import build_kv_cache_config, make_vllm_config
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.config import build_offloading_config
from vllm.v1.kv_offload.cpu.spec import CPUOffloadingSpec
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler import OffloadingConnectorScheduler

for pmu in (None, 8):
    kvc = build_kv_cache_config()
    vc = make_vllm_config(prefix_match_unit=pmu)
    print(f"--- prefix_match_unit={pmu}")
    try:
        oc = build_offloading_config(vc, kvc)
        print("tokens_per_hash", oc.cache.tokens_per_hash, "groups", [g.tokens_per_block for g in oc.groups],
              "worker_kv_bytes_per_block", oc.worker_kv_bytes_per_block, "replicated", oc.replicated_layout)
        spec = CPUOffloadingSpec(oc)
        print("cpu blocks", spec.num_blocks, "bytes/chunk", spec.kv_bytes_per_chunk)
        OffloadingConnectorScheduler(spec, vc, kvc)
        print("scheduler OK")
    except BaseException as e:
        print(type(e).__name__, str(e)[:200])
        print("".join(traceback.format_tb(e.__traceback__)[-1:]))
