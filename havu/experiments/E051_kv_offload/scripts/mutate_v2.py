"""Mutation check for test_dsv41_kv_offload_v2: each mutant of the v2 key builder must fail the randomized test."""
import sys, unittest
import test_dsv41_kv_offload_v2 as T
from vllm.distributed.kv_transfer.kv_connector.v1.offloading import scheduler as S
from vllm.v1.kv_offload.base import make_offload_key

def off_by_one(self):
    hs = self.req.block_hashes
    for gc, gs in zip(self.config.kv_group_configs, self.group_states):
        if gc.skip: continue
        st = gc.hashes_per_chunk; start = st * len(gs.offload_keys) + st  # mutant: missing -1
        if start >= len(hs): continue
        gs.offload_keys.extend(make_offload_key(h, gc.group_idx) for h in hs[start::st])

def no_step(self):
    hs = self.req.block_hashes
    for gc, gs in zip(self.config.kv_group_configs, self.group_states):
        if gc.skip: continue
        st = gc.hashes_per_chunk; start = st * len(gs.offload_keys) + st - 1
        if start >= len(hs): continue
        gs.offload_keys.extend(make_offload_key(h, gc.group_idx) for h in hs[start:])  # mutant: step ignored

def stale_len(self):
    hs = self.req.block_hashes; n = len(hs) - 1  # mutant: misses the newest hash
    for gc, gs in zip(self.config.kv_group_configs, self.group_states):
        if gc.skip: continue
        st = gc.hashes_per_chunk; start = st * len(gs.offload_keys) + st - 1
        if start >= n: continue
        gs.offload_keys.extend(make_offload_key(h, gc.group_idx) for h in hs[start:n:st])

bad = 0
for name, fn in [("off_by_one", off_by_one), ("no_step", no_step), ("stale_len", stale_len)]:
    S.RequestOffloadState.update_offload_keys = fn
    r = unittest.TextTestRunner(stream=open("/dev/null", "w")).run(
        unittest.TestLoader().loadTestsFromName("test_dsv41_kv_offload_v2.IncrementalKeysTest.test_randomized_histories_match_v1"))
    caught = not r.wasSuccessful()
    print(f"mutant {name}: {'caught' if caught else 'SURVIVED'}")
    bad += not caught
sys.exit(bad)
