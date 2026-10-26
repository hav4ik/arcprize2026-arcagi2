"""Worker extension for E051 KV-byte parity (test servers only).

Loaded with `--worker-extension-cls e051_ext.E051Ext` (PYTHONPATH must include this directory); called through the
dev-mode `POST /collective_rpc` endpoint (VLLM_SERVER_DEV_MODE=1). It adds no behaviour to serving itself.

- e051_layout(): how the KV pool is laid out (storages, sizes, page size, groups).
- e051_snapshot(): for every request in the runner's block tables, per KV-cache group, the block ids and a sha1 of
  each block's page bytes. Model numerics on this stack are nondeterministic, so parity is checked on the KV bytes a
  request attends to (GPU prefix hit vs host restore) instead of on sampled logprobs.
- e051_poison(): fill the whole KV pool with 0xA5. Only call it with no request running and after
  /reset_prefix_cache, so a host load that silently writes nothing cannot pass by reusing a block's stale bytes.
"""
import hashlib

import torch

PAGE = 113_920  # bytes per block in the packed DSv4.1 pool (all groups share one pool of these pages)


class E051Ext:
    def _e051_pool(self):
        mr = self.model_runner
        stores = {}
        for t in mr.kv_caches:
            s = t.untyped_storage()
            stores[s.data_ptr()] = s
        return mr, list(stores.values())

    def e051_layout(self):
        mr, stores = self._e051_pool()
        bt = mr.block_tables
        return {
            "num_layer_tensors": len(mr.kv_caches),
            "storages": [{"ptr": s.data_ptr(), "nbytes": s.nbytes(), "pages": s.nbytes() / PAGE} for s in stores],
            "layer0": {"shape": list(mr.kv_caches[0].shape), "dtype": str(mr.kv_caches[0].dtype),
                       "stride": list(mr.kv_caches[0].stride()),
                       "offset_bytes": mr.kv_caches[0].data_ptr() - stores[0].data_ptr()},
            "block_sizes": list(bt.block_sizes), "kernel_block_sizes": list(bt.kernel_block_sizes),
            "num_groups": bt.num_kv_cache_groups,
            "reqs": dict(mr.req_states.req_id_to_index),
        }

    def e051_snapshot(self):
        mr, stores = self._e051_pool()
        assert len(stores) == 1, f"expected one packed KV pool, got {len(stores)}"
        pool = torch.empty(0, dtype=torch.uint8, device=mr.kv_caches[0].device)
        pool.set_(stores[0])
        pool = pool[: stores[0].nbytes() // PAGE * PAGE].view(-1, PAGE)
        bt = mr.block_tables
        torch.cuda.synchronize()
        out = {}
        for req_id, idx in mr.req_states.req_id_to_index.items():
            groups = []
            for g in range(bt.num_kv_cache_groups):
                n = int(bt.num_blocks.np[g, idx])
                bpk = bt.blocks_per_kv_block[g]
                ids = bt.block_tables[g].gpu[idx, :n].tolist()
                if bpk > 1:  # kernel blocks -> KV blocks
                    ids = ids[::bpk]
                    ids = [b // bpk for b in ids]
                pages = pool[torch.tensor(ids, dtype=torch.long, device=pool.device)].cpu() if ids else None
                hashes = [hashlib.sha1(pages[i].numpy().tobytes()).hexdigest()[:16] for i in range(len(ids))]
                groups.append({"ids": ids, "sha": hashes})
            out[req_id] = groups
        return out

    def e051_poison(self):
        mr, stores = self._e051_pool()
        for s in stores:
            t = torch.empty(0, dtype=torch.uint8, device=mr.kv_caches[0].device)
            t.set_(s)
            t.fill_(0xA5)
        torch.cuda.synchronize()
        return "poisoned"
