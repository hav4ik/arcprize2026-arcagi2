"""E067 memory lab: per-step GPU memory logging for vLLM GPU workers.

Loaded by Python at startup when this directory is on PYTHONPATH (it is a
`sitecustomize`). It does nothing unless MEMLAB_LOG is set. When set, it
patches `vllm.v1.worker.gpu_worker.Worker` as soon as that module is imported
(in the engine-core / worker process) and writes JSON lines to
"$MEMLAB_LOG.<pid>.jsonl":

  startup events  loaded / profiled / kv_alloc / ready / cap
  one line per engine step (window = this execute_model .. next one):
    ntok   scheduled tokens, nreq requests, npf requests with >1 token,
    maxctx / sumctx  context length after the step (computed + scheduled),
    a / ap  allocated now / peak in the window (GiB),
    r / rp  reserved now / peak in the window, isp inactive split bytes,
    free    cudaMemGetInfo free, ret alloc retries, oom, seg segments, dalloc
  frag    every MEMLAB_SNAP_EVERY steps: segments, largest free block, ...
  oom     on torch.OutOfMemoryError (then re-raised)

Options (all optional):
  MEMLAB_CAP_MB=N           after warmup, cap the torch allocator at
                            reserved + free - N MiB (what the teammate fork's
                            VLLM_GPU_DRIVER_RESERVE_MB does)
  MEMLAB_EXPANDABLE_AFTER=1 after warmup, switch the allocator to expandable
                            segments (the fork does this together with the cap)
  MEMLAB_EMPTY_CACHE_EVERY=N  torch.cuda.empty_cache() every N steps
  MEMLAB_SNAP_EVERY=N       fragmentation snapshot period (default 400 steps)
"""

import os
import sys

if os.environ.get("MEMLAB_LOG"):
    import importlib.abc
    import json
    import time

    _TARGET = "vllm.v1.worker.gpu_worker"
    _GiB = float(1 << 30)

    def _g(x):
        return round(x / _GiB, 4)

    class _State:
        fh = None
        step = 0
        meta = None
        t0 = time.time()

    def _out(rec):
        if _State.fh is None:
            _State.fh = open(f"{os.environ['MEMLAB_LOG']}.{os.getpid()}.jsonl", "a", buffering=1)
        rec["t"] = round(time.time() - _State.t0, 3)
        _State.fh.write(json.dumps(rec) + "\n")

    def _stats(torch, dev=None):
        s = torch.cuda.memory_stats(dev)
        free, total = torch.cuda.mem_get_info(dev)
        return {
            "a": _g(s.get("allocated_bytes.all.current", 0)),
            "ap": _g(s.get("allocated_bytes.all.peak", 0)),
            "r": _g(s.get("reserved_bytes.all.current", 0)),
            "rp": _g(s.get("reserved_bytes.all.peak", 0)),
            "isp": _g(s.get("inactive_split_bytes.all.current", 0)),
            "free": _g(free),
            "ret": s.get("num_alloc_retries", 0),
            "oom": s.get("num_ooms", 0),
            "seg": s.get("segment.all.current", 0),
            "dalloc": s.get("num_device_alloc", 0),
        }

    def _frag(torch):
        segs = torch.cuda.memory_snapshot()
        largest_free = 0
        free_total = 0
        n_split = 0
        big = []
        for sg in segs:
            inact = 0
            for b in sg["blocks"]:
                if b["state"] == "inactive":
                    inact += b["size"]
                    largest_free = max(largest_free, b["size"])
            free_total += inact
            if 0 < inact < sg["total_size"]:
                n_split += 1
            big.append(sg["total_size"])
        big.sort(reverse=True)
        return {
            "ev": "frag",
            "nseg": len(segs),
            "nseg_split_with_free": n_split,
            "cached_free": _g(free_total),
            "largest_free": _g(largest_free),
            "top_segs_gib": [_g(x) for x in big[:6]],
        }

    def _meta(so):
        ctx = []
        npf = 0
        nst = so.num_scheduled_tokens
        for r in so.scheduled_new_reqs:
            n = nst.get(r.req_id, 0)
            ctx.append(r.num_computed_tokens + n)
            npf += n > 1
        c = so.scheduled_cached_reqs
        for i, rid in enumerate(c.req_ids):
            n = nst.get(rid, 0)
            ctx.append(c.num_computed_tokens[i] + n)
            npf += n > 1
        return {
            "ntok": so.total_num_scheduled_tokens,
            "nreq": len(nst),
            "npf": npf,
            "maxctx": max(ctx) if ctx else 0,
            "sumctx": sum(ctx),
            "pre": len(so.preempted_req_ids or ()),
        }

    def _patch(mod):
        import torch

        W = mod.Worker
        o_load = W.load_model
        o_det = W.determine_available_memory
        o_init = W.initialize_from_config
        o_warm = W.compile_or_warm_up_model
        o_exec = W.execute_model
        o_samp = getattr(W, "sample_tokens", None)
        every_empty = int(os.environ.get("MEMLAB_EMPTY_CACHE_EVERY", "0"))
        snap_every = int(os.environ.get("MEMLAB_SNAP_EVERY", "400"))

        def load_model(self, *a, **k):
            r = o_load(self, *a, **k)
            _out({"ev": "loaded", **_stats(torch),
                  "weights": _g(getattr(self.model_runner, "model_memory_usage", 0))})
            return r

        def determine_available_memory(self, *a, **k):
            r = o_det(self, *a, **k)
            snap = getattr(self, "init_snapshot", None)
            _out({
                "ev": "profiled", **_stats(torch),
                "kv_budget": _g(r),
                "requested": _g(getattr(self, "requested_memory", 0)),
                "init_free": _g(snap.free_memory) if snap else None,
                "init_used": _g(snap.cuda_memory) if snap else None,
                "init_torch_reserved": _g(snap.torch_memory) if snap else None,
                "total": _g(snap.total_memory) if snap else None,
                "peak_act": _g(getattr(self, "peak_activation_memory", 0)),
                "non_torch": _g(getattr(self, "non_torch_memory", 0)),
                "cg_est": _g(getattr(self, "cudagraph_memory_estimate", 0)),
            })
            return r

        def initialize_from_config(self, cfg, *a, **k):
            r = o_init(self, cfg, *a, **k)
            _out({"ev": "kv_alloc", **_stats(torch), "num_blocks": cfg.num_blocks})
            return r

        def compile_or_warm_up_model(self, *a, **k):
            r = o_warm(self, *a, **k)
            st = _stats(torch)
            _out({"ev": "ready", **st})
            _out(_frag(torch))
            cap_mb = int(os.environ.get("MEMLAB_CAP_MB", "0"))
            if cap_mb > 0:
                free, total = torch.cuda.mem_get_info()
                reserved = torch.cuda.memory_reserved()
                limit = reserved + free - (cap_mb << 20)
                torch.cuda.set_per_process_memory_fraction(limit / total)
                _out({"ev": "cap", "cap": _g(limit), "reserved": _g(reserved),
                      "free": _g(free), "keep_mb": cap_mb})
            if os.environ.get("MEMLAB_EXPANDABLE_AFTER") == "1":
                torch.cuda.memory._set_allocator_settings("expandable_segments:True")
                _out({"ev": "expandable_on"})
            torch.cuda.reset_peak_memory_stats()
            return r

        def _flush():
            if _State.meta is not None:
                _out({"ev": "step", "s": _State.step, **_State.meta, **_stats(torch)})
                if snap_every and _State.step % snap_every == 0:
                    _out(_frag(torch))
                if every_empty and _State.step % every_empty == 0:
                    torch.cuda.empty_cache()
                _State.meta = None

        def _oom(where, e):
            try:
                rec = {"ev": "oom", "where": where, "s": _State.step,
                       "msg": str(e)[:600], **(_State.meta or {}), **_stats(torch)}
                _out(rec)
                _out(_frag(torch))
            except Exception:
                pass

        def execute_model(self, so, *a, **k):
            _flush()
            _State.step += 1
            try:
                _State.meta = _meta(so)
            except Exception as e:  # never break serving because of logging
                _State.meta = {"meta_err": str(e)[:100]}
            torch.cuda.reset_peak_memory_stats()
            try:
                return o_exec(self, so, *a, **k)
            except torch.OutOfMemoryError as e:
                _oom("execute_model", e)
                raise

        W.load_model = load_model
        W.determine_available_memory = determine_available_memory
        W.initialize_from_config = initialize_from_config
        W.compile_or_warm_up_model = compile_or_warm_up_model
        W.execute_model = execute_model
        if o_samp is not None:
            def sample_tokens(self, *a, **k):
                try:
                    return o_samp(self, *a, **k)
                except torch.OutOfMemoryError as e:
                    _oom("sample_tokens", e)
                    raise
            W.sample_tokens = sample_tokens
        _out({"ev": "patched", "pid": os.getpid(), "t0": _State.t0,
              "alloc_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")})

    class _Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path, target=None):
            if name != _TARGET:
                return None
            for f in sys.meta_path:
                if f is self or not hasattr(f, "find_spec"):
                    continue
                spec = f.find_spec(name, path, target)
                if spec is not None:
                    break
            else:
                return None
            orig = spec.loader.exec_module

            def exec_module(module, _orig=orig):
                _orig(module)
                try:
                    _patch(module)
                except Exception as e:
                    _out({"ev": "patch_error", "err": repr(e)[:300]})

            spec.loader.exec_module = exec_module
            return spec

    sys.meta_path.insert(0, _Finder())
