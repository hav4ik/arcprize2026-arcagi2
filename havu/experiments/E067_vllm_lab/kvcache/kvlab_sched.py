"""Instrumented vLLM V1 scheduler for the E067 kvcache lab (read-only observer, no policy change).

Load with:  PYTHONPATH=<this dir> vllm serve ... --scheduler-cls kvlab_sched.LabScheduler
and KVLAB_SCHED_LOG=<path.jsonl>. Once per second it appends cumulative counters plus a snapshot:

  tok_total      tokens scheduled (prefill + decode)
  tok_decode     tokens scheduled in 1-token (decode) slots
  tok_prefill    tokens scheduled in prefill chunks
  tok_recompute  prefill tokens below the request's earlier high-water mark (= work redone after
                 preemption, net of whatever its own freed blocks still gave as prefix hits)
  preempt        preemptions
  admitted       requests moved WAITING -> RUNNING (new + resumed)
  hol_steps      steps where requests waited, running < max_num_seqs, and nothing was admitted
                 (head-of-line blocked on KV)
  snapshot: running, waiting, skipped_waiting (incl. WAITING_FOR_REMOTE_KVS), free blocks,
            free blocks that still hold a cached prefix (evictable cache), total blocks.
"""
import json
import os
import time

from vllm.v1.core.sched.async_scheduler import AsyncScheduler as _Base  # default when async scheduling is on


class LabScheduler(_Base):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        path = os.environ.get("KVLAB_SCHED_LOG")
        self._lab_f = open(path, "a", buffering=1) if path else None
        self._lab_t = 0.0
        self._lab_hw = {}
        self._lab = dict(steps=0, tok_total=0, tok_decode=0, tok_prefill=0, tok_recompute=0,
                         preempt=0, admitted=0, hol_steps=0)

    def schedule(self):
        n_wait_before = len(self.waiting) + len(self.skipped_waiting)
        out = super().schedule()
        a = self._lab
        a["steps"] += 1
        a["preempt"] += len(out.preempted_req_ids or ())
        n_new = len(out.scheduled_new_reqs) + len(out.scheduled_cached_reqs.resumed_req_ids
                                                    if hasattr(out.scheduled_cached_reqs, "resumed_req_ids") else ())
        a["admitted"] += n_new
        if n_wait_before and n_new == 0 and len(self.running) < self.max_num_running_reqs:
            a["hol_steps"] += 1
        hw = self._lab_hw
        for rid, n in out.num_scheduled_tokens.items():
            a["tok_total"] += n
            req = self.requests.get(rid)
            if req is None:
                continue
            end = req.num_computed_tokens  # already advanced by n in _update_after_schedule
            start = end - n
            if n == 1 and start >= req.num_prompt_tokens:
                a["tok_decode"] += 1
            else:
                a["tok_prefill"] += n
                prev = hw.get(rid, 0)
                if prev > start:
                    a["tok_recompute"] += min(prev, end) - start
            if end > hw.get(rid, 0):
                hw[rid] = end
        now = time.time()
        if self._lab_f is not None and now - self._lab_t >= 1.0:
            self._lab_t = now
            for rid in [r for r in hw if r not in self.requests]:
                del hw[rid]
            bp = self.kv_cache_manager.block_pool
            cached_free = sum(1 for b in bp.blocks if b.ref_cnt == 0 and b.block_hash is not None)
            rec = dict(a, t=now, running=len(self.running), waiting=len(self.waiting),
                       skipped_waiting=len(self.skipped_waiting), free_blocks=bp.get_num_free_blocks(),
                       cached_free_blocks=cached_free, total_blocks=bp.num_gpu_blocks)
            self._lab_f.write(json.dumps(rec) + "\n")
        return out
