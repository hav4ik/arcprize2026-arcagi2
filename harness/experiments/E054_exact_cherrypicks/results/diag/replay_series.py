import json, sys
R = __import__('sys').argv[1] if len(__import__('sys').argv) > 1 else '.'
for a in ("old", "both"):
    rows = [json.loads(l) for l in open(f"{R}/{a}/metrics.jsonl")]
    t_end = rows[-1]["t"]
    for name, lo, hi in (("C36", t_end - 3620, t_end - 1810), ("C48", t_end - 1810, t_end)):
        seg = [r for r in rows if lo <= r["t"] < hi]
        win = seg[len(seg) * 2 // 3:]  # the last ~600 s = the measurement window
        kv = [r.get("vllm:kv_cache_usage_perc", 0) for r in win]
        q0, q1 = win[0]["vllm:prefix_cache_queries"], win[-1]["vllm:prefix_cache_queries"]
        h0, h1 = win[0]["vllm:prefix_cache_hits"], win[-1]["vllm:prefix_cache_hits"]
        run = [r.get("vllm:num_requests_running", 0) for r in win]
        wait = [r.get("vllm:num_requests_waiting", 0) for r in win]
        print(f"{a:5s} {name}: window kv mean {sum(kv)/len(kv):.3f} max {max(kv):.3f} frac>0.95 {sum(k>0.95 for k in kv)/len(kv):.2f} | "
              f"hit {((h1-h0)/(q1-q0) if q1>q0 else float('nan')):.3f} | running mean {sum(run)/len(run):.1f} waiting mean {sum(wait)/len(wait):.1f} | "
              f"preempt {win[-1].get('vllm:num_preemptions')}")
