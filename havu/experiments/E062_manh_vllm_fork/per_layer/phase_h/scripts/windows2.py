"""Per-window table from the solver's metrics.jsonl (5-min windows as logged), with prompt tokens/s split into GPU
prefix-cached / CPU-restored (connector, external_prefix_cache_hits) / recomputed, using the solver's definitions:
q = prefix_cache_queries/s = gpu + host + uncached; gpu = prefix_hit * q. Usage: windows2.py <metrics.jsonl> [pool]"""
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1])]
pool = float(sys.argv[2]) if len(sys.argv) > 2 else 0
t0 = rows[0]["elapsed_h"] - rows[0]["window_s"] / 3600
print(f"{'min':>7} {'gen/s':>6} {'prompt/s':>8} {'GPU-hit/s':>9} {'CPU-rest/s':>10} {'recomp/s':>8} {'GPU%':>5} {'CPU%':>5} {'rec%':>5} {'KV':>5} {'run':>4} {'wait':>4} {'pre':>3} {'done':>4}" + (" KV/str" if pool else ""))
for r in rows:
    m = (r["elapsed_h"] - t0) * 60
    hit, host, unc = r.get("prefix_hit") or 0, r.get("host_tok_s") or 0, r.get("uncached_tok_s") or 0
    q = (host + unc) / (1 - hit) if hit < 1 else 0
    gpu = hit * q
    pct = lambda x: 100 * x / q if q else 0
    kvs = (r["kv"] * pool / r["srv_running"] / 1000) if pool and r.get("srv_running") else 0
    print(f"{m - r['window_s'] / 60:3.0f}-{m:<3.0f} {r['gen_tok_s']:6.0f} {q:8.0f} {gpu:9.0f} {host:10.0f} {unc:8.0f} {pct(gpu):5.1f} {pct(host):5.1f} {pct(unc):5.1f} {r['kv']:5.2f} {r['srv_running']:4.0f} {r['srv_waiting']:4.0f} {r['preemptions']:3.0f} {r['done']:4d}" + (f" {kvs:5.1f}k" if pool else ""))
