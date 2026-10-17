"""Aggregates GPU kernel time by category from a torch-profiler trace (rank 0) written by vLLM.
usage: python prof_parse.py <trace dir>"""
import glob, gzip, json, re, sys, collections

files = sorted(glob.glob(sys.argv[1] + "/**/*.json*", recursive=True))
files = [f for f in files if "rank0" in f or "rank_0" in f or "TP0" in f] or files[:1]
f = files[0]
ev = json.load(gzip.open(f) if f.endswith(".gz") else open(f))
ev = ev["traceEvents"] if isinstance(ev, dict) else ev
k = [e for e in ev if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")]
cats = [("nccl/allreduce", r"nccl|allreduce|all_reduce|cross_device"),
        ("gemm (marlin/gemm/cutlass)", r"marlin|gemm|cutlass|sm89_|ampere_|cublas|matmul|xmma|s16816|wgmma"),
        ("full attention", r"unified_attention|attention|attn|paged|flash|_fwd_kernel|reshape_and_cache"),
        ("deltanet / conv (fla)", r"chunk|recurrent|gated_delta|delta|fla|causal_conv|conv1d|l2norm|solve_tril|cumsum"),
        ("norm / rope / act", r"rms|norm|rotary|rope|silu|gelu|act_and_mul|sigmoid|swiglu"),
        ("sampling / logits", r"sampl|topk|top_k|softmax|argmax|gumbel|penalt|logit|sort|radix|scan"),
        ("memcpy/memset", r"memcpy|memset|Memcpy|Memset"),
        ("elementwise/other", r".")]
by = collections.Counter(); names = collections.defaultdict(collections.Counter)
for e in k:
    n = e.get("name", "")
    for c, pat in cats:
        if re.search(pat, n, re.I):
            by[c] += e["dur"]; names[c][n[:80]] += e["dur"]; break
t0 = min(e["ts"] for e in k); t1 = max(e["ts"] + e["dur"] for e in k)
busy = sum(by.values()); span = t1 - t0
print(f"trace {f.split('/')[-1]}: span {span / 1e6:.2f} s, GPU kernel time {busy / 1e6:.2f} s ({100 * busy / span:.0f}% busy), {len(k)} kernels")
for c, t in by.most_common():
    print(f"  {c:28s} {t / 1e6:7.3f} s  {100 * t / busy:5.1f}%   top: " +
          "; ".join(f"{n} {v / 1e6:.2f}s" for n, v in names[c].most_common(2)))
