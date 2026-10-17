"""Peak decode throughput on real ARC prompts: N concurrent /v1/completions, fixed output length, no prefix reuse.
usage: python bench_decode.py <base> <label> <out.json> [levels...]"""
import json, sys, time, uuid, threading, urllib.request
base, label, out = sys.argv[1:4]
levels = [int(x) for x in sys.argv[4:]] or [8, 16, 32, 48]
OUT_TOK = 3000
tasks = json.load(open("/root/arc/input/arc-prize-2026-arc-agi-2/arc-agi_evaluation_challenges.json"))
def grid(g): return "\n".join("".join(str(c) for c in r) for r in g)
def prompt_for(tid):
    t = tasks[tid]; s = f"[{uuid.uuid4().hex}]\nFind the rule that maps each input grid to its output grid.\n\n"
    for i, p in enumerate(t["train"]):
        s += f"Example {i+1} input:\n{grid(p['input'])}\nExample {i+1} output:\n{grid(p['output'])}\n\n"
    return s + f"Test input:\n{grid(t['test'][0]['input'])}\n\nThink step by step about the rule.\n<think>\n"
ids = sorted(tasks)
def one(i, res):
    body = json.dumps({"model": "qwen3.8-27b", "prompt": prompt_for(ids[i % len(ids)]), "max_tokens": OUT_TOK,
                       "min_tokens": OUT_TOK, "ignore_eos": True, "temperature": 1.0, "top_p": 0.95, "top_k": 20}).encode()
    t0 = time.time()
    r = json.load(urllib.request.urlopen(urllib.request.Request(base + "/v1/completions", data=body,
                                         headers={"Content-Type": "application/json"}), timeout=3600))
    res.append({"ptok": r["usage"]["prompt_tokens"], "ctok": r["usage"]["completion_tokens"], "s": time.time() - t0})
results = []
for n in levels:
    res = []; th = [threading.Thread(target=one, args=(i, res)) for i in range(n)]
    t0 = time.time(); [t.start() for t in th]; [t.join() for t in th]; wall = time.time() - t0
    ct = sum(r["ctok"] for r in res)
    row = {"label": label, "concurrency": n, "wall_s": round(wall, 1), "gen_tokens": ct, "tok_s": round(ct / wall, 1),
           "per_req_tok_s": round(sum(r["ctok"] / r["s"] for r in res) / len(res), 2),
           "mean_prompt_tok": round(sum(r["ptok"] for r in res) / len(res))}
    results.append(row); print(row, flush=True)
json.dump(results, open(out, "w"), indent=1)
