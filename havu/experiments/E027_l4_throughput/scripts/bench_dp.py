"""Box-level decode throughput over several servers (data parallel): bench_decode.py's prompts and output length,
with request i sent to server i % len(servers). Levels are total concurrent requests on the box.
usage: python bench_dp.py <base1,base2,...> <label> <out.json> [levels...]"""
import json, sys, time, uuid, threading, urllib.request
bases = sys.argv[1].split(","); label, out = sys.argv[2:4]
levels = [int(x) for x in sys.argv[4:]] or [8, 16, 32, 48]
OUT_TOK = 3000
tasks = json.load(open("/root/arc/input/arc-prize-2026-arc-agi-2/arc-agi_evaluation_challenges.json"))


def grid(g):
    return "\n".join("".join(str(c) for c in r) for r in g)


def prompt_for(tid):
    t = tasks[tid]
    s = f"[{uuid.uuid4().hex}]\nFind the rule that maps each input grid to its output grid.\n\n"
    for i, p in enumerate(t["train"]):
        s += f"Example {i+1} input:\n{grid(p['input'])}\nExample {i+1} output:\n{grid(p['output'])}\n\n"
    return s + f"Test input:\n{grid(t['test'][0]['input'])}\n\nThink step by step about the rule.\n<think>\n"


ids = sorted(tasks)


def one(i, res):
    body = json.dumps({"model": "qwen3.8-27b", "prompt": prompt_for(ids[i % len(ids)]), "max_tokens": OUT_TOK,
                       "min_tokens": OUT_TOK, "ignore_eos": True, "temperature": 1.0, "top_p": 0.95, "top_k": 20}).encode()
    t0 = time.time()
    try:
        r = json.load(urllib.request.urlopen(urllib.request.Request(
            bases[i % len(bases)] + "/v1/completions", data=body, headers={"Content-Type": "application/json"}),
            timeout=3600))
        res.append({"ptok": r["usage"]["prompt_tokens"], "ctok": r["usage"]["completion_tokens"], "s": time.time() - t0})
    except Exception as e:
        res.append({"ptok": 0, "ctok": 0, "s": time.time() - t0, "err": str(e)[:200]})


results = []
for n in levels:
    res = []
    th = [threading.Thread(target=one, args=(i, res)) for i in range(n)]
    t0 = time.time()
    [t.start() for t in th]
    [t.join() for t in th]
    wall = time.time() - t0
    ok = [r for r in res if not r.get("err")]
    ct = sum(r["ctok"] for r in ok)
    row = {"label": label, "concurrency": n, "servers": len(bases), "wall_s": round(wall, 1), "gen_tokens": ct,
           "tok_s": round(ct / wall, 1), "per_req_tok_s": round(sum(r["ctok"] / r["s"] for r in ok) / max(1, len(ok)), 2),
           "mean_prompt_tok": round(sum(r["ptok"] for r in ok) / max(1, len(ok))), "errors": len(res) - len(ok)}
    results.append(row)
    print(row, flush=True)
json.dump(results, open(out, "w"), indent=1)
