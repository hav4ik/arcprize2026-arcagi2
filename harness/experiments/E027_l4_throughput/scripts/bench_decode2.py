"""Decode throughput on real ARC text: N concurrent /v1/completions with a fixed prompt length and output length.
Prompts start with a unique nonce (no prefix reuse) and are padded with other ARC tasks to the target length.
usage: python bench_decode2.py <base> <label> <out.json> <prompt_tokens> <out_tokens> <levels...>"""
import json, sys, time, uuid, threading, urllib.request
base, label, out = sys.argv[1:4]
PTOK, OTOK = int(sys.argv[4]), int(sys.argv[5])
levels = [int(x) for x in sys.argv[6:]]
tasks = json.load(open("/root/arc/input/arc-prize-2026-arc-agi-2/arc-agi_evaluation_challenges.json"))
ids = sorted(tasks)


def grid(g):
    return "\n".join("".join(str(c) for c in r) for r in g)


def task_text(tid):
    t = tasks[tid]
    s = ""
    for i, p in enumerate(t["train"]):
        s += f"Example {i + 1} input:\n{grid(p['input'])}\nExample {i + 1} output:\n{grid(p['output'])}\n\n"
    return s + f"Test input:\n{grid(t['test'][0]['input'])}\n\n"


def prompt_for(i):
    # ~3.2 characters per token for these grids; pad with following tasks until the target length
    s = f"[{uuid.uuid4().hex}]\nFind the rule that maps each input grid to its output grid.\n\n"
    j = i
    while len(s) < PTOK * 3.2:
        s += task_text(ids[j % len(ids)])
        j += 1
    return s[: int(PTOK * 3.2)] + "\nThink step by step about the rule.\n<think>\n"


def one(i, res):
    body = json.dumps({"model": "qwen3.8-27b", "prompt": prompt_for(i), "max_tokens": OTOK, "min_tokens": OTOK,
                       "ignore_eos": True, "temperature": 1.0, "top_p": 0.95, "top_k": 20}).encode()
    t0 = time.time()
    try:
        r = json.load(urllib.request.urlopen(urllib.request.Request(
            base + "/v1/completions", data=body, headers={"Content-Type": "application/json"}), timeout=3600))
        res.append({"ptok": r["usage"]["prompt_tokens"], "ctok": r["usage"]["completion_tokens"], "s": time.time() - t0})
    except Exception as e:  # keep the level's other requests
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
    row = {"label": label, "concurrency": n, "prompt_tok": round(sum(r["ptok"] for r in ok) / max(1, len(ok))),
           "out_tok": OTOK, "wall_s": round(wall, 1), "gen_tokens": ct, "tok_s": round(ct / wall, 1),
           "per_req_tok_s": round(sum(r["ctok"] / r["s"] for r in ok) / max(1, len(ok)), 2),
           "errors": len(res) - len(ok)}
    results.append(row)
    print(row, flush=True)
json.dump(results, open(out, "w"), indent=1)
