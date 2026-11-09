"""Controlled multi-turn prefix-cache probe: does the next turn reuse the previous turn's *generated* tokens?
For each of N conversations (one at a time, server otherwise idle):
  turn 1: prompt P (an ARC task, ~T1 tokens), generate G tokens (temperature 0, ignore_eos)
  turn 2: prompt P + generated text + a short tool reply, max_tokens 1
Prefix-cache hits per request come from the server's Prometheus counters (vllm:prefix_cache_hits / queries).
If generated tokens are cached, turn-2 hits ~ len(P) + G (block-aligned); if not, ~ len(P) (block-aligned).
usage: python cache_probe.py <base> <label> <out.json> [n_conv] [gen_tokens]"""
import json, re, sys, time, urllib.request

base, label, out = sys.argv[1:4]
N = int(sys.argv[4]) if len(sys.argv) > 4 else 4
G = int(sys.argv[5]) if len(sys.argv) > 5 else 1500
tasks = json.load(open("/root/arc/input/arc-prize-2026-arc-agi-2/arc-agi_evaluation_challenges.json"))
ids = sorted(tasks)


def grid(g):
    return "\n".join(" ".join(str(c) for c in r) for r in g)


def prompt_for(tid):
    t = tasks[tid]
    s = "<|im_start|>user\nFind the transformation rule and use the python tool to check it.\n\n"
    for i, p in enumerate(t["train"]):
        s += f"Example {i+1} input:\n{grid(p['input'])}\nExample {i+1} output:\n{grid(p['output'])}\n\n"
    return s + f"Test input:\n{grid(t['test'][0]['input'])}<|im_end|>\n<|im_start|>assistant\n<think>\n"


def metrics():
    txt = urllib.request.urlopen(base + "/metrics", timeout=30).read().decode()
    def get(name):
        v = 0.0
        for m in re.finditer(rf"^{name}(?:_total)?(?:{{[^}}]*}})? ([0-9.e+]+)$", txt, re.M):
            v += float(m.group(1))
        return v
    return get("vllm:prefix_cache_hits"), get("vllm:prefix_cache_queries")


def complete(prompt, max_tokens, **kw):
    body = json.dumps({"model": "qwen3.8-27b", "prompt": prompt, "max_tokens": max_tokens, "temperature": 0.0, **kw}).encode()
    return json.load(urllib.request.urlopen(urllib.request.Request(base + "/v1/completions", data=body,
                                            headers={"Content-Type": "application/json"}), timeout=3600))


rows = []
for i in range(N):
    p = prompt_for(ids[(7 * i) % len(ids)])
    h0, q0 = metrics()
    r1 = complete(p, G, min_tokens=G, ignore_eos=True)
    h1, q1 = metrics()
    gen = r1["choices"][0]["text"]
    p2 = p + gen + "<|im_end|>\n<|im_start|>user\n<tool_response>\nok\n</tool_response><|im_end|>\n<|im_start|>assistant\n<think>\n"
    r2 = complete(p2, 1)
    h2, q2 = metrics()
    row = {"conv": i, "turn1_prompt": r1["usage"]["prompt_tokens"], "turn1_gen": r1["usage"]["completion_tokens"],
           "turn2_prompt": r2["usage"]["prompt_tokens"], "turn1_hits": h1 - h0, "turn2_hits": h2 - h1, "turn2_queries": q2 - q1}
    row["turn2_hit_frac"] = round(row["turn2_hits"] / max(1, row["turn2_prompt"]), 3)
    row["reused_beyond_turn1_prompt"] = row["turn2_hits"] - row["turn1_prompt"]
    rows.append(row)
    print(label, row, flush=True)
json.dump(rows, open(out, "w"), indent=1)
