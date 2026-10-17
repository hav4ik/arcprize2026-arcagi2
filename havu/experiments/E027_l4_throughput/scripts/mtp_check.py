"""Greedy multi-turn consistency check for speculative decoding with prefix caching (hybrid-model state bug).
run:     python mtp_check.py run <base> <out.json> [n_conv] [turn_tokens]
         Two passes over n_conv two-turn conversations (pass 2 hits the prefix cache for everything).
compare: python mtp_check.py compare <a.json> <b.json>   (first divergence per conversation and turn)"""
import json, sys, threading, urllib.request

TASKS = "/root/arc/input/arc-prize-2026-arc-agi-2/arc-agi_evaluation_challenges.json"


def grid(g):
    return "\n".join(" ".join(str(c) for c in r) for r in g)


def first_msg(t):
    s = "Find the transformation rule. Think step by step.\n\n"
    for i, p in enumerate(t["train"]):
        s += f"Example {i+1} input:\n{grid(p['input'])}\nExample {i+1} output:\n{grid(p['output'])}\n\n"
    return s + f"Test input:\n{grid(t['test'][0]['input'])}\n"


def chat(base, msgs, n):
    body = json.dumps({"model": "qwen3.8-27b", "messages": msgs, "max_tokens": n, "temperature": 0.0, "seed": 0}).encode()
    r = json.load(urllib.request.urlopen(urllib.request.Request(base + "/v1/chat/completions", data=body,
                                         headers={"Content-Type": "application/json"}), timeout=3600))
    m = r["choices"][0]["message"]
    return (m.get("reasoning_content") or m.get("reasoning") or "") + "\n<<C>>\n" + (m.get("content") or "")


def run(base, out, n_conv=16, n=400):
    tasks = json.load(open(TASKS)); ids = sorted(tasks)[:n_conv]
    res = {}
    for p in (1, 2):
        def one(tid):
            m1 = [{"role": "user", "content": first_msg(tasks[tid])}]
            t1 = chat(base, m1, n)
            m2 = m1 + [{"role": "assistant", "content": t1[-1500:]},
                       {"role": "user", "content": "Tool output:\n" + grid(tasks[tid]["train"][0]["output"]) + "\nContinue."}]
            t2 = chat(base, m2, n)
            res[f"p{p}:{tid}"] = [t1, t2]
        th = [threading.Thread(target=one, args=(t,)) for t in ids]
        [x.start() for x in th]; [x.join() for x in th]
    json.dump(res, open(out, "w"))
    print("wrote", out, len(res))


def div(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def compare(fa, fb):
    A, B = json.load(open(fa)), json.load(open(fb))
    for p in ("p1", "p2"):
        for turn in (0, 1):
            ds = [div(A[k][turn], B[k][turn]) for k in A if k.startswith(p) and k in B]
            same = sum(d is None for d in ds)
            early = sorted(d for d in ds if d is not None)
            print(f"{fa.split('/')[-1]} vs {fb.split('/')[-1]} {p} turn{turn + 1}: identical {same}/{len(ds)}, "
                  f"first-divergence chars {early[:8]}")


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run(sys.argv[2], sys.argv[3], *(int(x) for x in sys.argv[4:]))
    else:
        compare(sys.argv[2], sys.argv[3])
