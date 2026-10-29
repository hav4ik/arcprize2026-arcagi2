"""Greedy parity probe: send each request JSON (temperature 0, logprobs 5, max_tokens 200, cache_salt "parity") to a
server; save the generated tokens and logprobs. Compare mode: first-50-token agreement and mean |dlogprob| between two
saved runs. Usage: parity.py send <base> <out.json> <req.json>...  |  parity.py compare <a.json> <b.json>"""
import json, sys, urllib.request
if sys.argv[1] == "send":
    base, out, reqs = sys.argv[2], sys.argv[3], sys.argv[4:]
    res = {}
    for p in reqs:
        r = json.load(open(p))
        r.update(temperature=0, max_tokens=200, logprobs=True, top_logprobs=5, cache_salt=__import__("os").environ.get("PARITY_SALT", "parity"))
        r.pop("top_p", None)
        q = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(r).encode(), headers={"Content-Type": "application/json"})
        o = json.load(urllib.request.urlopen(q, timeout=900))
        c = o["choices"][0]
        lp = (c.get("logprobs") or {}).get("content") or []
        res[p] = {"prompt_tokens": o["usage"]["prompt_tokens"], "tokens": [x["token"] for x in lp], "logprobs": [x["logprob"] for x in lp],
                  "finish": c.get("finish_reason"), "reasoning": (c["message"].get("reasoning_content") or c["message"].get("reasoning") or "")[:300]}
        print(p.split("/")[-2] + "/" + p.split("/")[-1], "prompt", res[p]["prompt_tokens"], "gen", len(lp), "finish", res[p]["finish"], "|", res[p]["reasoning"][:120].replace("\n", " "))
    json.dump(res, open(out, "w"))
else:
    a, b = json.load(open(sys.argv[2])), json.load(open(sys.argv[3]))
    for k in a:
        if k not in b:
            continue
        ta, tb = a[k]["tokens"], b[k]["tokens"]
        first_diff = next((i for i, (x, y) in enumerate(zip(ta, tb)) if x != y), min(len(ta), len(tb)))
        n = min(len(ta), len(tb), first_diff)
        dl = [abs(x - y) for x, y in zip(a[k]["logprobs"][:n], b[k]["logprobs"][:n])]
        print(f"{k.split('/')[-2]}/{k.split('/')[-1]}: prompt {a[k]['prompt_tokens']} vs {b[k]['prompt_tokens']}; identical first "
              f"{first_diff} tokens (first 50 equal: {first_diff >= 50}); mean |dlogprob| over the common prefix {sum(dl) / max(1, len(dl)):.4f} (n {len(dl)})"
              + (f"; first diff {ta[first_diff:first_diff + 3]} vs {tb[first_diff:first_diff + 3]}" if first_diff < min(len(ta), len(tb)) else ""))
