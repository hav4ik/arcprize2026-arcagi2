"""Decode-vs-prefill logprob consistency test for Manh's vLLM server (text path).
  select  <traces dir> <tools.json> <out prompts.json> [n=32] [lo=30000] [hi=45000]
      real multi-turn transcripts (b32m_x2t traces), cut after a tool message so the next turn is the model's, rendered
      with the server's /tokenize (messages + tools, add_generation_prompt) to token ids of lo..hi tokens.
  decode  <prompts.json> <out.json> [concurrency=32] [max_tokens=256] [first_n=all]
      /v1/completions, prompt = token ids, temperature 0, logprobs 1, all requests at once (>= 8 concurrent decodes).
  prefill <prompts.json> <decode.json> <out.json> [step=1] [concurrency=4]
      for every k (step), prompt = prompt ids + generated[:k], max_tokens 1, logprobs 20: the logprob of generated[k]
      as computed by the prefill path (concurrency <= 4, so < 8 decodes are ever batched).
  compare <decode.json> <prefill.json> [<decode2.json>]  per-prompt and overall |d| stats (and decode-vs-decode)
Server: http://127.0.0.1:18084, model dsv41."""
import concurrent.futures as cf, glob, gzip, json, os, sys, time, urllib.request
URL = os.environ.get("AB_URL", "http://127.0.0.1:18084")
def post(path, body, timeout=1800):
    req = urllib.request.Request(URL + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())
def tid(tok):  # return_tokens_as_token_ids: "token_id:123"
    return int(tok.split(":", 1)[1])
cmd = sys.argv[1]
if cmd == "select":
    tdir, tools_f, out = sys.argv[2:5]
    n = int(sys.argv[5]) if len(sys.argv) > 5 else 32
    lo = int(sys.argv[6]) if len(sys.argv) > 6 else 30000
    hi = int(sys.argv[7]) if len(sys.argv) > 7 else 45000
    tools = json.load(open(tools_f))
    def ntok(msgs):
        r = post("/tokenize", {"model": "dsv41", "messages": msgs, "tools": tools, "add_generation_prompt": True})
        return r["tokens"]
    picks = []
    files = sorted(glob.glob(f"{tdir}/*.json.gz"), key=lambda f: -os.path.getsize(f))
    for f in files:
        msgs = json.load(gzip.open(f))["messages"]
        cuts = [i + 1 for i, m in enumerate(msgs) if m["role"] == "tool" and (i + 1 == len(msgs) or msgs[i + 1]["role"] == "assistant")]
        best = None
        for c in reversed(cuts):  # longest prefix within [lo, hi]
            try:
                ids = ntok(msgs[:c])
            except Exception as exc:
                print("tokenize failed", f, type(exc).__name__, str(exc)[:200]); break
            if len(ids) <= hi:
                if len(ids) >= lo:
                    best = (c, ids)
                break
        if best:
            picks.append({"trace": os.path.basename(f), "n_messages": best[0], "ids": best[1]})
            print(len(picks), os.path.basename(f), best[0], len(best[1]), flush=True)
        if len(picks) >= n:
            break
    json.dump(picks, open(out, "w"))
    print("selected", len(picks), "prompts, tokens", [len(p["ids"]) for p in picks])
elif cmd == "decode":
    prompts = json.load(open(sys.argv[2])); out = sys.argv[3]
    conc = int(sys.argv[4]) if len(sys.argv) > 4 else 32
    mt = int(sys.argv[5]) if len(sys.argv) > 5 else 256
    if len(sys.argv) > 6:
        prompts = prompts[:int(sys.argv[6])]
    def one(i):
        t = time.time()
        r = post("/v1/completions", {"model": "dsv41", "prompt": prompts[i]["ids"], "max_tokens": mt, "temperature": 0,
                                     "logprobs": 1, "return_tokens_as_token_ids": True, "ignore_eos": False})
        lp = r["choices"][0]["logprobs"]
        return {"i": i, "ids": [tid(x) for x in lp["tokens"]], "lps": lp["token_logprobs"],
                "finish": r["choices"][0].get("finish_reason"), "s": time.time() - t}
    t0 = time.time()
    with cf.ThreadPoolExecutor(conc) as ex:
        res = sorted(ex.map(one, range(len(prompts))), key=lambda r: r["i"])
    json.dump({"concurrency": conc, "wall_s": time.time() - t0, "results": res}, open(out, "w"))
    print(f"decode: {len(res)} prompts, conc {conc}, {sum(len(r['ids']) for r in res)} tokens, {time.time() - t0:.0f}s")
elif cmd == "prefill":
    prompts = json.load(open(sys.argv[2])); dec = json.load(open(sys.argv[3]))["results"]; out = sys.argv[4]
    step = int(sys.argv[5]) if len(sys.argv) > 5 else 1
    conc = int(sys.argv[6]) if len(sys.argv) > 6 else 4
    jobs = [(r["i"], k) for r in dec for k in range(0, len(r["ids"]), step)]
    def one(job):
        i, k = job; g = dec[i]["ids"]
        r = post("/v1/completions", {"model": "dsv41", "prompt": prompts[i]["ids"] + g[:k], "max_tokens": 1,
                                     "temperature": 0, "logprobs": 20, "return_tokens_as_token_ids": True})
        top = r["choices"][0]["logprobs"]["top_logprobs"][0] or {}
        top = {tid(t): v for t, v in top.items()}
        return {"i": i, "k": k, "lp": top.get(g[k]), "argmax": max(top, key=top.get) if top else None,
                "top1_lp": max(top.values()) if top else None}
    t0 = time.time(); res = []
    with cf.ThreadPoolExecutor(conc) as ex:
        for n, x in enumerate(ex.map(one, jobs)):
            res.append(x)
            if n % 500 == 0:
                print(f"prefill {n}/{len(jobs)} {time.time() - t0:.0f}s", flush=True)
    json.dump({"step": step, "concurrency": conc, "wall_s": time.time() - t0, "results": res}, open(out, "w"))
    print(f"prefill: {len(res)} positions, {time.time() - t0:.0f}s")
elif cmd == "compare":
    dec = json.load(open(sys.argv[2])); pre = json.load(open(sys.argv[3]))["results"]
    D = {r["i"]: r for r in dec["results"]}
    per = {}; alld = []; miss = 0; flips = 0
    for x in pre:
        dl = D[x["i"]]["lps"][x["k"]]
        if x["lp"] is None:
            miss += 1; continue
        d = dl - x["lp"]; per.setdefault(x["i"], []).append(d); alld.append(d)
        flips += x["argmax"] != D[x["i"]]["ids"][x["k"]]
    def stats(v):
        a = sorted(abs(t) for t in v)
        return (f"n {len(v)} mean|d| {sum(a) / len(a):.4f} p99 {a[min(len(a) - 1, int(0.99 * len(a)))]:.4f} max {a[-1]:.4f} "
                f">0.5: {sum(t > 0.5 for t in a)} mean d {sum(v) / len(v):+.4f}")
    for i in sorted(per):
        print(f"prompt {i:2d}: {stats(per[i])}")
    print(f"OVERALL decode(conc {dec['concurrency']}) vs prefill: {stats(alld)}; missing from prefill top-20 {miss}; "
          f"prefill argmax != decode token {flips}")
    if len(sys.argv) > 4:  # decode vs decode on the common prefix (same prompt, greedy)
        d2 = {r["i"]: r for r in json.load(open(sys.argv[4]))["results"]}
        dd = []; div = []
        for i, r in d2.items():
            a = D[i]; n = 0
            while n < min(len(a["ids"]), len(r["ids"])) and a["ids"][n] == r["ids"][n]:
                dd.append(a["lps"][n] - r["lps"][n]); n += 1
            div.append(n)
        print(f"decode vs decode2 on common prefixes: {stats(dd)}; common prefix lengths {div}")
