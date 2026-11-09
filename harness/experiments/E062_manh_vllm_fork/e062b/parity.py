#!/usr/bin/env python3
"""E062 server-level parity: B32's stack vs Manh's fork on identical token-id prompts.

  parity.py prompts --traces dsv41_traces.tar.gz --url http://127.0.0.1:8000 --out prompts.json [--n 24]
      cut real B32-replica tool transcripts after a tool message (sizes spread over ~2k-60k tokens), render each through
      the server's /tokenize (chat template, generation prompt), keep the ids; also keep the messages
  parity.py run --prompts prompts.json --url http://127.0.0.1:8000 --out run_X.json [--max-tokens 256] [--salt S]
      greedy /v1/completions on the token ids (temperature 0, top-5 logprobs), plus /tokenize of every prompt's messages
      (render parity)
  parity.py compare run_A.json run_B.json
      per prompt: first-token argmax agreement, |delta logprob| of A's first token, length of the common greedy prefix,
      fraction of A's greedy tokens that are B's top-1 at the same position while the prefixes agree, render equality
"""
import argparse
import gzip
import io
import json
import random
import statistics
import sys
import tarfile
import time
import urllib.request


def post(url, body, timeout=900):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def tokenize(url, messages):
    r = post(url + "/tokenize", {"model": "dsv41", "messages": messages, "add_generation_prompt": True})
    return r["tokens"]


def cmd_prompts(a):
    tf = tarfile.open(a.traces)
    names = sorted(m.name for m in tf.getmembers() if m.name.endswith(".json.gz"))
    random.Random(62).shuffle(names)
    out, sizes = [], []
    want = [2_000, 8_000, 15_000, 25_000, 35_000, 45_000, 60_000]
    for name in names:
        if len(out) >= a.n:
            break
        d = json.load(io.TextIOWrapper(gzip.GzipFile(fileobj=tf.extractfile(name)), encoding="utf-8"))
        msgs = d["messages"]
        cuts = [i + 1 for i, m in enumerate(msgs) if m["role"] == "tool"] or [2]
        cuts = [2] + cuts  # 2 = system + user (the first turn)
        target = want[len(out) % len(want)]
        best = None
        for c in cuts:
            try:
                ids = tokenize(a.url, msgs[:c])
            except Exception as exc:
                print("tokenize failed", name, c, exc, file=sys.stderr)
                break
            if best is None or abs(len(ids) - target) < abs(len(best[1]) - target):
                best = (c, ids)
            if len(ids) > target * 1.3:
                break
        if best:
            c, ids = best
            out.append({"trace": name, "cut": c, "messages": msgs[:c], "ids": ids})
            sizes.append(len(ids))
            print(f"{name} cut {c} -> {len(ids)} tokens (target {target})", flush=True)
    json.dump(out, open(a.out, "w"))
    print("prompts", len(out), "tokens min/median/max", min(sizes), statistics.median(sizes), max(sizes))


def cmd_run(a):
    ps = json.load(open(a.prompts))
    res = []
    for i, p in enumerate(ps):
        t = time.time()
        body = {"model": "dsv41", "prompt": p["ids"], "max_tokens": a.max_tokens, "temperature": 0.0, "logprobs": 5,
                "cache_salt": f"{a.salt}_{i}", "skip_special_tokens": False, "return_token_ids": True}
        r = post(a.url + "/v1/completions", body)
        ch = r["choices"][0]
        lp = ch.get("logprobs") or {}
        try:
            render = tokenize(a.url, p["messages"])
        except Exception as exc:
            render = f"error: {exc}"
        res.append({"i": i, "n_prompt": len(p["ids"]), "text": ch["text"], "tokens": lp.get("tokens"),
                    "token_ids": ch.get("token_ids"),
                    "token_logprobs": lp.get("token_logprobs"), "top_logprobs": lp.get("top_logprobs"),
                    "finish": ch.get("finish_reason"), "render_equal": render == p["ids"],
                    "render_len": len(render) if isinstance(render, list) else render, "s": time.time() - t})
        print(f"[{i}] {len(p['ids'])} tok prompt, {len(lp.get('tokens') or [])} out, {time.time() - t:.1f}s, "
              f"render_equal {render == p['ids']}", flush=True)
    json.dump(res, open(a.out, "w"))


def cmd_compare(a):
    A, B = json.load(open(a.a)), json.load(open(a.b))
    first_agree, dlp, common, top1_frac, render = [], [], [], [], []
    for x, y in zip(A, B):
        ta, tb = x["tokens"] or [], y["tokens"] or []
        first_agree.append(bool(ta and tb and ta[0] == tb[0]))
        if ta and tb and y["top_logprobs"] and x["token_logprobs"]:
            top_b = y["top_logprobs"][0] or {}
            if ta[0] in top_b:
                dlp.append(abs(top_b[ta[0]] - x["token_logprobs"][0]))
        n = 0
        while n < min(len(ta), len(tb)) and ta[n] == tb[n]:
            n += 1
        common.append(n)
        # at each position of the common prefix + the first divergence: is A's token B's argmax?
        k = min(len(ta), len(tb), n + 1)
        if k:
            top1_frac.append(sum(1 for j in range(k) if ta[j] == tb[j]) / k)
        render.append(y["render_equal"])
    n = len(first_agree)
    print(f"{n} prompts: first-token argmax agree {sum(first_agree)}/{n}; "
          f"|dlogprob| first token median {statistics.median(dlp) if dlp else float('nan'):.4f} "
          f"max {max(dlp) if dlp else float('nan'):.4f} (n={len(dlp)}); "
          f"common greedy prefix median {statistics.median(common)} mean {statistics.mean(common):.1f} "
          f"(full {sum(c >= len(x['tokens'] or []) for c, x in zip(common, A))}/{n}); "
          f"B render == prompt ids {sum(render)}/{n}")
    for x, c in zip(A, common):
        print(f"  [{x['i']}] prompt {x['n_prompt']}: common prefix {c}/{len(x['tokens'] or [])}")


def cmd_tf(a):
    """Teacher-forced: score a reference run's greedy continuation (token ids) under this server, one prefill each."""
    ps, ref = json.load(open(a.prompts)), json.load(open(a.ref))
    if a.limit:
        ps, ref = ps[:a.limit], ref[:a.limit]
    res = []
    for i, (p, r) in enumerate(zip(ps, ref)):
        cont = r.get("token_ids") or []
        if not cont:
            res.append(None)
            continue
        body = {"model": "dsv41", "prompt": p["ids"] + cont, "max_tokens": 1, "temperature": 0.0,
                "prompt_logprobs": 0, "cache_salt": f"{a.salt}_tf_{i}"}
        t = time.time()
        out = post(a.url + "/v1/completions", body)
        pl = out["choices"][0].get("prompt_logprobs") or out.get("prompt_logprobs")
        lps = []
        for j, tok in enumerate(cont):
            d = pl[len(p["ids"]) + j] or {}
            v = d.get(str(tok)) or d.get(tok)
            lps.append(v["logprob"] if isinstance(v, dict) else v)
        res.append(lps)
        print(f"[{i}] {len(cont)} tokens, nll {-sum(lps) / len(lps):.4f}, {time.time() - t:.1f}s", flush=True)
    json.dump(res, open(a.out, "w"))


def harness_tools(kit):
    """The tools list and chat-template kwargs the B32 solver's harness sends (solver_dsv41 HARNESS + B32's flags)."""
    sys.path.insert(0, kit + "/repo/evals")
    import run_deepseek_tools as h
    args = h.parser().parse_args([
        "--mode", "grid", "--effort", "low", "--prompt", "lean2sub", "--answer-tool", "--final-mode", "session",
        "--lenient-parse", "--max-tokens", "64000", "--max-total-output", "80000", "--budget-soft", "40000",
        "--budget-notes", "0.01", "--budget-warn", "0.75", "--effort-header", "minimal", "--turns-note", "5",
        "--train-check", "--rescue-code", "--token-exact", "--model", "dsv41", "--fallback-solve", "--multi-test",
        "--auto-repair", "--multi-budget", "0.5", "--context-limit", "163840", "--max-turns", "40",
        "--base-url", "http://127.0.0.1:1/v1", "--sandbox-url", "local", "--attempts", "2"])
    return h.tools_for(args), {"thinking": True, "reasoning_effort": 50}


def cmd_render(a):
    """Render every prompt's messages the way the harness does (tools + kwargs; full with the generation prompt, and
    the prefix before the last message without it, as --token-exact's prefix check does) and save the ids."""
    tools, kw = harness_tools(a.kit)
    out = []
    for i, p in enumerate(json.load(open(a.prompts))):
        def r(msgs, gen):
            return post(a.url + "/tokenize", {"model": "dsv41", "messages": msgs, "tools": tools,
                                              "chat_template_kwargs": kw, "add_generation_prompt": gen})["tokens"]
        full = r(p["messages"], True)
        upto = r(p["messages"][:-1], False)
        out.append({"full": full, "upto": upto, "prefix_stable": full[:len(upto)] == upto})
    json.dump(out, open(a.out, "w"))
    print(f"{len(out)} renders; prefix-stable {sum(x['prefix_stable'] for x in out)}/{len(out)}")


def cmd_rendercompare(a):
    A, B = json.load(open(a.a)), json.load(open(a.b))
    eq_full = sum(x["full"] == y["full"] for x, y in zip(A, B))
    eq_upto = sum(x["upto"] == y["upto"] for x, y in zip(A, B))
    print(f"{len(A)} renders: full equal {eq_full}, prefix equal {eq_upto}; prefix-stable A {sum(x['prefix_stable'] for x in A)} "
          f"B {sum(y['prefix_stable'] for y in B)}")
    for i, (x, y) in enumerate(zip(A, B)):
        if x["full"] != y["full"]:
            n = 0
            while n < min(len(x["full"]), len(y["full"])) and x["full"][n] == y["full"][n]:
                n += 1
            print(f"  [{i}] lengths {len(x['full'])} vs {len(y['full'])}, first difference at {n}")


def cmd_tfcompare(a):
    X, Y = json.load(open(a.a)), json.load(open(a.b))
    d, nx, ny = [], [], []
    for x, y in zip(X, Y):
        if not x or not y:
            continue
        d += [abs(u - v) for u, v in zip(x, y)]
        nx.append(-sum(x) / len(x))
        ny.append(-sum(y) / len(y))
    d.sort()
    q = lambda f: d[min(len(d) - 1, int(f * len(d)))]
    print(f"{len(nx)} continuations, {len(d)} tokens: per-token |dlogprob| mean {statistics.mean(d):.4f} "
          f"median {q(0.5):.4f} p90 {q(0.9):.4f} p99 {q(0.99):.4f} max {d[-1]:.3f}; "
          f"mean NLL A {statistics.mean(nx):.4f} B {statistics.mean(ny):.4f} (B-A {statistics.mean(ny) - statistics.mean(nx):+.4f})")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prompts")
    p.add_argument("--traces", required=True)
    p.add_argument("--url", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--n", type=int, default=24)
    r = sub.add_parser("run")
    r.add_argument("--prompts", required=True)
    r.add_argument("--url", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--max-tokens", type=int, default=256)
    r.add_argument("--salt", default="e062")
    c = sub.add_parser("compare")
    c.add_argument("a")
    c.add_argument("b")
    t = sub.add_parser("tf")
    t.add_argument("--prompts", required=True)
    t.add_argument("--ref", required=True)
    t.add_argument("--url", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--salt", default="e062")
    t.add_argument("--limit", type=int, default=0)
    tc = sub.add_parser("tfcompare")
    tc.add_argument("a")
    tc.add_argument("b")
    rr = sub.add_parser("render")
    rr.add_argument("--prompts", required=True)
    rr.add_argument("--url", required=True)
    rr.add_argument("--out", required=True)
    rr.add_argument("--kit", default="/root/kit_b")
    rc = sub.add_parser("rendercompare")
    rc.add_argument("a")
    rc.add_argument("b")
    a = ap.parse_args()
    {"render": cmd_render, "rendercompare": cmd_rendercompare, "prompts": cmd_prompts, "run": cmd_run, "compare": cmd_compare, "tf": cmd_tf, "tfcompare": cmd_tfcompare}[a.cmd](a)


if __name__ == "__main__":
    main()
