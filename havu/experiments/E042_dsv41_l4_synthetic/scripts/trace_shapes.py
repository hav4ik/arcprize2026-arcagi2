#!/usr/bin/env python3
"""E042 R: token shapes of the real DeepSeek-V4.1-Flash-1017 ARC runs, for trace-shaped load on the synthetic model.

notools: E034 (ARC Prize prompt, max effort, 334 calls): [prompt_tokens, completion_tokens] per call.
tools:   E030 hdrmin (sub80s40 + "Reasoning Effort: Minimal", the best tool config, 334 attempts): first prompt and per
         turn [completion_tokens, tool-result tokens] (= next prompt - this prompt - this completion).
Only token counts leave this script, no content. Also prints the steady-state KV fit: in a closed loop each slot sits
in a trace with probability proportional to its length, at a uniform position, and holds 1.6K + 0.355 x context
reported KV tokens (S4 fit).
--snapshot notools --conc C --out f.json writes C such slot contexts for decode_window.py --ctx-file.
"""

import argparse
import bisect
import gzip
import glob
import itertools
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RES = ROOT / "evals/results"


def notools():
    rows = [json.loads(l) for l in open(RES / "arcprize_deepseek-flash_max_e034_flash1017.jsonl")]
    return [[r["usage"]["prompt_tokens"], r["usage"]["completion_tokens"]] for r in rows if r.get("usage")]


def tools():
    out = []
    for f in sorted(glob.glob(str(RES / "tools_persistent_grid_lean2sub_deepseek-flash_low_e030_hdrmin/*.messages.json.gz"))):
        us = [e["usage"] for e in json.load(gzip.open(f))["events"] if isinstance(e, dict) and e.get("usage")]
        if not us:
            continue
        p = [u["prompt_tokens"] for u in us]
        c = [u["completion_tokens"] for u in us]
        r = [p[i + 1] - p[i] - c[i] for i in range(len(p) - 1)] + [0]
        out.append({"p0": p[0], "turns": [[ci, ri] for ci, ri in zip(c, r)]})
    return out


def kv(ctx):
    return 1600 + 0.355 * ctx


def pick(cum, rng):
    return bisect.bisect(cum, rng.random() * cum[-1])


def sampler_notools(shapes):
    cum = list(itertools.accumulate(n for _, n in shapes))
    def draw(rng):
        p, n = shapes[pick(cum, rng)]
        return p + rng.randrange(n)
    return draw


def sampler_tools(shapes):
    per = [list(itertools.accumulate(c for c, _ in s["turns"])) for s in shapes]
    cum = list(itertools.accumulate(c[-1] for c in per))
    def draw(rng):
        i = pick(cum, rng)
        a, k = shapes[i], pick(per[i], rng)
        return a["p0"] + sum(c + r for c, r in a["turns"][:k]) + rng.randrange(a["turns"][k][0])
    return draw


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "metadata/trace_shapes"))
    ap.add_argument("--snapshot", choices=["notools", "tools"])
    ap.add_argument("--shapes", help="shapes file for --snapshot (default: the extracted one)")
    ap.add_argument("--conc", type=int, default=64)
    ap.add_argument("--max-ctx", type=int, default=160000)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    if a.snapshot:
        d = Path(__file__).resolve().parents[1] / "metadata/trace_shapes"
        f = a.shapes or str(d / ("notools_e034.json" if a.snapshot == "notools" else "tools_e030_hdrmin.json"))
        draw = (sampler_notools if a.snapshot == "notools" else sampler_tools)(json.load(open(f)))
        rng = random.Random(a.seed)
        ctx = [min(draw(rng), a.max_ctx) for _ in range(a.conc)]
        json.dump(ctx, open(a.out, "w"))
        q = sorted(ctx)
        print(f"{a.conc} slot contexts -> {a.out}: mean {sum(q) / len(q):,.0f}, p50 {q[len(q) // 2]:,}, max {q[-1]:,}, "
              f"KV {sum(map(kv, q)):,.0f} reported tokens")
        return
    Path(a.out).mkdir(parents=True, exist_ok=True)
    nt, tl = notools(), tools()
    json.dump(nt, open(f"{a.out}/notools_e034.json", "w"))
    json.dump(tl, open(f"{a.out}/tools_e030_hdrmin.json", "w"))
    print(len(nt), "no-tools calls,", len(tl), "tool attempts ->", a.out)
    rng = random.Random(0)
    for name, shapes, mk in (("notools", nt, sampler_notools), ("tools", tl, sampler_tools)):
        snap = mk(shapes)
        ctx = sorted(snap(rng) for _ in range(20000))
        q = lambda f: ctx[int(f * len(ctx))]
        print(f"{name}: steady-state slot context mean {sum(ctx) / len(ctx):,.0f}, p50 {q(.5):,}, p90 {q(.9):,}, "
              f"p99 {q(.99):,}, max {ctx[-1]:,}")
        for cap_name, cap in (("E64", 1_267_921), ("E48", 2_451_609)):
            for c in (32, 48, 64, 80, 96, 112, 128):
                tot = sorted(sum(kv(snap(rng)) for _ in range(c)) for _ in range(2000))
                print(f"  {cap_name} C={c}: KV p50 {tot[1000] / cap:.0%}, p95 {tot[1900] / cap:.0%} of {cap:,}")


if __name__ == "__main__":
    main()
