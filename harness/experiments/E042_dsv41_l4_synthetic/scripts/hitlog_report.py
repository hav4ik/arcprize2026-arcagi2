#!/usr/bin/env python3
"""E042 RW: split prefix-cache misses in a prefix_hitlog.py JSONL into their causes.

Per turn (after the first): R = reusable tokens (previous prompt + sampled ids), d = first token where the new
prompt departs from them (d = R when it extends them exactly), C = tokens the server served from cache.
The best possible hit is the last full 128-token block before min(R, d); the new suffix past R is never cached.
- re-tokenization loss: R - min(R, d), the part of the previous turn that the re-rendered prompt spells differently;
- eviction loss: best possible hit - C, cache that existed for this exact text but was gone (or unusable) by now,
  split by whether the turn extended the previous one exactly or broke (a break moves the resume point back);
- new tokens: prompt - R (tool output, notes, next generation header) plus every first prompt.
Usage: hitlog_report.py <jsonl> [--from-min M]
"""
import argparse
import json
from collections import defaultdict

B = 128


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl")
    ap.add_argument("--from-min", type=float, default=0.0)
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.jsonl)]
    turns = [r for r in rows if "turn" in r and r["t_start"] >= a.from_min * 60 and r.get("cached") is not None]
    first = [r for r in turns if r["turn"] == 0]
    later = [r for r in turns if r["turn"] > 0]
    tot = defaultdict(float)
    cls = defaultdict(int)
    prev_dur = {}
    evict_by_prev_dur = defaultdict(lambda: [0, 0])
    for r in sorted(rows, key=lambda x: (x.get("attempt", ""), x.get("turn", -1))):
        if "turn" in r:
            r["_prev_dur"] = prev_dur.get(r["attempt"])
            prev_dur[r["attempt"]] = r["dur"]
    for r in first:
        tot["queried"] += r["prompt"]
        tot["cached"] += r["cached"]
        tot["new"] += r["prompt"] - r["cached"]
    for r in later:
        R, d, C, P = r["reusable"], r["first_diff"], r["cached"], r["prompt"]
        best = (min(R, d) // B) * B
        retok = R - min(R, d)
        evict = max(0, best - C)
        tot["queried"] += P
        tot["cached"] += C
        tot["new"] += P - R
        tot["retok"] += retok
        tot["evict"] += evict
        tot["evict_exact" if r["exact_extension"] else "evict_break"] += evict
        tot["block_tail"] += min(R, d) - best
        key = ("exact" if r["exact_extension"] else "break") + ("+evicted" if evict > 2 * B else "")
        cls[key] += 1
        pd = r["_prev_dur"]
        if pd is not None:
            b = "<5m" if pd < 300 else "5-10m" if pd < 600 else "10-20m" if pd < 1200 else ">20m"
            evict_by_prev_dur[b][0] += 1
            evict_by_prev_dur[b][1] += evict > 2 * B
    q = tot["queried"] or 1
    span = (max(r["t_start"] + r["dur"] for r in turns) - min(r["t_start"] for r in turns)) if turns else 0
    gen = sum(r["gen"] for r in turns)
    print(f"turns {len(turns)} ({len(first)} first), span {span / 60:.1f} min, sampled {gen} tokens "
          f"({gen / max(span, 1):.1f} tok/s)")
    print(f"queried {tot['queried']:.0f}  cached {tot['cached']:.0f}  hit {tot['cached'] / q:.3f}")
    for k in ("new", "retok", "evict", "evict_exact", "evict_break", "block_tail"):
        print(f"  miss {k:10s} {tot[k]:10.0f}  {tot[k] / q:6.3f} of queried")
    print("turn classes:", dict(cls))
    print("eviction (>2 blocks short) by previous turn's duration:",
          {k: f"{v[1]}/{v[0]}" for k, v in sorted(evict_by_prev_dur.items())})
    br = [r for r in later if not r["exact_extension"]]
    if br:
        print(f"breaks: {len(br)} of {len(later)} turns; tokens of the previous turn after the break, "
              f"p50 {sorted(r['reusable'] - r['first_diff'] for r in br)[len(br) // 2]}")


if __name__ == "__main__":
    main()
