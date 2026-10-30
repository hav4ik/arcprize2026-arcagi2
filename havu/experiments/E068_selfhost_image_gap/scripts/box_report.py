#!/usr/bin/env python3
"""E068: tables for the box runs (scripts/box_run.sh) against the DeepSeek API probe.

  box_report.py --probe results/probe_api.jsonl results/probe_ref.jsonl ... --nll results/nll_ref.jsonl ...
Probe: per file, overall and by grid size (cells <=36, 37-144, 145-400, >400): answered, size, colours, corners,
exact; each self-hosted arm minus the API, paired by grid (mean over its samples), 95% bootstrap CI over grids.
NLL: per tag, generated-token NLL over all rollouts and over the first assistant turn; paired differences between
tags per rollout (per-token NLL), bootstrap CI over rollouts.
"""
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

BINS = [(0, 36, "<=36"), (37, 144, "37-144"), (145, 400, "145-400"), (401, 10 ** 9, ">400")]
KEYS = ["parsed", "size", "colors", "corners", "exact"]


def boot(diffs, n=4000, seed=0):
    rng = random.Random(seed)
    m = sum(diffs) / len(diffs)
    bs = sorted(sum(rng.choice(diffs) for _ in diffs) / len(diffs) for _ in range(n))
    return m, bs[int(0.025 * n)], bs[int(0.975 * n)]


def load_probe(p):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from perception_probe import probes
    cells = {gid: len(g) * len(g[0]) for gid, g in probes(60, 0)}
    recs = {}
    for line in open(p):
        r = json.loads(line)
        if "error" not in r:
            recs[(r["gid"], r["sample"])] = r
    by_gid = defaultdict(list)
    for (gid, _), r in recs.items():
        by_gid[gid].append(r)
    return cells, recs, by_gid


def probe_tables(paths):
    loaded = {Path(p).stem: load_probe(p) for p in paths}
    names = list(loaded)
    print("## Perception probe (% of calls)")
    print("| run | n | " + " | ".join(KEYS) + " | exact by size " + "/".join(b[2] for b in BINS) +
          " | out tok |")
    print("|---|---|" + "---|" * len(KEYS) + "---|---|")
    for name in names:
        cells, recs, _ = loaded[name]
        rs = list(recs.values())
        row = [f"{100 * sum(bool(r[k]) for r in rs) / len(rs):.1f}" for k in KEYS]
        per_bin = []
        for lo, hi, _ in BINS:
            b = [r for r in rs if lo <= cells[r["gid"]] <= hi]
            per_bin.append(f"{100 * sum(bool(r['exact']) for r in b) / max(1, len(b)):.0f}")
        toks = [r["usage"]["completion_tokens"] for r in rs if r.get("usage")]
        print(f"| {name} | {len(rs)} | " + " | ".join(row) + f" | {'/'.join(per_bin)} | "
              f"{sum(toks) / max(1, len(toks)) / 1e3:.1f}k |")
    base = names[0]
    print(f"\nPaired by grid vs {base} (pp, 95% CI over grids):")
    for name in names[1:]:
        out = []
        for k in ("size", "colors", "corners", "exact"):
            gb, ga = loaded[base][2], loaded[name][2]
            common = sorted(set(gb) & set(ga))
            d = [100 * (sum(bool(r[k]) for r in ga[g]) / len(ga[g]) - sum(bool(r[k]) for r in gb[g]) / len(gb[g]))
                 for g in common]
            m, lo, hi = boot(d)
            out.append(f"{k} {m:+.1f} [{lo:+.1f}, {hi:+.1f}]")
        print(f"- {name} (n={len(common)} grids): " + "; ".join(out))


def nll_tables(paths):
    by_tag = defaultdict(dict)
    for p in paths:
        for line in open(p):
            r = json.loads(line)
            by_tag[r["tag"]][(r["task_id"], r["pair"], r["attempt"], r.get("batch"))] = r
    tags = list(by_tag)
    print("\n## Teacher-forced NLL of the API's image rollouts (per generated token)")
    print("| tag | rollouts | gen tokens | NLL gen | NLL first turn |")
    print("|---|---|---|---|---|")
    for t in tags:
        rs = list(by_tag[t].values())
        ng, n1 = sum(r["n_gen"] for r in rs), sum(r.get("n_t1", 0) for r in rs)
        t1 = f"{sum(r['nll_t1'] for r in rs) / n1:.4f}" if n1 else "-"
        print(f"| {t} | {len(rs)} | {ng} | {sum(r['nll_gen'] for r in rs) / ng:.4f} | {t1} |")
    print("\nPaired per rollout (b - a, per-token NLL; 95% CI over rollouts):")
    for i, a in enumerate(tags):
        for b in tags[i + 1:]:
            common = sorted(set(by_tag[a]) & set(by_tag[b]), key=str)
            if not common:
                continue
            for field, n in (("nll_gen", "n_gen"), ("nll_t1", "n_t1")):
                d = [by_tag[b][k][field] / by_tag[b][k][n] - by_tag[a][k][field] / by_tag[a][k][n]
                     for k in common if by_tag[a][k].get(n)]
                if d:
                    m, lo, hi = boot(d)
                    print(f"- {b} - {a}, {field}: {m:+.4f} [{lo:+.4f}, {hi:+.4f}] (n={len(d)})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", nargs="*", default=[])
    ap.add_argument("--nll", nargs="*", default=[])
    a = ap.parse_args()
    if a.probe:
        probe_tables(a.probe)
    if a.nll:
        nll_tables(a.nll)
