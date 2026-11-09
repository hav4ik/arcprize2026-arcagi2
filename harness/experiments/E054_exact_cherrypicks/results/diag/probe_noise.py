#!/usr/bin/env python3
"""E054 probe re-analysis against OLD's own server-to-server spread (stdlib only).

gpu/logprob_probe.py's d_lp0 check (<= 0.5 nats) scores a token present in one run's top-5 but not the other's as
-30, so any top-5 membership change reads ~27 nats; OLD vs OLD (two server starts, same code) already fails it.
Here token 0 is compared on the INTERSECTION of the two top-5 sets (and on the chosen token when both picked the
same one). Every arm pair is summarized the same way; an arm is "inside the noise" when its across-OLD statistics
lie within the OLD-vs-OLD2 (separate servers) range with modest slack:
  first_div median >= 0.5 x OLD-vs-OLD2's; mean |d lp| (prefix) median <= 1.5 x OLD-vs-OLD2's + 0.01;
  token-0 |d lp| on the top-5 intersection: max <= 1.5 x OLD-vs-OLD2's max + 0.05; argmax0 same on >= 4/5 prompts.

  python3 probe_noise.py OLD.json OLD2.json ARM.json [ARM2.json ...]   (names = parent dir names)
"""
import itertools
import json
import statistics
import sys
from pathlib import Path


def pair(x, y):
    ix, iy = x["ids"] or [], y["ids"] or []
    n = min(len(ix), len(iy))
    div = next((i for i in range(n) if ix[i] != iy[i]), n)
    tx, ty = (x["top"] or [{}])[0] or {}, (y["top"] or [{}])[0] or {}
    inter = set(tx) & set(ty)
    d0 = max((abs(tx[t] - ty[t]) for t in inter), default=0.0)
    ch = abs(x["lps"][0] - y["lps"][0]) if ix and iy and ix[0] == iy[0] else None
    common = range(min(div, len(x["lps"] or []), len(y["lps"] or [])))
    dm = statistics.mean(abs(x["lps"][i] - y["lps"][i]) for i in common) if len(common) else 0.0
    am = (max(tx, key=tx.get) if tx else None) == (max(ty, key=ty.get) if ty else None)
    return {"first_div": div, "d_lp0_inter": d0, "d_lp0_chosen": ch, "mean_d_lp": dm, "argmax0_same": am}


def summ(ps):
    col = lambda k: [p[k] for p in ps if p[k] is not None]  # noqa: E731
    return {"pairs": len(ps), "first_div_median": statistics.median(col("first_div")),
            "mean_d_lp_median": round(statistics.median(col("mean_d_lp")), 4),
            "d_lp0_inter_median": round(statistics.median(col("d_lp0_inter")), 4),
            "d_lp0_inter_max": round(max(col("d_lp0_inter")), 4),
            "d_lp0_chosen_max": round(max(col("d_lp0_chosen") or [0]), 4)}


def by(d, i):
    return [r for r in d["runs"] if r["prompt"] == i]


def across(a, b):
    n = len(a["prompts"])
    ps = [pair(x, y) for i in range(n) for x in by(a, i) for y in by(b, i)]
    am = sum(all(pair(x, y)["argmax0_same"] for x in by(a, i) for y in by(b, i)) for i in range(n))
    return summ(ps), am


def within(a):
    n = len(a["prompts"])
    return summ([pair(x, y) for i in range(n) for x, y in itertools.combinations(by(a, i), 2)])


def main():
    files = sys.argv[1:]
    arms = {Path(f).parent.name: json.load(open(f)) for f in files}
    names = list(arms)
    ref, ref2 = names[0], names[1]
    out = {"within": {k: within(v) for k, v in arms.items()}}
    floor, am_floor = across(arms[ref], arms[ref2])
    out["floor_" + ref + "_vs_" + ref2] = {**floor, "argmax0_same_prompts": am_floor}
    for k in names[2:]:
        s, am = across(arms[ref], arms[k])
        checks = {"first_div": s["first_div_median"] >= 0.5 * floor["first_div_median"],
                  "mean_d_lp": s["mean_d_lp_median"] <= 1.5 * floor["mean_d_lp_median"] + 0.01,
                  "d_lp0_inter": s["d_lp0_inter_max"] <= 1.5 * floor["d_lp0_inter_max"] + 0.05,
                  "argmax0_4of5": am >= 4}
        out[ref + "_vs_" + k] = {**s, "argmax0_same_prompts": am, "checks": checks, "inside_noise": all(checks.values())}
    for a, b in itertools.combinations(names[2:], 2):
        s, am = across(arms[a], arms[b])
        out[a + "_vs_" + b] = {**s, "argmax0_same_prompts": am}
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
