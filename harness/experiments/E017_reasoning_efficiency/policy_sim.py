"""E017 phase 1A: verifier quality and offline compute-allocation policies on the stratified label sample (no API).

Sample (label_sample.json): pairs are stratified by how many of their sub80s40 (3) + s20x60 (2) attempts are correct:
A = all correct, M = mixed, W = none correct. All attempts of the sampled pairs are labelled (train_labels.jsonl).
Population estimates weight each sampled pair by N_s / n_s (stratum size / sampled pairs). Policy outcomes are exact
expectations over every ordering of a pair's attempts. 95% CIs: bootstrap over sampled pairs within each stratum.

    .venv/bin/python experiments/E017_reasoning_efficiency/policy_sim.py > /tmp/policy_tables.md
"""
import json
import random
import statistics as st
import sys
from itertools import permutations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
E = ROOT / "experiments/E017_reasoning_efficiency"
sys.path.insert(0, str(E))
from label_train import RUNS, load_run  # noqa: E402

TASKS = {p.stem: json.loads(p.read_text()) for p in sorted((ROOT / "data/arc-agi-2-github/data/evaluation").glob("*.json"))}
NP = {t: len(v["test"]) for t, v in TASKS.items()}
POOL_RUNS = ["sub80s40", "s20x60"]
BOOT = 2000


def load():
    smp = json.load(open(E / "label_sample.json"))
    strata = {s: [tuple(k) for k in v] for s, v in smp["sample"].items()}
    sizes = smp["strata_sizes"]
    lab = {}
    for line in open(E / "train_labels.jsonl"):
        r = json.loads(line)
        if "error" not in r:
            lab[(r["run"], r["task_id"], r["pair"], r["attempt"])] = r
    pools, missing = {}, 0
    for s, keys in strata.items():
        for t, p in keys:
            atts = []
            for (run, tt, pp, a), r in sorted(lab.items()):
                if (tt, pp) == (t, p) and run in POOL_RUNS:
                    atts.append({"run": run, "c": bool(r["correct"]), "tp": r.get("train_pass") is True,
                                 "tf": r["train_frac"] if r.get("train_frac") is not None else -1.0,
                                 "ans": r.get("pred_hash") if r["trace"] != "no_answer" else None,
                                 "tok": r["out_tokens"], "cost": r.get("cost_usd") or 0.0, "trace": r["trace"]})
            pools[(t, p)] = atts
    high = {}
    for r in load_run(RUNS["sub80s40h"]):
        high.setdefault((r["task_id"], r["pair"]), []).append(
            {"c": bool(r.get("correct")), "tok": r["usage"]["completion"], "cost": r.get("est_cost_usd") or 0.0,
             "ans": json.dumps(r["pred"]) if r.get("pred") is not None else None, "tp": False, "tf": -1.0})
    return strata, sizes, pools, high


# ---------- selection ----------
def rank(atts, key):
    groups = {}
    for i, a in enumerate(atts):
        if a["ans"] is None:
            continue
        g = groups.setdefault(a["ans"], {"c": a["c"], "first": i, "n": 0, "ntp": 0, "tf": -1.0})
        g["n"] += 1
        g["ntp"] += a["tp"]
        g["tf"] = max(g["tf"], a["tf"])
    return sorted(groups.values(), key=key)


VERIFY = lambda g: (-(g["ntp"] > 0), -g["ntp"], -g["tf"], -g["n"], g["first"])  # noqa: E731
VOTE = lambda g: (-g["n"], g["first"])  # noqa: E731
ORDER = lambda g: g["first"]  # noqa: E731


def select(atts, key):
    gs = rank(atts, key)
    s1 = bool(gs and gs[0]["c"])
    return s1, s1 or (len(gs) > 1 and gs[1]["c"]), sum(a["tok"] for a in atts), sum(a["cost"] for a in atts)


def gated(atts, kmax, key=VERIFY):
    used = []
    for a in atts[:kmax]:
        used.append(a)
        if a["tp"]:
            break
    return select(used, key)


def two_plus(atts, kmax, key=VERIFY):
    """Always run 2 attempts; keep adding attempts (up to kmax) while none has passed the train check."""
    used = list(atts[:2])
    for a in atts[2:kmax]:
        if any(u["tp"] for u in used):
            break
        used.append(a)
    return select(used, key)


def truncate(a, B):
    if a["ans"] is not None and a["tok"] <= B:
        return a
    return {**a, "ans": None, "c": False, "tp": False, "tf": -1.0, "tok": min(a["tok"], B),
            "cost": a["cost"] * min(1.0, B / max(1, a["tok"]))}


def expect(atts, policy, k, high=None):
    """Exact mean over all orderings of the pair's attempts (first k used) [x orderings of high attempts]."""
    outs = []
    for perm in set(permutations(range(len(atts)), min(k, len(atts)))):
        d = [atts[i] for i in perm]
        if high is None:
            outs.append(policy(d))
        else:
            for h in high:
                outs.append(policy(d, h))
    return [st.mean(float(o[i]) for o in outs) for i in range(4)]


def aggregate(strata, sizes, per, idx=None):
    """Stratified estimate. per: {(t,p): [s1, s2, tok, cost]}. idx: optional bootstrap resample per stratum."""
    N = sum(sizes.values())
    pw = [0.0] * 4
    tw_num, tw_den = [0.0, 0.0], 0.0
    for s, keys in strata.items():
        ks = idx[s] if idx else keys
        w = sizes[s] / len(ks)
        for k in ks:
            v = per[k]
            for i in range(4):
                pw[i] += w * v[i] / N
            wt = w / NP[k[0]]
            tw_den += wt
            tw_num[0] += wt * v[0]
            tw_num[1] += wt * v[1]
    return {"s1": pw[0], "s2": pw[1], "tok": pw[2], "cost167": pw[3] * N,
            "s1_tw": tw_num[0] / tw_den, "s2_tw": tw_num[1] / tw_den}


def boot(strata, sizes, per, base_per=None, rng=None):
    vals = []
    for _ in range(BOOT):
        idx = {s: [rng.choice(keys) for _ in keys] for s, keys in strata.items()}
        a = aggregate(strata, sizes, per, idx)
        if base_per is not None:
            b = aggregate(strata, sizes, base_per, idx)
            a = {k: a[k] - b[k] for k in a}
        vals.append(a)
    ci = lambda key: (sorted(v[key] for v in vals)[int(0.025 * BOOT)], sorted(v[key] for v in vals)[int(0.975 * BOOT)])  # noqa: E731
    return {k: ci(k) for k in ("s1", "s2", "tok")}


def verifier_tables(strata, sizes, pools, rng):
    N = sum(sizes.values())
    rows = []
    for s, keys in strata.items():
        for k in keys:
            for a in pools[k]:
                rows.append((s, k, a))
    out = ["| stratum | attempts | answered | traceable | train_pass: correct / n | train_fail: correct / n | "
           "untraceable: correct / n | no answer |", "|---|---|---|---|---|---|---|---|"]
    for s in list(strata) + ["all (raw)"]:
        xs = [a for ss, _, a in rows if ss == s or s == "all (raw)"]
        ans = [a for a in xs if a["ans"] is not None]
        tr = [a for a in ans if a["trace"] != "untraceable"]
        f = lambda ys: f"{sum(a['c'] for a in ys)} / {len(ys)}"  # noqa: E731
        out.append(f"| {s} | {len(xs)} | {len(ans)} | {len(tr)} | {f([a for a in tr if a['tp']])} | "
                   f"{f([a for a in tr if not a['tp']])} | {f([a for a in ans if a['trace'] == 'untraceable'])} | "
                   f"{len(xs) - len(ans)} |")

    def est(idx):
        num = {"pass": [0, 0], "fail": [0, 0], "unt": [0, 0], "cor": [0, 0]}
        for s, keys in strata.items():
            ks = idx[s] if idx else keys
            w = sizes[s] / len(ks)
            for k in ks:
                for a in pools[k]:
                    if a["ans"] is None:
                        cat = None
                    elif a["trace"] == "untraceable":
                        cat = "unt"
                    else:
                        cat = "pass" if a["tp"] else "fail"
                    if cat:
                        num[cat][0] += w * a["c"]
                        num[cat][1] += w
                    if a["c"]:
                        num["cor"][0] += w * a["tp"]
                        num["cor"][1] += w
        return {k: v[0] / v[1] if v[1] else float("nan") for k, v in num.items()}
    point = est(None)
    bs = [est({s: [rng.choice(keys) for _ in keys] for s, keys in strata.items()}) for _ in range(BOOT)]
    ci = lambda key: (sorted(b[key] for b in bs)[int(0.025 * BOOT)], sorted(b[key] for b in bs)[int(0.975 * BOOT)])  # noqa: E731
    est_lines = ["| quantity (population-weighted) | estimate | 95% CI |", "|---|---|---|"]
    for key, name in [("pass", "P(correct | train_pass)"), ("fail", "P(correct | train_fail)"),
                      ("unt", "P(correct | untraceable)"), ("cor", "recall: P(train_pass | correct)")]:
        lo, hi = ci(key)
        est_lines.append(f"| {name} | {100 * point[key]:.0f}% | {100 * lo:.0f}–{100 * hi:.0f}% |")
    # pairs with >=1 train-passing attempt
    cov = ["| stratum (pop. size) | sampled pairs | ≥1 train_pass attempt | ≥1 train_pass & correct | train_pass only on wrong answers |",
           "|---|---|---|---|---|"]
    for s, keys in strata.items():
        a1 = sum(any(a["tp"] for a in pools[k]) for k in keys)
        a2 = sum(any(a["tp"] and a["c"] for a in pools[k]) for k in keys)
        a3 = sum(any(a["tp"] for a in pools[k]) and not any(a["tp"] and a["c"] for a in pools[k]) for k in keys)
        cov.append(f"| {s} ({sizes[s]}) | {len(keys)} | {a1} | {a2} | {a3} |")
    return "\n".join(out), "\n".join(est_lines), "\n".join(cov)


def main():
    rng = random.Random(0)
    strata, sizes, pools, high = load()
    t1, t2, t3 = verifier_tables(strata, sizes, pools, rng)
    print("### Raw counts (sampled attempts, sub80s40 + s20x60)\n\n" + t1)
    print("\n### Population-weighted verifier quality\n\n" + t2)
    print("\n### Test outputs with a train-passing attempt (all 4–5 attempts of the pair)\n\n" + t3)

    s80 = {k: [a for a in v if a["run"] == "sub80s40"] for k, v in pools.items()}
    policies = []  # (name, pools, policy, k, high?)
    single = lambda d: select(d[:1], ORDER)  # noqa: E731
    two = lambda d: select(d[:2], ORDER)  # noqa: E731
    policies += [
        ("(a) 1 attempt (pass@1)", pools, single, 1, None),
        ("(a) 2 attempts (pass@2 baseline)", pools, two, 2, None),
        ("(b) gated: 2nd attempt only if 1st fails train check", pools, lambda d: gated(d, 2), 2, None),
        ("(b) gated, up to 3 attempts", pools, lambda d: gated(d, 3), 3, None),
        ("(b) gated, up to 4 attempts", pools, lambda d: gated(d, 4), 4, None),
        ("(f) 2 attempts, +3rd only if neither passes train", pools, lambda d: two_plus(d, 3), 3, None),
        ("(f) 2 attempts, +up to 2 more while none passes train", pools, lambda d: two_plus(d, 4), 4, None),
        ("(c) best-of-3, train-check select", pools, lambda d: select(d, VERIFY), 3, None),
        ("(c) best-of-3, majority only", pools, lambda d: select(d, VOTE), 3, None),
        ("(c) best-of-4, train-check select", pools, lambda d: select(d, VERIFY), 4, None),
        ("(c) best-of-4, majority only", pools, lambda d: select(d, VOTE), 4, None),
        ("(c) sub80s40 only: best-of-3, train-check", s80, lambda d: select(d, VERIFY), 3, None),
        ("(c) sub80s40 only: 2 attempts (pass@2)", s80, two, 2, None),
    ]
    for B in (25000, 30000):
        tp = {k: [truncate(a, B) for a in v] for k, v in pools.items()}
        policies += [
            (f"(d) @{B // 1000}k: 1 attempt", tp, single, 1, None),
            (f"(d) @{B // 1000}k: 2 attempts", tp, two, 2, None),
            (f"(d) @{B // 1000}k: best-of-3, train-check", tp, lambda d: select(d, VERIFY), 3, None),
            (f"(d) @{B // 1000}k: best-of-4, train-check", tp, lambda d: select(d, VERIFY), 4, None),
            (f"(d) @{B // 1000}k: gated, up to 4", tp, lambda d: gated(d, 4), 4, None),
        ]

    def esc(d, h):
        a1 = d[0]
        if a1["tp"]:
            return select([a1], VERIFY)
        return select([a1, h], ORDER)  # slot1 = low attempt (high attempt is unlabelled), slot2 = high attempt

    def mix(d, h):
        return select([d[0], h], ORDER)
    policies += [("(e) low → sub80s40h 2nd attempt only if check fails", pools, esc, 1, "high"),
                 ("(e) low + sub80s40h always (mixed pass@2)", pools, mix, 1, "high")]

    per_all = {}
    for name, pl, pol, k, hi in policies:
        per_all[name] = {key: expect(pl[key], pol, k, high.get(key) if hi else None) for key in pl}
    base = per_all["(a) 2 attempts (pass@2 baseline)"]
    print("\n## Policies (population estimates; Δ vs 2-attempt baseline with 95% bootstrap CI)\n")
    print("| Policy | pass@1-slot pair-w / task-w | pass@2 pair-w / task-w | Δ pass@2 (pp) [95% CI] | "
          "out tok / pair | Δ tok / pair | $ / full set (167 pairs) |")
    print("|---|---|---|---|---|---|---|")
    res = []
    for name, *_ in policies:
        per = per_all[name]
        a = aggregate(strata, sizes, per)
        b = aggregate(strata, sizes, base)
        d = boot(strata, sizes, per, base, rng)
        res.append({"policy": name, **a, "d_s2_ci": d["s2"]})
        print(f"| {name} | {100 * a['s1']:.1f} / {100 * a['s1_tw']:.1f} | {100 * a['s2']:.1f} / {100 * a['s2_tw']:.1f} | "
              f"{100 * (a['s2'] - b['s2']):+.1f} [{100 * d['s2'][0]:+.1f}, {100 * d['s2'][1]:+.1f}] | "
              f"{a['tok'] / 1000:.1f}k | {(a['tok'] - b['tok']) / 1000:+.1f}k | ${a['cost167']:.2f} |")
    json.dump(res, open(E / "policy_sim.json", "w"), indent=1)


if __name__ == "__main__":
    main()
