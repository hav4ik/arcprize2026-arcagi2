#!/usr/bin/env python3
"""pass@2 for every Flash-1017 tool harness on the 120 eval tasks, task-weighted (a pair counts if either sample of
the two is right; a task scores the mean over its pairs). With k independent attempt samples of one harness,
pass@2 is averaged over all k(k-1)/2 sample pairs (less noisy than the fixed attempt 0+1 pair); pass@1 = mean over
samples. Tasks missing in a sample are skipped for the pairs that need them.
Tokens are per task-attempt (multi-test: one rollout per task; per-pair runs: summed over the task's pairs), pooled
over all samples: output mean / median / p90 / max, input (prompt, all turns) mean and its cache-hit share, turns.
Usage (repo root): pass2_table.py"""
import itertools, json, statistics as st
R = "evals/results/tools_persistent_grid_lean2sub_deepseek-flash_low_"
CONFIGS = [
    ("E029 sub80s40 (per pair)", [("e029_flash1017_sub80s40", None)]),
    ("E029 s20x60 (per pair)", [("e029_flash1017_s20x60", None)]),
    ("E030 sub80s40 + header", [("e030_hdrmin", None)]),
    ("E030 s20x60 + header", [("e030_s20x60_hdrmin", None)]),
    ("E046 best (text)", [("e046p_note_all", None), ("e047_ctrl", (1, 2))]),
    ("E046 best (text, DeepInfra)", [("e047or_ctrl", (1, 2, 3))]),
    ("E058 best + images", [("e058_pilot30", None)]),
    ("E064 + images + show_grid", [("e064_show", None)]),
    ("E064 + images + check", [("e064_check", None)]),
    ("images pooled (E058 + show + check)", [("e058_pilot30", None), ("e064_show", None), ("e064_check", None)]),
    ("E064 + images (DeepInfra)", [("e064or_base", None)]),
    ("E064 + images + strict (DeepInfra)", [("e064or_strict", None)]),
    ("E064 lazy: show_grid only (DeepInfra)", [("e064or_lazy", None)]),
]


def samples(parts):
    out, toks = [], []
    for tag, keep in parts:
        d = {}
        try:
            for r in map(json.loads, open(R + tag + ".jsonl")):
                if "task_id" in r and not r.get("error") and (keep is None or r["attempt"] in keep):
                    d[r["task_id"], r["pair"], r["attempt"]] = r
        except FileNotFoundError:
            continue
        for a in sorted({k[2] for k in d}):
            s = {(t, p): bool(r["correct"]) for (t, p, aa), r in d.items() if aa == a}
            if len({t for t, p in s}) >= 118:
                out.append(s)
                for t in {t for t, p in s}:
                    rs = [r for (tt, p, aa), r in d.items() if tt == t and aa == a and not r.get("multi_share")]
                    toks.append((sum(r["usage"]["completion"] for r in rs), sum(r["usage"]["prompt"] for r in rs),
                                 sum(r["usage"].get("cache_hit") or 0 for r in rs), sum(r["turns"] for r in rs)))
    return out, toks


def q(xs, f):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(f * len(xs)))]


def task_mean(scores):  # {(t, p): v} -> task-weighted mean
    by = {}
    for (t, p), v in scores.items():
        by.setdefault(t, []).append(v)
    return 100 * st.mean(st.mean(v) for v in by.values())


print(f"{'harness':40} {'samples':>7} {'pass@1':>7} {'pass@2':>7} {'p@2 range':>12} {'pairs':>5} | output tokens per "
      f"task-attempt: {'mean':>6} {'p50':>6} {'p90':>6} {'max':>6} | {'input':>6} {'cached':>6} {'turns':>5}")
for name, parts in CONFIGS:
    ss, toks = samples(parts)
    if len(ss) < 2:
        print(f"{name:40} {len(ss):7d}  incomplete"); continue
    p1 = st.mean(task_mean(s) for s in ss)
    p2s = []
    for a, b in itertools.combinations(ss, 2):
        common = set(a) & set(b)
        p2s.append(task_mean({k: a[k] or b[k] for k in common}))
    out = [t[0] for t in toks]
    print(f"{name:40} {len(ss):7d} {p1:6.1f}% {st.mean(p2s):6.1f}% {min(p2s):5.1f}-{max(p2s):5.1f}% {len(p2s):5d} |"
          f" {'':26}{st.mean(out)/1e3:5.1f}k {q(out, .5)/1e3:5.1f}k {q(out, .9)/1e3:5.1f}k {max(out)/1e3:5.0f}k |"
          f" {st.mean(t[1] for t in toks)/1e3:5.0f}k {100*sum(t[2] for t in toks)/max(1, sum(t[1] for t in toks)):5.0f}%"
          f" {st.mean(t[3] for t in toks):5.1f}")
