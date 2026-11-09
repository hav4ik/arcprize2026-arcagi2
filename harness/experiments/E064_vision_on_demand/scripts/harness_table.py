#!/usr/bin/env python3
"""All Flash-1017 tool-harness runs on the 120 eval tasks, scored the same way: task-weighted pass@1 (mean over every
attempt sample), pass@2 (attempts 0+1), output tokens per task-attempt (multi-test: one rollout per task), turns,
and pass@1 per 10k output tokens. Several files can form one config (extra attempt samples of the same harness).
Usage (repo root): harness_table.py"""
import json, statistics as st
R = "evals/results/tools_persistent_grid_lean2sub_deepseek-flash_low_"
CONFIGS = [  # name, [(tag, attempts or None = all)], provider
    ("E029 sub80s40 (per pair)", [("e029_flash1017_sub80s40", None)], "DS"),
    ("E029 s20x60 (per pair)", [("e029_flash1017_s20x60", None)], "DS"),
    ("E030 sub80s40 + header", [("e030_hdrmin", None)], "DS"),
    ("E030 s20x60 + header", [("e030_s20x60_hdrmin", None)], "DS"),
    ("E046 best: + multi-test 0.5 + fallback", [("e046p_note_all", None), ("e047_ctrl", (1, 2))], "DS"),
    ("E046 best (DeepInfra)", [("e047or_ctrl", (1, 2, 3))], "DI"),
    ("E058 best + images", [("e058_pilot30", None)], "DS"),
    ("E064 best + images + show_grid", [("e064_show", None)], "DS"),
    ("E064 best + images + check", [("e064_check", None)], "DS"),
    ("E064 best + images (DeepInfra)", [("e064or_base", None)], "DI"),
    ("E064 best + images + strict note (DI)", [("e064or_strict", None)], "DI"),
    ("E064 best + show_grid, no up-front images (DI)", [("e064or_lazy", None)], "DI"),
]
TASKS = None


def load(tag, keep):
    d = {}
    try:
        for r in map(json.loads, open(R + tag + ".jsonl")):
            if "task_id" in r and not r.get("error") and (keep is None or r["attempt"] in keep):
                d[r["task_id"], r["pair"], r["attempt"]] = r
    except FileNotFoundError:
        pass
    return d


rows = []
for name, parts, prov in CONFIGS:
    samples = []  # each (tag, attempt) is one sample: {task: (score, tokens, turns)}
    for tag, keep in parts:
        d = load(tag, keep)
        for a in sorted({k[2] for k in d}):
            s = {}
            for t in sorted({k[0] for k in d if k[2] == a}):
                rs = [d[k] for k in d if k[0] == t and k[2] == a]
                use = [r for r in rs if not r.get("multi_share")]
                s[t] = (st.mean(bool(r["correct"]) for r in rs), sum(r["usage"]["completion"] for r in use),
                        sum(r["turns"] for r in use))
            samples.append(s)
    if not samples:
        continue
    full = [s for s in samples if len(s) >= 118]
    if not full:
        rows.append((name, prov, None)); continue
    p1s = [100 * st.mean(v[0] for v in s.values()) for s in full]
    tok = st.mean(st.mean(v[1] for v in s.values()) for s in full)
    turns = st.mean(st.mean(v[2] for v in s.values()) for s in full)
    p2 = None
    if len(full) >= 2:
        common = set(full[0]) & set(full[1])
        # pass@2 on pairs needs pair detail; approximate with task-level max of the two samples' pair scores
        p2 = None
    rows.append((name, prov, (len(full), st.mean(p1s), (st.stdev(p1s) if len(p1s) > 1 else 0), tok, turns)))
print(f"{'config':48} {'prov':4} {'n':>2} {'pass@1':>7} {'sd':>4} {'out tok':>8} {'turns':>5} {'p@1/10k':>7}")
for name, prov, v in rows:
    if v is None:
        print(f"{name:48} {prov:4} incomplete"); continue
    n, p1, sd, tok, turns = v
    print(f"{name:48} {prov:4} {n:2d} {p1:6.1f}% {sd:4.1f} {tok/1e3:7.1f}k {turns:5.1f} {p1/(tok/1e4):7.1f}")
