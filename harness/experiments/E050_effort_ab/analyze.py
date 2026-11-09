#!/usr/bin/env python3
"""E050: DeepSeek-V4.1 reasoning effort 25 vs 50 on E128 (dsv41-tools-submit-mtf V6 pipeline, 1 attempt per task).

Per arm: task-level pass@1 (a multi-test task scores the fraction of its pairs right), pair-level pass@1, output tokens
per task (= per attempt: one multi-test rollout per task), turns, no-answer rate, stop reasons and answer sources;
the paired comparison by task (sign test, bootstrap CI of the pass@1 difference and of the token ratio); the second
control (E046 run 7 = the same harness at effort 50 on an 8x box, 2 attempts, GitHub eval set) on the tasks whose
prompt is identical in both eval sets.

Pairs are matched by test-input content: each record's test input (from the challenge file the run used) is looked up
in a content-keyed solution table. The Kaggle eval json and the GitHub set differ in 6 tasks (5 extra test pairs in
the Kaggle file; d8e07eb2 and faa9f03d have different train pairs), which are left out of the control comparison.

Usage: analyze.py [--a runs/eff25] [--b runs/eff50] [--json results.json]
"""
import argparse
import collections
import glob
import hashlib
import json
import math
import os
import statistics
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BIG = 10 ** 9  # completion tokens the solver reports for a cut stream (loop guard / force deadline)
LOOP_MIN_TOKENS = 10_000  # a loop cut needs ARC_LOOP_CONFIRM (10) windows of ARC_LOOP_EVERY (1000) thinking tokens
CONTROL = ROOT / "evals/results/tools_persistent_grid_lean2sub_dsv41_low_e043_e128t_final_all.jsonl"


def H(g):
    return hashlib.sha1(json.dumps(g, separators=(",", ":")).encode()).hexdigest()


def kaggle_set():
    ch = json.load(open(ROOT / "data/arc-agi_evaluation_challenges.json"))
    so = json.load(open(ROOT / "data/arc-agi_evaluation_solutions.json"))
    return ch, {(t, H(x["input"])): so[t][i] for t, v in ch.items() for i, x in enumerate(v["test"])}


def github_set():
    ch = {Path(f).stem: json.load(open(f)) for f in glob.glob(str(ROOT / "data/arc-agi-2-github/data/evaluation/*.json"))}
    return ch, {(t, H(x["input"])): x["output"] for t, v in ch.items() for x in v["test"]}


def identical_tasks(a, b):
    """Tasks whose prompt is the same in both sets: same train pairs and the same test inputs in the same order."""
    return sorted(t for t in a if t in b and json.dumps(a[t]["train"]) == json.dumps(b[t]["train"])
                  and [H(x["input"]) for x in a[t]["test"]] == [H(x["input"]) for x in b[t]["test"]])


def tokens(rec):
    """Output tokens of an attempt; for cut streams the solver adds BIG per cut instead of the turn's real count, so
    use the other turns plus the minimum a loop cut implies (lower bound, flagged)."""
    c = rec.get("usage", {}).get("completion") or 0
    ncut = sum((rec.get("cuts") or {}).values())
    if ncut and c >= BIG:
        return c - ncut * BIG + ncut * LOOP_MIN_TOKENS, True
    return c, False


def load_arm(path, ch, sol):
    """One row per task (attempt 0): pair results, tokens, turns, stop / answer source. The last record per
    (task, pair, attempt) wins (an errored attempt that was retried is logged before its retry)."""
    last = {}
    for line in open(path):
        r = json.loads(line)
        if "task_id" in r:
            last[r["task_id"], r["pair"], r["attempt"]] = r
    rows = {}
    for (t, p, a), r in last.items():
        if a != 0 or p != 0:
            continue
        tests = ch[t]["test"]
        n = len(tests)
        pred = r.get("pred")
        preds = (pred if r.get("multi_test") and isinstance(pred, list) and len(pred) == n else
                 [None] * n if r.get("multi_test") else [pred])
        right = [preds[q] is not None and preds[q] == sol[t, H(tests[q]["input"])] for q in range(n)]
        tok, cut = tokens(r)
        rows[t] = {"n": n, "right": right, "score": sum(right) / n, "tokens": tok, "cut": cut,
                   "turns": r.get("turns") or 0, "stop": r.get("stop_reason"), "source": r.get("answer_source"),
                   "answered": pred is not None, "train_pass": r.get("train_pass"), "error": r.get("error"),
                   "prompt": r.get("usage", {}).get("prompt") or 0, "wall_s": r.get("wall_s") or 0,
                   "t_end_h": r.get("t_end_h"), "rescues": r.get("rescues") or 0}
    return rows


def load_control(ch, sol):
    """E046 run 7: per-pair records, 2 attempts; usage on pair 0 for multi-test tasks. -> {attempt: {task: row}}."""
    last = {}
    for line in open(CONTROL):
        r = json.loads(line)
        if "task_id" in r:
            last[r["task_id"], r["pair"], r["attempt"]] = r
    out = collections.defaultdict(dict)
    for (t, p, a), r in last.items():
        if p != 0:
            continue
        tests = ch[t]["test"]
        n = len(tests)
        right = []
        for q in range(n):
            rq = last.get((t, q, a))
            pred = rq.get("pred") if rq else None
            right.append(pred is not None and pred == sol[t, H(tests[q]["input"])])
        out[a][t] = {"n": n, "right": right, "score": sum(right) / n, "tokens": r["usage"]["completion"],
                     "cut": False, "turns": r.get("turns") or 0, "stop": r.get("stop_reason"),
                     "source": r.get("answer_source"), "answered": r.get("pred") is not None,
                     "train_pass": r.get("train_pass"), "error": r.get("error"),
                     "prompt": r["usage"].get("prompt") or 0, "wall_s": r.get("seconds") or 0, "rescues": r.get("rescues") or 0}
    return out


def load_traces(arm_dir):
    """Per task (attempt 0): completion tokens per turn (trace events) and the characters of thinking vs visible output
    (content + tool-call arguments) over the assistant messages. Reads traces/ or dsv41_traces.tar.gz."""
    import gzip, io, tarfile
    files = {}
    tdir, tgz = Path(arm_dir) / "dsv41_logs/traces", Path(arm_dir) / "dsv41_traces.tar.gz"
    if tdir.is_dir():
        for f in tdir.glob("*_p0_a0.json.gz"):
            files[f.name] = f.read_bytes()
    elif tgz.exists():
        with tarfile.open(tgz) as tf:
            for m in tf.getmembers():
                if m.isfile() and m.name.endswith("_p0_a0.json.gz"):
                    files[os.path.basename(m.name)] = tf.extractfile(m).read()
    out = {}
    for name, blob in files.items():
        try:
            tr = json.load(gzip.open(io.BytesIO(blob)))
        except Exception:
            continue
        toks = [int(e["usage"]["completion_tokens"]) for e in tr.get("events", [])
                if isinstance(e.get("usage"), dict) and e["usage"].get("completion_tokens") is not None]
        think = vis = 0
        for m in tr.get("messages", []):
            if m.get("role") == "assistant":
                think += len(m.get("reasoning_content") or "")
                vis += len(m.get("content") or "") + len(json.dumps(m.get("tool_calls") or ""))
        out[name.split("_p0_a0")[0]] = {"turn_tokens": toks, "think_chars": think, "visible_chars": vis}
    return out


def trace_md(name, tr, tasks):
    ts = [t for t in tasks if t in tr and tr[t]["turn_tokens"]]
    if not ts:
        return f"| {name} | 0 | | | | |"
    first = [tr[t]["turn_tokens"][0] for t in ts]
    allt = [x for t in ts for x in tr[t]["turn_tokens"]]
    th = sum(tr[t]["think_chars"] for t in ts); vi = sum(tr[t]["visible_chars"] for t in ts)
    return (f"| {name} | {len(ts)} | {statistics.mean(first):.0f} / {statistics.median(first):.0f} | "
            f"{statistics.mean(allt):.0f} / {statistics.median(allt):.0f} / {np.percentile(allt, 90):.0f} | "
            f"{len(allt) / len(ts):.1f} | {th / max(1, th + vi):.2f} |")


def pct(x):
    return f"{100 * x:.1f}%"


def summary(rows, tasks=None):
    tasks = sorted(rows if tasks is None else [t for t in tasks if t in rows])
    R = [rows[t] for t in tasks]
    tok = np.array([r["tokens"] for r in R], float)
    pairs = [x for r in R for x in r["right"]]
    single = [r for r in R if r["n"] == 1]
    multi = [r for r in R if r["n"] > 1]
    return {
        "tasks": len(R), "pairs": len(pairs),
        "pass1_task": statistics.mean(r["score"] for r in R) if R else None,
        "pass1_pair": statistics.mean(pairs) if pairs else None, "pairs_right": int(sum(pairs)),
        "tok_mean": float(tok.mean()) if len(tok) else None, "tok_median": float(np.median(tok)) if len(tok) else None,
        "tok_p90": float(np.percentile(tok, 90)) if len(tok) else None, "tok_total": float(tok.sum()),
        "tok_per_pair": float(tok.sum() / len(pairs)) if pairs else None,
        "turns_mean": statistics.mean(r["turns"] for r in R) if R else None,
        "no_answer": sum(not r["answered"] for r in R), "no_answer_rate": statistics.mean(not r["answered"] for r in R) if R else None,
        "train_pass": sum(r["train_pass"] is True for r in R), "cut_attempts": sum(r["cut"] for r in R),
        "errors": sum(bool(r["error"]) for r in R), "rescues": sum(r["rescues"] for r in R),
        "stops": dict(collections.Counter(r["stop"] for r in R).most_common()),
        "sources": dict(collections.Counter(str(r["source"]) for r in R).most_common()),
        "single": {"tasks": len(single), "pass1": statistics.mean(r["score"] for r in single) if single else None,
                   "tok_mean": statistics.mean(r["tokens"] for r in single) if single else None},
        "multi": {"tasks": len(multi), "pass1_task": statistics.mean(r["score"] for r in multi) if multi else None,
                  "tok_mean": statistics.mean(r["tokens"] for r in multi) if multi else None},
        "prompt_mean": statistics.mean(r["prompt"] for r in R) if R else None,
        "wall_mean_s": statistics.mean(r["wall_s"] for r in R) if R else None,
    }


def sign_test(wins, losses):
    """Two-sided exact binomial test on the tasks where the arms differ."""
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def paired(a, b, tasks, n_boot=20000, seed=0):
    """a vs b on common tasks: mean task-pass@1 difference (a - b) with a bootstrap CI over tasks, sign test, and the
    token ratio sum(a)/sum(b) (and the geometric mean of per-task ratios) with bootstrap CIs."""
    ts = [t for t in tasks if t in a and t in b]
    sa = np.array([a[t]["score"] for t in ts]); sb = np.array([b[t]["score"] for t in ts])
    ta = np.array([a[t]["tokens"] for t in ts], float); tb = np.array([b[t]["tokens"] for t in ts], float)
    pa = sum(sum(a[t]["right"]) for t in ts); pb = sum(sum(b[t]["right"]) for t in ts)
    npairs = sum(a[t]["n"] for t in ts)
    d = sa - sb
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(ts), size=(n_boot, len(ts)))
    bd = d[idx].mean(1)
    br = ta[idx].sum(1) / tb[idx].sum(1)
    lr = np.log(np.maximum(ta, 1) / np.maximum(tb, 1))
    bg = np.exp(lr[idx].mean(1))
    wins, losses = int((d > 0).sum()), int((d < 0).sum())
    return {"tasks": len(ts), "pairs": npairs, "pass1_a": float(sa.mean()), "pass1_b": float(sb.mean()),
            "diff": float(d.mean()), "diff_ci": [float(np.percentile(bd, 2.5)), float(np.percentile(bd, 97.5))],
            "pairs_a": int(pa), "pairs_b": int(pb), "wins_a": wins, "wins_b": losses, "ties": len(ts) - wins - losses,
            "sign_p": sign_test(wins, losses),
            "tok_ratio": float(ta.sum() / tb.sum()), "tok_ratio_ci": [float(np.percentile(br, 2.5)), float(np.percentile(br, 97.5))],
            "geo_ratio": float(np.exp(lr.mean())), "geo_ratio_ci": [float(np.percentile(bg, 2.5)), float(np.percentile(bg, 97.5))],
            "tok_mean_a": float(ta.mean()), "tok_mean_b": float(tb.mean()),
            "tok_median_a": float(np.median(ta)), "tok_median_b": float(np.median(tb)),
            "turns_a": float(np.mean([a[t]["turns"] for t in ts])), "turns_b": float(np.mean([b[t]["turns"] for t in ts])),
            "tok_per_turn_a": float(ta.sum() / max(1, sum(a[t]["turns"] for t in ts))),
            "tok_per_turn_b": float(tb.sum() / max(1, sum(b[t]["turns"] for t in ts))),
            "no_answer_a": sum(not a[t]["answered"] for t in ts), "no_answer_b": sum(not b[t]["answered"] for t in ts)}


def metrics(path):
    """Server metrics.jsonl (every ~300 s): generated tok/s, prefix hits, KV, preemptions over the run."""
    if not os.path.exists(path):
        return None
    m = [json.loads(l) for l in open(path)]
    m = [x for x in m if x.get("window_s", 0) >= 120]
    if not m:
        return None
    busy = [x for x in m if (x.get("running") or 0) >= 16]  # the steady part, before the tail drains
    w = lambda xs, k: (sum(x[k] * x["window_s"] for x in xs if x.get(k) is not None)
                       / max(1, sum(x["window_s"] for x in xs if x.get(k) is not None)))
    return {"windows": len(m), "gen_tok_s": round(w(m, "gen_tok_s"), 1),
            "gen_tok_s_busy": round(w(busy, "gen_tok_s"), 1) if busy else None,
            "prefix_hit": round(w(m, "prefix_hit"), 3), "kv_peak": max(x.get("kv") or 0 for x in m),
            "preemptions": max(x.get("preemptions") or 0 for x in m), "hours": round(m[-1]["elapsed_h"], 2)}


def row_md(name, s, m=None):
    return (f"| {name} | {s['tasks']} | {pct(s['pass1_task'])} | {s['pairs_right']}/{s['pairs']} = {pct(s['pass1_pair'])} | "
            f"{s['tok_mean'] / 1e3:.1f}k | {s['tok_median'] / 1e3:.1f}k | {s['tok_p90'] / 1e3:.1f}k | {s['turns_mean']:.1f} | "
            f"{s['no_answer']} ({pct(s['no_answer_rate'])}) | {s['train_pass']} |"
            + (f" {m['gen_tok_s']:.0f} / {m['prefix_hit']:.3f} / {m['kv_peak']:.2f} / {m['preemptions']:.0f} / {m['hours']}h |" if m else " |"))


def paired_md(name, p):
    return (f"| {name} | {p['tasks']} ({p['pairs']} pairs) | {pct(p['pass1_a'])} vs {pct(p['pass1_b'])} | "
            f"{100 * p['diff']:+.1f} pp [{100 * p['diff_ci'][0]:+.1f}, {100 * p['diff_ci'][1]:+.1f}] | "
            f"{p['wins_a']}/{p['wins_b']}/{p['ties']} (p={p['sign_p']:.2f}) | "
            f"{p['tok_ratio']:.3f} [{p['tok_ratio_ci'][0]:.3f}, {p['tok_ratio_ci'][1]:.3f}] | "
            f"{p['geo_ratio']:.3f} [{p['geo_ratio_ci'][0]:.3f}, {p['geo_ratio_ci'][1]:.3f}] |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default=str(HERE / "runs/eff25"), help="treatment arm dir (effort 25)")
    ap.add_argument("--b", default=str(HERE / "runs/eff50"), help="control arm dir (effort 50)")
    ap.add_argument("--json", default=str(HERE / "results.json"))
    args = ap.parse_args()
    kch, ksol = kaggle_set()
    gch, gsol = github_set()
    same = identical_tasks(kch, gch)
    arms = {}
    for name, d in (("eff25", args.a), ("eff50", args.b)):
        f = Path(d) / "dsv41_logs/attempts.jsonl"
        if f.exists():
            arms[name] = (load_arm(f, kch, ksol), metrics(Path(d) / "dsv41_logs/metrics.jsonl"))
    ctrl = load_control(gch, gsol)
    res = {"identical_tasks": len(same), "arms": {}, "paired": {}, "control": {}}

    hdr = ("| arm | tasks | pass@1 (task) | pass@1 (pairs) | out tok/task mean | median | p90 | turns | no answer | "
           "train_pass | gen tok/s / prefix hit / KV peak / preempt / hours |\n|---|---|---|---|---|---|---|---|---|---|---|")
    print("## Arms (Kaggle eval set, 120 tasks / 172 pairs, 1 attempt)\n")
    print(hdr)
    for name, (rows, m) in arms.items():
        s = summary(rows)
        res["arms"][name] = {"summary": s, "metrics": m}
        print(row_md(name, s, m))
    for a in (0, 1):
        s = summary(ctrl[a], same)
        res["control"][f"run7_a{a}"] = s
        print(row_md(f"run7 a{a} (eff 50, 8x box, {len(same)} identical tasks)", s))
    for name, (rows, m) in arms.items():
        s = summary(rows, same)
        res["arms"][name]["identical_tasks"] = s
        print(row_md(f"{name} on the {len(same)} identical tasks", s))
    print()
    for name, (rows, _) in arms.items():
        s = res["arms"][name]["summary"]
        print(f"{name}: stops {s['stops']} | sources {s['sources']} | errors {s['errors']} | cut {s['cut_attempts']} | "
              f"rescues {s['rescues']} | single {s['single']['tasks']} tasks pass@1 {pct(s['single']['pass1'])} "
              f"{s['single']['tok_mean'] / 1e3:.1f}k tok | multi {s['multi']['tasks']} tasks pass@1 {pct(s['multi']['pass1_task'])} "
              f"{s['multi']['tok_mean'] / 1e3:.1f}k tok | mean wall {s['wall_mean_s'] / 60:.1f} min")

    phdr = ("| comparison (a vs b) | tasks | pass@1 a vs b | diff a-b [95% CI] | task wins a/b/ties (sign p) | "
            "token ratio a/b (sum) [95% CI] | geo-mean ratio [95% CI] |\n|---|---|---|---|---|---|---|")
    print("\n## Paired by task\n")
    print(phdr)
    comps = []
    if "eff25" in arms and "eff50" in arms:
        comps.append(("eff25 vs eff50 (all)", arms["eff25"][0], arms["eff50"][0], sorted(kch)))
        comps.append((f"eff25 vs eff50 ({len(same)} identical)", arms["eff25"][0], arms["eff50"][0], same))
    for name in arms:
        comps.append((f"{name} vs run7 a0", arms[name][0], ctrl[0], same))
        comps.append((f"{name} vs run7 a1", arms[name][0], ctrl[1], same))
    comps.append(("run7 a0 vs run7 a1 (same config: noise)", ctrl[0], ctrl[1], same))
    for label, a, b, ts in comps:
        p = paired(a, b, ts)
        res["paired"][label] = p
        print(paired_md(label, p))
    print("\n| comparison (a vs b) | out tok/task a vs b (mean; median) | turns a vs b | out tok/turn a vs b | no answer a vs b |\n|---|---|---|---|---|")
    for label, p in res["paired"].items():
        print(f"| {label} | {p['tok_mean_a'] / 1e3:.1f}k vs {p['tok_mean_b'] / 1e3:.1f}k; {p['tok_median_a'] / 1e3:.1f}k vs "
              f"{p['tok_median_b'] / 1e3:.1f}k | {p['turns_a']:.1f} vs {p['turns_b']:.1f} | {p['tok_per_turn_a']:.0f} vs "
              f"{p['tok_per_turn_b']:.0f} | {p['no_answer_a']} vs {p['no_answer_b']} |")

    traces = {name: load_traces(d) for name, d in (("eff25", args.a), ("eff50", args.b)) if name in arms}
    if any(traces.values()):
        common = sorted(set.intersection(*[set(v) for v in traces.values()])) if len(traces) == 2 else None
        print("\n## Per turn (traces, attempt 0; tasks traced in both arms)\n")
        print("| arm | tasks | turn-1 out tok mean / median | out tok per turn mean / median / p90 | turns | thinking share "
              "(chars) |\n|---|---|---|---|---|---|")
        for name, tr in traces.items():
            print(trace_md(name, tr, common if common is not None else sorted(tr)))

    if "eff25" in arms and "eff50" in arms:
        p = res["paired"]["eff25 vs eff50 (all)"]
        cut = 1 - p["tok_ratio"]
        acc_ok = p["diff"] >= -0.03
        tok_ok = cut >= 0.10
        verdict = ("PASS" if tok_ok and acc_ok else "FAIL") + (
            f": effort 25 changes output tokens per task by {-100 * cut:+.1f}% (sum ratio {p['tok_ratio']:.3f}, 95% CI "
            f"[{p['tok_ratio_ci'][0]:.3f}, {p['tok_ratio_ci'][1]:.3f}]) and task pass@1 by {100 * p['diff']:+.1f} pp "
            f"(95% CI [{100 * p['diff_ci'][0]:+.1f}, {100 * p['diff_ci'][1]:+.1f}]); rule: >= 10-15% fewer tokens and "
            f"pass@1 within 2-3 pp")
        res["verdict"] = verdict
        print("\nVerdict:", verdict)
    with open(args.json, "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
