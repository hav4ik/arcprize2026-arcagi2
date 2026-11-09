"""Reconcile submission.json with attempts.jsonl + solver.log for one run (read-only on the repo)."""
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

SCR = Path(__file__).resolve().parent
os.environ["ARC_OUT"] = str(SCR / "out")
REPO = Path("/home/vu/Workspace/ARC-AGI-2")
sys.path[:0] = [str(REPO / "kaggle/dsv41_tools/pinned_mtf_v6"), str(REPO / "kaggle/dsv41_tools/out/kit/repo/evals")]
import solver_dsv41 as S  # noqa: E402
from train_check import ranked_answers  # noqa: E402

CH = json.load(open(REPO / "data/arc-agi_evaluation_challenges.json"))
SOL = json.load(open(REPO / "data/arc-agi_evaluation_solutions.json"))


def indep_rank(recs):
    """Independent re-implementation: train_pass True first, then attempt index, stable; top-2 distinct."""
    good = [r for r in recs if S.valid_grid(r.get("pred"))]
    good = sorted(enumerate(good), key=lambda x: (0 if x[1].get("train_pass") is True else 1, x[1]["attempt"], x[0]))
    out = []
    for _, r in good:
        if r["pred"] not in out:
            out.append(r["pred"])
        if len(out) == 2:
            break
    return out


def run(d, multi_test=True):
    d = Path(d)
    logs = d / "dsv41_logs"
    recs = [json.loads(l) for l in open(logs / "attempts.jsonl")]
    log = open(logs / "solver.log").read().splitlines()
    requeue = defaultdict(int)
    done = []
    for l in log:
        m = re.search(r"\] (re-queue|retry) (\w+)\.(\d+)#(\d+)", l)
        if m:
            requeue[(m.group(2), int(m.group(3)), int(m.group(4)))] += 1
        m = re.search(r"\] \[(\d+)\] (\w+)\.(\d+)#(\d+) pred=", l)
        if m:
            done.append((m.group(2), int(m.group(3)), int(m.group(4))))
    seen = defaultdict(int)
    kept, dropped = [], []
    for r in recs:
        k = (r["task_id"], r["pair"], r["attempt"])
        seen[k] += 1
        if seen[k] <= requeue[k]:
            dropped.append(r)
        else:
            kept.append(r)
    kept_keys = [(r["task_id"], r["pair"], r["attempt"]) for r in kept]
    issues = []
    if kept_keys != done:
        issues.append(f"kept records != done log lines ({len(kept_keys)} vs {len(done)})")
    for r in dropped:
        if r.get("pred") is not None:
            issues.append(f"dropped (re-queued) record with a pred: {r['task_id']}.{r['pair']}#{r['attempt']}")
    # Rebuild RECORDS exactly as the solver does
    S.RECORDS.clear()
    for r in kept:
        for q in S.split_multi(r, CH[r["task_id"]]):
            S.RECORDS.setdefault((r["task_id"], q["pair"]), []).append(q)
    rebuilt = S.build_submission(CH)
    sub = json.load(open(d / "submission.json"))
    S.check_submission(sub, CH)
    n_diff = 0
    for t in CH:
        for p, row in enumerate(sub[t]):
            if row != rebuilt[t][p]:
                n_diff += 1
                issues.append(f"submission row differs from rebuild: {t}.{p}")
            # independent ranking
            prs = indep_rank(S.RECORDS.get((t, p), []))
            fb = CH[t]["test"][p]["input"]
            a1 = prs[0] if prs else fb
            a2 = prs[1] if len(prs) > 1 else fb
            if (a1, a2) != (row["attempt_1"], row["attempt_2"]):
                issues.append(f"independent ranking differs: {t}.{p}")
            if not S.RECORDS.get((t, p)) and (row["attempt_1"] != fb or row["attempt_2"] != fb):
                issues.append(f"unattempted pair without fallback: {t}.{p}")
    # Scoring
    attempted = set()
    best_ok, sub_ok, a1_ok, a2_ok = set(), set(), set(), set()
    for (t, p), rs in S.RECORDS.items():
        attempted.add((t, p))
        if any(r.get("pred") == SOL[t][p] for r in rs):
            best_ok.add((t, p))
    tot = 0
    for t in CH:
        for p, row in enumerate(sub[t]):
            tot += 1
            if row["attempt_1"] == SOL[t][p]:
                a1_ok.add((t, p))
            if row["attempt_2"] == SOL[t][p]:
                a2_ok.add((t, p))
            if SOL[t][p] in (row["attempt_1"], row["attempt_2"]):
                sub_ok.add((t, p))
    for k in sorted(best_ok - sub_ok):
        rs = S.RECORDS[k]
        issues.append(f"correct record not in submission: {k} recs={[(r['attempt'], r.get('train_pass'), r.get('pred') == SOL[k[0]][k[1]]) for r in rs]}")
    fallback_hits = sorted(sub_ok - best_ok)
    # Multi-test order: pred[q] equals another pair's solution but not its own
    swaps = []
    raw_multi = [r for r in kept if r.get("multi_test")]
    for r in raw_multi:
        t = r["task_id"]
        n = r["multi_test"]
        if n != len(CH[t]["test"]):
            issues.append(f"multi_test n {n} != test count {len(CH[t]['test'])} for {t}")
        if r["pair"] != 0:
            issues.append(f"multi record pair {r['pair']} != 0 for {t}")
        if isinstance(r.get("pred"), list) and len(r["pred"]) == n:
            for q in range(n):
                if r["pred"][q] != SOL[t][q] and any(r["pred"][q] == SOL[t][j] for j in range(n) if j != q):
                    swaps.append((t, q, r["attempt"]))
        elif r.get("pred") is not None:
            issues.append(f"multi record with non-list pred or wrong length: {t}#{r['attempt']}")
    # Single-test tasks must not carry multi_test; multi tasks must
    for r in kept:
        t = r["task_id"]
        if multi_test and len(CH[t]["test"]) > 1 and not r.get("multi_test"):
            issues.append(f"multi-input task without multi_test record: {t}.{r['pair']}#{r['attempt']}")
        if len(CH[t]["test"]) == 1 and r.get("multi_test"):
            issues.append(f"single-input task with multi_test: {t}")
    # train-pass-first ordering checks in the submission
    tp_order = 0
    for (t, p), rs in S.RECORDS.items():
        good = [r for r in rs if S.valid_grid(r.get("pred"))]
        tps = [r for r in good if r.get("train_pass") is True]
        if tps:
            row = sub[t][p]
            if row["attempt_1"] != min(tps, key=lambda r: r["attempt"])["pred"]:
                issues.append(f"train-passing answer not attempt_1: {t}.{p}")
            tp_order += 1
    # attempt_2 distinctness
    dup_a2 = sum(1 for t in CH for p, row in enumerate(sub[t])
                 if row["attempt_1"] == row["attempt_2"] and len({json.dumps(r["pred"]) for r in S.RECORDS.get((t, p), []) if S.valid_grid(r.get("pred"))}) > 1)
    if dup_a2:
        issues.append(f"{dup_a2} rows with attempt_1 == attempt_2 although 2 distinct preds exist")
    # source / stop stats
    src = defaultdict(int)
    for r in kept:
        src[(r.get("answer_source"), r.get("stop_reason"), bool(r.get("cuts")))] += 1
    errs = [(r["task_id"], r["pair"], r["attempt"], (r.get("error") or "")[:80]) for r in kept if r.get("error")]
    return {
        "run": str(d.relative_to(REPO)), "records": len(recs), "kept": len(kept), "dropped": len(dropped),
        "multi_kept": len(raw_multi), "pairs_total": tot, "pairs_attempted": len(attempted),
        "pairs_with_valid_pred": sum(1 for k, rs in S.RECORDS.items() if any(S.valid_grid(r.get("pred")) for r in rs)),
        "best_correct": len(best_ok), "sub_correct": len(sub_ok), "a1": len(a1_ok), "a2_only": len(a2_ok - a1_ok),
        "fallback_hits": fallback_hits, "rows_differ": n_diff, "swaps": swaps, "train_pass_pairs": tp_order,
        "sources": {str(k): v for k, v in sorted(src.items(), key=str)}, "errors_kept": errs, "issues": issues,
    }


if __name__ == "__main__":
    for d in sys.argv[1:]:
        res = run(d)
        print(json.dumps(res, indent=1, default=str))
