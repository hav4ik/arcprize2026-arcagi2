"""LB noise from attempt randomness: per pair (independent attempts per pair) vs multi (one attempt per task)."""
import json, collections, statistics, random
D = "/tmp/claude-1000/-home-vu-Workspace-ARC-AGI-2/81e68821-e1f4-4b05-bad6-9a606f77ad54/scratchpad/audit_decision/"
runs = {n: {tuple([k.split("|")[0], int(k.split("|")[1])]): v for k, v in r.items()} for n, r in json.load(open(D + "runs.json")).items()}
R = "/home/vu/Workspace/ARC-AGI-2/evals/results/tools_persistent_grid_lean2sub_dsv41_low_e043_"
# multi attempts as vectors per task: need attempt alignment -> rebuild from files
def multi_vectors(tags_attempts):
    out = collections.defaultdict(list)
    for tag, atts in tags_attempts:
        by = collections.defaultdict(dict)
        for l in open(R + tag + ".jsonl"):
            r = json.loads(l)
            if "task_id" not in r or r["attempt"] not in atts: continue
            by[(r["task_id"], r["attempt"])][r["pair"]] = bool(r.get("correct"))
        for (t, a), prs in by.items():
            if len(prs) > 1:
                out[t].append([prs[q] for q in sorted(prs)])
    return out
mv = multi_vectors([("e128t_mt3_sub80_tx_b05", {0, 1, 2, 3}), ("e128t_mt3b_sub80_tx_b05", {0, 1}), ("e128t_final_all", {0, 1})])
pp = runs["POOL per pair E043+ppK"]
tasks = sorted(mv)
var_pp = var_mt = 0; mean_pp = mean_mt = 0; npairs = 0
allsame = 0; mixed = 0
for t in tasks:
    n = len(mv[t][0]); npairs += n
    ps = [statistics.mean(pp[(t, q)]) for q in range(n)]
    var_pp += sum(p * (1 - p) for p in ps); mean_pp += sum(ps)
    s = [sum(v) for v in mv[t]]
    var_mt += statistics.pvariance(s); mean_mt += statistics.mean(s)
    for v in mv[t]:
        if all(v) or not any(v): allsame += 1
        else: mixed += 1
print(f"45 multi tasks, {npairs} pairs: expected pairs right per pair {mean_pp:.1f} multi {mean_mt:.1f}")
print(f"attempt-noise variance (pairs^2) on these tasks: per pair {var_pp:.1f}  multi {var_mt:.1f}  ratio {var_mt/var_pp:.2f}")
print(f"multi attempts all-or-nothing {allsame} / mixed {mixed}")
# per-pair within-task correlation for per pair attempts (same attempt index pairs both right?)
# Scale to a 240-task / ~335-pair hidden set: 2x the eval set. Single-test tasks identical in both modes.
single_var = 0
base = runs["E043 per pair (2)"]
for (t, q), v in base.items():
    if t not in mv:
        p = statistics.mean(v); single_var += p * (1 - p)
for lab, vm in (("per pair", var_pp), ("multi", var_mt)):
    tot = 2 * (vm + single_var)  # 240 tasks ~ 2x eval
    print(f"{lab:8s}: LB sd from attempt noise alone ~ {100 * tot ** 0.5 / 334:.2f} pp (335 pairs)")
