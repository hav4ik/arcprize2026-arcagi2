"""Accuracy of E128 per pair vs multi-test on the multi-pair eval tasks, task-clustered bootstrap."""
import json, collections, random, statistics, sys
R = "/home/vu/Workspace/ARC-AGI-2/evals/results/tools_persistent_grid_lean2sub_dsv41_low_e043_"
sol = json.load(open("/home/vu/Workspace/ARC-AGI-2/data/arc-agi_evaluation_solutions.json"))

def load(tag, attempts=None, drop_errors=False):
    """-> {(task, pair): [correct per attempt]} ; errors counted wrong unless drop_errors."""
    d = collections.defaultdict(dict)
    for l in open(R + tag + ".jsonl"):
        r = json.loads(l)
        if "task_id" not in r: continue
        if attempts is not None and r["attempt"] not in attempts: continue
        if r.get("error") and drop_errors: continue
        d[(r["task_id"], r["pair"])][r["attempt"]] = bool(r.get("correct"))
    return {k: [v[a] for a in sorted(v)] for k, v in d.items()}

base = load("e128t_w10_u2_gptq_eng64_sub80")
npairs = collections.Counter(t for t, _ in base)
multi_tasks = sorted(t for t in npairs if npairs[t] > 1)
single_tasks = sorted(t for t in npairs if npairs[t] == 1)
print("tasks", len(npairs), "multi", len(multi_tasks), "pairs in multi", sum(npairs[t] for t in multi_tasks))

runs = {
 "E043 per pair (2)": load("e128t_w10_u2_gptq_eng64_sub80"),
 "pp Kaggle flags (4, errors dropped)": load("e128t_pp_sub80_tx", drop_errors=True),
 "pp Kaggle flags (4, errors=wrong)": load("e128t_pp_sub80_tx"),
 "run1 mt 80k": load("e128t_mt_sub80"),
 "run2 mt 80k tx": load("e128t_mt2_sub80_tx"),
 "run3 mt b0.5 (a0-1)": load("e128t_mt3_sub80_tx_b05", {0, 1}),
 "run5 mt b0.5 (a2-3)": load("e128t_mt3_sub80_tx_b05", {2, 3}),
 "run4 mt b0.5": load("e128t_mt3b_sub80_tx_b05"),
 "run6 mt b0.25 ctx": load("e128t_mt6_sub80_tx_b025_ctx"),
 "run7 final (multi part)": load("e128t_final_all"),
}
def pool(*names):
    out = collections.defaultdict(list)
    for n in names:
        for k, v in runs[n].items(): out[k] += v
    return dict(out)
runs["POOL multi b0.5 runs3,4,5,7"] = pool("run3 mt b0.5 (a0-1)", "run5 mt b0.5 (a2-3)", "run4 mt b0.5", "run7 final (multi part)")
runs["POOL per pair E043+ppK"] = pool("E043 per pair (2)", "pp Kaggle flags (4, errors dropped)")

def p1_pairs(run, tasks):
    """pair-weighted pass@1 over pairs of tasks (mean over attempts per pair, mean over pairs)."""
    vals = [statistics.mean(run[(t, q)]) for t in tasks for q in range(npairs[t]) if run.get((t, q))]
    return 100 * statistics.mean(vals), len(vals)

def boot_diff(ra, rb, tasks, B=4000, seed=1):
    rng = random.Random(seed)
    def score(run, ts):
        num = den = 0
        for t in ts:
            for q in range(npairs[t]):
                v = run.get((t, q))
                if v: num += statistics.mean(v); den += 1
        return 100 * num / den
    obs = score(ra, tasks) - score(rb, tasks)
    ds = []
    for _ in range(B):
        ts = [rng.choice(tasks) for _ in tasks]
        ds.append(score(ra, ts) - score(rb, ts))
    ds.sort()
    return obs, ds[int(0.025 * B)], ds[int(0.975 * B)], sum(d < 0 for d in ds) / B

print("\n== pass@1 (pair-weighted, mean over attempts) on the 45 multi-pair tasks")
for n, r in runs.items():
    p, k = p1_pairs(r, multi_tasks)
    na = statistics.mean(len(r[(t, q)]) for t in multi_tasks for q in range(npairs[t]) if r.get((t, q)))
    print(f"  {n:40s} {p:5.1f}%  pairs {k}  attempts/pair {na:.1f}")
print("\n== single-test tasks (75): E043 vs run7")
for n in ("E043 per pair (2)", "run7 final (multi part)"):
    print("  ", n, "%.1f%% (%d)" % p1_pairs(runs[n], single_tasks))
print("\n== differences multi - per pair on multi tasks (task bootstrap 95% CI, P(diff<0))")
for a in ("run3 mt b0.5 (a0-1)", "run5 mt b0.5 (a2-3)", "run4 mt b0.5", "run7 final (multi part)", "run6 mt b0.25 ctx", "POOL multi b0.5 runs3,4,5,7"):
    for b in ("E043 per pair (2)", "pp Kaggle flags (4, errors dropped)", "POOL per pair E043+ppK"):
        o, lo, hi, pneg = boot_diff(runs[a], runs[b], multi_tasks)
        print(f"  {a:32s} - {b:36s}: {o:+5.1f} [{lo:+5.1f}, {hi:+5.1f}]  P<0 {pneg:.2f}")
json.dump({n: {f"{k[0]}|{k[1]}": v for k, v in r.items()} for n, r in runs.items()},
          open("/tmp/claude-1000/-home-vu-Workspace-ARC-AGI-2/81e68821-e1f4-4b05-bad6-9a606f77ad54/scratchpad/audit_decision/runs.json", "w"))
