import json, collections, statistics, sys, os
sol=json.load(open("/home/vu/Workspace/ARC-AGI-2/data/arc-agi_evaluation_solutions.json"))
E043="/home/vu/Workspace/ARC-AGI-2/evals/results/tools_persistent_grid_lean2sub_dsv41_low_e043_e128t_w10_u2_gptq_eng64_sub80.jsonl"
ref=collections.defaultdict(list)
for l in open(E043):
    r=json.loads(l)
    if "task_id" in r: ref[(r["task_id"],r["pair"])].append(bool(r["correct"]))
def cells_of(path):
    cells=[]; stops=collections.Counter(); errs=0; nrec=0
    for l in open(path):
        r=json.loads(l)
        if "task_id" not in r: continue
        nrec+=1
        t=r["task_id"]; n=len(sol[t])
        if r.get("error"): errs+=1
        if r.get("pred") is None: stops[str(r.get("stop_reason"))+("/err" if r.get("error") else "")]+=1
        if r.get("multi_test"):
            preds=r["pred"] if isinstance(r.get("pred"),list) and len(r["pred"])==n else [None]*n
            qs=[(q,preds[q]) for q in range(n)]
        else:
            qs=[(r["pair"],r.get("pred"))]
        for q,g in qs:
            if (t,q) not in ref: continue
            cells.append(dict(t=t,q=q,ok=None if g is None else g==sol[t][q],ref=statistics.mean(ref[(t,q)]),multi=bool(r.get("multi_test")),single=(n==1)))
    return cells,stops,errs,nrec
if __name__=="__main__":
    base="/home/vu/Workspace/ARC-AGI-2/experiments/E046_multi_test_rollout/pods/runs"
    runs=sys.argv[1:] or sorted(d for d in os.listdir(base) if os.path.exists(f"{base}/{d}/out/dsv41_logs/attempts.jsonl"))
    for run in runs:
        p=f"{base}/{run}/out/dsv41_logs/attempts.jsonl" if not run.endswith(".jsonl") else run
        cells,stops,errs,nrec=cells_of(p)
        if not cells: continue
        c=[x for x in cells if x["ok"] is not None]
        f=lambda xs,k: 100*statistics.mean((bool(x["ok"]) if k=="ok" else x["ref"]) for x in xs) if xs else float('nan')
        s=[x for x in cells if x["single"]]; m=[x for x in cells if not x["single"]]
        print(f"{run[-28:]:28s} recs {nrec:3d} cells {len(cells):3d} None {len(cells)-len(c):2d} errs {errs:2d} | None=wrong: {f(cells,'ok'):5.1f} vs {f(cells,'ref'):5.1f} ({f(cells,'ok')-f(cells,'ref'):+5.1f}) | single {f(s,'ok'):5.1f} vs {f(s,'ref'):5.1f} n{len(s):3d} | multi-pair {f(m,'ok'):5.1f} vs {f(m,'ref'):5.1f} n{len(m):3d} | None stops {dict(stops)}")
