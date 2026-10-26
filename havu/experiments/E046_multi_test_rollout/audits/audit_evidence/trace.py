import sys, random, statistics, collections
sys.path.insert(0, '/tmp/claude-1000/-home-vu-Workspace-ARC-AGI-2/81e68821-e1f4-4b05-bad6-9a606f77ad54/scratchpad/audit_evidence/sim')
import sim2, deadline_sim as ds
M='/home/vu/Workspace/ARC-AGI-2/evals/results/tools_persistent_grid_lean2sub_dsv41_low_e043_e128t_mt3_sub80_tx_b05.jsonl'
jobs, fpj, tasks, npairs, multi_tasks = sim2.build(M)
ds.THRASH=None
def trace(mode, ntask_mode, gate, reps=20, hours=(1,1.5,2,2.5,3,4,5,6,7,8,9,10,11), s=1.04, ratio=0.83):
    acc=collections.defaultdict(list); fg=[]
    for rep in range(reps):
        rng=random.Random(9000+rep)
        draw = list(tasks) if ntask_mode=='eval' else [rng.choice(tasks) for _ in range(240)]
        keys=[]
        for j,t in enumerate(draw):
            if mode=='multi' and t in multi_tasks: keys.append((t,'m',j))
            else: keys += [(t,q,j) for q in range(npairs[t])]
        jp={k:jobs[k[:2]] for k in keys}; fpk={k:fpj.get(k[:2],7000) for k in keys}
        tr=[]
        r=sim2.simulate(jp,keys,fpk,36,s,rng,1.556e6/ratio,gate,trace=tr,t0=0.1*3600 if ntask_mode=='eval' else None)
        fg.append((r['first_gate'] or 0)/3600)
        for h,u,n,act in tr: acc[round(h*4)/4].append(u)
    print(f"{mode:6s} {ntask_mode:5s} gate {gate}: first gate {statistics.mean(fg):.2f}h | " + ' '.join(f"{h}:{statistics.mean(acc[h]):.2f}" for h in hours if acc.get(h)))
for nt in ('eval','240'):
    trace('pp',nt,0); trace('multi',nt,0); trace('multi',nt,0.6)
