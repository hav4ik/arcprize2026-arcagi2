import sys, random, statistics, collections
sys.path.insert(0, '.')
import sim2, deadline_sim as ds
M='/home/vu/Workspace/ARC-AGI-2/evals/results/tools_persistent_grid_lean2sub_dsv41_low_e043_e128t_mt3_sub80_tx_b05.jsonl'
jobs, fpj, tasks, npairs, multi_tasks = sim2.build(M)
ds.THRASH=None
for nt in ('eval','240'):
  for mode,gate in (('pp',0),('multi',0.6)):
    acc=collections.defaultdict(list); tok=collections.defaultdict(list)
    for rep in range(20):
        rng=random.Random(9100+rep)
        draw=list(tasks) if nt=='eval' else [rng.choice(tasks) for _ in range(240)]
        keys=[]
        for j,t in enumerate(draw):
            if mode=='multi' and t in multi_tasks: keys.append((t,'m',j))
            else: keys+=[(t,q,j) for q in range(npairs[t])]
        jp={k:jobs[k[:2]] for k in keys}; fpk={k:fpj.get(k[:2],7000) for k in keys}
        tr=[]; sim2.simulate(jp,keys,fpk,36,1.0,rng,1.556e6/0.83,gate,trace=tr,t0=0.1*3600 if nt=='eval' else None)
        for h,u,n,a in tr:
            b=int(h); acc[b].append(n); tok[b].append(ds.G(a,1.0))
    print(nt,mode,'running by hour:',' '.join(f"{h}:{statistics.mean(acc[h]):.0f}/{statistics.mean(tok[h]):.0f}" for h in sorted(acc)))
