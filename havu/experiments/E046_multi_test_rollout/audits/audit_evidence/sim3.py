import sys, random, statistics
sys.path.insert(0, '/tmp/claude-1000/-home-vu-Workspace-ARC-AGI-2/81e68821-e1f4-4b05-bad6-9a606f77ad54/scratchpad/audit_evidence/sim')
import sim2, deadline_sim as ds
M='/home/vu/Workspace/ARC-AGI-2/evals/results/tools_persistent_grid_lean2sub_dsv41_low_e043_e128t_mt3_sub80_tx_b05.jsonl'
jobs, fpj, tasks, npairs, multi_tasks = sim2.build(M)
def run(s5, s6, r5, r6, thr, reps=60, acc5=1.0, acc6=1.0, gate=0.6, w6=36):
    ds.THRASH = thr
    o=[]
    for rep in range(reps):
        res={}
        for mode in ('v5','v6'):
            rng=random.Random(7000+rep); draw=[rng.choice(tasks) for _ in range(240)]
            keys=[]
            for j,t in enumerate(draw):
                if mode=='v6' and t in multi_tasks: keys.append((t,'m',j))
                else: keys+=[(t,q,j) for q in range(npairs[t])]
            jp={k:jobs[k[:2]] for k in keys}; fpk={k:fpj.get(k[:2],7000) for k in keys}
            res[mode]=sim2.simulate(jp,keys,fpk,36 if mode=='v5' else w6,s5 if mode=='v5' else s6,rng,1.556e6/(r5 if mode=='v5' else r6),gate if mode=='v6' else 0)
        o.append(res)
    v5=[r['v5']['pair']*acc5 for r in o]; v6=[r['v6']['pair']*acc6 for r in o]
    d=[b-a for a,b in zip(v5,v6)]
    return statistics.mean(v5),statistics.mean(v6),statistics.mean(d),statistics.pstdev(d),100*sum(x>0 for x in d)/len(d), statistics.mean(r['v5']['thrash_h'] for r in o), statistics.mean(r['v5']['a0miss'] for r in o)
sc=[('calibrated KV: V5 ratio .70, V6 .83, thrash .88:.6',dict(s5=1.04,s6=1.04,r5=0.70,r6=0.83,thr=(0.88,0.6))),
    ('calibrated KV, thrash .82:.6',dict(s5=1.04,s6=1.04,r5=0.70,r6=0.83,thr=(0.82,0.6))),
    ('calibrated KV, s -10% both',dict(s5=0.936,s6=0.936,r5=0.70,r6=0.83,thr=(0.88,0.6))),
    ('calibrated KV, V6 s -8% (gate tok/s cost)',dict(s5=1.04,s6=0.957,r5=0.70,r6=0.83,thr=(0.88,0.6))),
    ('calibrated KV, no thrash',dict(s5=1.04,s6=1.04,r5=0.70,r6=0.83,thr=None)),
    ('calibrated KV, no thrash, V5 acc x0.93 (pod gap)',dict(s5=1.04,s6=1.04,r5=0.70,r6=0.83,thr=None,acc5=0.93)),
    ('calibrated KV, no thrash, V6 acc x0.96 only',dict(s5=1.04,s6=1.04,r5=0.70,r6=0.83,thr=None,acc6=0.96)),
    ('calibrated KV, both acc x0.96',dict(s5=1.04,s6=1.04,r5=0.70,r6=0.83,thr=(0.88,0.6),acc5=0.96,acc6=0.96)),
]
for n,c in sc:
    v5,v6,d,sd,p,th,miss=run(**c)
    print(f'{n:50s} V5 {v5:5.1f} (thrash {th:.1f}h, a0miss {miss:.1f}) V6 {v6:5.1f} diff {d:+.1f} ±{sd:.1f} V6>V5 {p:.0f}%',flush=True)
