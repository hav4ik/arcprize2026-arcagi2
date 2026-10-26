import json, glob, os, random, statistics, collections, sys
R='/home/vu/Workspace/ARC-AGI-2/evals/results/tools_persistent_grid_lean2sub_dsv41_low_e043_e128t_'
G={os.path.basename(f)[:-5]:json.load(open(f)) for f in glob.glob('/home/vu/Workspace/ARC-AGI-2/data/arc-agi-2-github/data/evaluation/*.json')}
SOL={t:[x['output'] for x in g['test']] for t,g in G.items()}
def load(name):
    d={}
    n=0; dup=0
    for r in map(json.loads, open(R+name+'.jsonl')):
        if 'task_id' not in r: continue
        k=(r['task_id'],r['pair'],r['attempt'])
        if k in d: dup+=1
        d[k]=r
    return d,dup
def ok(r):
    if r is None or r.get('error'): return False
    p=r.get('pred')
    return p is not None and p==SOL[r['task_id']][r['pair']]
for name in ['w10_u2_gptq_eng64_sub80','final_all','pp_sub80_tx','mt3_sub80_tx_b05','mt3b_sub80_tx_b05','mt6_sub80_tx_b025_ctx']:
    d,dup=load(name)
    mism=sum(ok(r)!=bool(r.get('correct')) for r in d.values())
    errs=sum(1 for r in d.values() if r.get('error'))
    print(name,len(d),'dups',dup,'correct-field mismatches',mism,'errors',errs,'attempts',sorted(collections.Counter(k[2] for k in d).items()), 'tasks',len({k[0] for k in d}))

print('\n=== claim 1: 120 tasks, base vs final ===')
B,_=load('w10_u2_gptq_eng64_sub80'); F,_=load('final_all')
tasks=sorted(SOL)
multi=[t for t in tasks if len(SOL[t])>1]; single=[t for t in tasks if len(SOL[t])==1]
def task_scores(d,t,atts=(0,1)):
    n=len(SOL[t])
    per_att=[statistics.mean(ok(d.get((t,p,a))) for p in range(n)) for a in atts]
    p2=statistics.mean(any(ok(d.get((t,p,a))) for a in atts) for p in range(n))
    return per_att,p2
def summ(d,ts,label):
    p1=statistics.mean(statistics.mean(task_scores(d,t)[0]) for t in ts)
    p1a0=statistics.mean(task_scores(d,t,(0,))[0][0] for t in ts)
    p1a1=statistics.mean(task_scores(d,t,(1,))[0][0] for t in ts)
    p2=statistics.mean(task_scores(d,t)[1] for t in ts)
    pairs=[(t,p) for t in ts for p in range(len(SOL[t]))]
    pp1=statistics.mean(ok(d.get((t,p,a))) for t,p in pairs for a in (0,1))
    pp2=statistics.mean(any(ok(d.get((t,p,a))) for a in (0,1)) for t,p in pairs)
    print(f'{label:28s} tasks {len(ts)} task-p1 {100*p1:.1f} (a0 {100*p1a0:.1f}, a1 {100*p1a1:.1f}) task-p2 {100*p2:.1f} | pairs {len(pairs)} pair-p1 {100*pp1:.1f} pair-p2 {100*pp2:.1f}')
for lab,ts in (('all',tasks),('single',single),('multi',multi)):
    summ(B,ts,'base '+lab); summ(F,ts,'final '+lab)
# bootstrap over tasks
random.seed(0)
def diff_stat(ts,stat):
    return statistics.mean(stat(F,t)-stat(B,t) for t in ts)
s_p1=lambda d,t: statistics.mean(task_scores(d,t)[0])
s_p1a0=lambda d,t: task_scores(d,t,(0,))[0][0]
s_p2=lambda d,t: task_scores(d,t)[1]
for nm,st in (('task p1',s_p1),('task p1 a0',s_p1a0),('task p2',s_p2)):
  for lab,ts in (('all',tasks),('multi',multi)):
    obs=diff_stat(ts,st)
    per={t:st(F,t)-st(B,t) for t in ts}
    bs=sorted(statistics.mean(per[random.choice(ts)] for _ in ts) for _ in range(4000))
    print(f'{nm} {lab}: diff {100*obs:+.1f} pp, 95% CI [{100*bs[100]:+.1f}, {100*bs[3900]:+.1f}]  tasks better {sum(v>0 for v in per.values())} worse {sum(v<0 for v in per.values())}')
# paired cells same (t,p,a)
c=collections.Counter()
for t in tasks:
    for p in range(len(SOL[t])):
        for a in (0,1):
            c[(ok(B.get((t,p,a))),ok(F.get((t,p,a))))]+=1
print('cells (base,final):',dict(c))
c2=collections.Counter()
for t in tasks:
    for p in range(len(SOL[t])):
        c2[(any(ok(B.get((t,p,a))) for a in (0,1)),any(ok(F.get((t,p,a))) for a in (0,1)))]+=1
print('pair pass@2 (base,final):',dict(c2))
# tokens per task-attempt
def toks(d,t,a,mt):
    if mt and len(SOL[t])>1:
        r=d.get((t,0,a)); return (r.get('usage') or {}).get('completion') or 0, r.get('turns') or 0
    rs=[d.get((t,p,a)) for p in range(len(SOL[t]))]
    return sum((r.get('usage') or {}).get('completion') or 0 for r in rs), sum(r.get('turns') or 0 for r in rs)
for lab,ts in (('all',tasks),('single',single),('multi',multi)):
    bt=[toks(B,t,a,False) for t in ts for a in (0,1)]; ft=[toks(F,t,a,True) for t in ts for a in (0,1)]
    print(f'tokens {lab}: base {statistics.mean(x for x,_ in bt)/1e3:.1f}k turns {statistics.mean(y for _,y in bt):.1f} | final {statistics.mean(x for x,_ in ft)/1e3:.1f}k turns {statistics.mean(y for _,y in ft):.1f} | ratio {statistics.mean(x for x,_ in ft)/statistics.mean(x for x,_ in bt):.3f}')
# check multi-share records have zero usage
ms=[r for r in F.values() if r.get('multi_share')]
print('multi_share recs',len(ms),'nonzero usage',sum(1 for r in ms if (r.get('usage') or {}).get('completion')))
# no answer counts
na=collections.Counter((r.get('stop_reason'),r.get('answer_source')) for k,r in F.items() if k[1]==0)
print('final stop/answer sources (pair0):',na.most_common(12))
