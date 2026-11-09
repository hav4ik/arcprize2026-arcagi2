import json,glob,os,collections,random,statistics
W='/home/vu/Workspace/ARC-AGI-2/'
G={os.path.basename(f)[:-5]:json.load(open(f)) for f in glob.glob(W+'data/arc-agi-2-github/data/evaluation/*.json')}
KC=json.load(open(W+'data/arc-agi_evaluation_challenges.json')); KS=json.load(open(W+'data/arc-agi_evaluation_solutions.json'))
# map kaggle (t,q) -> github (t,p) by input
kmap={}
for t in KC:
    for q,x in enumerate(KC[t]['test']):
        for p,y in enumerate(G[t]['test']):
            if x['input']==y['input'] and KS[t][q]==y['output']: kmap[(t,q)]=(t,p)
print('kaggle pairs',sum(len(v) for v in KS.values()),'mapped to github',len(kmap))
R=W+'evals/results/tools_persistent_grid_lean2sub_dsv41_low_e043_e128t_'
def gl(name,atts=None):
    out=collections.defaultdict(list)
    for r in map(json.loads,open(R+name+'.jsonl')):
        if 'task_id' not in r: continue
        if atts and r['attempt'] not in atts: continue
        p=r.get('pred'); ok=(not r.get('error')) and p is not None and p==G[r['task_id']]['test'][r['pair']]['output']
        out[(r['task_id'],r['pair'])].append(ok)
    return out
refPP=gl('w10_u2_gptq_eng64_sub80')
refMT=collections.defaultdict(list)
for n in ['final_all','mt3_sub80_tx_b05','mt3b_sub80_tx_b05']:
    for k,v in gl(n).items(): refMT[k]+=v
P=W+'experiments/E046_multi_test_rollout/pods/runs/'
files={'kaggle':W+'experiments/E046_multi_test_rollout/kaggle_bench/dsv41_logs/attempts.jsonl'}
for r in ['mt36_b05_gate60','mt36_b05_gate60_kv90','c36_heavy','c36_heavy_kv90','c36_full','c36_restart','c36_deadline','v5_w36']:
    files[r]=P+r+'/out/dsv41_logs/attempts.jsonl'
def l4(f,count_noans=True,drop_cut=True):
    out=[]
    for r in map(json.loads,open(f)):
        if 'task_id' not in r or r.get('error'): continue
        if drop_cut and r.get('cuts'): continue
        t=r['task_id']; n=r.get('multi_test') or 0
        pr=r.get('pred')
        if pr is None and not count_noans: continue
        if n:
            for q in range(n):
                g=pr[q] if isinstance(pr,list) and len(pr)==n else None
                out.append(((t,q,r['attempt']), g is not None and g==KS[t][q]))
        else:
            q=r['pair']; out.append(((t,q,r['attempt']), pr is not None and pr==KS[t][q]))
    return out
def cmp(name,recs,ref,label):
    m=[(k,v) for k,v in recs if (k[0],k[1]) in kmap and ref.get(kmap[(k[0],k[1])])]
    if not m: return None
    obs=sum(v for _,v in m)/len(m); exp=statistics.mean(statistics.mean(ref[kmap[(k[0],k[1])]]) for k,_ in m)
    return m,obs,exp
random.seed(0)
def boot(m,ref):
    by=collections.defaultdict(list)
    for k,v in m: by[k[0]].append((v,statistics.mean(ref[kmap[(k[0],k[1])]])))
    ts=list(by); ds=[]
    for _ in range(3000):
        s=[x for t in (random.choice(ts) for _ in ts) for x in by[t]]
        ds.append(sum(a for a,_ in s)/len(s)-sum(b for _,b in s)/len(s))
    ds.sort(); return ds[75],ds[2925]
pooled_mt=[]
for n,f in files.items():
    for cn in (True,False):
        recs=l4(f,count_noans=cn)
        if n.startswith('c36') or n.startswith('mt36') or n=='kaggle':
            if cn: pooled_mt+= [((n,)+k,v) for k,v in recs]
        for rl,ref in (('E043pp',refPP),('8xMT',refMT)):
            res=cmp(n,recs,ref,rl)
            if res:
                m,obs,exp=res; lo,hi=boot(m,ref)
                print(f"{n:22s} noans={'wrong' if cn else 'excl '} ref {rl:6s} n={len(m):3d} L4 {100*obs:5.1f} ref {100*exp:5.1f} diff {100*(obs-exp):+5.1f} CI[{100*lo:+.1f},{100*hi:+.1f}]")
# pooled multi-test candidate runs (incl. kaggle), clustered by task
for rl,ref in (('E043pp',refPP),('8xMT',refMT)):
    m=[(k[1:],v) for k,v in pooled_mt if (k[1],k[2]) in kmap and ref.get(kmap[(k[1],k[2])])]
    obs=sum(v for _,v in m)/len(m); exp=statistics.mean(statistics.mean(ref[kmap[(k[0],k[1])]]) for k,_ in m)
    lo,hi=boot(m,ref)
    print(f"POOLED multi candidates+kaggle ref {rl}: n={len(m)} attempts on {len({(k[0],k[1]) for k,_ in m})} unique pairs / {len({k[0] for k,_ in m})} tasks: L4 {100*obs:.1f} ref {100*exp:.1f} diff {100*(obs-exp):+.1f} CI[{100*lo:+.1f},{100*hi:+.1f}]")

print('\n--- sim reference (E043 for single-test tasks, mt3 for multi-test tasks) ---')
refSIM={}
mt3=gl('mt3_sub80_tx_b05')
for k,v in refPP.items():
    refSIM[k]= mt3[k] if len(G[k[0]]['test'])>1 and mt3.get(k) else v
for rl,ref in (('SIMref',refSIM),):
    m=[(k[1:],v) for k,v in pooled_mt if (k[1],k[2]) in kmap and ref.get(kmap[(k[1],k[2])])]
    obs=sum(v for _,v in m)/len(m); exp=statistics.mean(statistics.mean(ref[kmap[(k[0],k[1])]]) for k,_ in m)
    lo,hi=boot(m,ref)
    print(f"POOLED V6-config L4 runs vs {rl}: n={len(m)} L4 {100*obs:.1f} ref {100*exp:.1f} diff {100*(obs-exp):+.1f} CI[{100*lo:+.1f},{100*hi:+.1f}]")
    for n in ['kaggle','mt36_b05_gate60','mt36_b05_gate60_kv90','c36_heavy','c36_heavy_kv90','c36_full']:
        recs=l4(files[n]); res=cmp(n,recs,ref,rl); m2,o2,e2=res
        print(f"   {n:22s} n={len(m2)} L4 {100*o2:.1f} ref {100*e2:.1f} diff {100*(o2-e2):+.1f}")
# v5 tokens and no-answer vs E043 on the same pairs
import itertools
E={}
for r in map(json.loads,open(R+'w10_u2_gptq_eng64_sub80.jsonl')):
    if 'task_id' in r: E.setdefault((r['task_id'],r['pair']),[]).append(r)
v5=[json.loads(l) for l in open(files['v5_w36'])]
lt=[];et=[];ln=0;en=[]
for r in v5:
    if r.get('error') or r.get('cuts'): continue
    k=kmap.get((r['task_id'],r['pair']))
    if not k or k not in E: continue
    lt.append(r['usage']['completion']); et.append(statistics.mean(x['usage']['completion'] for x in E[k]))
    ln+= r.get('pred') is None; en.append(statistics.mean(x.get('pred') is None for x in E[k]))
print(f"\nv5 pod vs E043 same pairs: n={len(lt)} tokens/attempt L4 {statistics.mean(lt)/1e3:.1f}k vs E043 {statistics.mean(et)/1e3:.1f}k; no-answer L4 {ln}/{len(lt)} vs E043 expected {sum(en):.1f}")

print('\n--- early-finisher check: tokens of finished L4 attempts vs reference attempts on the same job ---')
def tokref(name):
    d=collections.defaultdict(list)
    for r in map(json.loads,open(R+name+'.jsonl')):
        if 'task_id' in r and r['pair']==0 or ('task_id' in r and not r.get('multi_test')):
            if r.get('multi_share'): continue
            d[(r['task_id'],r['pair'])].append((r.get('usage') or {}).get('completion') or 0)
    return d
tE=tokref('w10_u2_gptq_eng64_sub80'); tM=collections.defaultdict(list)
for n in ['final_all','mt3_sub80_tx_b05','mt3b_sub80_tx_b05']:
    for k,v in tokref(n).items():
        if len(G[k[0]]['test'])>1: tM[k]+=v
for n in ['kaggle','mt36_b05_gate60','mt36_b05_gate60_kv90','c36_heavy','c36_heavy_kv90','c36_full']:
    lt=[];rt=[];
    for r in map(json.loads,open(files[n])):
        if r.get('error') or r.get('cuts') or 'task_id' not in r: continue
        t=r['task_id']
        if r.get('multi_test'):
            if len(G[t]['test'])!=r['multi_test'] or not tM.get((t,0)): continue
            lt.append(r['usage']['completion']); rt.append(statistics.mean(tM[(t,0)]))
        else:
            k=kmap.get((t,r['pair']))
            if not k or len(G[t]['test'])>1 or not tE.get(k): continue
            lt.append(r['usage']['completion']); rt.append(statistics.mean(tE[k]))
    print(f"{n:22s} n={len(lt)} L4 tokens/attempt {statistics.mean(lt)/1e3:.1f}k vs ref mean on same jobs {statistics.mean(rt)/1e3:.1f}k  (ratio {sum(lt)/sum(rt):.2f})")
