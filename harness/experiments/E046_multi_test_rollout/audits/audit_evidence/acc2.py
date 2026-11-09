exec(open('/tmp/claude-1000/-home-vu-Workspace-ARC-AGI-2/81e68821-e1f4-4b05-bad6-9a606f77ad54/scratchpad/audit_evidence/acc.py').read().split("print('\\n=== claim 1")[0].split("for name in [")[0])
multi=[t for t in SOL if len(SOL[t])>1]
pairs=[(t,p) for t in multi for p in range(len(SOL[t]))]
B,_=load('w10_u2_gptq_eng64_sub80'); F,_=load('final_all'); PP,_=load('pp_sub80_tx')
M3,_=load('mt3_sub80_tx_b05'); M3b,_=load('mt3b_sub80_tx_b05'); M6,_=load('mt6_sub80_tx_b025_ctx')
def cells(d,atts,skip_err=False):
    out=[]
    for t,p in pairs:
        for a in atts:
            r=d.get((t,p,a))
            if r is None: continue
            if skip_err and r.get('error'): continue
            # for multi: error on pair0 record means whole attempt
            out.append(((t,p,a),ok(r)))
    return out
def rate(c): return 100*sum(v for _,v in c)/len(c), len(c)
print('base 2 att', rate(cells(B,(0,1))))
print('pp kaggle flags 4 att, errors wrong', rate(cells(PP,(0,1,2,3))), 'errors skipped', rate(cells(PP,(0,1,2,3),True)))
pm=cells(M3,(0,1,2,3))+cells(M3b,(0,1))
print('multi pooled 6 (b0.5) errors wrong', rate(pm))
print('final multi part', rate(cells(F,(0,1))))
print('multi pooled 8 (b0.5 incl final)', rate(pm+cells(F,(0,1))))
print('mt6 b0.25', rate(cells(M6,(0,1))))
print('per-pair pooled 6 (base+pp, err skipped)', rate(cells(B,(0,1))+cells(PP,(0,1,2,3),True)))
# per-task-cluster bootstrap for pooled multi (8) vs pooled per pair (6)
import random
random.seed(1)
def per_task(c):
    d=collections.defaultdict(list)
    for (t,p,a),v in c: d[t].append(v)
    return d
A=per_task(pm+cells(F,(0,1))); Bp=per_task(cells(B,(0,1))+cells(PP,(0,1,2,3),True))
# pair-weighted rate difference with task bootstrap
def rd(ts): 
    a=[v for t in ts for v in A[t]]; b=[v for t in ts for v in Bp[t]]
    return sum(a)/len(a)-sum(b)/len(b)
ts=list(A)
bs=sorted(rd([random.choice(ts) for _ in ts]) for _ in range(4000))
print(f'pooled multi - pooled per pair: {100*rd(ts):+.1f} pp CI [{100*bs[100]:+.1f},{100*bs[3900]:+.1f}]')
# tokens for pp
def tk(d,atts,mt):
    v=[]
    for t in multi:
        for a in atts:
            if mt:
                r=d.get((t,0,a))
                if r and not r.get('error'): v.append(r['usage']['completion'])
            else:
                rs=[d.get((t,p,a)) for p in range(len(SOL[t]))]
                if all(r and not r.get('error') for r in rs): v.append(sum(r['usage']['completion'] for r in rs))
    return statistics.mean(v)/1e3, len(v)
print('tokens base',tk(B,(0,1),False),'pp',tk(PP,(0,1,2,3),False),'m3',tk(M3,(0,1,2,3),True),'m3b',tk(M3b,(0,1),True),'final',tk(F,(0,1),True))
