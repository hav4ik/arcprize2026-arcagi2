import json,sys,os,collections
f=sys.argv[1]; R=[json.loads(l) for l in open(f) if l.strip()]
a1=set(x[:-5] for d in ['training','evaluation'] for x in os.listdir('/home/vu/Workspace/auto-neurogolf/ARC-AGI/data/'+d))
T=collections.defaultdict(list)
for r in R: T[r['task_id']].append(r)
def tier(rs):
    ok=all(r.get('correct') for r in rs); tp=all(r.get('train_pass') for r in rs)
    tok=sum((r.get('usage') or {}).get('completion',0) for r in rs)/len(rs)
    if not any(r.get('correct') for r in rs): return '5 unsolved'
    if not ok: return '4 partial (some pairs wrong)'
    if not tp: return '3 correct, train_check fails'
    if tok<10000: return '0 easy (correct+tp, <10k tok)'
    if tok<40000: return '1 medium (correct+tp, 10-40k)'
    return '2 hard (correct+tp, >=40k)'
subs=[('all',lambda t:True)]
if len(sys.argv)>2: subs+=[('ARC-1 origin',lambda t:t in a1),('ARC-2 only',lambda t:t not in a1)]
C={n:collections.Counter(tier(rs) for t,rs in T.items() if p(t)) for n,p in subs}
ks=sorted(set(k for c in C.values() for k in c))
print('| tier | '+' | '.join(n for n,_ in subs)+' |'); print('|---'*(len(subs)+1)+'|')
for k in ks: print(f'| {k[2:]} | '+' | '.join(f"{C[n][k]} ({C[n][k]/sum(C[n].values()):.0%})" for n,_ in subs)+' |')
