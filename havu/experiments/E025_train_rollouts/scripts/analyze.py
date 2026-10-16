import json,sys,os,statistics as st,collections
f=sys.argv[1]; arc2=len(sys.argv)>2 and sys.argv[2]=='arc2'
R=[json.loads(l) for l in open(f) if l.strip()]
a1=set(x[:-5] for d in ['training','evaluation'] for x in os.listdir('/home/vu/Workspace/auto-neurogolf/ARC-AGI/data/'+d))
def tok(r): return (r.get('usage') or {}).get('completion',0)
def summ(rs,label):
    if not rs: return
    n=len(rs); c=sum(bool(r.get('correct')) for r in rs); tp=[r for r in rs if r.get('train_pass')]
    ctp=sum(bool(r.get('correct')) for r in tp)
    bt=collections.defaultdict(list)
    for r in rs: bt[r['task_id']].append(bool(r.get('correct')))
    tw=st.mean(sum(v)/len(v) for v in bt.values()); allc=st.mean(all(v) for v in bt.values())
    T=[tok(r) for r in rs]; cost=sum(r.get('est_cost_usd') or 0 for r in rs)
    nc=[r for r in rs if not r.get('correct')]
    print(f"| {label} | {len(bt)} | {n} | {c/n:.1%} | {tw:.1%} | {allc:.1%} | {len(tp)/n:.1%} | {ctp/max(1,len(tp)):.1%} | {sum(bool(r.get('correct')) and not r.get('train_pass') for r in rs)} | {st.mean(T):,.0f} | {st.median(T):,.0f} | ${cost:.2f} |")
print("| subset | tasks | pairs | pass@1 (pair) | pass@1 (task) | all pairs solved | train_pass | P(correct given train_pass) | correct w/o train_pass | out-tok mean | out-tok median | cost |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|")
summ(R,'all')
if arc2:
    summ([r for r in R if r['task_id'] in a1],'ARC-1 origin')
    summ([r for r in R if r['task_id'] not in a1],'ARC-2 only')
# difficulty histogram per pair: correct x train_pass x token bucket
bk=[(0,5000),(5000,10000),(10000,20000),(20000,40000),(40000,10**9)]
names=['<5k','5-10k','10-20k','20-40k','>=40k']
print("\n| out-tokens | pairs | correct+train_pass | correct only | train_pass only (wrong) | neither | pass@1 |")
print("|---|---|---|---|---|---|---|")
for (lo,hi),nm in zip(bk,names):
    rs=[r for r in R if lo<=tok(r)<hi]
    if not rs: print(f"| {nm} | 0 | | | | | |"); continue
    a=sum(bool(r.get('correct')) and bool(r.get('train_pass')) for r in rs)
    b=sum(bool(r.get('correct')) and not r.get('train_pass') for r in rs)
    c=sum((not r.get('correct')) and bool(r.get('train_pass')) for r in rs)
    d=len(rs)-a-b-c
    print(f"| {nm} | {len(rs)} | {a} | {b} | {c} | {d} | {(a+b)/len(rs):.1%} |")
sr=collections.Counter(r.get('stop_reason') for r in R); print("\nstop reasons:",dict(sr))
print("no pred:",sum(1 for r in R if not r.get('pred')),"errors:",sum(1 for r in R if r.get('error')),"peak $:",round(sum(r.get('est_cost_peak_usd') or 0 for r in R),2))
