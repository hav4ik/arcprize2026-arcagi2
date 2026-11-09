# task 97d7923e pair 0 attempt 1 run tools_persistent_grid_lean2sub_deepseek-flash_low_e009_full_sub96n (median-length of 13 function programs)
def runs_of(g):
    g=np.array(g); H,W=g.shape
    runs=[]
    for c in range(W):
        col=g[:,c]; nz=np.nonzero(col)[0]
        if len(nz)==0: continue
        start=nz[0]; prev=nz[0]
        for r in nz[1:]:
            if r==prev+1: prev=r
            else:
                runs.append((c,start,prev)); start=r; prev=r
        runs.append((c,start,prev))
    return runs

def solve(g):
    g=np.array(g).copy(); H,W=g.shape
    runs=runs_of(g)
    # group by color
    from collections import defaultdict
    bycolor=defaultdict(list)
    for (c,s,e) in runs:
        bycolor[g[s,c]].append((c,s,e))
    for color,rs in bycolor.items():
        # label = run starting at row 0
        labels=[r for r in rs if r[1]==0]
        if not labels: continue
        # if multiple labels? take the one... assume one
        lab=labels[0]
        rank=lab[2]-lab[1]+1  # length
        bars=[r for r in rs if r[1]>0]
        bars.sort(key=lambda r:r[1])  # by top ascending
        if rank-1 < len(bars):
            c,s,e=bars[rank-1]
            for r in range(s+1,e):
                g[r,c]=color
    return g

# submitted:
answer=solve(test_input)
answer
