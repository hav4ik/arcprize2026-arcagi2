# task 16de56c4 pair 1 attempt 0 run tools_persistent_grid_lean2sub_deepseek-flash_low_e009_full_sub80s40 (median-length of 31 function programs)
def process_line(vals):
    # vals: list of ints length n
    n=len(vals)
    pos={}
    for i,c in enumerate(vals):
        if c!=0:
            pos.setdefault(c,[]).append(i)
    if not pos: 
        return vals[:]
    # ruler = color with max count >=2
    ruler=None; rmax=0
    for c,p in pos.items():
        if len(p)>=2 and len(p)>rmax:
            rmax=len(p); ruler=c
    new=vals[:]
    if ruler is None:
        return new  # nothing to extend
    rp=sorted(pos[ruler])
    s=rp[1]-rp[0]
    # progression containing rp[0]
    start=rp[0]%s
    P=[i for i in range(n) if i%s==start]
    # lone
    loners=[(i,c) for c,p in pos.items() if c!=ruler for i in p]
    if len(loners)==1:
        L,lc=loners[0]
        if L in P:
            allidx=[i for i in range(n) if vals[i]!=0]
            lo,hi=min(allidx),max(allidx)
            for i in P:
                if lo<=i<=hi:
                    new[i]=lc
            # clear other nonzero cells not in P?
            for i in allidx:
                if i not in P:
                    new[i]=0
            return new
        else:
            for i in P:
                new[i]=ruler
            # keep lone
            for i in allidx if False else []:
                pass
            return new
    else:
        for i in P:
            new[i]=ruler
        return new

def solve_grid(g, axis):
    a=np.array(g)
    if axis=='col':
        a=a.T
    h,w=a.shape
    out=np.zeros_like(a)
    for r in range(h):
        out[r]=process_line(list(a[r]))
    if axis=='col':
        out=out.T
    return out.tolist()

# submitted:
answer=solve_grid(g,axis)
answer
