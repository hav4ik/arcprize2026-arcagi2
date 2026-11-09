# task 221dfab4 pair 1 attempt 0 run tools_persistent_grid_lean2sub_deepseek-flash_low_e009_full_sub64a (median-length of 21 function programs)
def solve(g):
    g=np.array(g)
    vals,counts=np.unique(g,return_counts=True)
    bg=int(vals[np.argmax(counts)])
    bys,bxs=np.where(g==4)
    H,W=g.shape
    horizontal = (bys.min()==bys.max())
    out=g.copy()
    if horizontal:
        R=int(bys[0]); barcols=set(bxs.tolist())
        d_of = (lambda r: R-r) if R > H-1-R else (lambda r: r-R)
        for r in range(H):
            d=d_of(r)%6
            if d==4:
                for c in range(W):
                    if g[r,c]!=bg: out[r,c]=3
                for c in barcols: out[r,c]=3
            elif d in (0,2):
                for c in barcols: out[r,c]=4
            else:
                for c in barcols: out[r,c]=bg
    else:
        C=int(bxs[0]); barrows=set(bys.tolist())
        d_of = (lambda c: C-c) if C > W-1-C else (lambda c: c-C)
        for c in range(W):
            d=d_of(c)%6
            if d==4:
                for r in range(H):
                    if g[r,c]!=bg: out[r,c]=3
                for r in barrows: out[r,c]=3
            elif d in (0,2):
                for r in barrows: out[r,c]=4
            else:
                for r in barrows: out[r,c]=bg
    return out

# submitted:
answer=solve(test_input)
answer
