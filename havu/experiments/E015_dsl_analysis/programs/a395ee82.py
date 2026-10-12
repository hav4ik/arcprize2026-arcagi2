# task a395ee82 pair 0 attempt 1 run tools_persistent_grid_lean2sub_deepseek-flash_low_e010_full_s20x60 (median-length of 18 function programs)
def solve(inp):
    inp=np.array(inp)
    vals,counts=np.unique(inp,return_counts=True)
    bg=vals[np.argmax(counts)]
    nz=[v for v in vals if v!=bg]
    # find components per color
    comps={}
    best=None
    for v in nz:
        mask=inp==v
        lab,n=ndimage.label(mask,structure=np.ones((3,3)))
        for k in range(1,n+1):
            p=np.argwhere(lab==k)
            comps.setdefault(v,[]).append(p)
            if best is None or len(p)>len(best[1]):
                best=(v,p)
    A,tile=best
    B=[v for v in nz if v!=A][0]
    # center: size-1 component of A
    centers=[p for p in comps[A] if len(p)==1]
    center=centers[0][0]
    # coarse cells: B cells + center
    coarse=list(comps[B])+centers
    cells=[]
    for p in coarse:
        for c in p:
            cells.append((int(c[0]),int(c[1])))
    c0=center
    diffs=[(r-c0[0],cc-c0[1]) for r,cc in cells]
    s=0
    for d in diffs: s=gcd(s,gcd(abs(d[0]),abs(d[1])))
    # tile bbox
    t0=tile.min(0); t1=tile.max(0)
    h=t1[0]-t0[0]+1; w=t1[1]-t0[1]+1
    rel=[(int(r-t0[0]),int(cc-t0[1])) for r,cc in tile]
    out=np.full(inp.shape,bg,dtype=int)
    for r,cc in cells:
        dr=(r-c0[0])//s; dc=(cc-c0[1])//s
        col = B if (dr==0 and dc==0) else A
        for rr,rcc in rel:
            R=t0[0]+dr*h+rr; C=t0[1]+dc*w+rcc
            out[R,C]=col
    return out

# submitted:
answer=solve(test_input)
answer
