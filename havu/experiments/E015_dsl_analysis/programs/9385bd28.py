# task 9385bd28 pair 1 attempt 0 run tools_persistent_grid_lean2sub_deepseek-flash_high_e011_full_sub80s40h (median-length of 22 function programs)
def find_legend(g):
    vals,counts=np.unique(g,return_counts=True)
    bg=vals[np.argmax(counts)]
    mask=g!=bg
    lab,n=label(mask)
    comps=[]
    for i in range(1,n+1):
        ys,xs=np.where(lab==i)
        comps.append((ys,xs))
    def key(c):
        ys,xs=c
        return (ys.max(), -xs.min())
    ly,lx=max(comps,key=key)
    r0,r1=ly.min(),ly.max()
    c0,c1=lx.min(),lx.max()
    cells=set(zip(ly.tolist(),lx.tolist()))
    return bg,(r0,r1,c0,c1),cells

def solve(g, mode='count2'):
    g=np.array(g)
    bg,(r0,r1,c0,c1),lcells=find_legend(g)
    pairs=[(int(g[r,c0]),int(g[r,c0+1])) for r in range(r0,r1+1)]
    out=g.copy()
    H,W=g.shape
    for src,tgt in reversed(pairs):
        if src==bg: 
            continue
        cells=[(r,c) for r in range(H) for c in range(W) if g[r,c]==src and (r,c) not in lcells]
        if not cells: 
            continue
        rs=[x[0] for x in cells]; cs=[x[1] for x in cells]
        a,b,cc,dd=min(rs),max(rs),min(cs),max(cs)
        fill=bg if tgt==0 else tgt
        cellset=set(cells)
        area=(b-a+1)*(dd-cc+1)
        if mode=='count2':
            preserve = len(cells)>=2
        elif mode=='solid':
            preserve = len(cells)<area
        for rr in range(a,b+1):
            for ccc in range(cc,dd+1):
                if (rr,ccc) in lcells: 
                    continue
                if preserve and (rr,ccc) in cellset:
                    out[rr,ccc]=src
                else:
                    out[rr,ccc]=fill
    return out

# submitted:
answer=solve(ti,'solid')
answer=solve(g,'solid')
answer = solve(test_input,'solid')
answer
