# task 16b78196 pair 0 attempt 1 run tools_persistent_grid_lean2subv_deepseek-flash_low_e011_full_subv80s40 (median-length of 4 function programs)
def bbox(cells):
    rs=[r for r,c in cells]; cs=[c for r,c in cells]
    return min(rs),max(rs),min(cs),max(cs)

def dock_gen(Acells, Bcells, H, W):
    Ac=set(Acells); br0,br1,bc0,bc1=bbox(Bcells); ar0,ar1,ac0,ac1=bbox(Ac)
    best=None
    for dr in range(ar0-br1-1, ar1-br0+2):
        for dc in range(ac0-bc1-2, ac1-bc0+3):
            Bt=set((r+dr,c+dc) for r,c in Bcells)
            if Bt & Ac: continue
            nbr0,nbr1=br0+dr,br1+dr; nbc0,nbc1=bc0+dc,bc1+dc
            if nbc0<0 or nbc1>=W or nbr0<0 or nbr1>=H: continue
            # row-overlap (vertical docking)
            orows=[r for r in range(max(nbr0,ar0),min(nbr1,ar1)+1)]
            okv=bool(orows)
            for r in orows:
                aset=set(c for (rr,c) in Ac if rr==r and nbc0<=c<=nbc1)
                bset=set(c for (rr,c) in Bt if rr==r)
                if aset & bset or (aset|bset)!=set(range(nbc0,nbc1+1)): okv=False;break
            # col-overlap (horizontal docking)
            ocols=[c for c in range(max(nbc0,ac0),min(nbc1,ac1)+1)]
            okh=bool(ocols)
            for c in ocols:
                aset=set(r for (r,cc) in Ac if cc==c and nbr0<=r<=nbr1)
                bset=set(r for (r,cc) in Bt if cc==c)
                if aset & bset or (aset|bset)!=set(range(nbr0,nbr1+1)): okh=False;break
            ov=max(len(orows) if okv else 0, len(ocols) if okh else 0)
            if ov>0:
                key=(ov,-(dr*dr+dc*dc))
                if best is None or key>best[0]: best=(key,(dr,dc),(nbr0,nbr1,nbc0,nbc1))
    return best

def cells_of(mask):
    return set(map(tuple, np.argwhere(mask)))

def components(grid):
    comps=[]
    for c in sorted(set(grid.flatten())-{0}):
        mask=grid==c
        lbl,n=ndimage.label(mask)
        for k in range(1,n+1):
            cells=cells_of(lbl==k)
            comps.append((c,cells))
    return comps

def chain_solve2(grid):
    H,W=grid.shape
    comps=components(grid); comps.sort(key=lambda t:-len(t[1]))
    fcol,fcells=comps[0]; rem=comps[1:]
    placed={ (r,c):fcol for r,c in fcells}
    pieces=[(fcol,set(fcells))]
    while rem:
        cand=[]
        for idx,(col,cells) in enumerate(rem):
            bo=None
            for (pcol,pcells) in pieces:
                b=dock_gen(pcells,cells,H,W)
                if b and (bo is None or b[0]>bo[0]): bo=b
            if bo: cand.append((bo[0][0],bo[0][1],idx,bo))
        if not cand: break
        cand.sort(key=lambda t:(-t[0],-t[1]))
        _,_,idx,b=cand[0]
        col,cells=rem[idx];(dr,dc)=b[1]
        nc=set((r+dr,c+dc) for r,c in cells)
        for (r,c) in nc: placed[(r,c)]=col
        pieces.append((col,nc)); rem.pop(idx)
    out=np.zeros((H,W),dtype=int)
    for (r,c),v in placed.items(): out[r,c]=v
    return out

# submitted:
chain_solve2(np.array(test_input))
