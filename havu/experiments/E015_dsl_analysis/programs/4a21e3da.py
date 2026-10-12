# task 4a21e3da pair 0 attempt 1 run tools_persistent_grid_lean2sub_deepseek-flash_high_e011_full_sub80s40h (median-length of 17 function programs)
def solve_single2(a):
    a=np.array(a); H,W=a.shape
    twos=[tuple(x) for x in np.argwhere(a==2)]
    assert len(twos)==1
    r0,c0=twos[0]
    cells=set(map(tuple,np.argwhere(a==7)))
    seven=set(); two=set()
    if r0==0 or r0==H-1:
        V=c0; top=(r0==0)
        left=[p for p in cells if p[1]<V]
        right=[p for p in cells if p[1]>V]
        on=[p for p in cells if p[1]==V]
        seven|=set(on)
        for pts,side in [(left,True),(right,False)]:
            if not pts: continue
            rs=[p[0] for p in pts]; cs=[p[1] for p in pts]
            dr=(0-min(rs)) if top else ((H-1)-max(rs))
            dc=(0-min(cs)) if side else ((W-1)-max(cs))
            for r,c in pts: seven.add((r+dr,c+dc))
        if on:
            rmin=0 if top else min(p[0] for p in on)
            rmax=max(p[0] for p in on) if top else H-1
            for r in range(rmin,rmax+1):
                if (r,V) not in seven: two.add((r,V))
        else:
            # draw from 2 to opposite edge? unknown
            pass
    else:
        Hrow=r0; right=(c0==W-1)
        top=[p for p in cells if p[0]<Hrow]
        bot=[p for p in cells if p[0]>Hrow]
        on=[p for p in cells if p[0]==Hrow]
        seven|=set(on)
        for pts,up in [(top,True),(bot,False)]:
            if not pts: continue
            rs=[p[0] for p in pts]; cs=[p[1] for p in pts]
            dr=(0-min(rs)) if up else ((H-1)-max(rs))
            dc=(0-min(cs)) if right else ((W-1)-max(cs))
            for r,c in pts: seven.add((r+dr,c+dc))
        if on:
            cmin=min(p[1] for p in on) if right else 0
            cmax=W-1 if right else max(p[1] for p in on)
            for c in range(cmin,cmax+1):
                if (Hrow,c) not in seven: two.add((Hrow,c))
    out=np.ones((H,W),dtype=int)
    for r,c in seven: out[r,c]=7
    for r,c in two: out[r,c]=2
    return out

# submitted:
answer = solve_single2(test_input)
answer
