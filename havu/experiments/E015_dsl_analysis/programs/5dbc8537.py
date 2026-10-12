# task 5dbc8537 pair 0 attempt 0 run tools_persistent_grid_lean2sub_deepseek-flash_low_e010_full_s20x40 (median-length of 19 function programs)
def exact_cover(holes,stamps):
    # holes: set of (r,c). stamps list. Return placements dict or None
    holes=set(holes)
    n=len(stamps)
    rels=[s['rel'] for s in stamps]
    used=[False]*n
    placements=[None]*n
    import sys
    sys.setrecursionlimit(10000)
    def rec(remaining):
        if not remaining:
            return all(used)
        # pick hole cell with fewest candidate placements
        best=None;bestc=None
        # to limit cost, just pick min cell
        h=min(remaining)  # topmost-leftmost
        # try each stamp, each rel cell mapping onto h
        for i in range(n):
            if used[i]: continue
            for (dr,dc) in rels[i]:
                # placement offset so that (dr,dc)->h
                orow=h[0]-dr; ocol=h[1]-dc
                cells=[(orow+r,ocol+c) for r,c in rels[i]]
                if all(x in remaining for x in cells):
                    # place
                    used[i]=True; placements[i]=(orow,ocol)
                    newrem=remaining-set(cells)
                    if rec(newrem): return True
                    used[i]=False; placements[i]=None
        return False
    if rec(holes):
        return placements
    return None

def components(mask):
    lab,n=label(mask,structure=cross)
    out=[]
    for i in range(1,n+1):
        cells=np.argwhere(lab==i)
        out.append(cells)
    return out

def get_stamps(panel,bgS):
    mask=panel!=bgS
    stamps=[]
    for cells in components(mask):
        r0,c0=cells.min(axis=0)
        rel=[(int(r-r0),int(c-c0)) for r,c in cells]
        cols={ (int(r-r0),int(c-c0)): int(panel[r,c]) for r,c in cells}
        stamps.append({'rel':rel,'cols':cols,'size':len(rel)})
    stamps.sort(key=lambda s:-s['size'])
    return stamps

def try_solve(target,stamp):
    tvals=set(target.flatten().tolist())
    if len(tvals)!=2: return None
    svals,scounts=np.unique(stamp,return_counts=True)
    bgS=int(svals[np.argmax(scounts)])
    if bgS not in tvals: return None
    holes=set(map(tuple,np.argwhere(target==bgS)))
    if not holes: return None
    if holes==set(map(tuple,np.argwhere(np.ones_like(target,dtype=bool)))): return None
    stamps=get_stamps(stamp,bgS)
    if not stamps: return None
    if sum(s['size'] for s in stamps)!=len(holes):
        return None
    pl=exact_cover(holes,stamps)
    if pl is None: return None
    res=target.copy()
    for s,(orow,ocol) in zip(stamps,pl):
        for (r,c),col in s['cols'].items():
            res[orow+r,ocol+c]=col
    return res

def solve_full(G):
    G=np.array(G); H,W=G.shape
    for k in range(1,W):
        left=G[:,:k]; right=G[:,k:]
        for target,stamp in [(left,right),(right,left)]:
            r=try_solve(target,stamp)
            if r is not None: return r,'V',k
    for k in range(1,H):
        top=G[:k,:]; bot=G[k:,:]
        for target,stamp in [(top,bot),(bot,top)]:
            r=try_solve(target,stamp)
            if r is not None: return r,'H',k
    return None

# submitted:
answer,o,k=solve_full(test_input)
answer=np.array(answer)
answer
