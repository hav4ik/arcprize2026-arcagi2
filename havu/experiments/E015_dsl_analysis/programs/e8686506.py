# task e8686506 pair 0 attempt 0 run tools_persistent_grid_lean2sub_deepseek-flash_low_e010_full_s20x40 (median-length of 12 function programs)
def tile_holes(base,smallpieces):
    H=len(base); W=len(base[0])
    free=[(r,c) for r in range(H) for c in range(W) if base[r][c] is None]
    N=len(smallpieces)
    P=[tuple(sorted(p)) for p in smallpieces]
    used=[False]*N
    grid=[[base[r][c] for c in range(W)] for r in range(H)]
    sols=[]
    def fits(idx,ar,ac):
        for r,c,col in P[idx]:
            y,x=ar+r,ac+c
            if y<0 or y>=H or x<0 or x>=W or grid[y][x] is not None: return False
        return True
    def place(idx,ar,ac):
        for r,c,col in P[idx]: grid[ar+r][ac+c]=col
    def unplace(idx,ar,ac):
        for r,c,col in P[idx]: grid[ar+r][ac+c]=None
    def bt():
        if len(sols)>=6: return
        pos=None
        for r in range(H):
            for c in range(W):
                if grid[r][c] is None: pos=(r,c);break
            if pos:break
        if pos is None:
            sols.append([row[:] for row in grid]); return
        tr,tc=pos
        for idx in range(N):
            if used[idx]:continue
            for r,c,col in P[idx]:
                ar,ac=tr-r,tc-c
                if fits(idx,ar,ac):
                    place(idx,ar,ac);used[idx]=True
                    bt()
                    used[idx]=False;unplace(idx,ar,ac)
    bt()
    return sols

def build_anchor_and_holes(inp):
    bg=Counter(inp.flatten()).most_common(1)[0][0]
    # group cells by color
    bycol={}
    for c in sorted(set(inp.flatten())):
        if c==bg: continue
        bycol[c]=[tuple(x) for x in np.argwhere(inp==c)]
    maxn=max(len(v) for v in bycol.values())
    anchors=[c for c,v in bycol.items() if len(v)==maxn]
    acells=[p for c in anchors for p in bycol[c]]
    r0=min(r for r,c in acells); c0=min(c for r,c in acells)
    r1=max(r for r,c in acells); c1=max(c for r,c in acells)
    H=r1-r0+1; W=c1-c0+1
    base=[[None]*W for _ in range(H)]
    for r,c in acells:
        base[r-r0][c-c0]=inp[r,c]
    return bg,anchors,base,bycol

def solve(inp):
    inp=np.array(inp)
    bg,anchors,base,bycol=build_anchor_and_holes(inp)
    small=[]
    for c,v in bycol.items():
        if c in anchors: continue
        cs=set(v); seen=set()
        for p in v:
            if p in seen: continue
            stack=[p];seen.add(p);comp=[]
            while stack:
                y,x=stack.pop();comp.append((y,x))
                for dy,dx in((1,0),(-1,0),(0,1),(0,-1)):
                    q=(y+dy,x+dx)
                    if q in cs and q not in seen: seen.add(q);stack.append(q)
            mr=min(r for r,x in comp);mc=min(x for r,x in comp)
            small.append([(r-mr,x-mc,c) for r,x in comp])
    sols=tile_holes(base,small)
    return sols[0], len(sols)

# submitted:
answer=sols[0]
answer = solve(test_input)[0]
answer
