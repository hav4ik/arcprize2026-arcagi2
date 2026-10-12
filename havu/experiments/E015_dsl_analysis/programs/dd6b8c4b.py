# task dd6b8c4b pair 0 attempt 1 run tools_persistent_grid_lean2sub_deepseek-flash_high_e011_full_sub80s40h (median-length of 29 function programs)
def solve(grid):
    g=np.array(grid).copy(); H,W=g.shape
    blk=np.argwhere(g==3)
    r0,r1=blk[:,0].min(),blk[:,0].max()
    c0,c1=blk[:,1].min(),blk[:,1].max()
    starts=set()
    for r in range(r0,r1+1):
        for c in range(c0,c1+1):
            for dr,dc in((1,0),(-1,0),(0,1),(0,-1)):
                nr,nc=r+dr,c+dc
                if 0<=nr<H and 0<=nc<W and g[nr,nc] in (7,9):
                    starts.add((nr,nc))
    dist={s:0 for s in starts}; q=deque(starts)
    while q:
        r,c=q.popleft()
        for dr,dc in((1,0),(-1,0),(0,1),(0,-1)):
            nr,nc=r+dr,c+dc
            if 0<=nr<H and 0<=nc<W and g[nr,nc] in (7,9) and (nr,nc) not in dist:
                dist[(nr,nc)]=dist[(r,c)]+1; q.append((nr,nc))
    nines=[tuple(x) for x in np.argwhere(g==9)]
    reach=[(dist[p],p) for p in nines if p in dist]
    reach.sort()
    cells=[(r,c) for r in range(r0,r1+1) for c in range(c0,c1+1)]
    consume=reach[:len(cells)]
    for d,p in consume: g[p]=7
    for i,(d,p) in enumerate(consume): g[cells[i]]=9
    return g

# submitted:
answer = solve(test_input)
answer
