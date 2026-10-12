# task 8f3a5a89 pair 0 attempt 0 run tools_persistent_grid_lean2sub_deepseek-flash_low_e009_full_sub80s40 (median-length of 13 function programs)
def solve2(inp, adj=8, grp=4, erase_adj=8):
    inp=np.array(inp); h,w=inp.shape
    passable=(inp!=1)
    start=tuple(np.argwhere(inp==6)[0])
    comp=np.zeros((h,w),bool); dq=deque([start]); comp[start]=True
    while dq:
        r,c=dq.popleft()
        for dr,dc in((1,0),(-1,0),(0,1),(0,-1)):
            nr,nc=r+dr,c+dc
            if 0<=nr<h and 0<=nc<w and passable[nr,nc] and not comp[nr,nc]:
                comp[nr,nc]=True; dq.append((nr,nc))
    onb=lambda r,c: r==0 or c==0 or r==h-1 or c==w-1
    lab=np.full((h,w),-1); gdirs=[(1,0),(-1,0),(0,1),(0,-1)]
    if grp==8: gdirs+=[(1,1),(1,-1),(-1,1),(-1,-1)]
    gid=0; comps={}
    for r in range(h):
        for c in range(w):
            if inp[r,c]==1 and lab[r,c]<0:
                dq=deque([(r,c)]); lab[r,c]=gid; cells=[(r,c)]
                while dq:
                    x,y=dq.popleft()
                    for dr,dc in gdirs:
                        nx,ny=x+dr,y+dc
                        if 0<=nx<h and 0<=ny<w and inp[nx,ny]==1 and lab[nx,ny]<0:
                            lab[nx,ny]=gid; dq.append((nx,ny)); cells.append((nx,ny))
                comps[gid]=cells; gid+=1
    eadirs=[(1,0),(-1,0),(0,1),(0,-1)]
    if erase_adj==8: eadirs+=[(1,1),(1,-1),(-1,1),(-1,-1)]
    wallB=np.zeros((h,w),bool)
    for g,cells in comps.items():
        touches_border=any(onb(r,c) for r,c in cells)
        adj_comp=any(0<=r+dr<h and 0<=c+dc<w and comp[r+dr,c+dc] for r,c in cells for dr,dc in eadirs)
        if touches_border and adj_comp:
            for r,c in cells: wallB[r,c]=True
    adirs=[(1,0),(-1,0),(0,1),(0,-1)]
    if adj==8: adirs+=[(1,1),(1,-1),(-1,1),(-1,-1)]
    S=np.zeros((h,w),bool)
    for r in range(h):
        for c in range(w):
            if comp[r,c]:
                if onb(r,c): S[r,c]=True
                else:
                    for dr,dc in adirs:
                        nr,nc=r+dr,c+dc
                        if 0<=nr<h and 0<=nc<w and wallB[nr,nc]:
                            S[r,c]=True; break
    out=inp.copy()
    for g,cells in comps.items():
        adj_comp=any(0<=r+dr<h and 0<=c+dc<w and comp[r+dr,c+dc] for r,c in cells for dr,dc in eadirs)
        if not adj_comp:
            for r,c in cells: out[r,c]=8
    for r in range(h):
        for c in range(w):
            if S[r,c] and out[r,c]==8: out[r,c]=7
    return out

# submitted:
solve2(test_input, adj=8, erase_adj=8, grp=4)
