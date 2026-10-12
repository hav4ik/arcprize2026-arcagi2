# task 67e490f4 pair 0 attempt 0 run tools_persistent_grid_lean2sub_deepseek-flash_low_e009_full_sub96n (median-length of 16 function programs)
def comps(grid, bg):
    g=np.array(grid); H,W=g.shape
    seen=np.zeros(g.shape,bool); out=[]
    for i in range(H):
        for j in range(W):
            if not seen[i,j] and g[i,j]!=bg:
                col=g[i,j]; q=deque([(i,j)]); seen[i,j]=True; cells=[]
                while q:
                    x,y=q.popleft(); cells.append((x,y))
                    for dx,dy in((1,0),(-1,0),(0,1),(0,-1)):
                        nx,ny=x+dx,y+dy
                        if 0<=nx<H and 0<=ny<W and not seen[nx,ny] and g[nx,ny]==col:
                            seen[nx,ny]=True; q.append((nx,ny))
                out.append((col,cells))
    return out

def canon(cells):
    # canonical under D4 (rotations + reflections), translation-normalized
    pts=[tuple(c) for c in cells]
    forms=[]
    for t in range(4):
        r=[(x,y) for x,y in pts]
        for _ in range(t):
            r=[(y,-x) for x,y in r]
        forms.append(r)
    # reflections
    for t in range(4):
        r=[(x,-y) for x,y in pts]
        for _ in range(t):
            r=[(y,-x) for x,y in r]
        forms.append(r)
    best=None
    for f in forms:
        mnx=min(p[0] for p in f); mny=min(p[1] for p in f)
        s=tuple(sorted((x-mnx,y-mny) for x,y in f))
        if best is None or s<best: best=s
    return best

def solve_attempt(grid):
    g=np.array(grid)
    bg=np.bincount(g.flatten()).argmax()
    comps_all=comps(g,bg); comps_all.sort(key=lambda t:-len(t[1]))
    framecol=comps_all[0][0]
    ys,xs=np.where(g==framecol); r0,r1,c0,c1=ys.min(),ys.max(),xs.min(),xs.max()
    mask=np.zeros(g.shape,bool); mask[r0:r1+1,c0:c1+1]=True
    from collections import defaultdict
    shape_count=defaultdict(Counter)
    for col,cells in comps(g,bg):
        if col==framecol: continue
        rr=[c[0] for c in cells]; cc=[c[1] for c in cells]
        if mask[min(rr),min(cc)] and mask[max(rr),max(cc)]: continue
        shape_count[canon(cells)][col]+=1
    sub=g[r0:r1+1,c0:c1+1]
    o=np.full(sub.shape,framecol,dtype=int)
    for col,cells in comps(sub,framecol):
        s=canon(cells); cand=shape_count.get(s)
        if not cand: continue
        best=max(cand.items(), key=lambda kv:kv[1])[0]
        for x,y in cells: o[x,y]=best
    return o

# submitted:
answer = solve_attempt(test_input)
answer
