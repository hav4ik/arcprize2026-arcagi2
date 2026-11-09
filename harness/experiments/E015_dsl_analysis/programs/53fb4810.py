# task 53fb4810 pair 0 attempt 0 run tools_persistent_grid_lean2subv_deepseek-flash_low_e011_full_subv80s40 (median-length of 9 function programs)
def components(g):
    H,W=g.shape
    seen=np.zeros_like(g,bool)
    comps=[]
    for i in range(H):
        for j in range(W):
            if g[i,j] not in (1,8) and not seen[i,j]:
                q=deque([(i,j)]); seen[i,j]=True; cells=[]
                while q:
                    r,c=q.popleft(); cells.append((r,c))
                    for dr,dc in((1,0),(-1,0),(0,1),(0,-1)):
                        nr,nc=r+dr,c+dc
                        if 0<=nr<H and 0<=nc<W and not seen[nr,nc] and g[nr,nc] not in (1,8):
                            seen[nr,nc]=True;q.append((nr,nc))
                comps.append(cells)
    return comps

def solve(g):
    g=np.array(g).copy()
    H,W=g.shape
    out=g.copy()
    for cells in components(g):
        rs=[r for r,c in cells]; cs=[c for r,c in cells]
        r0,r1,c0,c1=min(rs),max(rs),min(cs),max(cs)
        # find adjacent 1 cells
        dirs=[]
        for r in range(r0,r1+1):
            for c in range(c0,c1+1):
                if (r,c) in set(cells):
                    for dr,dc in((1,0),(-1,0),(0,1),(0,-1)):
                        nr,nc=r+dr,c+dc
                        if 0<=nr<H and 0<=nc<W and g[nr,nc]==1:
                            # object at (nr,nc)
                            if nr<r0: dirs.append('down')
                            elif nr>r1: dirs.append('up')
                            elif nc<c0: dirs.append('right')
                            elif nc>c1: dirs.append('left')
        if not dirs: continue
        d=dirs[0]
        h=r1-r0+1; w=c1-c0+1
        block=g[r0:r1+1,c0:c1+1]
        if d=='up':
            for r in range(0,r0):
                out[r,c0:c1+1]=block[(r-r0)%h]
        elif d=='down':
            for r in range(r1+1,H):
                out[r,c0:c1+1]=block[(r-r0)%h]
        elif d=='left':
            for c in range(0,c0):
                out[r0:r1+1,c]=block[:,(c-c0)%w]
        elif d=='right':
            for c in range(c1+1,W):
                out[r0:r1+1,c]=block[:,(c-c0)%w]
    return out

# submitted:
solve(test_input)
