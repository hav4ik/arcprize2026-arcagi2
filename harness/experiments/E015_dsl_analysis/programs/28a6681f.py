# task 28a6681f pair 0 attempt 1 run tools_persistent_grid_lean2subv_deepseek-flash_low_e011_full_subv80s40 (median-length of 6 function programs)
def settle2(g, maxit=10000):
    g=[row[:] for row in g]
    H=len(g); W=len(g[0])
    def move_cell(r,c):
        moved=True
        while moved:
            moved=False
            if r+1<H and g[r+1][c]==0:
                g[r][c]=0; r=r+1; g[r][c]=1; moved=True; continue
            target=None
            x=c-1
            while x>=0 and g[r][x]==0:
                if r+1<H and g[r+1][x]==0: target=x; break
                x-=1
            if target is None:
                x=c+1
                while x<W and g[r][x]==0:
                    if r+1<H and g[r+1][x]==0: target=x; break
                    x+=1
            if target is not None:
                step=1 if target>c else -1
                g[r][c]=0; c+=step; g[r][c]=1; moved=True
        return r,c
    for it in range(maxit):
        changed=False
        water=[(r,c) for r in range(H) for c in range(W) if g[r][c]==1]
        for (r,c) in water:
            if g[r][c]!=1: continue
            nr,nc=move_cell(r,c)
            if (nr,nc)!=(r,c): changed=True
        if not changed: break
    return np.array(g)

def solve(grid):
    return settle2(np.array(grid))

# submitted:
solve(test_input)
