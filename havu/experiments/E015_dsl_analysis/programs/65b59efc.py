# task 65b59efc pair 0 attempt 0 run tools_persistent_grid_lean2subv_deepseek-flash_low_e011_full_subv80s40 (median-length of 32 function programs)
def solve(grid):
    g=np.array(grid); H,W=g.shape
    row_seps=[r for r in range(H) if g[r,0]==5]
    col_seps=[c for c in range(W) if g[0,c]==5]
    s=row_seps[0]
    row_starts=[0, s+1]
    col_starts=[0]+[c+1 for c in col_seps]
    legend_colors={}
    legend_row=None
    for r in range(H-1, row_starts[1]+s-1, -1):
        ok=True
        for c in col_starts:
            win=g[r,c:c+s]
            nz=[v for v in win if v!=0]
            if len(nz)!=1 or win[s//2]==0:
                ok=False;break
        if ok:
            legend_row=r;break
    for c in col_starts:
        legend_colors[c]=g[legend_row, c+s//2]
    layouts={}
    for c in col_starts:
        tile=g[s+1:2*s+1, c:c+s]
        vals=[v for v in tile.flatten() if v!=0]
        if not vals: continue
        layouts[vals[0]]=tile.copy()
    out=np.zeros((s*s,s*s),dtype=int)
    for c in col_starts:
        tile=g[0:s,c:c+s]
        vals=[v for v in tile.flatten() if v!=0]
        tcol=vals[0]
        newcol=legend_colors[c]
        lay=layouts.get(tcol)
        if lay is None: continue
        for i in range(s):
            for j in range(s):
                if lay[i,j]!=0:
                    out[i*s:(i+1)*s, j*s:(j+1)*s]=np.where(tile!=0,newcol,0)
    return out

# submitted:
answer = solve(test_input)
answer
