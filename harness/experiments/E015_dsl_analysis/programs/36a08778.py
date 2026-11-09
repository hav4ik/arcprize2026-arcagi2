# task 36a08778 pair 0 attempt 1 run tools_persistent_grid_lean2sub_deepseek-flash_high_e011_full_sub80s40h (median-length of 24 function programs)
def solve(gin):
    g=np.array(gin).copy()
    H,W=g.shape
    # find bars: list of (row,a,b) for maximal horizontal runs of 2
    bars=[]
    for r in range(H):
        c=0
        while c<W:
            if g[r,c]==2:
                a=c
                while c+1<W and g[r,c+1]==2: c+=1
                bars.append((r,a,c))
            c+=1
    barrow={}
    for (r,a,b) in bars:
        barrow.setdefault(r,[]).append((a,b))
    changed=True
    while changed:
        changed=False
        cells=[(r,c) for r in range(H) for c in range(W) if g[r,c]==6]
        for (r,c) in cells:
            if r+1>=H: continue
            v=g[r+1,c]
            if v==7:
                if g[r+1,c]==7:  # redundant
                    g[r+1,c]=6; changed=True
            elif v==2:
                # find bar at row r+1 containing c
                for (a,b) in barrow.get(r+1,[]):
                    if a<=c<=b:
                        # ceiling row r span [a-1,b+1]
                        for cc in range(max(0,a-1),min(W-1,b+1)+1):
                            if g[r,cc]==7:
                                g[r,cc]=6; changed=True
                        for wc in (a-1,b+1):
                            if 0<=wc<W and g[r,wc]==7:
                                g[r,wc]=6; changed=True
                        break
    return g

# submitted:
solve(test_input)
