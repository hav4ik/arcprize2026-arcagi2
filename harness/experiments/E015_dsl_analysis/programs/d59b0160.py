# task d59b0160 pair 0 attempt 1 run tools_persistent_grid_lean2sub_deepseek-flash_low_e011_full_sub80s40v (median-length of 16 function programs)
def solve(g):
    a=np.array(g)
    key=a[0:3,0:3]
    keyset=set(int(v) for v in key.flatten() if v not in (0,7))
    m=(a!=7)
    lab,n=ndimage.label(m, structure=np.array([[0,1,0],[1,1,1],[0,1,0]]))
    out=a.copy()
    for i in range(1,n+1):
        ys,xs=np.where(lab==i)
        # skip key region components (overlap rows0-3 cols0-3)
        if ys.min()<=3 and xs.min()<=3 and ys.max()>=0 and xs.max()>=0 and (ys.min()<=3 and xs.min()<=3 and ys.min()<=3):
            if ys.max()<=3 and xs.max()<=3:
                continue
        vals=set(int(v) for v in a[lab==i])
        if keyset and keyset<=vals:
            out[lab==i]=7
    return out

# submitted:
answer = solve(test_input)
answer
