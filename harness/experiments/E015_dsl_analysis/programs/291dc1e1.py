# task 291dc1e1 pair 0 attempt 0 run tools_persistent_grid_lean2sub_deepseek-flash_low_e010_full_s20x40 (median-length of 16 function programs)
def solve(gin):
    g=np.array(gin)
    R,C=g.shape
    r0,c0=np.argwhere(g==0)[0]
    # corner
    top = (r0==0)
    left=(c0==0)
    # marker row is r0 (either 0 or R-1), marker col is c0
    # horizontal arm label = value in row r0 (excluding the 0)
    hvals=[g[r0,j] for j in range(C) if j!=c0]
    hLbl=hvals[0]
    vvals=[g[i,c0] for i in range(R) if i!=r0]
    vLbl=vvals[0]
    # canvas: exclude row r0 and col c0
    rows=[i for i in range(R) if i!=r0]
    cols=[j for j in range(C) if j!=c0]
    canvas=g[np.ix_(rows,cols)]
    mask=canvas!=8
    lab,n=ndimage.label(mask)
    shapes=[]
    for k in range(1,n+1):
        ys,xs=np.where(lab==k)
        sub=canvas[ys.min():ys.max()+1, xs.min():xs.max()+1].copy()
        # keep only this comp (in case bbox includes other cells) - but bbox of comp only contains comp cells? not necessarily
        m=(lab[ys.min():ys.max()+1, xs.min():xs.max()+1]==k)
        sub[~m]=8
        h,w=sub.shape
        if h>w:
            sub=np.rot90(sub)  # CCW
        shapes.append((ys.min(),xs.min(),sub))
    # fast axis
    fast_vertical = (vLbl==1)
    if fast_vertical:
        # slow horizontal
        slow_dec = (not left)   # corner right -> decreasing col
        fast_dec = (not top)    # corner bottom -> decreasing row
        def key(s):
            return ((-s[1] if slow_dec else s[1]), (-s[0] if fast_dec else s[0]))
    else:
        slow_dec = (not top)
        fast_dec = (not left)
        def key(s):
            return ((-s[0] if slow_dec else s[0]), (-s[1] if fast_dec else s[1]))
    shapes.sort(key=key)
    W=max(s[2].shape[1] for s in shapes)
    out=[]
    for _,_,s in shapes:
        h,w=s.shape
        pad=(W-w)//2
        for i in range(h):
            row=[8]*pad+list(s[i])+[8]*(W-w-pad)
            out.append(row)
    return np.array(out)

# submitted:
answer=solve(test_input)
answer
