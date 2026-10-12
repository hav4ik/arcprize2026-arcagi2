# task 6e453dd6 pair 0 attempt 0 run tools_persistent_grid_lean2sub_deepseek-flash_low_e010_full_s20x60 (median-length of 7 function programs)
def solve(inp):
    inp=np.array(inp)
    H,W=inp.shape
    sep=[c for c in range(W) if (inp[:,c]==5).all()][0]
    left=inp[:,:sep].copy()
    out=np.full((H,W),6)
    out[:,sep]=5
    lab,n=label(left==0, structure=np.array([[0,1,0],[1,1,1],[0,1,0]]))
    two=np.zeros(H,bool)
    for c in range(1,n+1):
        rows,cols=np.where(lab==c)
        r0,r1=rows.min(),rows.max(); c0,c1=cols.min(),cols.max()
        shift=(sep-1)-c1
        sub=(left[r0:r1+1,c0:c1+1]==0)
        # holes within sub (4-conn background not touching bbox border)
        bg=~sub
        l2,n2=label(bg)
        border=set(l2[0,:])|set(l2[-1,:])|set(l2[:,0])|set(l2[:,-1])
        holecells=[(r,cc) for r in range(sub.shape[0]) for cc in range(sub.shape[1])
                   if bg[r,cc] and l2[r,cc] not in border]
        # right-edge adjacency: hole cell whose right neighbor col == c1
        has_right_hole=False
        for (r,cc) in holecells:
            if c0+cc+1==c1:
                two[r0+r]=True
        # place shifted shape
        for (r,cc) in zip(*np.where(sub)):
            out[r0+r, c0+cc+shift]=0
    for r in range(H):
        if two[r]:
            out[r,sep+1:]=2
    return out

# submitted:
answer=solve(test_input)
answer
