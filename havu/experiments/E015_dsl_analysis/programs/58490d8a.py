# task 58490d8a pair 0 attempt 1 run tools_persistent_grid_lean2sub_deepseek-flash_low_e009_full_sub80s40 (median-length of 17 function programs)
def solve(inp):
    inp=np.array(inp)
    # find 0 region bounding box
    ys,xs=np.where(inp==0)
    r0,r1,c0,c1=ys.min(),ys.max(),xs.min(),xs.max()
    legend=inp[r0:r1+1,c0:c1+1]
    H,W=legend.shape
    out=np.zeros((H,W),dtype=int)
    # legend entries: single non-zero cells inside legend
    vals,counts=np.unique(inp,return_counts=True)
    bg=vals[np.argmax(counts)]
    # count objects per color outside legend region
    for v in np.unique(legend):
        if v==0: continue
        # position in legend
        ly,lx=np.where(legend==v)
        # count objects in whole inp color v with size>1
        mask=(inp==v)
        lab,n=ndimage.label(mask,structure=np.ones((3,3)))
        cnt=0
        for j in range(1,n+1):
            sz=np.sum(lab==j)
            if sz>1:
                # ensure it's not the legend single cell; legend cells are size1
                cnt+=1
        # place markers
        row=ly[0]; col=lx[0]
        for k in range(cnt):
            out[row, col+2*k]=v
    return out

# submitted:
answer = solve(test_input)
answer
