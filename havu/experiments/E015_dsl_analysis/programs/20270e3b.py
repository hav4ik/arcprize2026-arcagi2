# task 20270e3b pair 0 attempt 2 run tools_persistent_grid_lean2sub_deepseek-flash_low_e009_full_sub80s40 (median-length of 21 function programs)
def solve(inp, verbose=False):
    inp=np.array(inp)
    H,W=inp.shape
    seven=np.argwhere(inp==7)
    src=[]; tgt=[]
    for r,c in seven:
        if r+1<H and inp[r+1,c]==4:
            src.append((r,c))
        else:
            tgt.append((r,c))
    src=np.array(src); tgt=np.array(tgt)
    if verbose: print('src',src,'tgt',tgt)
    # centroids
    sc=src.mean(0); tc=tgt.mean(0)
    T=np.round(tc-sc-np.array([1,0])).astype(int)
    if verbose: print('T',T)
    # piece = 8-connected component containing (src_min + (1,0))
    lab,n=ndimage.label(inp==4, structure=np.ones((3,3),int))
    P=tuple(src[np.lexsort((src[:,1],src[:,0]))[0]]+np.array([1,0]))
    P=(int(P[0]),int(P[1]))
    if verbose: print('P',P,'lab',lab[P])
    comp_id=lab[P]
    piece=np.argwhere(lab==comp_id)
    pieceset=set(map(tuple,piece))
    # output 4 cells
    all4=np.argwhere(inp==4)
    out4=set()
    for r,c in all4:
        if (r,c) not in pieceset:
            out4.add((int(r),int(c)))
    for r,c in piece:
        out4.add((int(r)+T[0], int(c)+T[1]))
    pts=np.array(sorted(out4))
    r0,c0=pts[:,0].min(),pts[:,1].min()
    r1,c1=pts[:,0].max(),pts[:,1].max()
    out=np.ones((r1-r0+1,c1-c0+1),dtype=int)
    for r,c in out4:
        out[r-r0,c-c0]=4
    return out

# submitted:
answer = solve(test_input)
answer
