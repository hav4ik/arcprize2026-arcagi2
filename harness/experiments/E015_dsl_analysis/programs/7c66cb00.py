# task 7c66cb00 pair 0 attempt 0 run tools_persistent_grid_lean2sub_deepseek-flash_low_e011_full_sub80s40v (median-length of 16 function programs)
def solve(g):
    g=np.array(g)
    H,W=g.shape
    vals,counts=np.unique(g,return_counts=True)
    bg=vals[np.argmax(counts)]
    mask=(g!=bg).astype(int)
    lab,n=ndimage.label(mask, structure=np.array([[0,1,0],[1,1,1],[0,1,0]]))
    shapes=[];bands=[]
    for i in range(1,n+1):
        ys,xs=np.where(lab==i)
        r0,r1,c0,c1=ys.min(),ys.max(),xs.min(),xs.max()
        if c0==0 and c1==W-1:
            sub=g[r0:r1+1]
            side=sub[0,0]; fill=sub[0,W//2]
            bands.append((r0,r1,side,fill))
        else:
            shapes.append((r0,r1,c0,c1,g[r0:r1+1,c0:c1+1].copy()))
    bands.sort(key=lambda b:b[0])
    out=np.full((H,W),bg)
    for (r0,r1,side,fill) in bands:
        out[r0:r1+1,0]=side
        out[r0:r1+1,1:W-1]=fill
        out[r0:r1+1,W-1]=side
    for (sr0,sr1,sc0,sc1,mat) in shapes:
        cols=set(mat.flatten().tolist())
        band_list=[b for b in bands if b[3] in cols]
        for idx,(br0,br1,side,fill) in enumerate(band_list):
            h=mat.shape[0]
            top=br1+idx-h+1
            for i in range(h):
                tr=top+i
                if tr<br0 or tr>br1: continue
                for j in range(mat.shape[1]):
                    tc=sc0+j
                    if tc<0 or tc>=W: continue
                    out[tr,tc]= side if mat[i,j]==fill else fill
    return out

# submitted:
answer = solve(test_input)
answer
