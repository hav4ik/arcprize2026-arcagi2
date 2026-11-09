# task e3721c99 pair 0 attempt 1 run tools_persistent_grid_lean2sub_deepseek-flash_low_e011_full_sub80s40v (median-length of 27 function programs)
def holes_of(shape):
    m=shape==0
    lab,n=ndimage.label(m)
    border=set(lab[0,:])|set(lab[-1,:])|set(lab[:,0])|set(lab[:,-1]); border.discard(0)
    return sum(1 for k in range(1,n+1) if k not in border)

def legend_map(inp):
    H,W=inp.shape
    mask=(inp!=0)&(inp!=5)
    lab,n=ndimage.label(mask)
    d={}
    for k in range(1,n+1):
        ys,xs=np.where(lab==k); y0,y1,x0,x1=ys.min(),ys.max(),xs.min(),xs.max()
        if y0==0 or x0==0 or y1==H-1 or x1==W-1:  # touches border -> separator
            continue
        shape=(lab[y0:y1+1,x0:x1+1]==k).astype(int)
        h=holes_of(shape)
        d[h]=inp[ys[0],xs[0]]
    return d

def solve(inp):
    inp=np.array(inp); out=inp.copy()
    d=legend_map(inp)
    lab,n=ndimage.label(inp==5)
    for k in range(1,n+1):
        ys,xs=np.where(lab==k); y0,y1,x0,x1=ys.min(),ys.max(),xs.min(),xs.max()
        shape=(lab[y0:y1+1,x0:x1+1]==k).astype(int)
        h=holes_of(shape)
        c=d.get(h,0)
        out[lab==k]=c
    return out

# submitted:
answer=solve(test_input)
answer
