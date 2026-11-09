# task 6ffbe589 pair 0 attempt 1 run tools_persistent_grid_lean2sub_deepseek-flash_high_e011_full_sub80s40h (median-length of 3 function programs)
def solve2(inp, ret_extra=False):
    inp=np.array(inp)
    mask=inp!=0
    dil=ndimage.binary_dilation(mask, iterations=1)
    lab,n=ndimage.label(dil, np.ones((3,3)))
    counts={l:int(((lab==l)&mask).sum()) for l in range(1,n+1)}
    biglabel=max(counts,key=counts.get)
    shapemask=mask&(lab==biglabel)
    ys,xs=np.where(shapemask)
    r0,r1,c0,c1=ys.min(),ys.max(),xs.min(),xs.max()
    crop=inp[r0:r1+1,c0:c1+1]
    legend={}
    legmask=mask&(lab!=biglabel)
    lys,lxs=np.where(legmask)
    for v in inp[lys,lxs]: legend[int(v)]=legend.get(int(v),0)+1
    out=np.zeros_like(crop)
    for color in sorted(set(crop.flatten())-{0}):
        m=crop==color
        k=legend.get(int(color),0)%4
        out[np.rot90(m,(4-k)%4)]=color
    if ret_extra: return out,crop,legend,(r0,r1,c0,c1)
    return out

# submitted:
answer = solve2(test_input)
answer
