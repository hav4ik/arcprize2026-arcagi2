# task 2ba387bc pair 0 attempt 1 run tools_persistent_grid_lean2sub_deepseek-flash_low_e011_full_sub80s40v (median-length of 18 function programs)
def shapes(grid):
    g=np.array(grid)
    lab,n=ndimage.label(g>0)
    out=[]
    for i in range(1,n+1):
        ys,xs=np.where(lab==i)
        r0,r1,c0,c1=ys.min(),ys.max(),xs.min(),xs.max()
        crop=g[r0:r1+1,c0:c1+1]
        col=int(g[ys[0],xs[0]])
        hollow = (crop==0).any()
        out.append((r0,c0,col,hollow,crop))
    return out

def solve(grid):
    sh=shapes(grid)
    hollow=[s for s in sh if s[3]]
    solid=[s for s in sh if not s[3]]
    hollow.sort(key=lambda s:(s[0],s[1]))
    solid.sort(key=lambda s:(s[0],s[1]))
    R=max(len(hollow),len(solid))
    H,W=4,4
    out=np.zeros((R*H,2*W),int)
    for i,s in enumerate(hollow):
        out[i*H:i*H+s[4].shape[0], 0:W]=s[4]
    for i,s in enumerate(solid):
        out[i*H:i*H+s[4].shape[0], W:2*W]=s[4]
    return out

# submitted:
solve(test_input)
