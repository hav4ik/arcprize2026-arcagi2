# task aa4ec2a5 pair 0 attempt 0 run tools_persistent_grid_lean2sub_deepseek-flash_low_e009_full_sub64a (median-length of 17 function programs)
def solve(grid):
    g=np.array(grid)
    H,W=g.shape
    out=g.copy()
    obj = (g==1)
    # background = 4
    bg = (g==4)
    # label bg with 4-conn
    lbl,n = ndimage.label(bg, structure=S4)
    # find labels touching border
    border_labels=set(lbl[0,:]) | set(lbl[-1,:]) | set(lbl[:,0]) | set(lbl[:,-1])
    border_labels.discard(0)
    hole_mask = bg.copy()
    for l in border_labels:
        hole_mask &= (lbl!=l)
    # holes
    # object components
    olbl,on = ndimage.label(obj, structure=S4)
    # which objects are holed: dilate each hole, find adjacent object labels
    holed = set()
    hl,hn = ndimage.label(hole_mask, structure=S4)
    for i in range(1,hn+1):
        m = (hl==i)
        dil = ndimage.binary_dilation(m, structure=S8)
        labs = set(olbl[dil & obj].flatten()); labs.discard(0)
        holed |= labs
    # apply
    # object cells -> 8 if holed else 1
    for i in range(1,on+1):
        if i in holed:
            out[olbl==i]=8
        else:
            out[olbl==i]=1
    # holes -> 6
    out[hole_mask]=6
    # border: bg cells (non-hole, i.e., outer region) 8-adjacent to object -> 2
    obj_dil = ndimage.binary_dilation(obj, structure=S8)
    bordercells = obj_dil & bg & (~hole_mask)
    out[bordercells]=2
    return out

# submitted:
answer = solve(test_input)
answer
