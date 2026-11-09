# task dfadab01 pair 0 attempt 0 run tools_persistent_grid_lean2sub_deepseek-flash_low_e010_full_s20x40 (median-length of 10 function programs)
def obj_origin(g):
    from scipy import ndimage
    best=None
    for c in np.unique(g):
        if c==0: continue
        lab,n=ndimage.label(g==c)
        for k in range(1,n+1):
            yy,xx=np.where(lab==k)
            if len(yy)>1: best=c
    if best is None: return None
    ys,xs=np.where(g==best)
    return (int(ys.min()),int(xs.min()))

def solve3(g):
    g=np.array(g); H,W=g.shape; out=np.zeros_like(g)
    O=obj_origin(g)
    skip=set()
    if O: skip.add((O[0]+4,O[1]+4))
    for c in np.unique(g):
        if c==0: continue
        lab,n=ndimage.label(g==c)
        for k in range(1,n+1):
            ys,xs=np.where(lab==k)
            if len(ys)==1 and c in font and (ys[0],xs[0]) not in skip:
                name,fill=font[c]; pat=glyphs[name]
                r,cc=ys[0],xs[0]
                for dr in range(4):
                    for dc in range(4):
                        if pat[dr,dc] and r+dr<H and cc+dc<W:
                            out[r+dr,cc+dc]=fill
    return out

# submitted:
ans=solve3(test_input)
ans
