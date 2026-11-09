# task 58f5dbd5 pair 0 attempt 0 run tools_persistent_grid_lean2sub_deepseek-flash_low_e010_full_s20x40 (median-length of 17 function programs)
def solve(g):
    g=np.array(g)
    H,W=g.shape
    bg=int(np.argmax(np.bincount(g.ravel())))
    # find top-left corners of 5x5 solid non-bg squares
    corners=[]
    for i in range(H-4):
        for j in range(W-4):
            sub=g[i:i+5,j:j+5]
            c=sub[0,0]
            if c!=bg and np.all(sub==c):
                corners.append((i,j,int(c)))
    rows=sorted(set(i for i,j,c in corners))
    cols=sorted(set(j for i,j,c in corners))
    colmap={j:b for b,j in enumerate(cols)}
    rowmap={i:a for a,i in enumerate(rows)}
    # mark solid cells
    solid=np.zeros_like(g,dtype=bool)
    gridc={}
    for i,j,c in corners:
        solid[i:i+5,j:j+5]=True
        gridc[(rowmap[i],colmap[j])]=c
    nr,nc=len(rows),len(cols)
    out=np.full((nr*5+(nr-1)+2, nc*5+(nc-1)+2), bg,dtype=int)
    for (a,b),c in gridc.items():
        oy=1+a*6; ox=1+b*6
        out[oy:oy+5, ox:ox+5]=c
        # object pattern for color c: non-solid cells of color c
        mask=(g==c)&(~solid)
        ys,xs=np.where(mask)
        y0,y1,x0,x1=ys.min(),ys.max(),xs.min(),xs.max()
        pat=(g[y0:y1+1,x0:x1+1]==c)
        ph,pw=pat.shape
        # place centered in 3x3 interior
        inner=np.full((5,5),False)
        # interior is rows oy+1..oy+3, cols ox+1..ox+3 (3x3)
        # map pattern into it (pattern should be 3x3)
        inner[1:4,1:4]=pat
        for yy in range(5):
            for xx in range(5):
                if inner[yy,xx]:
                    out[oy+yy,ox+xx]=bg
    return out

# submitted:
answer = solve(test_input)
answer
