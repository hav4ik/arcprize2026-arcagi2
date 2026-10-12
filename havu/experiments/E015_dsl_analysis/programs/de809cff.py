# task de809cff pair 0 attempt 0 run tools_persistent_grid_lean2sub_deepseek-flash_low_e011_full_sub80s40v (median-length of 2 function programs)
def solve3(inp):
    inp=np.array(inp); H,W=inp.shape
    colors=sorted([c for c in np.unique(inp) if c!=0])
    rects=[]  # (startrow, r0,r1,c0,c1,color)
    for c in colors:
        m=(inp==c)
        nb=ndimage.convolve(m.astype(int),np.ones((3,3)),mode='constant')-m.astype(int)
        keep=m&(nb>0)
        span={}
        for r in range(H):
            cols=np.where(keep[r])[0]
            if len(cols)>=3:
                lo,hi=cols.min(),cols.max()
                if len(cols)/(hi-lo+1)>=0.5:
                    span[r]=(lo,hi)
        rows=sorted(span)
        groups=[]; cur=[]
        for r in rows:
            if cur and r==cur[-1]+1:
                lo,hi=span[r]; plo,phi=span[cur[-1]]
                if abs(lo-plo)<=1 and abs(hi-phi)<=1:
                    cur.append(r)
                else:
                    groups.append(cur); cur=[r]
            else:
                if cur: groups.append(cur)
                cur=[r]
        if cur: groups.append(cur)
        for g in groups:
            los=sorted(span[r][0] for r in g); his=sorted(span[r][1] for r in g)
            c0=los[len(los)//2]; c1=his[len(his)//2]
            rects.append((g[0],g[0],g[-1],c0,c1,c))
    # z-order by start row
    rects.sort(key=lambda t:t[0])
    out=np.zeros((H,W),dtype=int)
    for (_,r0,r1,c0,c1,c) in rects:
        out[r0:r1+1,c0:c1+1]=c
    # holes: 0 cells inside a rectangle
    holemask=np.zeros((H,W),dtype=bool)
    holecolor={}
    for idx,(_,r0,r1,c0,c1,c) in enumerate(rects):
        sub=(inp[r0:r1+1,c0:c1+1]==0)
        for y,x in np.argwhere(sub):
            Y,X=y+r0,x+c0
            holemask[Y,X]=True
            holecolor[(Y,X)]=c   # later rects override (top)
    # stamp
    for (Y,X),c in list(holecolor.items()):
        other=[cc for cc in colors if cc!=c][0]
        for dy in (-1,0,1):
            for dx in (-1,0,1):
                yy,xx=Y+dy,X+dx
                if 0<=yy<H and 0<=xx<W:
                    out[yy,xx]=other
    for (Y,X),c in holecolor.items():
        out[Y,X]=8
    return out

# submitted:
answer=solve3(test_input)
answer
