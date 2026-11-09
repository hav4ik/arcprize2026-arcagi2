# task dbff022c pair 0 attempt 0 run tools_persistent_grid_lean2sub_deepseek-flash_low_e009_full_sub96s40 (median-length of 2 function programs)
def solve(grid):
    g=np.array(grid)
    H,W=g.shape
    lab,n=ndimage.label(g>0, structure=np.array([[0,1,0],[1,1,1],[0,1,0]]))
    # find key: component with >1 distinct color
    key=None
    for c in range(1,n+1):
        ys,xs=np.where(lab==c)
        cols=set(g[ys,xs].tolist())
        if len(cols)>1:
            key=c
    ys,xs=np.where(lab==key)
    r0,r1,c0,c1=ys.min(),ys.max(),xs.min(),xs.max()
    kh,kw=r1-r0+1, c1-c0+1
    # build mapping
    mp={}
    def add(a,b): mp[a]=b
    if kh==2 and (kw!=2 or r0==0 or r1==H-1):
        # horizontal strip, vertical pairs
        # source lane = outer long side
        if r0==0:
            src,dst=0,1
        elif r1==H-1:
            src,dst=1,0
        else:
            src,dst=0,1
        for cc in range(c0,c1+1):
            add(g[r0+src,cc], g[r0+dst,cc])
    else:
        # vertical strip
        if c0==0:
            src,dst=0,1
        elif c1==W-1:
            src,dst=1,0
        else:
            src,dst=0,1
        for rr in range(r0,r1+1):
            add(g[rr,c0+src], g[rr,c0+dst])
    # flood fill background from border zeros (not crossing key? key is nonzero so fine)
    out=g.copy()
    bg=np.zeros_like(g,dtype=bool)
    from collections import deque
    dq=deque()
    for i in range(H):
        for j in [0,W-1]:
            if g[i,j]==0 and not bg[i,j]:
                bg[i,j]=True; dq.append((i,j))
    for j in range(W):
        for i in [0,H-1]:
            if g[i,j]==0 and not bg[i,j]:
                bg[i,j]=True; dq.append((i,j))
    while dq:
        i,j=dq.popleft()
        for di,dj in((1,0),(-1,0),(0,1),(0,-1)):
            a,b=i+di,j+dj
            if 0<=a<H and 0<=b<W and g[a,b]==0 and not bg[a,b]:
                bg[a,b]=True; dq.append((a,b))
    hole=(g==0)&(~bg)
    # find connected holes
    hl,hn=ndimage.label(hole, structure=np.array([[0,1,0],[1,1,1],[0,1,0]]))
    for hc in range(1,hn+1):
        yy,xx=np.where(hl==hc)
        nbrs=set()
        for a,b in zip(yy,xx):
            for di,dj in((1,0),(-1,0),(0,1),(0,-1)):
                p,q=a+di,b+dj
                if 0<=p<H and 0<=q<W and g[p,q]!=0:
                    nbrs.add(g[p,q])
        nbrs.discard(0)
        if len(nbrs)==1:
            col=next(iter(nbrs))
            if col in mp:
                out[yy,xx]=mp[col]
    return out

# submitted:
ans=solve(test_input)
ans
