# task 446ef5d2 pair 1 attempt 1 run tools_persistent_grid_lean2subv_deepseek-flash_low_e011_full_subv80s40 (median-length of 2 function programs)
def fits(r,c,t):
    h,w=t.shape
    if r+h>H or c+w>W: return False
    return all(grid[r+i,c+j]==-1 for i in range(h) for j in range(w))

def rec(used):
    pos=None
    for r in range(H):
        for c in range(W):
            if grid[r,c]==-1: pos=(r,c);break
        if pos:break
    if pos is None: sols.append(grid.copy());return
    r,c=pos
    for idx,t in enumerate(tin):
        if used[idx]:continue
        if fits(r,c,t):
            place(r,c,t,idx);used[idx]=True;rec(used);used[idx]=False;place(r,c,t,-1)

def ncomp(m):
    if m.sum()==0: return 0
    return ndimage.label(m,structure=neigh)[1]

def place(r,c,t,v):
    for i in range(t.shape[0]):
        for j in range(t.shape[1]): grid[r+i,c+j]=v

def solve(grid):
    inp=np.array(grid)
    bg=8
    mask=inp!=bg
    lab,n=ndimage.label(mask)
    tiles=[]; marker_idx=None; marker_corner=None
    marker_cent=None; content_cornercell=None
    for c in range(1,n+1):
        ys,xs=np.where(lab==c)
        sub=inp[ys.min():ys.max()+1, xs.min():xs.max()+1]
        m4=(sub==4)
        mcont=(sub!=8)&(sub!=4)
        yy,xx=np.where(mcont)
        r0,c0=ys.min()+yy.min(), xs.min()+xx.min()
        ct=sub[yy.min():yy.max()+1, xx.min():xx.max()+1]
        tiles.append(ct)
        if m4.any():
            marker_idx=len(tiles)-1
            # global marker centroid
            my,mx=np.where(m4)
            gmy=ys.min()+my.mean(); gmx=xs.min()+mx.mean()
            gcy=r0+ (ct.shape[0]-1)/2; gcx=c0+(ct.shape[1]-1)/2
            dr=gmy-gcy; dc=gmx-gcx
            corner=('T' if dr<0 else 'B')+('L' if dc<0 else 'R')
            marker_corner=corner
            marker_cent=(gmy,gmx)
            # content corner cell global
            h,w=ct.shape
            cr = r0 if 'T' in corner else r0+h-1
            cc = c0 if 'L' in corner else c0+w-1
            content_cornercell=(cr,cc)
    area=sum(t.size for t in tiles)
    R,C=inp.shape
    dims=[]
    for h in range(1,R+1):
        if area%h==0:
            w=area//h
            if w<=C:
                dims.append((h,w))
    best=None
    for (H,W) in dims:
        # tiling search
        g=np.full((H,W),-1)
        sols=[]
        def fits(r,c,t):
            if r+t.shape[0]>H or c+t.shape[1]>W: return False
            return np.all(g[r:r+t.shape[0],c:c+t.shape[1]]==-1)
        def place(r,c,t,v):
            g[r:r+t.shape[0],c:c+t.shape[1]]=v
        import sys
        def rec(used):
            pos=None
            for r in range(H):
                for cc2 in range(W):
                    if g[r,cc2]==-1: pos=(r,cc2);break
                if pos:break
            if pos is None:
                sols.append(g.copy());return
            r,c=pos
            for idx,t in enumerate(tiles):
                if used[idx]:continue
                if fits(r,c,t):
                    place(r,c,t,idx);used[idx]=True;rec(used);used[idx]=False;place(r,c,t,-1)
        sys.setrecursionlimit(10000)
        rec([False]*len(tiles))
        for gg in sols:
            # marker at corner?
            ys,xs=np.where(gg==marker_idx)
            rr,cc=ys.min(),xs.min()
            th,tw=tiles[marker_idx].shape
            okc=False
            if 'T' in marker_corner and rr!=0: continue
            if 'B' in marker_corner and rr+th!=H: continue
            if 'L' in marker_corner and cc!=0: continue
            if 'R' in marker_corner and cc+tw!=W: continue
            cg=np.full((H,W),-1)
            for idx,t in enumerate(tiles):
                yy2,xx2=np.where(gg==idx)
                if len(yy2)==0: continue
                r2,c2=yy2.min(),xx2.min(); cg[r2:r2+t.shape[0],c2:c2+t.shape[1]]=t
            colors=[v for v in np.unique(cg) if v>=0]
            sc=sum(ncomp(cg==v) for v in colors)
            if best is None or sc<best[0]:
                best=(sc,cg,(H,W))
    cg=best[1]; H,W=best[2]
    out=np.full((R,C),bg,int)
    cr,cc=content_cornercell
    if 'T' in marker_corner: rs=cr
    else: rs=cr-H+1
    if 'L' in marker_corner: cs=cc
    else: cs=cc-W+1
    out[rs:rs+H, cs:cs+W]=cg
    return out

# submitted:
solve(test_input)
