# task a32d8b75 pair 1 attempt 1 run tools_persistent_grid_lean2subv_deepseek-flash_low_e011_full_subv80s40 (median-length of 7 function programs)
def interior(box):
    b=box.copy()
    # trim all-6 rows/cols
    while b.shape[0]>0 and (b[0]==6).all(): b=b[1:]
    while b.shape[0]>0 and (b[-1]==6).all(): b=b[:-1]
    while b.shape[1]>0 and (b[:,0]==6).all(): b=b[:,1:]
    while b.shape[1]>0 and (b[:,-1]==6).all(): b=b[:,:-1]
    return b

def parse(grid):
    g=np.array(grid)
    H,W=g.shape
    # find separator L = first column all 6
    L=None
    for c in range(W):
        if (g[:,c]==6).all():
            L=c; break
    left=g[:,:L]
    right=g[:,L+1:]
    return g,L,left,right

def solve(grid):
    g,L,left,right=parse(grid)
    H,W=g.shape
    s=L-2
    motif=left[1:L-1,1:L-1]
    cols=[v for v in np.unique(motif) if v!=0]
    a,b=cols[0],cols[1]
    cell=np.where(motif==a,b,np.where(motif==b,a,motif))
    r_top=None
    for r in range(L,H):
        if (left[r]==6).all():
            r_top=r; break
    region=left[L:r_top,:]
    nz=(region!=0)&(region!=6)
    ys,xs=np.where(nz)
    bb=region[ys.min():ys.max()+1, xs.min():xs.max()+1]
    bb=(bb!=0)&(bb!=6)
    sixrows=[r for r in range(r_top,H) if (left[r]==6).all()]
    r0,r1=sixrows[0],sixrows[1]
    box=left[r0:r1+1,:]
    inner=interior(box)
    pos4=np.argwhere(inner==4)[0]
    dr=pos4[0]/(inner.shape[0]-1) if inner.shape[0]>1 else 0
    dc=pos4[1]/(inner.shape[1]-1) if inner.shape[1]>1 else 0
    corner=('T' if dr<0.5 else 'B')+('L' if dc<0.5 else 'R')
    k={'TL':0,'TR':3,'BR':2,'BL':1}[corner]
    bbr=np.rot90(bb,k=k)
    mh,mw=bbr.shape
    Hout,Wout=right.shape
    top=0 if 'T' in corner else Hout-mh*s
    lft=0 if 'L' in corner else Wout-mw*s
    out=right.copy()
    for i in range(mh):
        for j in range(mw):
            if bbr[i,j]:
                out[top+i*s:top+(i+1)*s, lft+j*s:lft+(j+1)*s]=cell
    return out

# submitted:
answer = solve(test_input)
answer
