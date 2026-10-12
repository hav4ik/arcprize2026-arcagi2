# task 4c416de3 pair 0 attempt 1 run tools_persistent_grid_lean2subv_deepseek-flash_low_e011_full_subv80s40 (median-length of 9 function programs)
def frames_of(gi,bg):
    z=(gi==0)
    zd=ndimage.binary_dilation(z,iterations=1)
    lab,n=ndimage.label(zd,structure=np.ones((3,3)))
    fr=[]
    for k in range(1,n+1):
        pts=np.argwhere((lab==k)&z)
        if len(pts)>=6:
            fr.append((int(pts[:,0].min()),int(pts[:,1].min()),int(pts[:,0].max()),int(pts[:,1].max())))
    return fr

def corners(fr):
    rmin,cmin,rmax,cmax=fr
    return {'TL':(rmin,cmin),'TR':(rmin,cmax),'BL':(rmax,cmin),'BR':(rmax,cmax)}

def derive_T(gi,bg,fr):
    H,W=gi.shape
    rmin,cmin,rmax,cmax=fr
    cor=corners(fr)
    counts={k:[] for k in cor}
    for r in range(max(0,rmin-5),min(H,rmax+6)):
        for c in range(max(0,cmin-5),min(W,cmax+6)):
            v=gi[r,c]
            if v==bg or v==0: continue
            best=None;bd=None
            for k,(cr,cc) in cor.items():
                d=max(abs(r-cr),abs(c-cc))
                if bd is None or d<bd: bd=d;best=k
            if bd<=5: counts[best].append(((r,c),v))
    bestk=max(counts,key=lambda k:len(counts[k]))
    T=[]
    for (r,c),v in counts[bestk]:
        if bestk in ('TL','TR'): dr=r-rmin
        else: dr=rmax-r
        if bestk in ('TL','BL'): dc=c-cmin
        else: dc=cmax-c
        T.append((dr,dc))
    return T,counts

def place(fr,corner,T):
    rmin,cmin,rmax,cmax=fr
    res=[]
    for (dr,dc) in T:
        if corner=='TL': r,c=rmin+dr,cmin+dc
        elif corner=='TR': r,c=rmin+dr,cmax-dc
        elif corner=='BL': r,c=rmax-dr,cmin+dc
        else: r,c=rmax-dr,cmax-dc
        res.append((r,c))
    return res

def solve(grid):
    gi=np.array(grid).copy()
    bg=np.bincount(gi.flatten()).argmax()
    frs=frames_of(gi,bg)
    T=None;bestcnt=-1
    for fr in frs:
        Tc,counts=derive_T(gi,bg,fr)
        if len(Tc)>bestcnt: bestcnt=len(Tc);T=Tc
    out=gi.copy()
    for fr in frs:
        for k in ['TL','TR','BL','BR']:
            cells=place(fr,k,T)
            present=[(r,c) for (r,c) in cells if 0<=r<gi.shape[0] and 0<=c<gi.shape[1] and gi[r,c]!=bg and gi[r,c]!=0]
            if not present: continue
            col=gi[present[0][0],present[0][1]]
            for (r,c) in cells:
                if 0<=r<gi.shape[0] and 0<=c<gi.shape[1]: out[r,c]=col
    return out

# submitted:
solve(test_input)
