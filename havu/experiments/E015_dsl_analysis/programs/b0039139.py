# task b0039139 pair 0 attempt 1 run tools_persistent_grid_lean2sub_deepseek-flash_low_e010_full_s20x60 (median-length of 22 function programs)
def segments(positions, size):
    # positions are indices of separators (rows or cols); return list of (start,end) inclusive
    segs=[]; prev=-1
    for p in positions+[size]:
        if p-prev>1:
            segs.append((prev+1,p-1))
        prev=p
    return segs

def components_of(mask):
    lab,n = ndimage.label(mask, structure=np.array([[0,1,0],[1,1,1],[0,1,0]]))
    return n

def solve(g):
    g = np.array(g); H,W = g.shape
    hsep=[i for i in range(H) if np.all(g[i]==1)]
    vsep=[j for j in range(W) if np.all(g[:,j]==1)]
    panels=[]
    if hsep:
        for (a,b) in segments(hsep,H): panels.append(g[a:b+1,:])
    else:
        for (a,b) in segments(vsep,W): panels.append(g[:,a:b+1])
    tile=None;n=None;solids=[]
    for p in panels:
        nz=p[p!=0]; u=np.unique(nz)
        if len(u)==1 and np.all(p!=0): solids.append(u[0])
        elif 4 in u:
            ys,xs=np.where(p==4);r0,r1,c0,c1=ys.min(),ys.max(),xs.min(),xs.max()
            tile=p[r0:r1+1,c0:c1+1]
        elif 3 in u: n=components_of(p==3)
    C,D=solids[0],solids[1]
    A=np.where(tile==4,C,D)
    if hsep:
        sep=np.full((1,A.shape[1]),D);parts=[]
        for k in range(n):
            parts.append(A)
            if k<n-1: parts.append(sep)
        return np.vstack(parts)
    else:
        sep=np.full((A.shape[0],1),D);parts=[]
        for k in range(n):
            parts.append(A)
            if k<n-1: parts.append(sep)
        return np.hstack(parts)

# submitted:
answer=solve(test_input)
answer
