# task e376de54 pair 0 attempt 1 run tools_persistent_grid_lean2sub_deepseek-flash_low_e009_full_sub80s40 (median-length of 16 function programs)
def get_dir(pos):
    # find direction as integer step vector
    r=pos[:,0]; c=pos[:,1]
    if r.max()==r.min():
        return np.array([0,1]) if c.max()!=c.min() else np.array([0,1])
    if c.max()==c.min():
        return np.array([1,0])
    # diagonal: determine sign
    # pixel with min row
    idx=np.lexsort((c,r))
    p0=pos[idx[0]]; p1=pos[idx[-1]]
    d=np.array([1 if p1[0]>p0[0] else -1, 1 if p1[1]>p0[1] else -1])
    return d

def analyze(grid):
    g=np.array(grid)
    comps=[]
    for c in np.unique(g):
        if c==7: continue
        mask=(g==c)
        lab,n=ndimage.label(mask, structure=np.ones((3,3)))
        for i in range(1,n+1):
            pos=np.argwhere(lab==i)
            comps.append((c,pos))
    return comps

def solve(grid):
    g=np.array(grid)
    comps=analyze(grid)
    infos=[]
    for c,pos in comps:
        m=pos.mean(axis=0)
        d=get_dir(pos)
        n=np.array([-d[1],d[0]])  # perpendicular
        k=float(m[0]*n[0]+m[1]*n[1])
        infos.append(dict(c=c,pos=pos,m=m,d=d,n=n,k=k,L=len(pos)))
    # order by perpendicular coordinate k
    order=sorted(range(len(infos)), key=lambda i:infos[i]['k'])
    ref=infos[order[len(order)//2]]
    L=ref['L']
    dm=ref['d']; nm=ref['n']; refm=ref['m']
    kn=float(refm[0]*nm[0]+refm[1]*nm[1]); nn=float(nm@nm)
    out=g.copy()
    out[g!=7]=7
    for inf in infos:
        t=(inf['k']-kn)/nn
        mp=refm+t*nm
        # place L pixels
        d=inf['d']
        pts=[]
        for i in range(L):
            off=i-(L-1)/2.0
            p=mp+off*d
            pts.append((int(round(p[0])),int(round(p[1]))))
        for (r,cc) in pts:
            out[r,cc]=inf['c']
    return out

# submitted:
answer=solve(test_input)
answer
