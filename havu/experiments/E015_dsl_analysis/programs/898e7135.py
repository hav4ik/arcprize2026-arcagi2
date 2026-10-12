# task 898e7135 pair 0 attempt 0 run tools_persistent_grid_lean2sub_deepseek-flash_low_e009_full_sub80s40 (median-length of 14 function programs)
def gcd_list(xs):
    return reduce(gcd, xs)

def components(a, col):
    m=(a==col)
    lbl,n=ndimage.label(m, structure=np.ones((3,3)))
    out=[]
    for k in range(1,n+1):
        ys,xs=np.where(lbl==k)
        out.append((ys.min(),ys.max(),xs.min(),xs.max(),len(ys)))
    return out

def solve_full(a, verbose=False):
    a=np.array(a)
    cols=[c for c in sorted(set(a.flatten())) if c!=0]
    allc=[]
    for c in cols:
        for (r0,r1,c0,c1,area) in components(a,c):
            allc.append((area,c,r0,r1,c0,c1))
    canvas=max(allc,key=lambda t:t[0])
    _,canvascol,cr0,cr1,cc0,cc1=canvas
    objs=[t for t in allc if t is not canvas and t[0]>=4]
    dims=[]
    for (area,c,r0,r1,c0,c1) in objs:
        dims+= [r1-r0+1, c1-c0+1]
    s=gcd_list(dims)
    H=cr1-cr0+1; W=cc1-cc0+1
    sub=a[cr0:cr1+1,cc0:cc1+1]
    logical=np.where(sub==canvascol,canvascol,0)
    holes=set((r,c) for r in range(H) for c in range(W) if logical[r,c]==0)
    # build placements with hole sets
    objlist=[]
    for (area,c,r0,r1,c0,c1) in objs:
        obj=a[r0:r1+1,c0:c1+1]
        h=(r1-r0+1)//s; w=(c1-c0+1)//s
        pat=np.array([[(obj[i*s:(i+1)*s,j*s:(j+1)*s]==c).sum()>0 for j in range(w)] for i in range(h)])
        pls=[]
        for rot in range(4):
            for flip in range(2):
                p=pat[:, ::-1] if flip else pat.copy()
                p=np.rot90(p,rot)
                ph,pw=p.shape
                for dy in range(H-ph+1):
                    for dx in range(W-pw+1):
                        cells=set()
                        ok=True
                        for i in range(ph):
                            for j in range(pw):
                                r,cc=dy+i,dx+j
                                if (r,cc) in holes:
                                    if not p[i,j]: ok=False;break
                                    cells.add((r,cc))
                                else:
                                    if p[i,j]: ok=False;break
                            if not ok:break
                        if ok: pls.append(cells)
        # dedupe
        pls=list(set(frozenset(x) for x in pls))
        objlist.append((c,frozenset(pls)))
    # backtracking exact cover
    sol=[None]*len(objlist)
    def bt(idx, rem):
        if idx==len(objlist):
            return len(rem)==0
        c,pls=objlist[idx]
        for cells in pls:
            if cells<=rem:
                sol[idx]=(c,cells)
                if bt(idx+1, rem-cells): return True
        return False
    bt(0,set(holes))
    grid=logical.copy()
    for (c,cells) in sol:
        for (r,cc) in cells:
            grid[r,cc]=c
    out=np.kron(grid, np.ones((s,s),int))
    return out

# submitted:
answer = solve_full(test_input)
answer
