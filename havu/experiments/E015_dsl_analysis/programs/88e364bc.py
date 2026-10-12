# task 88e364bc pair 1 attempt 0 run tools_persistent_grid_lean2subv_deepseek-flash_low_e011_full_subv80s40 (median-length of 14 function programs)
def wall_color_and_steps(a,r,c,dr,dc):
    H,W=a.shape
    rr,cc=r,c;steps=0
    while True:
        rr+=dr;cc+=dc
        if not(0<=rr<H and 0<=cc<W): return None,steps
        v=a[rr,cc]
        if v!=0 and v!=4: return int(v),steps
        steps+=1

def find_templates(a):
    H,W=a.shape
    st=np.array([[0,1,0],[1,1,1],[0,1,0]])
    colors=[c for c in set(a.flatten()) if c not in (0,1,2,4)]
    templates={}
    p1=np.argwhere(a==1); p2=np.argwhere(a==2)
    for c in colors:
        mask=(a==c)
        lab,n=ndimage.label(mask,structure=st)
        for j in range(1,n+1):
            ys,xs=np.where(lab==j)
            r0,r1,c0,c1=int(ys.min()),int(ys.max()),int(xs.min()),int(xs.max())
            b1=[(r,cc) for r,cc in p1 if r0<=r<=r1 and c0<=cc<=c1]
            b2=[(r,cc) for r,cc in p2 if r0<=r<=r1 and c0<=cc<=c1]
            if b1 and b2:
                cen1=np.mean(b1,axis=0); cen2=np.mean(b2,axis=0)
                d=cen2-cen1
                step=(int(np.sign(d[0])),int(np.sign(d[1])))
                templates[c]=(step,(r0,r1,c0,c1),len(ys))
    return templates

def solve(grid):
    a=np.array(grid).copy()
    H,W=a.shape
    tpl=find_templates(a)
    dirs={'U':(-1,0),'D':(1,0),'L':(0,-1),'R':(0,1),'DR':(1,1),'UL':(-1,-1),'DL':(1,-1),'UR':(-1,1)}
    fours=[(int(r),int(c)) for r,c in np.argwhere(a==4)]
    for (r,c) in fours:
        wc={}
        for name,(dr,dc) in dirs.items():
            w,s=wall_color_and_steps(a,r,c,dr,dc)
            if w is not None: wc[w]=wc.get(w,0)+1
        col=max(wc,key=wc.get)
        dr,dc=tpl[col][0]
        rr,cc=r,c
        while True:
            nr,nc=rr+dr,cc+dc
            if not(0<=nr<H and 0<=nc<W): break
            if a[nr,nc]!=0 and a[nr,nc]!=4: break
            rr,cc=nr,nc
        a[r,c]=0
        a[rr,cc]=4
    return a

# submitted:
solve(test_input)
