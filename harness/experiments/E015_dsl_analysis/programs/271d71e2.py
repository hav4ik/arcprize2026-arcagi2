# task 271d71e2 pair 0 attempt 0 run tools_persistent_grid_lean2sub_deepseek-flash_low_e009_full_sub96s40 (median-length of 3 function programs)
def comps(g):
    g=np.array(g)
    mask=(g!=6)
    lab,n=ndimage.label(mask, structure=np.array([[0,1,0],[1,1,1],[0,1,0]]))
    for i in range(1,n+1):
        ys,xs=np.where(lab==i)
        print(' comp',i,'rows',ys.min(),ys.max(),'cols',xs.min(),xs.max(),'size',len(ys))

def solve(grid):
    g=np.array(grid); H,W=g.shape
    out=np.full((H,W),6,int)
    mask=(g!=6)
    lab,n=ndimage.label(mask, structure=np.array([[0,1,0],[1,1,1],[0,1,0]]))
    comps=[]
    for i in range(1,n+1):
        ys,xs=np.where(lab==i)
        comps.append((ys,xs,set(g[ys,xs].tolist())))
    rooms=[c for c in comps if c[2]&{0,5,7}]
    nines=[c for c in comps if c[2]=={9}]
    for (ys2,xs2,v2) in nines:
        out[ys2,xs2]=9
    for (ys,xs,vals) in rooms:
        r0,r1,c0,c1=ys.min(),ys.max(),xs.min(),xs.max()
        sub=g[r0:r1+1,c0:c1+1].copy()
        np9=np.argwhere(sub==9)
        rm=np.isin(sub,[0,5,7]); rys,rxs=np.where(rm)
        ir0,ir1,ic0,ic1=rys.min(),rys.max(),rxs.min(),rxs.max()
        above = np9[:,0].max()<ir0
        below = np9[:,0].min()>ir1
        left  = np9[:,1].max()<ic0
        right = np9[:,1].min()>ic1
        interior=sub[ir0+1:ir1, ic0+1:ic1]; h,w=interior.shape
        cnt5=int((interior==5).sum())
        if above: side='top'; t=ir0-np9[:,0].min(); wall=int(np9[:,0].max())
        elif below: side='bottom'; t=np9[:,0].max()-ir1; wall=int(np9[:,0].min())
        elif left: side='left'; t=ic0-np9[:,1].min(); wall=int(np9[:,1].max())
        elif right: side='right'; t=np9[:,1].max()-ic1; wall=int(np9[:,1].min())
        else: side='none'; t=0; wall=0
        gap=None
        if t>=2: gap=t-1
        elif t==1:
            if side in ('left','right'):
                abscol=wall+c0; cand=[]
                for (ys2,xs2,v2) in nines:
                    if ys2.min()>=r0 and ys2.max()<=r1:
                        cc=xs2.min()
                        if side=='left' and cc<abscol: cand.append(cc)
                        if side=='right' and cc>abscol: cand.append(cc)
                if cand: tc=max(cand) if side=='left' else min(cand); gap=abs(tc-abscol)
            else:
                absrow=wall+r0; cand=[]
                for (ys2,xs2,v2) in nines:
                    if xs2.min()>=c0 and xs2.max()<=c1:
                        rr=ys2.min()
                        if side=='top' and rr<absrow: cand.append(rr)
                        if side=='bottom' and rr>absrow: cand.append(rr)
                if cand: tr=max(cand) if side=='top' else min(cand); gap=abs(tr-absrow)
        if gap is None: gap=0
        s=min(gap,cnt5)
        cells5=[(r,c) for r in range(h) for c in range(w) if interior[r,c]==5]
        def key(rc):
            r,c=rc
            if side=='right': return (w-1-c, r)
            if side=='left': return (c, -r)
            if side=='top': return (r, c)
            if side=='bottom': return (h-1-r, -c)
            return (0,0)
        cells5.sort(key=key)
        newinter=interior.copy()
        for (r,c) in cells5[:s]: newinter[r,c]=7
        sub[ir0+1:ir1, ic0+1:ic1]=newinter
        def blit(part,orow,ocol):
            for rr in range(part.shape[0]):
                for cc in range(part.shape[1]):
                    Y=orow+rr; X=ocol+cc
                    if 0<=Y<H and 0<=X<W: out[Y,X]=part[rr,cc]
        if side=='top':
            blit(sub[:wall,:],r0,c0); blit(sub[wall:,:],r0+wall-s,c0)
        elif side=='bottom':
            blit(sub[wall+1:,:],r0+wall+1,c0); blit(sub[:wall+1,:],r0+s,c0)
        elif side=='left':
            blit(sub[:,:wall],r0,c0); blit(sub[:,wall:],r0,c0+wall-s)
        elif side=='right':
            blit(sub[:,wall+1:],r0,c0+wall+1); blit(sub[:,:wall+1],r0,c0+s)
    return out

# submitted:
solve(test_input)
