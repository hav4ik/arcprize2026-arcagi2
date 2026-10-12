# task a25697e4 pair 0 attempt 1 run tools_persistent_grid_lean2sub_deepseek-flash_low_e010_full_s20x60 (median-length of 8 function programs)
def components(grid,bg):
    mask=(grid!=bg); lab,n=ndimage.label(mask,np.ones((3,3)))
    return [np.where(lab==L) for L in range(1,n+1)]

def TF(m,k,fl):
    r=np.rot90(m,k)
    if fl: r=np.fliplr(r)
    return r

def group_comps(comps,grid,thr=2):
    groups=[]; used=[False]*len(comps)
    for i,(ys,xs) in enumerate(comps):
        if used[i]: continue
        cols=set(grid[ys,xs].tolist())
        if len(cols)>1:
            groups.append([i]); used[i]=True; continue
        group=[i]; used[i]=True; changed=True
        while changed:
            changed=False
            allc=np.concatenate([comps[j][1] for j in group]); allr=np.concatenate([comps[j][0] for j in group])
            r0,r1,c0,c1=allr.min(),allr.max(),allc.min(),allc.max()
            for k,(ys2,xs2) in enumerate(comps):
                if used[k]: continue
                if set(grid[ys2,xs2].tolist())!=cols: continue
                rr0,rr1,cc0,cc1=ys2.min(),ys2.max(),xs2.min(),xs2.max()
                gapr=max(r0-rr1, rr0-r1, 0); gapc=max(c0-cc1, cc0-c1, 0)
                if gapr<=thr and gapc<=thr:
                    group.append(k); used[k]=True; changed=True
        groups.append(group)
    return groups

def region(r0,r1,c0,c1):
    return ti[r0:r1+1,c0:c1+1].copy()

def solve(grid):
    grid=np.array(grid); bg=int(np.bincount(grid.flatten()).argmax())
    comps=components(grid,bg); groups=group_comps(comps,grid)
    mains=[]; secs=[]
    for g in groups:
        cols=set()
        for j in g: ys,xs=comps[j]; cols|=set(grid[ys,xs].tolist())
        (secs if len(cols)>1 else mains).append(g)
    def minfo(g):
        mask=np.zeros(grid.shape,bool)
        for j in g: ys,xs=comps[j]; mask[ys,xs]=True
        ys,xs=np.where(mask); r0,r1,c0,c1=ys.min(),ys.max(),xs.min(),xs.max()
        region=grid[r0:r1+1,c0:c1+1]
        hole=set((r0+a,c0+b) for a in range(region.shape[0]) for b in range(region.shape[1]) if region[a,b]==bg)
        return dict(mask=mask,hole=hole,col=set(grid[mask].tolist()).pop())
    def sinfo(g):
        mask=np.zeros(grid.shape,bool)
        for j in g: ys,xs=comps[j]; mask[ys,xs]=True
        ys,xs=np.where(mask); r0,r1,c0,c1=ys.min(),ys.max(),xs.min(),xs.max()
        region=grid[r0:r1+1,c0:c1+1].copy()
        region[~mask[r0:r1+1,c0:c1+1]]=0
        return dict(off=(r0,c0),region=region)
    maininfos=[minfo(g) for g in mains]; secinfos=[sinfo(g) for g in secs]
    out=np.full(grid.shape,bg,int)
    for mi in maininfos: out[mi['mask']]=mi['col']
    secused=[False]*len(secinfos)
    for mi in maininfos:
        h=mi['hole']; hs=len(h)
        for si,sd in enumerate(secinfos):
            if secused[si]: continue
            reg=sd['region']; colors=sorted(set(reg.flatten().tolist())-{0})
            key=[c for c in colors if (reg==c).sum()==hs]
            if not key: continue
            key=key[0]; keymask=(reg==key); placed=None
            for k in range(4):
                for fl in [False,True]:
                    M=TF(keymask,k,fl); mh,mw=M.shape
                    mcells=[(a,b) for a in range(mh) for b in range(mw) if M[a,b]]
                    for R in range(-3,grid.shape[0]):
                        for Cc in range(-3,grid.shape[1]):
                            if set((R+a,Cc+b) for a,b in mcells)==h:
                                placed=(k,fl,R,Cc);break
                        if placed:break
                    if placed:break
                if placed:break
            if not placed: continue
            k,fl,R,Cc=placed; full=TF(reg,k,fl).copy()
            if fl and len(colors)==2:
                full2=full.copy()
                full2[full==colors[0]]=colors[1]; full2[full==colors[1]]=colors[0]
                # careful because overlapping assignment
                tmp=full.copy(); full=np.where(tmp==colors[0],colors[1],np.where(tmp==colors[1],colors[0],tmp))
            for a in range(full.shape[0]):
                for b in range(full.shape[1]):
                    if full[a,b]!=0: out[R+a,Cc+b]=full[a,b]
            secused[si]=True; break
    return out

# submitted:
answer=pred
answer=solve(np.array(test_input))
answer
