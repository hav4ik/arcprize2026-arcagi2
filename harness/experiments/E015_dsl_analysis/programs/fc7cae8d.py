# task fc7cae8d pair 0 attempt 0 run tools_persistent_grid_lean2sub_deepseek-flash_low_e009_full_sub96n (median-length of 8 function programs)
def transforms(a):
    res={}
    for k in range(4):
        res['rot%d'%k]=np.rot90(a,k)
        res['rot%d_flip'%k]=np.fliplr(np.rot90(a,k))
    return res

def solve(inp):
    inp=np.array(inp); H,W=inp.shape
    edges={'N':inp[0,1:-1],'S':inp[-1,1:-1],'W':inp[1:-1,0],'E':inp[1:-1,-1]}
    solids={}
    for s,arr in edges.items():
        vals,cnts=np.unique(arr,return_counts=True)
        d={int(v):int(c) for v,c in zip(vals,cnts) if v!=0}
        if d:
            dom=max(d,key=d.get)
            if d[dom]/len(arr)>=0.9 and dom!=0:
                solids[s]=dom
    framecolors=set(solids.values())
    best_h=None; best_v=None
    for r in range(1,H-1):
        c=1
        while c<W-1:
            col=inp[r,c]
            if col in framecolors:
                c2=c
                while c2<W-1 and inp[r,c2]==col: c2+=1
                if c2-c>=3 and (best_h is None or (c2-c)>(best_h[2]-best_h[1])):
                    best_h=(r,c,c2,int(col))
                c=c2
            else: c+=1
    for c in range(1,W-1):
        r=1
        while r<H-1:
            col=inp[r,c]
            if col in framecolors:
                r2=r
                while r2<H-1 and inp[r2,c]==col: r2+=1
                if r2-r>=3 and (best_v is None or (r2-r)>(best_v[2]-best_v[1])):
                    best_v=(c,r,r2,int(col))
                r=r2
            else: r+=1
    hr,hc0,hc1,hcol=best_h
    vcol,vr0,vr1,vcolor=best_v
    panel=inp[vr0:vr1,hc0:hc1]
    sideV=2 if vcol<hc0 else 3
    sideH=0 if hr<vr0 else 1
    Ex=[s for s,cc in solids.items() if cc==vcolor][0]
    Ey=[s for s,cc in solids.items() if cc==hcol][0]
    Eside={'N':0,'S':1,'W':2,'E':3}
    tx,ty=Eside[Ex],Eside[Ey]
    for name,sm in SM.items():
        if sm[sideV]==tx and sm[sideH]==ty:
            return transforms(panel)[name]
    return None

# submitted:
answer = solve(test_input)
answer
