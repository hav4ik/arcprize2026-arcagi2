# task a6f40cea pair 0 attempt 1 run tools_persistent_grid_lean2sub_deepseek-flash_low_e009_full_sub80s40 (median-length of 5 function programs)
def find_container(a):
    H,W=a.shape
    best=None
    for r in range(H):
        for c in range(W):
            C=a[r,c]
            w=0
            while c+w<W and a[r,c+w]==C: w+=1
            h=0
            while r+h<H and a[r+h,c]==C: h+=1
            if h>=3 and w>=3:
                ok=True
                for x in range(c,c+w):
                    if a[r,x]!=C or a[r+h-1,x]!=C: ok=False;break
                if ok:
                    for y in range(r,r+h):
                        if a[y,c]!=C or a[y,c+w-1]!=C: ok=False;break
                if ok:
                    inner=a[r+1:r+h-1,c+1:c+w-1]
                    vals=np.unique(inner)
                    if len(vals)==1 and vals[0]!=C:
                        if best is None or h*w>best[0]:
                            best=(h*w,r,c,h,w,C,int(vals[0]))
    return best

def solve(grid):
    a=np.array(grid)
    _,r0,c0,h,w,C,F=find_container(a)
    out=np.full((h-2,w-2),F,dtype=int)
    itop,ileft=r0+1,c0+1
    colors=[int(v) for v in np.unique(a) if v!=C and v!=F]
    groups=defaultdict(list)
    for v in colors:
        ys,xs=np.where(a==v)
        groups[(ys.min(),ys.max(),xs.min(),xs.max())].append(v)
    for bb,vs in groups.items():
        rmin,rmax,cmin,cmax=bb
        cells=set()
        for x in range(cmin,cmax+1):
            cells.add((rmin,x)); cells.add((rmax,x))
        for y in range(rmin,rmax+1):
            cells.add((y,cmin)); cells.add((y,cmax))
        inside=[(y,x) for (y,x) in cells if r0+1<=y<=r0+h-2 and c0+1<=x<=c0+w-2]
        if not inside: continue
        if len(vs)==1:
            for (y,x) in inside:
                out[y-itop,x-ileft]=vs[0]
        else:
            s=sorted(vs); mn,mx=s[0],s[-1]
            def ins(p): return r0+1<=p[0]<=r0+h-2 and c0+1<=p[1]<=c0+w-2
            def dist(p):
                yy=min(max(p[0],r0+1),r0+h-2); xx=min(max(p[1],c0+1),c0+w-2)
                return abs(p[0]-yy)+abs(p[1]-xx)
            corners=[(rmin,cmin),(rmin,cmax),(rmax,cmin),(rmax,cmax)]
            corners.sort(key=lambda p:(0 if ins(p) else 1, dist(p)))
            near=corners[0]; npar=(near[0]+near[1])%2
            for (y,x) in inside:
                out[y-itop,x-ileft]= mn if (y+x)%2==npar else mx
    return out.tolist()

# submitted:
answer=solve(test_input)
answer
