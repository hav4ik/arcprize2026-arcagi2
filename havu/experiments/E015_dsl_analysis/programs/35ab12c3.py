# task 35ab12c3 pair 0 attempt 1 run tools_persistent_grid_lean2sub_deepseek-flash_low_e009_full_sub80s40 (median-length of 7 function programs)
def line_cells(a,b):
    r0,c0=a; r1,c1=b
    dr=r1-r0; dc=c1-c0
    n=max(abs(dr),abs(dc))
    cells=[]
    for k in range(n+1):
        r=r0+ (dr*k//n if n else 0)
        c=c0+ (dc*k//n if n else 0)
        cells.append((r,c))
    return cells

def order_points(pos):
    if len(pos)<=1: return list(pos)
    cr=sum(p[0] for p in pos)/len(pos)
    cc=sum(p[1] for p in pos)/len(pos)
    return sorted(pos, key=lambda p: math.atan2(p[0]-cr, p[1]-cc))

def solve2(inp):
    inp=np.array(inp)
    out=inp.copy()
    colors=[v for v in sorted(set(inp.flatten())) if v!=0]
    multi={}; single={}
    for v in colors:
        pos=[tuple(p) for p in np.argwhere(inp==v)]
        if len(pos)>=2: multi[v]=pos
        else: single[v]=pos
    shapes={}
    for v,pos in multi.items():
        ordp=order_points(pos)
        drawn=set(pos)
        if len(ordp)>=2:
            n=len(ordp)
            rng=range(n) if n>2 else range(1)
            for i in rng:
                a=ordp[i]; b=ordp[(i+1)%n]
                dr=b[0]-a[0]; dc=b[1]-a[1]
                if dr==0 or dc==0 or abs(dr)==abs(dc):
                    for (r,c) in line_cells(a,b): drawn.add((r,c))
        shapes[v]=drawn
        for (r,c) in drawn:
            if 0<=r<out.shape[0] and 0<=c<out.shape[1] and out[r,c]==0: out[r,c]=v
    for s,pos in single.items():
        p=pos[0]; best=None;bestd=1e9
        for v,pts in multi.items():
            for (r,c) in pts:
                d=(r-p[0])**2+(c-p[1])**2
                if d<bestd: bestd=d;best=(r,c);bestv=v
        if best is None: continue
        off=(p[0]-best[0],p[1]-best[1])
        for (r,c) in shapes[bestv]:
            tr,tc=r+off[0],c+off[1]
            if 0<=tr<out.shape[0] and 0<=tc<out.shape[1] and out[tr,tc]==0: out[tr,tc]=s
    return out

# submitted:
answer=solve2(test_input)
answer
