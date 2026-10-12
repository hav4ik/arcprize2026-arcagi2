# task 9bbf930d pair 0 attempt 0 run tools_persistent_grid_lean2subv_deepseek-flash_low_e011_full_subv80s40 (median-length of 1 function programs)
def last_corner(path):
    for j in range(len(path)-2,0,-1):
        d1=(path[j][0]-path[j-1][0],path[j][1]-path[j-1][1])
        d2=(path[j+1][0]-path[j][0],path[j+1][1]-path[j][1])
        if d1!=d2: return path[j]
    return path[0]

def blank(a,r,c):
    H,W=a.shape; return 0<=r<H and 0<=c<W and a[r,c]==7

def trace(a,start):
    H,W=a.shape; r,c=start; d=(0,1); path=[(r,c)]
    while True:
        fr,fc=r+d[0],c+d[1]
        if blank(a,fr,fc): r,c=fr,fc; path.append((r,c)); continue
        if not(0<=fr<H and 0<=fc<W): return path,'border'
        perp=[(d[1],d[0]),(-d[1],-d[0])]
        opts=[pd for pd in perp if blank(a,r+pd[0],c+pd[1])]
        if len(opts)==1:
            d=opts[0]; r,c=r+d[0],c+d[1]; path.append((r,c)); continue
        return path,('T' if len(opts)==2 else 'deadend')

def row_color(a,r):
    H,W=a.shape; runs=[]; c=0
    while c<W:
        v=a[r,c]
        if v!=7 and v!=6:
            l=0
            while c<W and a[r,c]==v: c+=1;l+=1
            runs.append((c-l,v,l))
        else: c+=1
    runs=[x for x in runs if x[2]>=3]
    if not runs: return None
    runs.sort(key=lambda x:x[0]); return runs[0][1]

def solve(grid):
    a=np.array(grid).copy(); H,W=a.shape
    col=[row_color(a,r) for r in range(H)]
    lost=[r for r in range(1,H-1) if col[r-1] is not None and col[r-1]==col[r+1]]
    for r in lost:
        path,reason=trace(a,(r,0))
        target=path[-1]
        if reason=='T' and target[1]==1:
            target=last_corner(path)
        a[r,0]=7; a[target[0],target[1]]=6
    return a

# submitted:
solve(test_input)
