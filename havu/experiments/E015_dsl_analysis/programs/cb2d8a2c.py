# task cb2d8a2c pair 0 attempt 2 run tools_persistent_grid_lean2sub_deepseek-flash_low_e009_full_sub80s40 (median-length of 31 function programs)
def find_walls(g):
    H,W=g.shape
    seen=np.zeros_like(g,bool)
    walls=[]
    for r in range(H):
        for c in range(W):
            if g[r,c] in (1,2) and not seen[r,c]:
                # BFS
                comp=[]; dq=deque([(r,c)]); seen[r,c]=True
                while dq:
                    x,y=dq.popleft(); comp.append((x,y))
                    for dx,dy in((1,0),(-1,0),(0,1),(0,-1)):
                        nx,ny=x+dx,y+dy
                        if 0<=nx<H and 0<=ny<W and not seen[nx,ny] and g[nx,ny] in (1,2):
                            seen[nx,ny]=True; dq.append((nx,ny))
                rows=[p[0] for p in comp]; cols=[p[1] for p in comp]
                orient='V' if (max(cols)-min(cols))==0 else ('H' if (max(rows)-min(rows))==0 else '?')
                ones=sum(1 for p in comp if g[p]==1)
                walls.append({'cells':set(comp),'orient':orient,'rmin':min(rows),'rmax':max(rows),'cmin':min(cols),'cmax':max(cols),'g':ones+1})
    return walls

def solve(g):
    g=np.array(g); H,W=g.shape
    walls=find_walls(g)
    wallcell={}
    for w in walls:
        for p in w['cells']: wallcell[p]=w
    out=g.copy(); out[out==1]=2
    # start
    rs,cs=np.where(g==3); r,c=int(rs[0]),int(cs[0])
    if r==0: d=(1,0)
    elif r==H-1: d=(-1,0)
    elif c==0: d=(0,1)
    elif c==W-1: d=(0,-1)
    else: d=(1,0)
    out[r,c]=3
    dirs=[(1,0),(0,1)]  # down,right
    for _ in range(1000):
        dr,dc=d
        k=0; hit=None
        while True:
            k+=1
            nr,nc=r+dr*k,c+dc*k
            if not(0<=nr<H and 0<=nc<W):
                # no wall, go to edge
                for j in range(1,k):
                    out[r+dr*j,c+dc*j]=3
                return out
            if (nr,nc) in wallcell:
                hit=wallcell[(nr,nc)]; break
        w=hit; gg=w['g']
        stopk=max(0,k-gg)
        for j in range(1,stopk+1):
            out[r+dr*j,c+dc*j]=3
        r=r+dr*stopk; c=c+dc*stopk
        # choose perpendicular direction
        if w['orient']=='V':
            # wall vertical, move up/down (rows). p0=r
            p0=r; lo=w['rmin']-gg; hi=w['rmax']+gg
            if (abs(p0-w['rmin'])<=abs(p0-w['rmax'])):
                primary, other = lo, hi
            else:
                primary, other = hi, lo
            # check primary in grid
            target=primary
            if not(0<=target<H): target=other
        else:
            p0=c; lo=w['cmin']-gg; hi=w['cmax']+gg
            if (abs(p0-w['cmin'])<=abs(p0-w['cmax'])):
                primary, other = lo, hi
            else:
                primary, other = hi, lo
            target=primary
            if not(0<=target<W): target=other
        # move along perpendicular axis
        if w['orient']=='V':
            step=1 if target>p0 else -1
            for j in range(1,abs(target-p0)+1):
                out[r+step*j,c]=3
            r=target
        else:
            step=1 if target>p0 else -1
            for j in range(1,abs(target-p0)+1):
                out[r,c+step*j]=3
            c=target
        # d stays same
    return out

# submitted:
answer=solve(test_input)
answer
