# task 142ca369 pair 0 attempt 0 run tools_persistent_grid_lean2sub_deepseek-flash_low_e010_full_s20x60 (median-length of 16 function programs)
def solve(a):
    a=np.array(a)
    H,W=a.shape
    # per-color 4-connected components
    seen=np.zeros_like(a,bool)
    objs=[]
    for r in range(H):
        for c in range(W):
            if a[r,c] and not seen[r,c]:
                col=a[r,c]; q=deque([(r,c)]); seen[r,c]=True; cells=[]
                while q:
                    x,y=q.popleft(); cells.append((x,y))
                    for dx,dy in((1,0),(-1,0),(0,1),(0,-1)):
                        nx,ny=x+dx,y+dy
                        if 0<=nx<H and 0<=ny<W and a[nx,ny]==col and not seen[nx,ny]:
                            seen[nx,ny]=True; q.append((nx,ny))
                objs.append((col,set(cells)))
    out=a.copy()
    rays=[]
    for col,cells in objs:
        if len(cells)!=3: continue
        rs=[x for x,y in cells]; cs=[y for x,y in cells]
        r0,r1,c0,c1=min(rs),max(rs),min(cs),max(cs)
        if r1-r0==1 and c1-c0==1:
            full={(r0,c0),(r0,c1),(r1,c0),(r1,c1)}
            missing=list(full-cells)[0]
            corner=list(cells & { (r1-missing[0]+r0, c1-missing[1]+c0) })[0] if False else None
            # opposite corner:
            opp=(r0+r1-missing[0], c0+c1-missing[1])
            dr=opp[0]-missing[0]; dc=opp[1]-missing[1]
            rays.append([col,opp,(dr,dc),[]])
    occ=set()
    for r in range(H):
        for c in range(W):
            if a[r,c]: occ.add((r,c))
    # synchronous growth
    changed=True
    while changed:
        changed=False
        want=[]
        for ray in rays:
            col,start,d,tail=ray
            if tail:
                nxt=(tail[-1][0]+d[0], tail[-1][1]+d[1])
            else:
                nxt=(start[0]+d[0], start[1]+d[1])
            if 0<=nxt[0]<H and 0<=nxt[1]<W and nxt not in occ:
                want.append((ray,nxt))
        # resolve: only apply those not conflicting with each other
        chosen=[]
        for ray,nxt in want:
            if all(nxt!=o[1] for o in chosen):
                chosen.append((ray,nxt))
        if chosen:
            changed=True
            for ray,nxt in chosen:
                occ.add(nxt); ray[3].append(nxt); out[nxt]=ray[0]
    return out

# submitted:
answer = solve(test_input)
answer
