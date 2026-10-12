# task 13e47133 pair 1 attempt 2 run tools_persistent_grid_lean2sub_deepseek-flash_low_e009_full_sub80s40 (median-length of 24 function programs)
def solve_grid2(a,div):
    H,W=a.shape
    vals,cnts=np.unique(a,return_counts=True)
    d=dict(zip(vals.tolist(),cnts.tolist())); d.pop(div,None)
    bg=max(d,key=d.get)
    passable=(a!=div)
    comp=np.full((H,W),-1,dtype=int); cid=0
    for i in range(H):
        for j in range(W):
            if passable[i,j] and comp[i,j]<0:
                q=deque([(i,j)]); comp[i,j]=cid
                while q:
                    r,c=q.popleft()
                    for dr,dc in((1,0),(-1,0),(0,1),(0,-1)):
                        nr,nc=r+dr,c+dc
                        if 0<=nr<H and 0<=nc<W and passable[nr,nc] and comp[nr,nc]<0:
                            comp[nr,nc]=cid;q.append((nr,nc))
                cid+=1
    out=a.copy()
    for k in range(cid):
        cells=[(i,j) for i in range(H) for j in range(W) if comp[i,j]==k]
        cellset=set(cells)
        dist={}; q=deque()
        for (i,j) in cells:
            b=False
            for dr in(-1,0,1):
                for dc in(-1,0,1):
                    if dr==0 and dc==0: continue
                    if (i+dr,j+dc) not in cellset: b=True
            if b: dist[(i,j)]=0;q.append((i,j))
        while q:
            r,c=q.popleft()
            for dr in(-1,0,1):
                for dc in(-1,0,1):
                    if dr==0 and dc==0: continue
                    nr,nc=r+dr,c+dc
                    if (nr,nc) in cellset and (nr,nc) not in dist:
                        dist[(nr,nc)]=dist[(r,c)]+1;q.append((nr,nc))
        seedmap={}
        for (i,j) in cells:
            if a[i,j]!=bg: seedmap[dist[(i,j)]]=a[i,j]
        if not seedmap:
            continue
        mx=max(seedmap)
        seq=[]
        if 0 not in seedmap: seq.append(bg)
        for dd in range(mx+1):
            if dd in seedmap: seq.append(seedmap[dd])
        for (i,j) in cells:
            out[i,j]=seq[dist[(i,j)]%len(seq)]
    return out

# submitted:
answer=solve_grid2(np.array(test_input),1)
answer
