# task 5961cc34 pair 0 attempt 1 run tools_persistent_grid_lean2sub_deepseek-flash_low_e009_full_sub80s40 (median-length of 13 function programs)
def solve(grid):
    g=np.array(grid); H,W=g.shape
    out=np.full((H,W),8,int)
    src=np.argwhere(g==4); s=tuple(src[0])
    d=None
    for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
        nr,nc=s[0]+dr,s[1]+dc
        if 0<=nr<H and 0<=nc<W and g[nr,nc]==2: d=(dr,dc); break
    comp_id=np.full((H,W),-1,int); comps=[]
    for i in range(H):
        for j in range(W):
            if g[i,j] in (1,3) and comp_id[i,j]==-1:
                q=deque([(i,j)]); comp_id[i,j]=len(comps); cells=[]
                while q:
                    r,c=q.popleft(); cells.append((r,c))
                    for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr,nc=r+dr,c+dc
                        if 0<=nr<H and 0<=nc<W and g[nr,nc] in (1,3) and comp_id[nr,nc]==-1:
                            comp_id[nr,nc]=len(comps); q.append((nr,nc))
                comps.append(cells)
    def port(comp):
        cells=set(comp); threes=[(r,c) for (r,c) in comp if g[r,c]==3]; dirs=[]
        for (r,c) in threes:
            for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr,nc=r+dr,c+dc
                if (nr,nc) in cells and g[nr,nc]==1:
                    dirs.append((-dr,-dc))
        od=Counter(dirs).most_common(1)[0][0]
        front=[(r+od[0],c+od[1]) for (r,c) in threes]
        return od,front
    visited=set()
    def process(front,direction):
        front=list(front)
        for (r,c) in front:
            if 0<=r<H and 0<=c<W and out[r,c]==8: out[r,c]=2
        stack=[(tuple(front),direction)]
        while stack:
            fr,di=stack.pop(); state=(tuple(sorted(fr)),di)
            if state in visited: continue
            visited.add(state)
            nxt=[(r+di[0],c+di[1]) for (r,c) in fr]
            hit=None
            for (r,c) in nxt:
                if 0<=r<H and 0<=c<W and g[r,c] in (1,3): hit=(r,c); break
            if hit is not None:
                cid=comp_id[hit[0],hit[1]]
                for (r,c) in comps[cid]: out[r,c]=2
                od,pf=port(comps[cid])
                for (r,c) in pf:
                    if 0<=r<H and 0<=c<W and out[r,c]==8: out[r,c]=2
                stack.append((tuple(pf),od)); continue
            newfront=[(r,c) for (r,c) in nxt if 0<=r<H and 0<=c<W]
            if not newfront: continue
            for (r,c) in newfront:
                if out[r,c]==8: out[r,c]=2
            stack.append((tuple(newfront),di))
    out[s]=2
    process([s],d); process([s],(-d[0],-d[1]))
    return out

# submitted:
answer=solve(test_input)
answer
