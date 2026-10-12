# task 7b80bb43 pair 0 attempt 0 run tools_persistent_grid_lean2sub_deepseek-flash_low_e009_full_sub96n (median-length of 1 function programs)
def dnb(p,Set):
    r,c=p
    return [(r+dr,c+dc) for dr in(-1,1) for dc in(-1,1) if (r+dr,c+dc) in Set]

def runs(g,bg):
    H,W=g.shape
    hruns=[];vruns=[]
    for r in range(H):
        c=0
        while c<W:
            if g[r,c]!=bg:
                s=c
                while c<W and g[r,c]!=bg: c+=1
                hruns.append((r,s,c-1,c-s))
            else: c+=1
    for c in range(W):
        r=0
        while r<H:
            if g[r,c]!=bg:
                s=r
                while r<H and g[r,c]!=bg: r+=1
                vruns.append((c,s,r-1,r-s))
            else: r+=1
    return hruns,vruns

def onlong(m):
    on=set()
    for r in range(H):
        c=0
        while c<W:
            if m[r,c]:
                s=c
                while c<W and m[r,c]: c+=1
                if c-s>=3:
                    for cc in range(s,c): on.add((int(r),int(cc)))
            else: c+=1
    for c in range(W):
        r=0
        while r<H:
            if m[r,c]:
                s=r
                while r<H and m[r,c]: r+=1
                if r-s>=3:
                    for rr in range(s,r): on.add((int(rr),int(c)))
            else: r+=1
    return on

def solve(grid):
    g=np.array(grid); H,W=g.shape
    vals,counts=np.unique(g,return_counts=True); bg=vals[np.argmax(counts)]
    fcol=[v for v in vals if v!=bg][0]
    fg=(g!=bg)
    def onlong(m):
        on=set()
        for r in range(H):
            c=0
            while c<W:
                if m[r,c]:
                    s=c
                    while c<W and m[r,c]: c+=1
                    if c-s>=3:
                        for cc in range(s,c): on.add((int(r),int(cc)))
                else: c+=1
        for c in range(W):
            r=0
            while r<H:
                if m[r,c]:
                    s=r
                    while r<H and m[r,c]: r+=1
                    if r-s>=3:
                        for rr in range(s,r): on.add((int(rr),int(c)))
                else: r+=1
        return on
    S=set((int(rr),int(cc)) for rr,cc in zip(*np.where(fg))); on0=onlong(fg)
    def dnb(p,Set):
        a,b=p
        return [(a+dr,b+dc) for dr in(-1,1) for dc in(-1,1) if (a+dr,b+dc) in Set]
    def orth(p,Set):
        a,b=p
        return [(a+dr,b+dc) for dr,dc in((-1,0),(1,0),(0,-1),(0,1)) if (a+dr,b+dc) in Set]
    stub={p for p in S if p not in on0 and not any(q in on0 for q in orth(p,S))}
    runs=[]; runid={}
    for p in stub:
        if p in runid: continue
        stack=[p]; comp={p}
        while stack:
            u=stack.pop()
            for q in dnb(u,stub):
                if q not in comp:
                    comp.add(q); stack.append(q)
        idx=len(runs); runs.append(comp)
        for q in comp: runid[q]=idx
    runlen=[len(cc) for cc in runs]
    fills=set()
    for rr in range(H):
        cols=[c for c in range(W) if fg[rr,c]]
        for a,b in zip(cols,cols[1:]):
            size=b-a-1
            if size<1: continue
            for cell,need in (((rr,a+1),+1),((rr,b-1),-1)):
                for p in ((cell[0]-1,cell[1]),(cell[0]+1,cell[1])):
                    if p in runid and runlen[runid[p]]==size-1:
                        if any((p[0]+dr,p[1]+dc) in runid and runid[(p[0]+dr,p[1]+dc)]==runid[p] and dc==need for dr in(-1,1) for dc in(-1,1)):
                            for cc in range(a+1,b): fills.add((rr,cc))
    for cc in range(W):
        rows=[rr for rr in range(H) if fg[rr,cc]]
        for a,b in zip(rows,rows[1:]):
            size=b-a-1
            if size<1: continue
            for cell,need in (((a+1,cc),+1),((b-1,cc),-1)):
                for p in ((cell[0],cell[1]-1),(cell[0],cell[1]+1)):
                    if p in runid and runlen[runid[p]]==size-1:
                        if any((p[0]+dr,p[1]+dc) in runid and runid[(p[0]+dr,p[1]+dc)]==runid[p] and dr==need for dr in(-1,1) for dc in(-1,1)):
                            for rr in range(a+1,b): fills.add((rr,cc))
    out=g.copy()
    for (rr,cc) in fills: out[rr,cc]=fcol
    fg2=out!=bg
    on2=onlong(fg2); S2=set((int(rr),int(cc)) for rr,cc in zip(*np.where(fg2)))
    for p in S2:
        if p in on2: continue
        if any(q in S2 and q not in on2 for q in dnb(p,S2)):
            out[p[0],p[1]]=bg
    return out

# submitted:
answer=solve(test_input)
answer
