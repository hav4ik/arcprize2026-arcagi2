# task a47bf94d pair 0 attempt 1 run tools_persistent_grid_lean2subv_deepseek-flash_low_e011_full_subv80s40 (median-length of 6 function programs)
def nbrs(r,c):
    res=[]
    for dr,dc in((1,0),(-1,0),(0,1),(0,-1)):
        a,b=r+dr,c+dc
        if 0<=a<H and 0<=b<W and mask[a,b]: res.append((a,b))
    return res

def trace(start):
    nb=nbrs(*start); d=(nb[0][0]-start[0],nb[0][1]-start[1])
    cur=nb[0]; prev=start
    while True:
        if len(nbrs(*cur))==1: return cur
        d1=(cur[0]+d[0],cur[1]+d[1])
        if 0<=d1[0]<H and 0<=d1[1]<W and mask[d1]:
            prev,cur=cur,d1; continue
        cand=[n for n in nbrs(*cur) if n!=prev]
        if len(cand)==1:
            d=(cand[0][0]-cur[0],cand[0][1]-cur[1]); prev,cur=cur,cand[0]; continue
        return cur

def solve(g):
    I=np.array(g).copy(); H,W=I.shape
    mask=(I==8)|(I==5)|(I==9)
    def nbrs(r,c,m=mask):
        res=[]
        for dr,dc in((1,0),(-1,0),(0,1),(0,-1)):
            a,b=r+dr,c+dc
            if 0<=a<H and 0<=b<W and m[a,b]: res.append((a,b))
        return res
    eps=[(r,c) for r in range(H) for c in range(W) if mask[r,c] and len(nbrs(r,c))==1]
    def trace(start):
        nb=nbrs(*start); d=(nb[0][0]-start[0],nb[0][1]-start[1])
        cur=nb[0]; prev=start
        while True:
            if len(nbrs(*cur))==1: return cur
            d1=(cur[0]+d[0],cur[1]+d[1])
            if 0<=d1[0]<H and 0<=d1[1]<W and mask[d1]:
                prev,cur=cur,d1; continue
            cand=[n for n in nbrs(*cur) if n!=prev]
            if len(cand)==1:
                d=(cand[0][0]-cur[0],cand[0][1]-cur[1]); prev,cur=cur,cand[0]; continue
            return cur
    seen=set(); lanes=[]
    for e in eps:
        if e in seen: continue
        t=trace(e); seen.add(e); seen.add(t); lanes.append((e,t))
    # pair stuck lanes
    good=[]; stuck=[]
    for e,t in lanes:
        if len(nbrs(*t))==1:
            good.append((e,t))
        else:
            stuck.append((e,t))
    if len(stuck)==2:
        good.append((stuck[0][0], stuck[1][0]))
    elif len(stuck)>0:
        # fallback nearest pairing
        good+=stuck
    def info(e):
        nb=nbrs(*e); d=(nb[0][0]-e[0],nb[0][1]-e[1])
        return (e[0]-2*d[0], e[1]-2*d[1]), (-d[0],-d[1])
    def slot_color(center):
        cr,cc=center; cols=set()
        for r in range(cr-1,cr+2):
            for c in range(cc-1,cc+2):
                if 0<=r<H and 0<=c<W:
                    v=I[r,c]
                    if v not in (0,5,8,9): cols.add(int(v))
        return cols
    marker_colors={1,2,3,4,6,7}
    out=I.copy()
    allslots=[info(e)[0] for pair in good for e in pair]
    for (cr,cc) in allslots:
        for r in range(cr-1,cr+2):
            for c in range(cc-1,cc+2):
                if 0<=r<H and 0<=c<W and out[r,c] in marker_colors:
                    out[r,c]=0
    for e1,e2 in good:
        s1,o1=info(e1); s2,o2=info(e2)
        cols=slot_color(s1)|slot_color(s2)
        if not cols: continue
        col=sorted(cols)[0]
        def isplus(o): return (o[0]==-1) or (o[1]==-1)
        if isplus(o1)!=isplus(o2):
            plus,xside=(s1,s2) if isplus(o1) else (s2,s1)
        else:
            plus,xside=(s1,s2) if (s1[0],s1[1])<(s2[0],s2[1]) else (s2,s1)
        cr,cc=plus
        out[cr-1,cc]=col; out[cr,cc-1]=col; out[cr,cc+1]=col; out[cr+1,cc]=col
        cr,cc=xside
        out[cr-1,cc-1]=col; out[cr-1,cc+1]=col; out[cr,cc]=col; out[cr+1,cc-1]=col; out[cr+1,cc+1]=col
    return out

# submitted:
solve(test_input)
