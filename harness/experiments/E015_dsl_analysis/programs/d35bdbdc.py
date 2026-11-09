# task d35bdbdc pair 2 attempt 0 run tools_persistent_grid_lean2sub_deepseek-flash_low_e010_full_s20x40 (median-length of 18 function programs)
def find_objects_full(g):
    g=np.array(g); H,W=g.shape
    border_off=[(0,0),(0,1),(0,2),(1,0),(1,2),(2,0),(2,1),(2,2)]
    objs=[]
    for r in range(H):
        for c in range(W):
            if not (r+1<H and c+1<W): continue
            b=g[r,c]
            if b==0: continue
            ok=True
            for dr,dc in border_off:
                rr,cc=r+dr,c+dc
                if 0<=rr<H and 0<=cc<W:
                    if g[rr,cc]!=b: ok=False;break
            if not ok: continue
            ctr=g[r+1,c+1]
            if ctr==b or ctr==0: continue
            objs.append((r,c,int(b),int(ctr)))
    return objs

def solve(g):
    g=np.array(g).copy(); H,W=g.shape
    objs=find_objects_full(g)
    n=len(objs)
    # map border->index
    border_to_idx={o[2]:i for i,o in enumerate(objs)}
    link=[-1]*n
    for i,o in enumerate(objs):
        c=o[3]
        if c in border_to_idx:
            j=border_to_idx[c]
            if j!=i: link[i]=j
    incoming=set(j for j in link if j!=-1)
    starts=[i for i in range(n) if i not in incoming]
    removed=set(range(n)); survivors=[]
    for s in starts:
        chain=[]; cur=s
        while cur!=-1 and cur not in chain:
            chain.append(cur); cur=link[cur]
        for k in range(0,len(chain)-1,2):
            a=chain[k]; b=chain[k+1]
            survivors.append((a,b))
    # removed = all objects except survivors a
    surv_set=set(a for a,b in survivors)
    removed=set(range(n))-surv_set
    for i in removed:
        r,c,_,_=objs[i]
        for dr in range(3):
            for dc in range(3):
                rr,cc=r+dr,c+dc
                if 0<=rr<H and 0<=cc<W: g[rr,cc]=0
    for a,b in survivors:
        _,_,_,ctr=objs[b]
        r,c,_,_=objs[a]
        g[r+1,c+1]=ctr
    return g

# submitted:
answer = solve(test_input)
answer
