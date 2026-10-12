# task cbebaa4b pair 0 attempt 0 run tools_persistent_grid_lean2sub_deepseek-flash_low_e010_full_s20x40 (median-length of 16 function programs)
def get_pieces(G):
    mask=(G!=0)&(G!=2)
    lab,n=ndimage.label(mask,structure=np.ones((3,3)))
    ps=[]
    for c in range(1,n+1):
        ys,xs=np.where(lab==c)
        col=int(G[ys[0],xs[0]])
        C=set(zip(ys.tolist(),xs.tolist()))
        M=set()
        for y,x in C:
            for dy in(-1,0,1):
                for dx in(-1,0,1):
                    yy,xx=y+dy,x+dx
                    if (yy,xx) not in C and 0<=yy<G.shape[0] and 0<=xx<G.shape[1] and G[yy,xx]==2:
                        M.add((yy,xx))
        ps.append({'col':col,'C':C,'M':M,'bbox':tuple(int(v) for v in (ys.min(),ys.max(),xs.min(),xs.max()))})
    return ps

def candidates(ps):
    edges=[]
    for i in range(len(ps)):
        for j in range(i+1,len(ps)):
            Mi=ps[i]['M']; Mj=ps[j]['M']
            seen={}
            for a in Mi:
                for b in Mj:
                    t=(a[0]-b[0],a[1]-b[1])
                    seen.setdefault(t,set()).add((a,b))
            for t,pairs in seen.items():
                coinc=set()
                for m in Mj:
                    mm=(m[0]+t[0],m[1]+t[1])
                    if mm in Mi: coinc.add(mm)
                if len(coinc)==2:
                    usedj=frozenset((c[0]-t[0],c[1]-t[1]) for c in coinc)
                    edges.append({'i':i,'j':j,'t':t,'coinci':frozenset(coinc),'coincj':usedj})
    return edges

def solve_assembly(G, verbose=False):
    ps=get_pieces(G)
    edges=candidates(ps)
    n=len(ps)
    allmarks=set()
    for i,p in enumerate(ps):
        for m in p['M']: allmarks.add((i,m))
    edge_list=list(range(len(edges)))
    # backtracking
    best=[]
    def rec(idx, used, chosen):
        if len(used)==len(allmarks):
            # check connectivity
            parent=list(range(n))
            def find(x):
                while parent[x]!=x: parent[x]=parent[parent[x]]; x=parent[x]
                return x
            for e in chosen:
                a=find(edges[e]['i']); b=find(edges[e]['j'])
                if a!=b: parent[a]=b
            roots={find(k) for k in range(n)}
            if len(roots)==1:
                best.append(list(chosen))
            return
        if idx>=len(edges): return
        # prune: choose smallest unused marker, try edges containing it
        # (skip for now, simple)
        # option skip
        rec(idx+1, used, chosen)
        e=edges[idx]
        i=e['i']
        mi=frozenset((i,m) for m in e['coinci'])
        mj=frozenset((e['j'],m) for m in e['coincj'])
        if not (mi & used) and not (mj & used):
            rec(idx+1, used|mi|mj, chosen+[idx])
    rec(0,set(),[])
    if verbose:
        print("solutions",len(best))
        for sol in best:
            for e in sol:
                print("  ",edges[e])
    return ps,edges,best

def assemble_full2(G):
    ps,edges,best=solve_assembly(G)
    sol=[edges[k] for k in best[0]]
    n=len(ps)
    # root = solid piece with largest area
    best_i=None;best_a=-1
    for i,p in enumerate(ps):
        r0,r1,c0,c1=p['bbox']
        if len(p['C'])==(r1-r0+1)*(c1-c0+1):
            a=(r1-r0+1)*(c1-c0+1)
            if a>best_a: best_a=a; best_i=i
    root=best_i
    trans={root:(0,0)}
    for _ in range(n):
        for e in sol:
            i,j,t=e['i'],e['j'],e['t']
            if i in trans and j not in trans: trans[j]=(trans[i][0]+t[0],trans[i][1]+t[1])
            if j in trans and i not in trans: trans[i]=(trans[j][0]-t[0],trans[j][1]-t[1])
    out=np.zeros_like(G)
    for i,p in enumerate(ps):
        tr=trans[i]
        for (y,x) in p['C']: out[y+tr[0],x+tr[1]]=p['col']
    for e in sol:
        tr=trans[e['i']]
        for (y,x) in e['coinci']: out[y+tr[0],x+tr[1]]=2
    return out

# submitted:
answer=assemble_full2(np.array(test_input))
answer
