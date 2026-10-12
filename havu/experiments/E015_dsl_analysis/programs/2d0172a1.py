# task 2d0172a1 pair 0 attempt 0 run tools_persistent_grid_lean2sub_deepseek-flash_low_e011_full_sub80s40v (median-length of 6 function programs)
def build_children(comps,parent):
    children={i:[] for i in range(len(comps))}; roots=[]
    for i,p in enumerate(parent):
        if p is None: roots.append(i)
        else: children[p].append(i)
    return children,roots

def analyze(grid):
    a=np.array(grid); vals,counts=np.unique(a,return_counts=True)
    bg=vals[np.argmax(counts)]; obj=[v for v in vals if v!=bg][0]
    m=(a==obj); lab,n=ndimage.label(m)
    comps=[]
    for k in range(1,n+1):
        mask=(lab==k); ys,xs=np.where(mask); filled=ndimage.binary_fill_holes(mask)
        hole=filled&~mask
        comps.append(dict(mask=mask,filled=filled,hole=hole,cy=ys.mean(),cx=xs.mean(),
                          nh=int(ndimage.label(hole)[1]),size=int(mask.sum())))
    parent=[None]*n
    for i,A in enumerate(comps):
        best=None;bestarea=1e18
        for j,B in enumerate(comps):
            if i==j or B['nh']==0: continue
            if np.any(A['mask'] & ~B['filled']): continue
            area=int(B['filled'].sum())
            if area<bestarea: bestarea=area; best=j
        parent[i]=best
    return bg,obj,comps,parent

def render(i,comps,children,obj,bg):
    ch=children[i]
    if comps[i]['nh']==0: return np.array([[obj]])
    base=arrange(ch,comps,children,obj,bg) if ch else np.full((0,0),bg)
    h,w=base.shape
    inner=np.full((h+2,w+2),bg); inner[1:-1,1:-1]=base
    out=np.full((h+4,w+4),obj); out[1:-1,1:-1]=inner
    return out

def arrange(nodes,comps,children,obj,bg):
    if len(nodes)==1: return render(nodes[0],comps,children,obj,bg)
    arrs=[render(i,comps,children,obj,bg) for i in nodes]
    cys=[comps[i]['cy'] for i in nodes]; cxs=[comps[i]['cx'] for i in nodes]
    if (max(cys)-min(cys))<=(max(cxs)-min(cxs)):
        order=sorted(range(len(nodes)),key=lambda t:cxs[t])
        H=max(arrs[t].shape[0] for t in order); W=sum(arrs[t].shape[1] for t in order)+(len(nodes)-1)
        out=np.full((H,W),bg); cur=0
        for t in order:
            ar=arrs[t];h,w=ar.shape;off=(H-h)//2; out[off:off+h,cur:cur+w]=ar; cur+=w+1
        return out
    else:
        order=sorted(range(len(nodes)),key=lambda t:cys[t])
        H=sum(arrs[t].shape[0] for t in order)+(len(nodes)-1); W=max(arrs[t].shape[1] for t in order)
        out=np.full((H,W),bg); cur=0
        for t in order:
            ar=arrs[t];h,w=ar.shape;off=(W-w)//2; out[cur:cur+h,off:off+w]=ar; cur+=h+1
        return out

def solve(grid):
    bg,obj,comps,parent=analyze(grid)
    children,roots=build_children(comps,parent)
    return arrange(roots,comps,children,obj,bg)

# submitted:
answer=solve(test_input)
answer
