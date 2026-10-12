# task 7b0280bc pair 0 attempt 1 run tools_persistent_grid_lean2sub_deepseek-flash_low_e011_full_sub80s40v (median-length of 16 function programs)
def comps(a,color):
    mask=(a==color)
    lab,n=ndimage.label(mask, structure=np.array([[0,1,0],[1,1,1],[0,1,0]]))
    out=[]
    for i in range(1,n+1):
        ys,xs=np.where(lab==i)
        out.append((len(ys),ys.min(),ys.max(),xs.min(),xs.max(),list(zip(ys.tolist(),xs.tolist()))))
    return out

def solve(grid):
    a=np.array(grid)
    bgc=int(np.bincount(a.flatten()).argmax())
    nonbg=[int(c) for c in np.unique(a) if c!=bgc]
    struct8=np.ones((3,3))
    def is_solid_square(mask):
        ys,xs=np.where(mask)
        h=ys.max()-ys.min()+1; w=xs.max()-xs.min()+1
        return len(ys)==h*w and h==w and h>=2
    block_colors=[]; wire_colors=[]
    colorcomp={}
    for c in nonbg:
        m=(a==c); lab,n=ndimage.label(m,structure=struct8)
        comps=[]
        allsquare=True
        for k in range(1,n+1):
            mask=(lab==k)
            if not is_solid_square(mask): allsquare=False
            comps.append(mask)
        colorcomp[c]=comps
        if allsquare: block_colors.append(c)
        else: wire_colors.append(c)
    # blocks
    blocks=[]
    for c in block_colors:
        for mask in colorcomp[c]:
            ys,xs=np.where(mask)
            blocks.append({'color':c,'mask':mask,'bbox':(ys.min(),ys.max(),xs.min(),xs.max())})
    # choose B = block color with fewest blocks
    counts=collections.Counter(b['color'] for b in blocks)
    B=min(counts,key=lambda c:counts[c])
    Bidx=[i for i,b in enumerate(blocks) if b['color']==B]
    # wires
    edges=[]
    for c in wire_colors:
        for mask in colorcomp[c]:
            dil=ndimage.binary_dilation(mask,structure=struct8)
            touches=[i for i,b in enumerate(blocks) if (dil&b['mask']).any()]
            edges.append({'mask':mask,'touches':touches})
    # build adjacency
    adj=collections.defaultdict(list)
    for ei,e in enumerate(edges):
        t=e['touches']
        for i,j in itertools.combinations(t,2):
            adj[i].append((j,ei)); adj[j].append((i,ei))
    # BFS shortest path from Bidx[0] to Bidx[1]
    start,goal=Bidx[0],Bidx[1]
    prev={start:None}
    q=collections.deque([start])
    while q:
        u=q.popleft()
        for v,ei in adj[u]:
            if v not in prev:
                prev[v]=(u,ei); q.append(v)
    # reconstruct
    path=[]; node=goal
    while prev[node] is not None:
        u,ei=prev[node]
        path.append((u,node,ei)); node=u
    path=path[::-1]
    out=a.copy()
    used_edges=set(); used_nodes=set()
    for u,v,ei in path:
        used_edges.add(ei); used_nodes.add(u); used_nodes.add(v)
    for ei in used_edges:
        out[edges[ei]['mask']]=5
    for ni in used_nodes:
        b=blocks[ni]
        if b['color']!=B:  # only intermediate A blocks
            out[b['mask']]=3
    return out

# submitted:
solve(test_input)
