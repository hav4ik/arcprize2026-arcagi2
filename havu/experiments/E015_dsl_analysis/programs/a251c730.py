# task a251c730 pair 0 attempt 2 run tools_persistent_grid_lean2sub_deepseek-flash_low_e009_full_sub80s40 (median-length of 13 function programs)
def expand_box(inp,box):
    r0,r1,c0,c1=box
    bc=inp[r0,c0]
    changed=True
    while changed:
        changed=False
        # right
        if c1+1<inp.shape[1] and np.all(inp[r0:r1+1,c1+1]==bc):
            c1+=1; changed=True
        if c0-1>=0 and np.all(inp[r0:r1+1,c0-1]==bc):
            c0-=1; changed=True
        if r0-1>=0 and np.all(inp[r0-1,c0:c1+1]==bc):
            r0-=1; changed=True
        if r1+1<inp.shape[0] and np.all(inp[r1+1,c0:c1+1]==bc):
            r1+=1; changed=True
    return (r0,r1,c0,c1)

def components_of(fg):
    lab,n=ndimage.label(fg,structure=np.ones((3,3)))
    comps=[]
    for k in range(1,n+1):
        ys,xs=np.where(lab==k)
        comps.append((ys.min(),ys.max(),xs.min(),xs.max(),len(ys)))
    return comps

def get_boxes(inp):
    H,W=inp.shape
    cands=[np.array([inp[0,c] for c in range(W)])[None,:].repeat(H,0),
           np.array([inp[r,0] for r in range(H)])[:,None].repeat(W,1)]
    best=None
    for bg in cands:
        fg=inp!=bg
        comps=components_of(fg)
        score=(fg.sum(),len(comps))
        if best is None or score<best[0]:
            best=(score,comps)
    return best[1]

def analyze(inp):
    inp=np.array(inp)
    comps=get_boxes(inp)
    infos=[]
    for (r0,r1,c0,c1,sz) in comps:
        r0,r1,c0,c1=expand_box(inp,(r0,r1,c0,c1))
        interior=inp[r0+1:r1,c0+1:c1]
        vals,cnts=np.unique(interior,return_counts=True)
        fill=vals[np.argmax(cnts)]
        nonfill=set(vals.tolist())-{fill}
        infos.append(dict(box=(r0,r1,c0,c1),fill=fill,nonfill=nonfill,interior=interior))
    # find ref and tgt
    ref=tgt=None
    for i in infos:
        for j in infos:
            if i is not j and j['nonfill']< i['nonfill']:
                ref,tgt=i,j
    return infos,ref,tgt

def solve(inp):
    inp=np.array(inp)
    infos,ref,tgt=analyze(inp)
    markers=set(tgt['nonfill'])
    shapes=ref['nonfill']-markers
    interior=ref['interior']; h,w=interior.shape
    markpos=[(y,x,interior[y,x]) for y in range(h) for x in range(w) if interior[y,x] in markers]
    shapepos=[(y,x,interior[y,x]) for y in range(h) for x in range(w) if interior[y,x] in shapes]
    clusters=defaultdict(list)  # marker color -> list of (dy,dx,col)
    for (sy,sx,sc) in shapepos:
        dm=min(markpos,key=lambda m:(m[0]-sy)**2+(m[1]-sx)**2)
        clusters[dm[2]].append((sy-dm[0],sx-dm[1],sc))
    template={mc:list(set(v)) for mc,v in clusters.items()}
    tr0,tr1,tc0,tc1=tgt['box']
    out=inp[tr0:tr1+1,tc0:tc1+1].copy()
    tint=tgt['interior']; th,tw=tint.shape
    for (iy,ix) in [(y,x) for y in range(th) for x in range(tw)]:
        mc=tint[iy,ix]
        if mc in markers and mc in template:
            for (dy,dx,col) in template[mc]:
                oy,ox=iy+1+dy,ix+1+dx
                if 0<=oy<out.shape[0] and 0<=ox<out.shape[1]:
                    out[oy,ox]=col
    return out

# submitted:
answer = solve(test_input)
answer
