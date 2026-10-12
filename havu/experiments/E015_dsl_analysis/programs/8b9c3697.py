# task 8b9c3697 pair 0 attempt 1 run tools_persistent_grid_lean2sub_deepseek-flash_low_e010_full_s25x80 (median-length of 15 function programs)
def comps(cells):
    cells=set(cells); res=[]
    while cells:
        s=cells.pop(); comp={s}; q=deque([s])
        while q:
            y,x=q.popleft()
            for dy,dx in((1,0),(-1,0),(0,1),(0,-1)):
                n=(y+dy,x+dx)
                if n in cells:
                    cells.discard(n); comp.add(n); q.append(n)
        res.append(comp)
    return res

def solve(gi):
    gi=np.array(gi); H,W=gi.shape
    bg=np.bincount(gi.flatten()).argmax()
    cols=[c for c in set(gi.flatten()) if c!=bg and c!=2]
    shapemask=np.isin(gi,cols)
    shape_cells=[tuple(p) for p in np.argwhere(shapemask)]
    shapecomps=comps(shape_cells)
    cell2comp={}; bboxes=[]
    for idx,sc in enumerate(shapecomps):
        ys=[y for y,x in sc]; xs=[x for y,x in sc]
        bboxes.append((min(ys),max(ys),min(xs),max(xs)))
        for c in sc: cell2comp[c]=idx
    shape=set(shape_cells)
    twos=list(map(tuple,np.argwhere(gi==2)))
    out=gi.copy()
    for comp in comps(twos):
        best=None; bestpath=None
        for d in [(1,0),(-1,0),(0,1),(0,-1)]:
            pos=set(comp); path=[set(comp)]
            while True:
                nxt={(y+d[0],x+d[1]) for y,x in pos}
                if any(y<0 or y>=H or x<0 or x>=W for y,x in nxt): break
                hit=nxt & shape
                if hit:
                    # bbox check
                    idxs={cell2comp[c] for c in hit}
                    b0=min(bboxes[i][0] for i in idxs); b1=max(bboxes[i][1] for i in idxs)
                    b2=min(bboxes[i][2] for i in idxs); b3=max(bboxes[i][3] for i in idxs)
                    inside=all(b0<=y<=b1 and b2<=x<=b3 for y,x in pos)
                    # tip check along perpendicular p
                    p=(d[1],d[0])
                    coords=sorted(y*p[0]+x*p[1] for y,x in hit)
                    # contiguous
                    cont = all(coords[i+1]-coords[i]==1 for i in range(len(coords)-1))
                    tip=False
                    if cont:
                        ends=[]
                        for c0 in (coords[0]-1, coords[-1]+1):
                            # reconstruct cell: hit cells lie on a line; need one representative
                            y0,x0=next(iter(hit))
                            # cell along p at coord c0: since along p only, and d-coord same as hit
                            base=y0*p[0]+x0*p[1]
                            dy=(c0-base)*p[0]; dx=(c0-base)*p[1]
                            cell=(y0+dy,x0+dx)
                            ends.append(cell not in shape)
                        tip=all(ends)
                    if inside and tip:
                        if best is None: best=pos; bestpath=[s for s in path]
                    break
                pos=nxt; path.append(set(nxt))
        for y,x in comp: out[y,x]=bg
        if best is not None:
            for s in bestpath:
                for y,x in s: out[y,x]=0
            for y,x in best: out[y,x]=2
    return out

# submitted:
answer=solve(test_input)
answer
