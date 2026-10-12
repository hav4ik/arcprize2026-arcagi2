# task 409aa875 pair 0 attempt 1 run tools_persistent_grid_lean2sub_deepseek-flash_high_e011_full_sub80s40h (median-length of 14 function programs)
def find_shapes(grid, bg=7):
    g=np.array(grid); H,W=g.shape
    seen=np.zeros_like(g,bool); shapes=[]
    for i in range(H):
        for j in range(W):
            if g[i,j]!=bg and not seen[i,j]:
                q=deque([(i,j)]); seen[i,j]=True; cells=[]
                while q:
                    x,y=q.popleft(); cells.append((x,y))
                    for dx in(-1,0,1):
                        for dy in(-1,0,1):
                            if dx==0 and dy==0: continue
                            a,b=x+dx,y+dy
                            if 0<=a<H and 0<=b<W and g[a,b]!=bg and not seen[a,b]:
                                seen[a,b]=True; q.append((a,b))
                shapes.append(cells)
    return g,shapes

def solve(grid, bg=7):
    g,shapes=find_shapes(grid,bg)
    H,W=g.shape
    out=g.copy()
    markers={}   # cell -> count
    shape_of_cell={}
    grey=set()
    for cells in shapes:
        for c in cells: shape_of_cell[c]=cells
        cr=sum(c[0] for c in cells)/len(cells)
        cc=sum(c[1] for c in cells)/len(cells)
        tip=min(cells, key=lambda c:(c[0]-cr)**2+(c[1]-cc)**2)
        vr=tip[0]-cr; vc=tip[1]-cc
        dr=1 if vr>0.05 else (-1 if vr<-0.05 else 0)
        dc=1 if vc>0.05 else (-1 if vc<-0.05 else 0)
        mk=(tip[0]+5*dr, tip[1]+5*dc)
        markers[mk]=markers.get(mk,0)+1
    # marker landing on shape cell -> grey that shape
    for mk in markers:
        if mk in shape_of_cell:
            for c in shape_of_cell[mk]: grey.add(c)
    for c in grey: out[c]=9
    for mk,ct in markers.items():
        if 0<=mk[0]<H and 0<=mk[1]<W:
            out[mk]=1 if ct>1 else 9
    return out

# submitted:
answer=solve(test_input)
answer
