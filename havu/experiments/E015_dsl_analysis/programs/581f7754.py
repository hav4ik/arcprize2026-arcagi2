# task 581f7754 pair 1 attempt 1 run tools_persistent_grid_lean2sub_deepseek-flash_low_e010_full_s20x60 (median-length of 17 function programs)
def comps(g, bg):
    H,W=g.shape
    seen=np.zeros_like(g,bool)
    out=[]
    for r in range(H):
        for c in range(W):
            if g[r,c]!=bg and not seen[r,c]:
                q=deque([(r,c)]); seen[r,c]=True; cells=[]
                while q:
                    x,y=q.popleft(); cells.append((x,y))
                    for dx in(-1,0,1):
                        for dy in(-1,0,1):
                            nx,ny=x+dx,y+dy
                            if 0<=nx<H and 0<=ny<W and not seen[nx,ny] and g[nx,ny]!=bg:
                                seen[nx,ny]=True; q.append((nx,ny))
                out.append(cells)
    return out

def solve(g):
    g=np.array(g); H,W=g.shape
    bg=Counter(g.flatten()).most_common(1)[0][0]
    cs=comps(g,bg)
    markers={}; objects=[]
    for c in cs:
        if len(c)==1:
            r,y=c[0]
            if r in(0,H-1) or y in(0,W-1):
                markers[g[r,y]]=(r,y); continue
        objects.append(c)
    out=np.full((H,W),bg)
    for col,(r,y) in markers.items():
        out[r,y]=col
    for c in objects:
        colc=Counter(g[x,y] for x,y in c)
        # key = color in object that is a marker color
        keycols=[col for col in colc if col in markers]
        key=keycols[0]
        kr,kc=[(x,y) for x,y in c if g[x,y]==key][0]
        mr,mc=markers[key]
        if mr in(0,H-1) and mc not in(0,W-1):
            # horizontal shift
            dy=mc-kc; dx=0
        elif mc in(0,W-1) and mr not in(0,H-1):
            dx=mr-kr; dy=0
        else:
            # corner fallback: shift both? choose horizontal
            dy=mc-kc; dx=0
        for x,y in c:
            out[x+dx,y+dy]=g[x,y]
    return out

# submitted:
answer=solve(test_input)
answer
