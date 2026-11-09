# task 62593bfd pair 0 attempt 0 run tools_persistent_grid_lean2sub_deepseek-flash_low_e010_full_s25x80 (median-length of 10 function programs)
def components(g):
    g=np.array(g)
    H,W=g.shape
    bg=Counter(g.flatten()).most_common(1)[0][0]
    seen=np.zeros_like(g,bool)
    comps=[]
    for i in range(H):
        for j in range(W):
            if g[i,j]!=bg and not seen[i,j]:
                col=g[i,j]
                q=deque([(i,j)]); seen[i,j]=True; cells=[]
                while q:
                    x,y=q.popleft(); cells.append((x,y))
                    for dx in(-1,0,1):
                        for dy in(-1,0,1):
                            nx,ny=x+dx,y+dy
                            if 0<=nx<H and 0<=ny<W and not seen[nx,ny] and g[nx,ny]==col:
                                seen[nx,ny]=True; q.append((nx,ny))
                comps.append({'color':col,'cells':cells})
    return bg,comps

def solve(g):
    g=np.array(g); H,W=g.shape
    bg,comps=components(g)
    sizes=Counter(len(c['cells']) for c in comps)
    out=np.full((H,W),bg,dtype=int)
    for c in comps:
        dup = sizes[len(c['cells'])]>1
        rs=[x for x,y in c['cells']]
        top=min(rs)
        if dup:
            shift=-top
        else:
            shift=(H-1)-max(rs)
        for x,y in c['cells']:
            out[x+shift,y]=c['color']
    return out

# submitted:
answer=solve(test_input)
answer
