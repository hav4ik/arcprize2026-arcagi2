# task 7491f3cf pair 0 attempt 1 run tools_persistent_grid_lean2sub_deepseek-flash_high_e011_full_sub80s40h (median-length of 4 function programs)
def comps8(cells):
    cells=set(cells); out=[]
    while cells:
        s=cells.pop(); comp=[s]; dq=deque([s])
        while dq:
            r,c=dq.popleft()
            for dr in(-1,0,1):
                for dc in(-1,0,1):
                    if dr==0 and dc==0: continue
                    n=(r+dr,c+dc)
                    if n in cells: cells.discard(n); comp.append(n); dq.append(n)
        out.append(comp)
    return out

def build(P0,P2,P3,bg):
    cells0=[(r,c) for r in range(5) for c in range(5) if P0[r][c]!=bg]
    cs=comps8(cells0); cs.sort(key=len)
    dot=set(cs[0]); boundary=set()
    for comp in cs[1:]: boundary|=set(comp)
    start=next(iter(dot)); A=set([start]); dq=deque([start])
    while dq:
        r,c=dq.popleft()
        for dr,dc in((1,0),(-1,0),(0,1),(0,-1)):
            n=(r+dr,c+dc)
            if 0<=n[0]<5 and 0<=n[1]<5 and n not in boundary and n not in A:
                A.add(n); dq.append(n)
    B=set((r,c) for r in range(5) for c in range(5) if (r,c) not in boundary and (r,c) not in A)
    out=[[bg]*5 for _ in range(5)]
    for r in range(5):
        for c in range(5):
            p=(r,c)
            if p in boundary:
                if P2[r][c]!=bg: out[r][c]=P2[r][c]
                elif P3[r][c]!=bg: out[r][c]=P3[r][c]
            elif p in A:
                if P2[r][c]!=bg: out[r][c]=P2[r][c]
            else:
                if P3[r][c]!=bg: out[r][c]=P3[r][c]
    return out

def solve(grid):
    g=np.array(grid)
    P=[g[1:-1,1+6*k:6+6*k] for k in range(4)]
    bg=Counter(P[3].flatten()).most_common(1)[0][0]
    out=build(P[0],P[1],P[2],bg)
    res=g.copy()
    for r in range(5):
        for c in range(5):
            res[r+1][c+19]=out[r][c]
    return res.tolist()

# submitted:
solve(test_input)
