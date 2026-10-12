# task 332f06d7 pair 0 attempt 2 run tools_persistent_grid_lean2sub_deepseek-flash_low_e009_full_sub80s40 (median-length of 10 function programs)
def fits(py,px):
    if py<0 or px<0 or py+s>H or px+s>W: return False
    return shape[py:py+s,px:px+s].all()

def solve_one(a, include2=True):
    a=np.array(a)
    ys,xs=np.where(a==0)
    y0,y1,x0,x1=ys.min(),ys.max(),xs.min(),xs.max()
    s=y1-y0+1
    shape=(a==1)|(a==0)|((a==2) if include2 else False)
    H,W=shape.shape
    def fits(py,px):
        if py<0 or px<0 or py+s>H or px+s>W: return False
        return shape[py:py+s,px:px+s].all()
    start=(y0,x0)
    dist={start:0}; dq=deque([start])
    while dq:
        py,px=dq.popleft()
        for dy,dx in((1,0),(-1,0),(0,1),(0,-1)):
            q=(py+dy,px+dx)
            if q not in dist and fits(*q):
                dist[q]=dist[(py,px)]+1; dq.append(q)
    maxd=max(dist.values())
    cands=[p for p,d in dist.items() if d==maxd]
    return start,s,cands

def solve(a):
    a=np.array(a).copy()
    start,s,cands=solve_one(a)
    assert len(cands)==1, cands
    py,px=cands[0]
    # fill old 0 with 1
    a[a==0]=1
    # set new block to 0
    a[py:py+s,px:px+s]=0
    return a

# submitted:
answer = solve(np.array(test_input))
answer
