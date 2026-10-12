# task e87109e9 pair 0 attempt 0 run tools_persistent_grid_lean2sub_deepseek-flash_low_e010_full_s20x40 (median-length of 6 function programs)
def solve2(inp):
    inp=np.array(inp)
    top=inp[:6]; bot=inp[6:].copy()
    row=top[1]; leg={}
    for c,v in enumerate(row):
        if v not in (0,5): leg[v]='L' if c%6==1 else 'R'
    nrows,ncols=bot.shape
    base=Counter(bot.flatten().tolist()).most_common(1)[0][0]
    ys,xs=np.where(bot==8)
    rs0,rs1=ys.min(),ys.max(); cs0,cs1=xs.min(),xs.max()
    grid=bot.copy()
    marked=np.zeros_like(bot,bool)
    marked[ys,xs]=True
    visited=set()
    def passable(r,c):
        if r<0 or r>=nrows or c<0 or c>=ncols: return False
        v=grid[r,c]; return v==base or v==8
    left={'N':'W','W':'S','S':'E','E':'N'}
    right={'N':'E','E':'S','S':'W','W':'N'}
    delta={'N':(-1,0),'S':(1,0),'E':(0,1),'W':(0,-1)}
    def trace(orient,perp,start,direction):
        key=(orient,perp,direction,start)
        if key in visited: return
        visited.add(key)
        dr,dc=delta[direction]; w=len(perp)
        if orient=='v':
            cur=start
            while True:
                for c in perp: marked[cur,c]=True
                nxt=cur+dr
                if nxt<0 or nxt>=nrows: return
                rectcol=None
                for c in perp:
                    if not passable(nxt,c): rectcol=grid[nxt,c]
                if rectcol is None:
                    cur=nxt; continue
                turn=leg.get(rectcol,'L'); nd=left[direction] if turn=='L' else right[direction]
                adjacent=tuple(range(nxt+1,nxt+1+w)) if direction=='N' else tuple(range(nxt-w,nxt))
                newstart=perp[0] if nd=='W' else perp[-1]
                trace('h',adjacent,newstart,nd); return
        else:
            cur=start
            while True:
                for r in perp: marked[r,cur]=True
                nxt=cur+dc
                if nxt<0 or nxt>=ncols: return
                rectcol=None
                for r in perp:
                    if not passable(r,nxt): rectcol=grid[r,nxt]
                if rectcol is None:
                    cur=nxt; continue
                turn=leg.get(rectcol,'L'); nd=left[direction] if turn=='L' else right[direction]
                adjacent=tuple(range(nxt-w,nxt)) if direction=='E' else tuple(range(nxt+1,nxt+1+w))
                newstart=perp[0] if nd=='N' else perp[-1]
                trace('v',adjacent,newstart,nd); return
    R=tuple(range(rs0,rs1+1)); C=tuple(range(cs0,cs1+1))
    trace('v',C,rs0,'N'); trace('v',C,rs1,'S'); trace('h',R,cs0,'W'); trace('h',R,cs1,'E')
    out=bot.copy(); out[marked]=8
    return out

# submitted:
answer=out
answer=solve2(test_input)
answer = solve2(test_input)
answer
