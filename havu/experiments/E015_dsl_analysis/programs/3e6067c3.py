# task 3e6067c3 pair 1 attempt 0 run tools_persistent_grid_lean2sub_deepseek-flash_high_e011_full_sub80s40h (median-length of 29 function programs)
def solve(grid):
    g = np.array(grid, dtype=int)
    H, W = g.shape
    cnt = Counter(g.flatten().tolist())
    neutrals = set(c for c,_ in cnt.most_common(2))
    colored = np.zeros_like(g, dtype=bool)
    for c in cnt:
        if c not in neutrals:
            colored |= (g==c)
    rowcounts = colored.sum(axis=1)
    legend_row = int(np.argmax(rowcounts))
    legend = [int(g[legend_row,c]) for c in range(W) if colored[legend_row,c]]
    boxmask = colored.copy(); boxmask[legend_row,:]=False
    visited = np.zeros_like(boxmask)
    boxes=[]
    for r in range(H):
        for c in range(W):
            if boxmask[r,c] and not visited[r,c]:
                col=int(g[r,c]); stack=[(r,c)]; visited[r,c]=True; cells=[]
                while stack:
                    x,y=stack.pop(); cells.append((x,y))
                    for dx,dy in ((1,0),(-1,0),(0,1),(0,-1)):
                        nx,ny=x+dx,y+dy
                        if 0<=nx<H and 0<=ny<W and boxmask[nx,ny] and not visited[nx,ny] and g[nx,ny]==col:
                            visited[nx,ny]=True; stack.append((nx,ny))
                rs=[x for x,y in cells]; cs=[y for x,y in cells]
                # box color = neutral adjacent cell
                bc_color = int(g[min(rs)-1, min(cs)])
                # flood fill box color component from that cell
                bvis=np.zeros_like(g,dtype=bool)
                st=[(min(rs)-1,min(cs))]; bvis[min(rs)-1,min(cs)]=True; bcells=[]
                while st:
                    x,y=st.pop(); bcells.append((x,y))
                    for dx,dy in ((1,0),(-1,0),(0,1),(0,-1)):
                        nx,ny=x+dx,y+dy
                        if 0<=nx<H and 0<=ny<W and not bvis[nx,ny] and g[nx,ny]==bc_color:
                            bvis[nx,ny]=True; st.append((nx,ny))
                brs=[x for x,y in bcells]; bcs=[y for x,y in bcells]
                boxes.append({'color':col,'rmin':min(rs),'rmax':max(rs),'cmin':min(cs),'cmax':max(cs),
                              'box_rmin':min(brs),'box_rmax':max(brs),'box_cmin':min(bcs),'box_cmax':max(bcs)})
    rcenters=sorted(set((b['rmin']+b['rmax']) for b in boxes))
    ccenters=sorted(set((b['cmin']+b['cmax']) for b in boxes))
    for b in boxes:
        b['br']=rcenters.index(b['rmin']+b['rmax']); b['bc']=ccenters.index(b['cmin']+b['cmax']); b['used']=False
    def bt(pos, prev, path):
        if pos==len(legend): return list(path)
        for b in boxes:
            if b['used'] or b['color']!=legend[pos]: continue
            if prev is not None:
                if not ((b['br']==prev['br'] and abs(b['bc']-prev['bc'])==1) or
                        (b['bc']==prev['bc'] and abs(b['br']-prev['br'])==1)): continue
            b['used']=True; path.append(b)
            r=bt(pos+1,b,path)
            if r: return r
            path.pop(); b['used']=False
        return None
    path=bt(0,None,[])
    assert path, 'no path'
    out=g.copy()
    for i in range(len(path)-1):
        b1=path[i]; b2=path[i+1]; color=b1['color']
        if b1['br']==b2['br']:
            bL,bR=(b1,b2) if b1['box_cmin']<b2['box_cmin'] else (b2,b1)
            for c in range(bL['box_cmax']+1, bR['box_cmin']):
                for r in range(bL['rmin'], bL['rmax']+1):
                    out[r,c]=color
        else:
            bT,bB=(b1,b2) if b1['box_rmin']<b2['box_rmin'] else (b2,b1)
            for r in range(bT['box_rmax']+1, bB['box_rmin']):
                for c in range(bT['cmin'], bT['cmax']+1):
                    out[r,c]=color
    return out

# submitted:
answer = solve(test_input)
answer
