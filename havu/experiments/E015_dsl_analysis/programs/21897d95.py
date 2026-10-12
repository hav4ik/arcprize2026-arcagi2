# task 21897d95 pair 0 attempt 0 run tools_persistent_grid_lean2sub_deepseek-flash_low_e010_full_s25x80 (median-length of 1 function programs)
def detect_digits2(g):
    H,W=g.shape
    dig=np.zeros((H,W),bool)
    for y in range(H):
        for x in range(W):
            c=g[y,x]
            if c==1: continue
            nb={}; ones=0
            orth_wild=0
            up = 0<=y-1<H and g[y-1,x]==1
            dn = 0<=y+1<H and g[y+1,x]==1
            lf = 0<=x-1<W and g[y,x-1]==1
            rt = 0<=x+1<W and g[y,x+1]==1
            opp=(up and dn) or (lf and rt)
            same8=False
            for dy in (-1,0,1):
                for dx in (-1,0,1):
                    if dy==0 and dx==0: continue
                    yy,xx=y+dy,x+dx
                    if 0<=yy<H and 0<=xx<W:
                        v=g[yy,xx]
                        if v==1: ones+=1
                        else:
                            nb[v]=nb.get(v,0)+1
                            if v==c: same8=True
            if ones>=2 and nb.get(c,0)==0 and len(nb)==1 and opp and not same8:
                dig[y,x]=True
    return dig

def solve4(g):
    g=np.array(g); H,W=g.shape
    dig=detect_digits2(g)
    wild=(g==1)|dig
    lab=np.zeros((H,W),int); cur=0
    for y in range(H):
        for x in range(W):
            if wild[y,x] or lab[y,x]: continue
            cur+=1; col=g[y,x]; dq=deque([(y,x)]); lab[y,x]=cur
            while dq:
                cy,cx=dq.popleft()
                for dy,dx in ((1,0),(-1,0),(0,1),(0,-1)):
                    yy,xx=cy+dy,cx+dx
                    if 0<=yy<H and 0<=xx<W and not wild[yy,xx] and lab[yy,xx]==0 and g[yy,xx]==col:
                        lab[yy,xx]=cur; dq.append((yy,xx))
    parent=list(range(cur+1))
    def find(a):
        while parent[a]!=a: parent[a]=parent[parent[a]]; a=parent[a]
        return a
    def union(a,b):
        ra,rb=find(a),find(b)
        if ra!=rb: parent[rb]=ra
    for y in range(H):
        for x in range(W):
            if wild[y,x]:
                grp={}
                for dy,dx in ((1,0),(-1,0),(0,1),(0,-1)):
                    yy,xx=y+dy,x+dx
                    if 0<=yy<H and 0<=xx<W and lab[yy,xx]:
                        c=g[yy,xx]; grp.setdefault(c,[]).append(lab[yy,xx])
                for c,ls in grp.items():
                    for l in ls[1:]: union(ls[0],l)
    root_color={}
    for y in range(H):
        for x in range(W):
            if lab[y,x]: root_color[find(lab[y,x])]=g[y,x]
    lab2=lab.copy(); dq=deque()
    for y in range(H):
        for x in range(W):
            if not wild[y,x]: dq.append((y,x))
    while dq:
        cy,cx=dq.popleft()
        for dy,dx in ((1,0),(-1,0),(0,1),(0,-1)):
            yy,xx=cy+dy,cx+dx
            if 0<=yy<H and 0<=xx<W and wild[yy,xx] and lab2[yy,xx]==0:
                lab2[yy,xx]=lab2[cy,cx]; dq.append((yy,xx))
    newcolor=dict(root_color)
    for y in range(H):
        for x in range(W):
            for name,(bar,nub,d,barcenter) in TEMPLATES.items():
                ok=all(0<=y+dy<H and 0<=x+dx<W and wild[y+dy,x+dx] for dy,dx in bar)
                if not ok: continue
                by,bx=y+barcenter[0],x+barcenter[1]
                sy,sx=by-d[0],bx-d[1]
                if not(0<=sy<H and 0<=sx<W) or wild[sy,sx]: continue
                ny,nx=y+nub[0],x+nub[1]
                ty,tx=ny+d[0],nx+d[1]
                if not(0<=ty<H and 0<=tx<W) or wild[ty,tx]: continue
                src=find(lab[sy,sx]); tgt=find(lab[ty,tx])
                carried=g[by,bx] if dig[by,bx] else root_color[src]
                newcolor[tgt]=carried
                # glyph cells -> source region
                for (dy,dx) in bar+[nub]+[barcenter]:
                    lab2[y+dy,x+dx]=src
    out=np.zeros_like(g)
    for y in range(H):
        for x in range(W):
            l=lab2[y,x]; out[y,x]=newcolor[find(l)] if l else 0
    return out

# submitted:
answer = solve4(np.array(test_input))
answer
