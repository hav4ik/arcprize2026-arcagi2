# task f560132c pair 0 attempt 0 run tools_persistent_grid_lean2sub_deepseek-flash_low_e011_full_sub80s40v (median-length of 12 function programs)
def get_pieces(g):
    g=np.array(g); mask=g>0
    lab,n=ndimage.label(mask,structure=np.ones((3,3)))
    pieces=[]
    for i in range(1,n+1):
        ys,xs=np.where(lab==i)
        colors=sorted(set(g[ys,xs].tolist()))
        m=(lab==i)[ys.min():ys.max()+1,xs.min():xs.max()+1]
        pieces.append({'colors':colors,'mask':m,'n':int(m.sum())})
    return pieces

def orientations(m):
    outs=[]; seen=set()
    for k in range(4):
        r=np.rot90(m,k)
        for t in (r,np.fliplr(r)):
            key=t.tobytes()+bytes(t.shape)
            if key not in seen:
                seen.add(key); outs.append(t)
    return outs

def try_tile(pieces,keyidx,H,W):
    ops=[]
    for pi,p in enumerate(pieces):
        if pi==keyidx: continue
        for o in orientations(p['mask']):
            h,w=o.shape
            cells=list(zip(*np.where(o)))
            ops.append((pi,o,h,w,cells))
    keym=pieces[keyidx]['mask']; kh,kw=keym.shape
    if kh>H or kw>W: return None
    canvas=np.full((H,W),-1); canvas[0:kh,0:kw][keym]=keyidx+1
    sols=[]
    def solve(canvas,used):
        pos=None
        for r in range(H):
            for c in range(W):
                if canvas[r,c]==-1:
                    pos=(r,c);break
            if pos:break
        if pos is None:
            sols.append(canvas.copy());return True
        r,c=pos
        for pi,o,h,w,cells in ops:
            if pi in used: continue
            for (i,j) in cells:
                r0,c0=r-i,c-j
                if r0<0 or c0<0 or r0+h>H or c0+w>W: continue
                reg=canvas[r0:r0+h,c0:c0+w]
                if np.any(reg[o]!=-1): continue
                nc=canvas.copy(); nc[r0:r0+h,c0:c0+w][o]=pi+1
                if solve(nc,used|{pi}): return True
        return False
    solve(canvas,set())
    return sols[0] if sols else None

def solve(grid):
    ps=get_pieces(grid)
    ki=[i for i,p in enumerate(ps) if len(p['colors'])>1][0]
    key=ps[ki]['colors']
    # frame = majority color
    g=np.array(grid)
    frame=max(key,key=lambda c:int((g==c).sum()))
    emb=[c for c in key if c!=frame]
    # positions of embedded in full grid
    pos={}
    for c in emb:
        ys,xs=np.where(g==c); pos[c]=(ys.mean(),xs.mean())
    rs=sorted(set(round(v[0],3) for v in pos.values()))
    cs=sorted(set(round(v[1],3) for v in pos.values()))
    TL=TR=BL=BR=None
    for c,(r,cc) in pos.items():
        if abs(r-rs[0])<1e-6:
            if abs(cc-cs[0])<1e-6: TL=c
            else: TR=c
        else:
            if abs(cc-cs[0])<1e-6: BL=c
            else: BR=c
    # total area
    area=sum(p['n'] for p in ps)
    # find rectangle dims and tiling with key at 0,0
    sol=None
    for H in range(1,area+1):
        if area%H: continue
        W=area//H
        if H>W: continue
        s=try_tile(ps,ki,H,W)
        if s is not None:
            sol=(s,H,W); break
    s,H,W=sol
    out=np.zeros((H,W),dtype=int)
    for r in range(H):
        for c in range(W):
            v=s[r,c]
            if v==ki+1: out[r,c]=TL
            elif v==0: out[r,c]=0
    # corners
    def corner_piece(r,c):
        return s[r,c]
    trp=corner_piece(0,W-1); blp=corner_piece(H-1,0); brp=corner_piece(H-1,W-1)
    for r in range(H):
        for c in range(W):
            v=s[r,c]
            if v==ki+1: out[r,c]=TL
            elif v==trp: out[r,c]=TR
            elif v==blp: out[r,c]=BL
            elif v==brp: out[r,c]=BR
    return out

# submitted:
answer=solve(test_input)
answer
