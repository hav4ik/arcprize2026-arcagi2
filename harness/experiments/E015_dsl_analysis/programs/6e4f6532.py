# task 6e4f6532 pair 0 attempt 1 run tools_persistent_grid_lean2sub_deepseek-flash_low_e009_full_sub96n (median-length of 25 function programs)
def wall_dir2(c, cen, info, H, W):
    i=info[c]
    hc=i['maxr']-i['minr']+1; wc=i['maxc']-i['minc']+1
    if wc<=2 and hc>=0.7*H:  # vertical
        if i['minc']<=1: return np.array([0.0,-1.0])
        if i['maxc']>=W-2: return np.array([0.0,1.0])
        return np.array([0.0,1.0 if i['cc']>cen[1] else -1.0])
    if hc<=2 and wc>=0.7*W:  # horizontal
        if i['minr']<=1: return np.array([-1.0,0.0])
        if i['maxr']>=H-2: return np.array([1.0,0.0])
        return np.array([1.0 if i['cr']>cen[0] else -1.0, 0.0])
    return np.array([0.0,-1.0])

def choose_transform4(g, cells, target_set, info, H, W):
    key=[p for p in cells if g[p]==9]
    body=[p for p in cells if g[p]!=9]
    bc=np.mean(body,axis=0)
    cc=defaultdict(list)
    for (r,c) in body:
        if g[r,c]!=8: cc[int(g[r,c])].append((r,c))
    tab_dir={c: np.mean(v,axis=0)-bc for c,v in cc.items()}
    sols=[]
    for name,f in OPS8.items():
        fk=[f(r,c) for r,c in key]
        for t in target_set:
            s=(t[0]-fk[0][0], t[1]-fk[0][1])
            if set((fr+s[0],fc+s[1]) for fr,fc in fk)!=target_set: continue
            fb=[(f(r,c)[0]+s[0], f(r,c)[1]+s[1]) for r,c in body]
            cen=np.mean(fb,axis=0)
            ok=True
            for c,dv in tab_dir.items():
                if c not in info: ok=False;break
                d=wall_dir2(c,cen,info,H,W)
                fd=np.array(f(dv[0],dv[1]),dtype=float)
                if abs(fd[0])>=abs(fd[1]):
                    if not (abs(d[0])>=abs(d[1]) and np.sign(fd[0])==np.sign(d[0])): ok=False;break
                else:
                    if not (abs(d[1])>=abs(d[0]) and np.sign(fd[1])==np.sign(d[1])): ok=False;break
            if ok:
                sols.append((name,s,[(int(g[r,c]),(f(r,c)[0]+s[0],f(r,c)[1]+s[1])) for r,c in body]))
    return sols

def get_parts2(g):
    g=np.array(g); H,W=g.shape
    vals,cnts=np.unique(g,return_counts=True)
    bg=int(vals[np.argmax(cnts)])
    wall=np.zeros((H,W),dtype=bool); rowblocked=np.zeros(H,dtype=bool)
    for r in range(H):
        if np.all(g[r]==g[r,0]) and g[r,0]!=bg: wall[r,:]=True; rowblocked[r]=True
    for col in range(W):
        colv=g[~rowblocked,col]
        if len(colv)>0 and np.all(colv==colv[0]) and colv[0]!=bg: wall[~rowblocked,col]=True
    nonbg=(g!=bg); objregion=nonbg & ~wall
    lbl,n=ndimage.label(objregion,structure=np.ones((3,3)))
    objects=[];targets=[];wcc=defaultdict(list)
    for i in range(1,n+1):
        ys,xs=np.where(lbl==i)
        cells=[(int(y),int(x)) for y,x in zip(ys,xs)]
        cols=set(int(g[y,x]) for y,x in cells)
        if 8 in cols: objects.append(cells)
        elif cols=={9}: targets.append(cells)
    for r in range(H):
        for c in range(W):
            if wall[r,c]: wcc[int(g[r,c])].append((r,c))
    return bg,objects,targets,dict(wcc)

def classify_wall_dirs(g, wcc):
    g=np.array(g); H,W=g.shape
    info={}
    for c,v in wcc.items():
        rs=[p[0] for p in v]; cs=[p[1] for p in v]
        fullw=(max(cs)-min(cs)+1)==W; fullh=(max(rs)-min(rs)+1)==H
        info[c]=dict(fullw=fullw,fullh=fullh,minr=min(rs),maxr=max(rs),minc=min(cs),maxc=max(cs),
                     cr=(min(rs)+max(rs))/2, cc=(min(cs)+max(cs))/2)
    return info,H,W

def solve4(g, debug=False):
    g=np.array(g); H,W=g.shape
    bg,objects,targets,wcc=get_parts2(g)
    info,H,W=classify_wall_dirs(g,wcc)
    target_sets=[set(t) for t in targets]
    out=g.copy()
    for cells in objects:
        for (y,x) in cells: out[y,x]=bg
    objcands=[]
    for oi,cells in enumerate(objects):
        pc=[]
        for ti,T in enumerate(target_sets):
            for name,s,placed in choose_transform4(g,cells,T,info,H,W):
                pc.append((ti,name,s,placed))
        rots=[c for c in pc if c[1] in ROT]
        if rots: pc=rots
        objcands.append(pc)
        if debug: print("obj",oi,"cands",[(c[0],c[1]) for c in pc])
    chosen=None
    def search(idx,used,acc):
        nonlocal chosen
        if idx==len(objects):
            if chosen is None: chosen=list(acc)
            return
        for cand in objcands[idx]:
            if cand[0] in used: continue
            search(idx+1,used|{cand[0]},acc+[cand])
            if chosen is not None: return
    search(0,set(),[])
    for cand in (chosen or []):
        for col,(y,x) in cand[3]: out[y,x]=col
    for T in target_sets:
        for (y,x) in T: out[y,x]=9
    return out

# submitted:
answer=solve4(test_input)
answer
