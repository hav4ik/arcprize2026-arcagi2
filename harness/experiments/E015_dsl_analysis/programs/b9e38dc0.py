# task b9e38dc0 pair 0 attempt 0 run tools_persistent_grid_lean2sub_deepseek-flash_low_e009_full_sub96n (median-length of 3 function programs)
def extension_cells(inp):
    inp,bg,curvec,curve,h,w=getcurve(inp)
    def cnbrs(r,c):
        return [(r+dr,c+dc) for dr in(-1,0,1) for dc in(-1,0,1) if (dr,dc)!=(0,0) and (r+dr,c+dc) in curve]
    ends=[c for c in curve if len(cnbrs(*c))==1]
    res=[]
    for E in ends:
        N=cnbrs(*E)[0]
        d0=(E[0]-N[0],E[1]-N[1])
        N2s=[x for x in cnbrs(*N) if x!=E]
        if not N2s:
            pat=[d0]
        else:
            N2=N2s[0]
            d1=(N[0]-N2[0],N[1]-N2[1])
            if d1[0]*d0[0]+d1[1]*d0[1]>0:
                pat=[d1,d0]
            else:
                pat=[d0]
        nr,nc=E
        i=0
        while True:
            d=pat[i%len(pat)]; i+=1
            nr,nc=nr+d[0],nc+d[1]
            if not(0<=nr<h and 0<=nc<w): break
            res.append((nr,nc))
    return res,ends,curve

def getcurve(inp):
    inp=np.array(inp); h,w=inp.shape
    bg=int(np.bincount(inp.ravel()).argmax())
    nonbg=[int(v) for v in np.unique(inp) if v!=bg]
    curvec=max(nonbg,key=lambda v:(inp==v).sum())
    curve=set(map(tuple,np.argwhere(inp==curvec)))
    return inp,bg,curvec,curve,h,w

def solve4(inp, fillcolor, flow=None):
    inp,bg,curvec,curve,h,w=getcurve(inp)
    ext,ends,_=extension_cells(inp)
    walls=set(curve)|set(ext)
    if flow is None:
        cr=np.mean([x[0] for x in curve]); cc=np.mean([x[1] for x in curve])
        er=np.mean([x[0] for x in ends]); ec=np.mean([x[1] for x in ends])
        dr=er-cr; dc=ec-cc
        if abs(dr)>=abs(dc)*2: flow=(1 if dr>0 else -1,0)
        elif abs(dc)>=abs(dr)*2: flow=(0,1 if dc>0 else -1)
        else: flow=(1 if dr>0 else -1,1 if dc>0 else -1)
    distractors=set(map(tuple,np.argwhere((inp!=bg)&(inp!=curvec)&(inp!=fillcolor))))
    fr,fcc=flow
    nonbgmask=(inp!=bg)
    for (r,c) in distractors:
        nr,nc=r+fr,c+fcc
        while 0<=nr<h and 0<=nc<w and inp[nr,nc]==bg:
            walls.add((nr,nc)); nr+=fr;nc+=fcc
    # flood
    filled=np.zeros((h,w),bool); dq=deque()
    for r in range(h):
        for c in range(w):
            if inp[r,c]==fillcolor:
                filled[r,c]=True; dq.append((r,c))
    while dq:
        r,c=dq.popleft()
        for dr,dc in((1,0),(-1,0),(0,1),(0,-1)):
            nr,nc=r+dr,c+dc
            if 0<=nr<h and 0<=nc<w and not filled[nr,nc] and (nr,nc) not in walls and inp[nr,nc]==bg:
                filled[nr,nc]=True; dq.append((nr,nc))
    out=inp.copy(); out[filled]=fillcolor
    return out,flow

# submitted:
answer,_=solve4(np.array(test_input),2)
answer
