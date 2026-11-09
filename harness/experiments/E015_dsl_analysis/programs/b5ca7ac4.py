# task b5ca7ac4 pair 0 attempt 0 run tools_persistent_grid_lean2sub_deepseek-flash_low_e009_full_sub80s40 (median-length of 12 function programs)
def parse_objects_arr(g):
    g=np.array(g); H,W=g.shape; objs=[]
    for i in range(H-4):
        for j in range(W-4):
            block=g[i:i+5,j:j+5]
            border=block[0,0]
            if border not in (2,8): continue
            if not (np.all(block[0,:]==border) and np.all(block[4,:]==border) and np.all(block[:,0]==border) and np.all(block[:,4]==border)):
                continue
            inner=block[1:4,1:4]; col=inner[0,0]
            if np.all(inner==col):
                objs.append((i,j,border,col))
    return objs

def slide_group(g, objs, ks, direction):
    H,W=np.array(g).shape; placed=[]; pos={}
    if direction=='left':
        ks=sorted(ks, key=lambda k: objs[k][1])
    else:
        ks=sorted(ks, key=lambda k: objs[k][1], reverse=True)
    for k in ks:
        i,j,b,c=objs[k]; rows=set(range(i,i+5)); nj=j
        while True:
            cand=nj-1 if direction=='left' else nj+1
            if cand<0 or cand+5>W: break
            ok=True
            for (orows,ocols) in placed:
                if rows & orows and not (cand+4<ocols[0] or cand>ocols[1]):
                    ok=False; break
            if not ok: break
            nj=cand
        pos[k]=(i,nj); placed.append((rows,(nj,nj+4)))
    return pos

def solve_grid(g):
    g=np.array(g); objs=parse_objects_arr(g)
    pos={}
    pos.update(slide_group(g,objs,[k for k in range(len(objs)) if objs[k][2]==8],'left'))
    pos.update(slide_group(g,objs,[k for k in range(len(objs)) if objs[k][2]==2],'right'))
    out=np.full(g.shape, np.bincount(g.flatten()).argmax(), dtype=int)
    for k,(i,j,b,c) in enumerate(objs):
        ni,nj=pos[k]
        out[ni:ni+5,nj:nj+5]=np.array([[b,b,b,b,b],[b,c,c,c,b],[b,c,c,c,b],[b,c,c,c,b],[b,b,b,b,b]])
    return out

# submitted:
answer = solve_grid(test_input)
answer
