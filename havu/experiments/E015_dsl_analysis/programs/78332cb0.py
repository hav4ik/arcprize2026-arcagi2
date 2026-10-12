# task 78332cb0 pair 1 attempt 1 run tools_persistent_grid_lean2sub_deepseek-flash_low_e009_full_sub80s40 (median-length of 20 function programs)
def parse(g):
    g=np.array(g); H,W=g.shape
    hsep=[i for i in range(H) if (g[i]==6).all()]
    vsep=[j for j in range(W) if (g[:,j]==6).all()]
    def segs(seps,n):
        b=[];s=0
        for x in seps: b.append((s,x)); s=x+1
        b.append((s,n)); return [t for t in b if t[0]<t[1]]
    rs=segs(hsep,H); cs=segs(vsep,W)
    blocks={}
    for i,(r0,r1) in enumerate(rs):
        for j,(c0,c1) in enumerate(cs):
            blocks[(i,j)]=g[r0:r1,c0:c1]
    return blocks

def edges(b):
    return dict(T=set(np.where(b[0]!=7)[0]), B=set(np.where(b[-1]!=7)[0]),
                L=set(np.where(b[:,0]!=7)[0]), R=set(np.where(b[:,-1]!=7)[0]))

def build_chain(blocks):
    keys=list(blocks)
    E={k:edges(blocks[k]) for k in keys}
    # directed connections out_edge in {R,B}, in_edge in {L,T}
    conns=[]  # (src,dst,kind)
    for a in keys:
        for b in keys:
            if a==b: continue
            if E[a]['R'] & E[b]['L']:
                conns.append((a,b,'H'))
            if E[a]['B'] & E[b]['T']:
                conns.append((a,b,'V'))
    # outgoing/incoming counts
    def outdeg(k): return [c for c in conns if c[0]==k]
    def indeg(k): return [c for c in conns if c[1]==k]
    starts=[k for k in keys if not indeg(k)]
    chain=[]
    cur=starts[0]; kinds=[]
    used=set()
    while cur is not None:
        chain.append(cur); used.add(cur)
        outs=outdeg(cur)
        nxt=None
        for c in outs:
            if c[1] not in used:
                nxt=c[1]; kinds.append(c[2]); break
        cur=nxt
    return chain,kinds,conns,E

def solve(g):
    blocks=parse(g)
    chain,kinds,conns,E=build_chain(blocks)
    h=blocks[chain[0]].shape[0]; w=blocks[chain[0]].shape[1]
    mats=[blocks[c] for c in chain]
    if all(k=='H' for k in kinds):
        sep=np.full((h,1),6)
        row=[mats[0]]
        for m in mats[1:]:
            row.append(sep); row.append(m)
        return np.concatenate(row,axis=1)
    else:
        sep=np.full((1,w),6)
        col=[mats[0]]
        for m in mats[1:]:
            col.append(sep); col.append(m)
        return np.concatenate(col,axis=0)

# submitted:
answer = solve(test_input)
answer
