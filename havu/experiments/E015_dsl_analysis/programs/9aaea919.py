# task 9aaea919 pair 0 attempt 0 run tools_persistent_grid_lean2sub_deepseek-flash_low_e011_full_sub80s40v (median-length of 17 function programs)
def draw(slots,bg,H=30,W=30):
    g=np.full((H,W),bg)
    for i in range(7):
        y=6+4*i
        if y>=H: continue
        for j in range(4):
            c=slots[i,j]
            if c<0: continue
            x=5+6*j
            g[y,x-2:x+3]=c
            g[y-1,x-1:x+2]=c
            g[y+1,x-1:x+2]=c
    return g

def read_slots(g):
    bg=g[0,0]; H,W=g.shape
    M=np.full((7,4),-1)
    for i in range(7):
        y=6+4*i
        yy=y if y<H else y-1  # clipped -> use row29
        for j in range(4):
            x=5+6*j
            v=g[yy,x]
            M[i,j]=v if v!=bg else -1
    return M,bg

def solve2(g):
    g=np.array(g); M,bg=read_slots(g)
    bars={}  # value -> list of cols
    for j in range(4):
        if M[6,j]>=0: bars.setdefault(int(M[6,j]),[]).append(j)
    # target = largest bar value; sources = all others
    tval=max(bars); tcol=bars[tval]
    scol=[]
    for v,cols in bars.items():
        if v!=tval: scol+=cols
    def stack(j): return [(i,j) for i in range(6) if M[i,j]>=0]
    ext=sum(len(stack(j)) for j in scol)
    out=M.copy()
    recol=int(sum(bars.keys()))  # sum of distinct bar labels; 2+3=5
    for j in scol:
        for (i,jj) in stack(j): out[i,jj]=recol
    col=int(M[stack(tcol[0])[0][0],tcol[0]]); top=min(i for i,j in stack(tcol[0]))
    for k in range(1,ext+1):
        ni=top-k
        if ni>=0: out[ni,tcol[0]]=col
    out[6,:]=-1
    return draw(out,bg)

# submitted:
solve2(test_input)
