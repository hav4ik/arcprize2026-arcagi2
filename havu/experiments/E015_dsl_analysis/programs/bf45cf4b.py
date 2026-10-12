# task bf45cf4b pair 0 attempt 0 run tools_persistent_grid_lean2sub_deepseek-flash_low_e010_full_s20x60 (median-length of 13 function programs)
def solve(g):
    g=np.array(g)
    vals,counts=np.unique(g,return_counts=True)
    bg=vals[np.argmax(counts)]
    mask=g!=bg
    lab,n=ndimage.label(mask, structure=np.ones((3,3),int))
    comps=[]
    for k in range(1,n+1):
        ys,xs=np.where(lab==k)
        colors=set(g[ys,xs].tolist())
        comps.append((ys,xs,colors))
    marker=None; tile=None
    for ys,xs,colors in comps:
        if len(colors)==1 and marker is None:
            marker=(ys,xs,list(colors)[0])
        else:
            tile=(ys,xs)
    (my,mx,mc)=marker
    mr0,mr1,Mc0,Mc1=my.min(),my.max(),mx.min(),mx.max()
    (ty,tx)=tile
    tr0,tr1,tc0,tc1=ty.min(),ty.max(),tx.min(),tx.max()
    tiler=g[tr0:tr1+1,tc0:tc1+1]
    th,tw=tiler.shape
    mh,mw=mr1-mr0+1,Mc1-Mc0+1
    out=np.full((mh*th,mw*tw),bg,dtype=int)
    for i in range(mh):
        for j in range(mw):
            if g[mr0+i,Mc0+j]==mc:
                out[i*th:(i+1)*th, j*tw:(j+1)*tw]=tiler
    return out

# submitted:
answer=solve(test_input)
answer
