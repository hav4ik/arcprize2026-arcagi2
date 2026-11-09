# task 7ed72f31 pair 0 attempt 2 run tools_persistent_grid_lean2sub_deepseek-flash_low_e009_full_sub80s40 (median-length of 29 function programs)
def components(mask):
    lab,n=ndimage.label(mask, structure=np.array([[0,1,0],[1,1,1],[0,1,0]]))
    return [(lab==k) for k in range(1,n+1)]

def solve(grid):
    g=np.array(grid)
    bg=np.bincount(g.flatten()).argmax()
    out=g.copy()
    # 2 markers
    m2=(g==2)
    comps=components(m2)
    markers=[]
    for comp in comps:
        ys,xs=np.where(comp)
        markers.append(list(zip(ys.tolist(),xs.tolist())))
    for c in np.unique(g):
        if c==bg or c==2: continue
        ys,xs=np.where(g==c)
        cells=list(zip(ys.tolist(),xs.tolist()))
        # nearest marker
        best=None;bestd=1e9
        for mi,m in enumerate(markers):
            for (i,j) in cells:
                for (a,b) in m:
                    d=abs(i-a)+abs(j-b)
                    if d<bestd:
                        bestd=d;best=mi
        m=markers[best]
        rows=[a for a,b in m]; cols=[b for a,b in m]
        if len(m)==1:
            r0,c0=m[0]
            for (i,j) in cells:
                out[2*r0-i,2*c0-j]=c
        elif len(set(cols))==1:
            c0=cols[0]
            for (i,j) in cells:
                out[i,2*c0-j]=c
        elif len(set(rows))==1:
            r0=rows[0]
            for (i,j) in cells:
                out[2*r0-i,j]=c
        else:
            print("diagonal marker?")
    return out

# submitted:
answer=solve(test_input)
answer
