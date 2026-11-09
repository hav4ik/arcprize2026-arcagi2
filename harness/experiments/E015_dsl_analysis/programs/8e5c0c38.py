# task 8e5c0c38 pair 1 attempt 1 run tools_persistent_grid_lean2sub_deepseek-flash_low_e011_full_sub80s40v (median-length of 30 function programs)
def solve_grid(gi):
    gi=gi.copy()
    bg=np.bincount(gi.flatten()).argmax() if False else np.bincount(gi.flatten()).argmax()
    out=gi.copy()
    H,W=gi.shape
    for col in sorted(set(gi.flatten().tolist())):
        if col==bg: continue
        cells=np.argwhere(gi==col)
        r0,r1=cells[:,0].min(),cells[:,0].max(); c0,c1=cells[:,1].min(),cells[:,1].max()
        m=(gi[r0:r1+1,c0:c1+1]==col).astype(int)
        h,w=m.shape
        best=None
        for K in range(-w,2*w+2):
            cnt=0; V=np.zeros_like(m)
            for r in range(h):
                for c in range(w):
                    if m[r,c]:
                        cc=K-c
                        if not(0<=cc<w and m[r,cc]):
                            V[r,c]=1; cnt+=1
            if best is None or cnt<best[0]: best=(cnt,V)
        V=best[1]
        for r in range(h):
            for c in range(w):
                if V[r,c]: out[r0+r,c0+c]=bg
    return out

# submitted:
answer=solve_grid(np.array(test_input))
answer
