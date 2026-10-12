# task db695cfb pair 0 attempt 1 run tools_persistent_grid_lean2subv_deepseek-flash_low_e011_full_subv80s40 (median-length of 17 function programs)
def solve(grid):
    g=np.array(grid)
    H,W=g.shape
    bg=np.bincount(g.flatten()).argmax()
    ones=[tuple(p) for p in zip(*np.where(g==1))]
    sixs=[tuple(p) for p in zip(*np.where(g==6))]
    out=np.full((H,W),bg,dtype=int)
    for (r,c) in ones:
        out[r,c]=1
    segments=[]
    n=len(ones)
    for i in range(n):
        for j in range(i+1,n):
            r1,c1=ones[i]; r2,c2=ones[j]
            dr=r2-r1; dc=c2-c1
            if dr!=0 and abs(dr)==abs(dc):
                sr=1 if dr>0 else -1; sc=1 if dc>0 else -1
                cells=set()
                for k in range(abs(dr)+1):
                    cells.add((r1+sr*k,c1+sc*k))
                segments.append((cells,(sr,sc)))
    for cells,_ in segments:
        for (r,c) in cells:
            out[r,c]=1
    for cells,(sr,sc) in segments:
        pr,pc=sc,-sr
        for (r,c) in sixs:
            if (r,c) in cells:
                for k in range(-max(H,W),max(H,W)+1):
                    rr,cc=r+pr*k,c+pc*k
                    if 0<=rr<H and 0<=cc<W:
                        out[rr,cc]=6
    # sixes not on any segment stay
    onseg=set()
    for cells,_ in segments: onseg|=cells
    for (r,c) in sixs:
        if (r,c) not in onseg:
            out[r,c]=6
    return out

# submitted:
solve(test_input)
