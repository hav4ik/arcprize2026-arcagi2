# task 4e34c42c pair 1 attempt 0 run tools_persistent_grid_lean2sub_deepseek-flash_low_e009_full_sub96s40 (median-length of 8 function programs)
def add(arr,R,C):
    h,w=arr.shape
    for r in range(h):
        for c in range(w):
            known[(R+r,C+c)]=int(arr[r,c])

def get_pieces(inp):
    vals,counts=np.unique(inp,return_counts=True)
    bg=int(vals[np.argmax(counts)])
    mask=inp!=bg
    lab,n=ndimage.label(mask,structure=np.ones((3,3)))
    pieces=[]
    for k in range(1,n+1):
        ys,xs=np.where(lab==k)
        r0,r1,c0,c1=ys.min(),ys.max(),xs.min(),xs.max()
        pieces.append(inp[r0:r1+1,c0:c1+1].copy())
    return bg,pieces

def solve(inp):
    bg,pieces=get_pieces(inp)
    placed=[]  # (arr,R,C)
    known={}
    def add(arr,R,C):
        h,w=arr.shape
        for r in range(h):
            for c in range(w):
                known[(R+r,C+c)]=int(arr[r,c])
    add(pieces[0],0,0); placed.append((pieces[0],0,0))
    unplaced=list(range(1,len(pieces)))
    while unplaced:
        best=None
        rows=[k[0] for k in known]; cols=[k[1] for k in known]
        minR,maxR=min(rows),max(rows); minC,maxC=min(cols),max(cols)
        for idx in unplaced:
            arr=pieces[idx]; h,w=arr.shape
            for R in range(minR-h+1,maxR+1):
                for C in range(minC-w+1,maxC+1):
                    ov=0; ok=True
                    for r in range(h):
                        for c in range(w):
                            pos=(R+r,C+c)
                            if pos in known:
                                ov+=1
                                if known[pos]!=int(arr[r,c]): ok=False;break
                        if not ok: break
                    if ok and ov>=1:
                        if best is None or ov>best[0]:
                            best=(ov,idx,R,C)
        if best is None:
            # place remaining anywhere far (disconnected)
            idx=unplaced[0]; arr=pieces[idx]
            R=maxR+2; C=minC
            add(arr,R,C); placed.append((arr,R,C)); unplaced.remove(idx); continue
        ov,idx,R,C=best
        add(pieces[idx],R,C); placed.append((pieces[idx],R,C)); unplaced.remove(idx)
    # render
    rows=[k[0] for k in known]; cols=[k[1] for k in known]
    minR,maxR=min(rows),max(rows); minC,maxC=min(cols),max(cols)
    H=maxR-minR+1; W=maxC-minC+1
    out=np.full((H,W),bg,dtype=int)
    for (r,c),v in known.items():
        out[r-minR,c-minC]=v
    return out

# submitted:
answer=solve(np.array(test_input))
answer
