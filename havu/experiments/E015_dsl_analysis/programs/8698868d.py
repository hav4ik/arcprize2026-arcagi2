# task 8698868d pair 0 attempt 1 run tools_persistent_grid_lean2subv_deepseek-flash_low_e011_full_subv80s40 (median-length of 3 function programs)
def comps_count(mask):
    lab,n=ndimage.label(mask)  # 4-connectivity
    return n

def solve(grid):
    gi=np.array(grid)
    vals,counts=np.unique(gi,return_counts=True)
    marker=vals[np.argmax(counts)]
    tiles=[]
    for col in np.unique(gi):
        if col==marker: continue
        m=gi==col
        lab,n=ndimage.label(m)
        for k in range(1,n+1):
            ys,xs=np.where(lab==k)
            if len(ys)<8: continue
            r0,r1,c0,c1=ys.min(),ys.max(),xs.min(),xs.max()
            hh=r1-r0+1; ww=c1-c0+1
            if hh!=ww: continue
            # allow only marker inside bbox
            blk=gi[r0:r1+1,c0:c1+1]
            if not np.all((blk==col)|(blk==marker)): continue
            tiles.append((col,int(r0),int(c0),int(hh)))
    sizes=sorted(set(t[3] for t in tiles))
    small=sizes[0]; large=sizes[-1]
    bg_tiles=[t for t in tiles if t[3]==large]
    pat_tiles=[t for t in tiles if t[3]==small]
    def hcomps(col,r,c,s):
        blk=gi[r:r+s,c:c+s]; return comps_count(blk==marker)
    pat_info={}
    for (col,r,c,s) in pat_tiles:
        pat_info.setdefault(hcomps(col,r,c,s),[]).append((col,r,c,s))
    rmin=min(t[1] for t in bg_tiles); cmin=min(t[2] for t in bg_tiles)
    rmax=max(t[1]+t[3] for t in bg_tiles); cmax=max(t[2]+t[3] for t in bg_tiles)
    H=rmax-rmin; W=cmax-cmin
    out=np.full((H,W),marker,dtype=int)
    off=(large-small)//2
    for (col,r,c,s) in bg_tiles:
        cc=hcomps(col,r,c,s)
        pc,pr,pcc,ps=pat_info[cc][0]
        pblk=gi[pr:pr+ps,pcc:pcc+ps]
        sub=out[r-rmin:r-rmin+s, c-cmin:c-cmin+s]
        sub[:]=col
        for i in range(ps):
            for j in range(ps):
                if pblk[i,j]!=marker:
                    sub[off+i,off+j]=pblk[i,j]
    return out.tolist()

# submitted:
solve(test_input)
