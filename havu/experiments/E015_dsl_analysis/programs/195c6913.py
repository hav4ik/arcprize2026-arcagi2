# task 195c6913 pair 1 attempt 0 run tools_persistent_grid_lean2sub_deepseek-flash_low_e010_full_s20x60 (median-length of 13 function programs)
def markers(inp):
    inp=np.array(inp)
    bg=np.bincount(inp.flatten()).argmax()
    return bg

def find_blocks(inp, colors_present):
    blocks=[]
    for col in colors_present:
        pos=np.argwhere(inp==col); s=set(map(tuple,pos))
        for (r,c) in pos:
            if (r+1,c) in s and (r,c+1) in s and (r+1,c+1) in s:
                blocks.append((int(col),int(r),int(c)))
    return blocks

def solve(inp):
    inp=np.array(inp); grid=inp.copy(); H,W=inp.shape
    vals,cnts=np.unique(inp,return_counts=True); order=np.argsort(-cnts)
    big=list(vals[order[:2]]); special=[v for v in vals if v not in big]
    blocks=find_blocks(inp,special)
    legend=[b for b in blocks if b[1]<=2]
    legend_cols=set(b[0] for b in legend)
    hit_blocks=[b for b in blocks if b[0] not in legend_cols]
    hitcolor=hit_blocks[0][0]
    covered=set()
    for (col,r,c) in blocks:
        for dr in (0,1):
            for dc in (0,1): covered.add((r+dr,c+dc))
    markers=[]
    for col in special:
        for (r,c) in np.argwhere(inp==col):
            if (int(r),int(c)) not in covered: markers.append((int(r),int(c),int(col)))
    seq=[b[0] for b in sorted(legend,key=lambda b:b[2])]
    mr,mc,mcol=markers[0]; floor=None
    for dd in [(0,1),(0,-1),(1,0),(-1,0)]:
        rr,cc=mr+dd[0],mc+dd[1]
        if 0<=rr<H and 0<=cc<W and inp[rr,cc] in big: floor=inp[rr,cc]; break
    wall=[b for b in big if b!=floor][0]
    for (col,r,c) in blocks:
        for dr in (0,1):
            for dc in (0,1): grid[r+dr,c+dc]=wall
    L=len(seq)
    for (mr,mc,mcol) in markers:
        pos=(mr,mc); d='R'; idx=1
        while True:
            dr,dc=DIRS[d]; nr,nc=pos[0]+dr,pos[1]+dc
            if not(0<=nr<H and 0<=nc<W): break
            if grid[nr,nc]==floor:
                grid[nr,nc]=seq[idx%L]; idx+=1; pos=(nr,nc)
            else:
                grid[nr,nc]=hitcolor
                d='U' if d=='R' else 'R'
                dr,dc=DIRS[d]; nr2,nc2=pos[0]+dr,pos[1]+dc
                if not(0<=nr2<H and 0<=nc2<W) or grid[nr2,nc2]!=floor: break
    return grid

# submitted:
ans=solve(test_input)
ans
