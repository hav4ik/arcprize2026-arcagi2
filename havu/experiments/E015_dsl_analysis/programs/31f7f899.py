# task 31f7f899 pair 0 attempt 1 run tools_persistent_grid_lean2sub_deepseek-flash_low_e011_full_sub80s40v (median-length of 17 function programs)
def solve(g):
    g=np.array(g)
    H,W=g.shape
    axis=int(np.argmax((g==6).sum(axis=1)))
    colors=sorted(set(g.flatten().tolist())-{8,6})
    bars=[]
    for c in colors:
        cols=np.where((g==c).any(axis=0))[0]
        groups=[]; cur=[cols[0]]
        for x in cols[1:]:
            if x==cur[-1]+1: cur.append(x)
            else: groups.append(cur); cur=[x]
        groups.append(cur)
        for grp in groups:
            rows=np.where((g==c)[:,grp].any(axis=1))[0]
            bars.append([grp[0],grp[-1],c,rows.max()-rows.min()+1])
    bars.sort(key=lambda b:b[0])
    heights=sorted(b[3] for b in bars)
    out=np.full((H,W),8,dtype=int)
    out[axis]=g[axis]
    for b,h in zip(bars,heights):
        c0,c1,c,_=b
        up=h//2; down=h-1-up
        out[axis-up:axis+down+1, c0:c1+1]=c
    return out

# submitted:
answer=solve(test_input)
answer
