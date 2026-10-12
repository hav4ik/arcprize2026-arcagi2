# task 7666fa5d pair 0 attempt 1 run tools_persistent_grid_lean2sub_deepseek-flash_low_e009_full_sub80s40 (median-length of 11 function programs)
def solve(I):
    I=np.array(I); H,W=I.shape
    from collections import Counter
    bg=Counter(I.flatten().tolist()).most_common(1)[0][0]
    runs=[]
    for u in range(0,H+W-1):
        vs=sorted(r-c for r in range(H) for c in range(W) if r+c==u and I[r,c]!=bg)
        if not vs: continue
        start=vs[0]; prev=vs[0]
        for v in vs[1:]:
            if v-prev!=2:
                runs.append((u,start,prev)); start=v
            prev=v
        runs.append((u,start,prev))
    def ov(a,b):
        lo=max(a[1],b[1]); hi=min(a[2],b[2])
        return hi-lo+1 if hi>=lo else 0
    res=I.copy()
    for idx,r in enumerate(runs):
        best=None;bo=0
        for j,s in enumerate(runs):
            if j==idx or s[0]==r[0]: continue
            o=ov(r,s)
            if o>bo or (o==bo and o>0 and best is not None and abs(s[0]-r[0])<abs(best[0]-r[0])):
                bo=o;best=s
        if best is None or bo<=0: continue
        u1,u2=sorted([r[0],best[0]])
        lo=max(r[1],best[1]); hi=min(r[2],best[2])
        for u in range(u1+1,u2):
            for v in range(lo,hi+1):
                if (u+v)%2: continue
                i=(u+v)//2; c=(u-v)//2
                if 0<=i<H and 0<=c<W and res[i,c]==bg:
                    res[i,c]=2
    return res

# submitted:
solve(test_input)
