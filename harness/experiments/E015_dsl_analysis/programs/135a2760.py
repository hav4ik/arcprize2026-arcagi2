# task 135a2760 pair 0 attempt 0 run tools_persistent_grid_lean2sub_deepseek-flash_low_e009_full_sub96s40 (median-length of 9 function programs)
def get_bands(g):
    g=np.array(g); H,W=g.shape
    bg=Counter(g.flatten()).most_common(1)[0][0]
    urows=[i for i in range(H) if len(set(g[i]))==1]
    ucols=[j for j in range(W) if len(set(g[:,j]))==1]
    def ranges(lines,n):
        rs=[];start=0
        for l in lines:
            if l>start: rs.append((start,l-1))
            start=l+1
        if start<=n-1: rs.append((start,n-1))
        return rs
    return bg, ranges(urows,H), ranges(ucols,W)

def best_tile3(interior, div=4):
    H,W=interior.shape
    best=None
    for pr in range(1,H+1):
        for pc in range(1,W+1):
            if pr*pc > (H*W)//div:
                continue
            groups={}
            for r in range(H):
                for c in range(W):
                    groups.setdefault((r%pr,c%pc),[]).append(interior[r,c])
            err=0; pat=np.zeros((pr,pc),dtype=int)
            for key,vals in groups.items():
                cnt=Counter(vals)
                v=max(cnt,key=lambda k:(cnt[k],k))
                pat[key]=v
                err+=len(vals)-cnt[v]
            meta=(err/(H*W), pr*pc, pr, pc)
            if best is None or meta<best[0]:
                best=(meta,pat)
    return best

def solve3(g,div=4):
    g=np.array(g).copy()
    bg,rrs,ccs=get_bands(g)
    out=g.copy()
    for r0,r1 in rrs:
        for c0,c1 in ccs:
            inter=g[r0+1:r1,c0+1:c1]
            meta,pat=best_tile3(inter,div)
            pr,pc=meta[2],meta[3]
            H,W=inter.shape
            for r in range(H):
                for c in range(W):
                    out[r0+1+r,c0+1+c]=pat[r%pr,c%pc]
    return out

# submitted:
answer = solve3(test_input, 4)
answer
