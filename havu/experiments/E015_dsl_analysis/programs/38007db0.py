# task 38007db0 pair 1 attempt 0 run tools_persistent_grid_lean2sub_deepseek-flash_low_e009_full_sub96n (median-length of 33 function programs)
def find_groups(g, line_color):
    # separator rows: full row == line_color
    row_sep=[i for i in range(g.shape[0]) if np.all(g[i]==line_color)]
    col_sep=[i for i in range(g.shape[1]) if np.all(g[:,i]==line_color)]
    def groups(seps, n):
        res=[]; prev=-1
        for s in seps:
            if s-prev-1>0: res.append((prev+1,s-1))
            prev=s
        if n-prev-1>0: res.append((prev+1,n-1))
        return res
    return groups(row_sep,g.shape[0]), groups(col_sep,g.shape[1]), row_sep, col_sep

def solve(inp):
    g=np.array(inp)
    lc=g[0,0]
    rg,cg,rsep,csep=find_groups(g,lc)
    blocks=[[g[r0:r1+1, c0:c1+1] for (c0,c1) in cg] for (r0,r1) in rg]
    unique=[]
    for row in blocks:
        keys=[b.tobytes() for b in row]
        from collections import Counter
        cnt=Counter(keys)
        # pick block whose pattern count==1, else min
        uniq_idx=min(range(len(row)), key=lambda i:cnt[keys[i]])
        unique.append(row[uniq_idx])
    # build output
    bh=unique[0].shape[0]; bw=unique[0].shape[1]
    H=len(rg)*bh + (len(rg)+1)
    W=bw+2
    out=np.full((H,W),lc)
    # columns: left border col 0, block cols1..bw, right border bw+1
    for ri,blk in enumerate(unique):
        r_top=ri*(bh+1)+1
        out[r_top:r_top+bh,1:1+bw]=blk
    return out

# submitted:
solve(test_input)
