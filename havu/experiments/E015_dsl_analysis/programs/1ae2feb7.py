# task 1ae2feb7 pair 1 attempt 0 run tools_persistent_grid_lean2sub_deepseek-flash_low_e009_full_sub80s40 (median-length of 54 function programs)
def find_barrier(a):
    H,W=a.shape
    for c in range(W):
        col=a[:,c]
        if col[0]!=0 and np.all(col==col[0]):
            return c,col[0]
    return None,None

def solve(grid):
    a=np.array(grid).copy()
    H,W=a.shape
    bc,bcol=find_barrier(a)
    left=a[:,:bc]; right_cols=list(range(bc+1,W))
    out=a.copy()
    for r in range(H):
        row=left[r]
        nz=[(c,int(row[c])) for c in range(bc) if row[c]!=0]
        if not nz: continue
        # counts per color
        from collections import defaultdict
        pos=defaultdict(list)
        for c,v in nz: pos[v].append(c)
        counts={v:len(ps) for v,ps in pos.items()}
        rightmost={v:max(ps) for v,ps in pos.items()}
        for idx,cc in enumerate(right_cols):
            p=idx  # 0-based offset
            cand=[v for v in counts if p%counts[v]==0]
            if not cand:
                out[r,cc]=0
            else:
                v=max(cand,key=lambda v:rightmost[v])
                out[r,cc]=v
    return out

# submitted:
answer=solve(test_input)
answer
