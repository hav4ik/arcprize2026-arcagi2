# task 981571dc pair 0 attempt 1 run tools_persistent_grid_lean2subv_deepseek-flash_low_e011_full_subv80s40 (median-length of 12 function programs)
def solve(grid):
    g=np.array(grid); n=g.shape[0]; out=g.copy()
    zeros=np.argwhere(g==0)
    # iterate to propagate (some images may themselves be zeros resolved later); do multiple passes
    changed=True
    while changed:
        changed=False
        for r,c in zeros:
            vals=set()
            for M,v in grp:
                p=M@np.array([r,c])+v
                rr,cc=p
                if 0<=rr<n and 0<=cc<n:
                    val=out[rr,cc]
                    if val!=0: vals.add(val)
            if len(vals)==1:
                nv=vals.pop()
                if out[r,c]!=nv:
                    out[r,c]=nv; changed=True
            elif len(vals)==0:
                pass
            else:
                pass # conflict
    return out

# submitted:
solve(test_input)
