# task f931b4a8 pair 0 attempt 1 run tools_persistent_grid_lean2sub_deepseek-flash_low_e009_full_sub80s40 (median-length of 34 function programs)
def solve(a):
    a=np.array(a); n=a.shape[0]; s=n//2
    TL=a[:s,:s]; TR=a[:s,s:]; BL=a[s:,:s]; BR=a[s:,s:]
    H=int(np.count_nonzero(TL)); W=int(np.count_nonzero(TR))
    def has_mix(q): return np.any(q==0) and np.any(q!=0)
    if has_mix(BR):
        stamp, fillq = BR, BL
    else:
        stamp, fillq = BL, BR
    out=np.zeros((H,W),dtype=int)
    for R in range(H):
        for C in range(W):
            v=stamp[R%s, C%s]
            if v==0:
                I=(R//s)%s; J=(C//s)%s
                v=fillq[I,J]
            out[R,C]=v
    return out

# submitted:
answer=solve(test_input)
answer
