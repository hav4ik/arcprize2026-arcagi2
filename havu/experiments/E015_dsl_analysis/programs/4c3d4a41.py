# task 4c3d4a41 pair 1 attempt 1 run tools_persistent_grid_lean2sub_deepseek-flash_low_e009_full_sub96n (median-length of 27 function programs)
def solve(inp):
    g=np.array(inp); H,W=g.shape
    L=g[:, :9]
    out=g.copy()
    out[:, :9]=0
    out[1:6, 11:18]=0
    for r in range(H):
        for c in range(9):
            if L[r,c]==5:
                out[r,c+10]=5
    for vc in [2,4,6,8]:
        gc=9+vc
        col=g[:,gc]
        colors=col[(col!=0)&(col!=5)]
        lc=vc-1
        rows5=[r for r in range(H) if L[r,lc]==5]
        top=min(rows5)
        # fill whole value column rows1..5 with 5
        for r in range(1,6): out[r,gc]=5
        if len(colors)==0: continue
        color=colors[0]; k=len(colors)
        avail=top-1
        cnt=min(k,avail)
        for i in range(cnt):
            out[top-1-i,gc]=color
    return out.tolist()

# submitted:
answer = solve(test_input)
