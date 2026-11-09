# task d8e07eb2 pair 0 attempt 0 run tools_persistent_grid_lean2sub_deepseek-flash_low_e010_full_s20x40 (median-length of 34 function programs)
def solve(grid):
    a=np.array(grid)
    out=a.copy()
    trows=[8,13,18,23]; tcols=[2,7,12,17]
    # top tiles
    tops=[]
    for c in tcols:
        sub=a[1:4,c:c+3]
        s=set(sub.flatten())-{8}
        if not s:
            tops.append((None,None)); continue
        col=list(s)[0]
        shap=tuple((sub==col).astype(int).flatten())
        tops.append((col,shap))
    # body tiles
    matched=[]
    for ri,r in enumerate(trows):
        for ci,c in enumerate(tcols):
            sub=a[r:r+3,c:c+3]
            s=set(sub.flatten())-{8}
            col=list(s)[0] if s else None
            shap=tuple((sub==col).astype(int).flatten()) if s else None
            for (tc,ts) in tops:
                if tc is None: continue
                if tc==col and ts==shap:
                    matched.append((ri,ci,r,c))
    # recolor frames
    for (ri,ci,r,c) in matched:
        reg=out[r-1:r+4,c-1:c+4]
        reg[reg==8]=3
    # line check
    rows=[m[0] for m in matched]; cols=[m[1] for m in matched]
    is_line = (len(matched)==4) and (len(set(rows))==1 or len(set(cols))==1)
    if is_line:
        top=out[0:5,:]; top[top==8]=3
        out[28:30,:]=3
    else:
        out[28:30,:]=2
    return out

# submitted:
answer = solve(test_input)
answer
