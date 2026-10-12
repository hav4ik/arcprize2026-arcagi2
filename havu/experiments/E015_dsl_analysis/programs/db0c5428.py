# task db0c5428 pair 0 attempt 1 run tools_persistent_grid_lean2sub_deepseek-flash_low_e009_full_sub80s40 (median-length of 11 function programs)
def solve(gin):
    gi=np.array(gin)
    mask=gi!=8
    rs=np.where(mask.any(1))[0]; cs=np.where(mask.any(0))[0]
    r0,c0=rs.min(),cs.min()
    O=gi[r0:r0+9,c0:c0+9].copy()
    H,W=gi.shape
    # majority color
    cnt=Counter(O[O!=8].tolist())
    maj=cnt.most_common(1)[0][0]
    # hole fill 3x3
    hf=np.zeros((3,3),int)
    for i in range(3):
        for j in range(3):
            if i==1 and j==1:
                hf[i,j]=maj
            else:
                hf[i,j]=O[2+2*i,2+2*j]
    out=np.full((H,W),8,int)
    def fill_val(rr,cc):
        # rr,cc relative to r0,c0
        return O[rr%9,cc%9]
    for r in range(H):
        for c in range(W):
            rr=r-r0; cc=c-c0
            ci=1+ (rr//3); cj=1+(cc//3)
            if 0<=ci<5 and 0<=cj<5:
                if ci==2 or cj==2 or ci==cj or ci+cj==4:
                    sr=rr%9; sc=cc%9
                    if 3<=sr<=5 and 3<=sc<=5:
                        out[r,c]=hf[sr-3,sc-3]
                    else:
                        out[r,c]=O[sr,sc]
    return out

# submitted:
answer = solve(test_input)
answer
