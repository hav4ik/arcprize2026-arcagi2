# task eee78d87 pair 0 attempt 0 run tools_persistent_grid_lean2sub_deepseek-flash_high_e011_full_sub80s40h (median-length of 16 function programs)
def solve(grid):
    I=np.array(grid)
    ys,xs=np.where(I!=7)
    r0,r1,c0,c1=ys.min(),ys.max(),xs.min(),xs.max()
    P=(I!=7).astype(int)[r0:r1+1,c0:c1+1]
    B=set()
    for i in range(3):
        for j in range(3):
            if P[i][j]==0:
                B.add(map_pos[(i,j)])
    O=np.zeros((16,16),int)
    for r in range(16):
        for c in range(16):
            if 5<=r<=10 and 5<=c<=10:
                O[r][c]=9 if P[(r-5)%3][(c-5)%3]==1 else 7
            else:
                O[r][c]=7 if (r%3,c%3) in B else 0
    return O

# submitted:
solve(test_input)
