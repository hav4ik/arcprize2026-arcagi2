# task 1818057f pair 0 attempt 0 run tools_persistent_grid_lean2sub_deepseek-flash_low_e010_full_s20x40 (median-length of 18 function programs)
def solve(a):
    a=np.array(a)
    R,C=a.shape
    mask=np.zeros_like(a,dtype=bool)
    for r in range(1,R-1):
        for c in range(1,C-1):
            if a[r,c]==4 and a[r-1,c]==4 and a[r+1,c]==4 and a[r,c-1]==4 and a[r,c+1]==4:
                for dr,dc in [(0,0),(-1,0),(1,0),(0,-1),(0,1)]:
                    mask[r+dr,c+dc]=True
    return np.where(mask,8,a)

# submitted:
answer=solve(test_input)
answer
