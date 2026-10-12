# task 45a5af55 pair 0 attempt 1 run tools_persistent_grid_lean2sub_deepseek-flash_high_e011_full_sub80s40h (median-length of 18 function programs)
def bands(grid):
    grid=np.array(grid)
    res=[]
    i=0
    while i<len(grid):
        c=grid[i,0]
        j=i
        while j<len(grid) and grid[j,0]==c:
            j+=1
        res.append((c,j-i))
        i=j
    return res

def solve(grid):
    b=bands(grid)
    t=[x[1] for x in b]
    S=2*sum(t)-t[-1]
    A=np.zeros((S,S),dtype=int)
    o=0
    for c,th in b:
        A[o:S-o,o:S-o]=c
        o+=th
    return A.tolist()

# submitted:
ans=solve(test_input)
ans
