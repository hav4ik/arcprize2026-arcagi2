# task 71e489b6 pair 0 attempt 1 run tools_persistent_grid_lean2sub_deepseek-flash_low_e011_full_sub80s40v (median-length of 17 function programs)
def solve(a, S0=4, S1=4):
    a=np.array(a); R,C=a.shape
    b=a.copy()
    # 0-holes
    H=np.zeros_like(a,bool)
    lab,n=ndimage.label(a==0)  # 4-conn default
    sizes=np.bincount(lab.ravel())
    for r in range(R):
        for c in range(C):
            if a[r,c]==0 and 0<lab[r,c]<n+1 and sizes[lab[r,c]]<=S0:
                H[r,c]=True
    # dents 0-cells with exactly 3 orthogonal 1-neighbors
    for r in range(R):
        for c in range(C):
            if a[r,c]==0:
                o=[(r-1,c),(r+1,c),(r,c-1),(r,c+1)]
                o=[(i,j) for i,j in o if 0<=i<R and 0<=j<C]
                if sum(a[i,j] for i,j in o)==3:
                    H[r,c]=True
    ring=np.zeros_like(a,bool)
    for r in range(R):
        for c in range(C):
            if H[r,c]:
                for i in range(r-1,r+2):
                    for j in range(c-1,c+2):
                        if 0<=i<R and 0<=j<C and not H[i,j]:
                            ring[i,j]=True
    b[ring]=7
    # erase 1s
    lab1,n1=ndimage.label(a==1)
    sz1=np.bincount(lab1.ravel())
    for r in range(R):
        for c in range(C):
            if a[r,c]==1:
                if 0<lab1[r,c]<n1+1 and sz1[lab1[r,c]]<=S1:
                    b[r,c]=0
                    continue
                o=[(r-1,c),(r+1,c),(r,c-1),(r,c+1)]
                o=[(i,j) for i,j in o if 0<=i<R and 0<=j<C]
                if sum(a[i,j] for i,j in o)<=1:
                    b[r,c]=0
    return b

# submitted:
solve(test_input)
