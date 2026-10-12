# task 89565ca0 pair 0 attempt 1 run tools_persistent_grid_lean2sub_deepseek-flash_low_e009_full_sub96n (median-length of 8 function programs)
def get_wall(a,c):
    ys,xs=np.where(a==c)
    r0,r1,s0,s1=ys.min(),ys.max(),xs.min(),xs.max()
    sub=a[r0:r1+1,s0:s1+1]
    W=(sub==c)
    return W,r0,r1,s0,s1

def fill_gaps2(W, iters=4):
    W=W.copy()
    for _ in range(iters):
        new=W.copy()
        R,C=W.shape
        def wa(i,j):
            if 0<=i<R and 0<=j<C: return W[i,j]
            return False
        def is_h(i,j): return wa(i,j-1) or wa(i,j+1)
        def is_v(i,j): return wa(i-1,j) or wa(i+1,j)
        for i in range(R):
            for j in range(C):
                if W[i,j]: continue
                u,d,l,r=wa(i-1,j),wa(i+1,j),wa(i,j-1),wa(i,j+1)
                fill=False
                if u and d and not (is_h(i-1,j) and is_h(i+1,j)):
                    fill=True
                if l and r and not (is_v(i,j-1) and is_v(i,j+1)):
                    fill=True
                if fill: new[i,j]=True
        if (new==W).all(): break
        W=new
    return W

def count_final2(a,c):
    W,r0,r1,s0,s1=get_wall(a,c)
    W=W.copy()
    W[0,:]=True; W[-1,:]=True; W[:,0]=True; W[:,-1]=True
    Wf=fill_gaps2(W)
    free=~Wf
    lab,n=ndimage.label(free, structure=np.array([[0,1,0],[1,1,1],[0,1,0]]))
    return n

def solve(grid, noise):
    a=np.array(grid)
    cols=sorted(set(a.flatten())-{0,noise})
    res=[(count_final2(a,c),int(c)) for c in cols]
    res.sort()
    W=max(r for r,_ in res)
    out=[[c]*cnt+[noise]*(W-cnt) for cnt,c in res]
    return out

# submitted:
answer = solve(test_input, 9)
answer
