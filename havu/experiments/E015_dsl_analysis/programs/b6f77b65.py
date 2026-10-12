# task b6f77b65 pair 0 attempt 0 run tools_persistent_grid_lean2sub_deepseek-flash_low_e009_full_sub64a (median-length of 1 function programs)
def mono_pieces(a,S=S4):
    P=[]
    for col in np.unique(a):
        if col==0: continue
        lab,n=ndimage.label(a==col, structure=S)
        for k in range(1,n+1):
            m=lab==k; ys,xs=np.where(m)
            P.append({'col':int(col),'cells':[(int(r),int(c)) for r,c in zip(ys,xs)]})
    return P

def gravity_sim(a,maxit=5000):
    R,C=a.shape
    P=mono_pieces(a)
    occ=np.zeros((R,C),dtype=int)
    for idx,p in enumerate(P):
        for r,c in p['cells']: occ[r,c]=idx+1
    for it in range(maxit):
        moved=False
        for idx,p in enumerate(P):
            cells=p['cells']; can=True
            for (r,c) in cells:
                nr=r+1
                if nr>=R: can=False;break
                if occ[nr,c]!=0 and occ[nr,c]!=idx+1: can=False;break
            if can:
                for (r,c) in cells: occ[r,c]=0
                p['cells']=[(r+1,c) for r,c in cells]
                for (r,c) in p['cells']: occ[r,c]=idx+1
                moved=True
        if not moved: break
    out=np.zeros((R,C),dtype=int)
    for p in P:
        for r,c in p['cells']: out[r,c]=p['col']
    return out

def solve(g):
    a=np.array(g).copy()
    m=int(a[0,0])
    if (a==m).sum()<=1:
        return a
    a[a==m]=0; a[0,0]=0
    out=gravity_sim(a); out[0,0]=m
    return out

# submitted:
answer=solve(test_input)
answer
