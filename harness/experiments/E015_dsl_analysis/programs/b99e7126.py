# task b99e7126 pair 0 attempt 0 run tools_persistent_grid_lean2sub_deepseek-flash_low_e009_full_sub80s40 (median-length of 15 function programs)
def blocks(a):
    B=np.zeros((7,7,3,3),int)
    for bi in range(7):
        for bj in range(7):
            B[bi,bj]=a[1+4*bi:4+4*bi,1+4*bj:4+4*bj]
    return B

def solve(grid):
    a=np.array(grid)
    sep=a[0,0]
    def blocks(a):
        B=np.zeros((7,7,3,3),int)
        for bi in range(7):
            for bj in range(7):
                B[bi,bj]=a[1+4*bi:4+4*bi,1+4*bj:4+4*bj]
        return B
    Ba=blocks(a)
    c=Counter(Ba[bi,bj].tobytes() for bi in range(7) for bj in range(7))
    base=c.most_common(1)[0][0]
    baseT=np.frombuffer(base,int).reshape(3,3)
    special=[(bi,bj) for bi in range(7) for bj in range(7) if Ba[bi,bj].tobytes()!=base]
    st=Ba[special[0][0],special[0][1]]
    newcolor=[x for x in set(st.flatten().tolist()) if x not in set(baseT.flatten().tolist())][0]
    M=set((r,cc) for r in range(3) for cc in range(3) if st[r,cc]==newcolor)
    S=set(special)
    sols=[]
    for aa in range(-2,7):
        for bb in range(-2,7):
            region=set((aa+r,bb+cc) for r,cc in M)
            if all(0<=r<7 and 0<=c<7 for r,c in region) and S<=region:
                sols.append((aa,bb,region))
    assert len(sols)==1, sols
    aa,bb,region=sols[0]
    out=np.full_like(a,sep)
    for bi in range(7):
        for bj in range(7):
            tile = st if (bi,bj) in region else baseT
            out[1+4*bi:4+4*bi,1+4*bj:4+4*bj]=tile
    return out.tolist()

# submitted:
answer=solve(test_input)
answer
