# task 247ef758 pair 0 attempt 0 run tools_persistent_grid_lean2sub_deepseek-flash_low_e010_full_s20x60 (median-length of 32 function programs)
def parse(a):
    a=np.array(a);H,W=a.shape
    for sep in range(1,W-3):
        reg=a[:,sep+1:]
        if reg[:,0].tolist()==reg[:,-1].tolist() and (reg[1:-1,1:-1].size==0 or reg[1:-1,1:-1].max()==0):
            return sep,a[:,:sep].copy(),reg.copy()
    return None

def solve(a):
    a=np.array(a);H,W=a.shape
    sep,key,reg=parse(a)
    rh,rw=reg.shape;out=reg.copy();newkey=key.copy()
    rows=key.any(axis=1);chunks=[];i=0
    while i<len(rows):
        if rows[i]:
            j=i
            while j<len(rows) and rows[j]: j+=1
            r=i
            while r<j:
                e=min(r+3,j);chunks.append((r,e-1));r=e
            i=j
        else:i+=1
    for (r0,r1) in chunks:
        sh=key[r0:r1+1]; h,w=sh.shape
        nz=np.argwhere(sh>0)
        cmin=nz[:,1].min(); ww=nz[:,1].max()-cmin+1
        col=int(sh[sh>0][0])
        colpos=[c for c in range(rw) if reg[0,c]==col or reg[-1,c]==col]
        rowpos=[r for r in range(rh) if reg[r,0]==col or reg[r,-1]==col]
        if colpos and rowpos:
            for r in rowpos:
                for c in colpos:
                    for di in range(h):
                        for dj in range(ww):
                            rr=r-(h//2)+di; cc=c-(ww//2)+dj
                            v=sh[di,cmin+dj]
                            if v==col and 0<=rr<rh and 0<=cc<rw and out[rr,cc]==0:
                                out[rr,cc]=col
            newkey[r0:r1+1]=0
    res=a.copy();res[:,:sep]=newkey;res[:,sep+1:]=out
    return res

# submitted:
answer=solve(test_input)
answer
