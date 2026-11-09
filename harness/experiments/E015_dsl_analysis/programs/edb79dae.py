# task edb79dae pair 0 attempt 1 run tools_persistent_grid_lean2sub_deepseek-flash_low_e009_full_sub96n (median-length of 14 function programs)
def ranges(seps,size):
    res=[];prev=-1
    for s in seps+[size]:
        if s-prev-1>0: res.append((prev+1,s-1))
        prev=s
    return res

def solve(inp):
    g=np.array(inp)
    vals,counts=np.unique(g,return_counts=True)
    bg=int(vals[np.argmax(counts)])
    mask5=(g==5)
    rr=np.where(mask5.sum(axis=1)>=5)[0]; cc=np.where(mask5.sum(axis=0)>=5)[0]
    r0,r1,c0,c1=int(rr.min()),int(rr.max()),int(cc.min()),int(cc.max())
    interior=g[r0+1:r1,c0+1:c1]
    n,m=interior.shape
    sepR=[i for i in range(n) if len(set(interior[i].tolist()))==1]
    sepC=[j for j in range(m) if len(set(interior[:,j].tolist()))==1]
    def ranges(seps,size):
        res=[];prev=-1
        for s in seps+[size]:
            if s-prev-1>0: res.append((prev+1,s-1))
            prev=s
        return res
    cellR=ranges(sepR,n); cellC=ranges(sepC,m)
    key=g.copy(); key[r0:r1+1,c0:c1+1]=-1
    work=(key!=bg)&(key!=-1)
    lab,ncomp=ndimage.label(work,structure=np.ones((3,3)))
    shapes={};mapd={}
    for i in range(1,ncomp+1):
        ys,xs=np.where(lab==i)
        h=ys.max()-ys.min()+1;w=xs.max()-xs.min()+1
        block=g[ys.min():ys.max()+1,xs.min():xs.max()+1]
        uc=[int(x) for x in np.unique(block) if x!=bg]
        if h==w:
            shapes[uc[0]]=block==uc[0]
        else:
            if len(uc)==2:
                src=None
                for cxx in uc:
                    if np.all(block[:,:2]==cxx): src=cxx
                if src is None: src=uc[0]
                mapd[src]=[x for x in uc if x!=src][0]
    H=r1-r0+1;W=c1-c0+1
    out=np.full((H,W),bg,dtype=int)
    out[0,:]=5;out[-1,:]=5;out[:,0]=5;out[:,-1]=5
    for (ra,rb) in cellR:
        for (ca,cb) in cellC:
            block=interior[ra:rb+1,ca:cb+1]
            u,cnt=np.unique(block,return_counts=True)
            col=int(u[np.argmax(cnt)])
            if col==bg: continue
            out[1+ra:1+rb+1,1+ca:1+cb+1]=np.where(shapes[col],mapd[col],bg)
    return out

# submitted:
answer=solve(test_input)
answer
