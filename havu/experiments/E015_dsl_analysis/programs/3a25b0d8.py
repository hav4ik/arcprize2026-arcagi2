# task 3a25b0d8 pair 0 attempt 1 run tools_persistent_grid_lean2sub_deepseek-flash_low_e011_full_sub80s40v (median-length of 15 function programs)
def comps(a,bg):
    lbl,n=ndimage.label(a!=bg, structure=np.ones((3,3)))
    out=[]
    for k in range(1,n+1):
        ys,xs=np.where(lbl==k)
        out.append(a[ys.min():ys.max()+1, xs.min():xs.max()+1])
    return out

def nscale(img,H,W):
    h,w=img.shape
    ri=np.clip(np.arange(H)*h//H,0,h-1); ci=np.clip(np.arange(W)*w//W,0,w-1)
    return img[np.ix_(ri,ci)]

def solve(a):
    a=np.array(a)
    vals,cnts=np.unique(a,return_counts=True)
    bg=vals[np.argmax(cnts)]
    lbl,n=ndimage.label(a!=bg, structure=np.ones((3,3)))
    comps=[]
    for k in range(1,n+1):
        ys,xs=np.where(lbl==k)
        comps.append(a[ys.min():ys.max()+1, xs.min():xs.max()+1])
    comps.sort(key=lambda c: len(np.unique(c)))
    pure=comps[0]; colr=comps[1]
    H,W=pure.shape
    nonbg=pure[pure!=bg]; line=np.bincount(nonbg).argmax()
    out=pure.copy()
    out[(out!=bg)&(out!=line)]=bg
    # remove line pixels not supported by scaled colored silhouette and isolated
    s=nscale(colr,H,W); sil=(s!=bg)
    sup=ndimage.binary_dilation(sil, structure=np.ones((3,3)))
    out[(out!=bg)&(~sup)]=bg
    l2,n2=ndimage.label(out==bg)
    border=set(l2[0,:])|set(l2[-1,:])|set(l2[:,0])|set(l2[:,-1])
    ch,cw=colr.shape
    cnon=colr[(colr!=bg)]; cline=np.bincount(cnon).argmax() if len(cnon) else bg
    colored_mask=(colr!=bg)&(colr!=cline)
    cys,cxs=np.where(colored_mask)
    for k in range(1,n2+1):
        if k in border: continue
        ys,xs=np.where(l2==k)
        cy,cx=ys.mean(),xs.mean()
        mr=min(max(int(round(cy*ch/H)),0),ch-1); mc=min(max(int(round(cx*cw/W)),0),cw-1)
        if colored_mask[mr,mc]: v=colr[mr,mc]
        else:
            d=(cys-cy*ch/H)**2+(cxs-cx*cw/W)**2; idx=np.argmin(d); v=colr[cys[idx],cxs[idx]]
        out[l2==k]=v
    return out

# submitted:
solve(test_input)
