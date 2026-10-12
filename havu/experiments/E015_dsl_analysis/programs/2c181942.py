# task 2c181942 pair 0 attempt 0 run tools_persistent_grid_lean2sub_deepseek-flash_low_e009_full_sub80s40 (median-length of 13 function programs)
def comps(g):
    g=np.array(g)
    res=[]
    for c in sorted(set(g.flatten().tolist())):
        if c==8: continue
        mask=(g==c)
        lab,n=ndimage.label(mask, structure=np.ones((3,3)))
        for k in range(1,n+1):
            ys,xs=np.where(lab==k)
            res.append((c,ys.min(),ys.max(),xs.min(),xs.max(),len(ys)))
    return res

def solve(g):
    g=np.array(g).copy()
    H,W=g.shape
    # find components per color
    comps=[]  # (color, mask, bbox)
    for c in sorted(set(g.flatten().tolist())):
        if c==8: continue
        m=(g==c)
        lab,n=ndimage.label(m,structure=np.ones((3,3)))
        for k in range(1,n+1):
            ys,xs=np.where(lab==k)
            comps.append((c,lab==k,ys.min(),ys.max(),xs.min(),xs.max(),len(ys)))
    arms=[comp for comp in comps if comp[6]==2]
    srcs={}
    for comp in comps:
        if comp[6]>2:
            srcs.setdefault(comp[0],[]).append(comp)
    # center
    allarmcells=[]
    for comp in arms:
        ys,xs=np.where(comp[1])
        allarmcells+=[(y,x) for y,x in zip(ys,xs)]
    cy=np.mean([p[0] for p in allarmcells]); cx=np.mean([p[1] for p in allarmcells])
    # arm direction
    arm_dir={}
    for comp in arms:
        c,mask,r0,r1,c0,c1,sz=comp
        if r0==r1:  # horizontal arm
            dirn='top' if r0<cy else 'bottom'
        else:
            dirn='left' if c0<cx else 'right'
        arm_dir[c]=(dirn,comp)
    out=np.full((H,W),8,dtype=int)
    # keep arm-only colors as is
    # build
    for comp in comps:
        c,mask,r0,r1,c0,c1,sz=comp
        if sz==2:
            # arm: keep
            out[mask]=c
    # place sources
    for c,comps_list in srcs.items():
        if c not in arm_dir:
            continue  # removed
        comp=comps_list[0]
        S=comp[1][comp[2]:comp[3]+1, comp[4]:comp[5]+1].astype(int)
        dirn,armcomp=arm_dir[c]
        ar=armcomp
        # arm cells
        ays,axs=np.where(armcomp[1])
        # find rotation
        found=None
        for rot in range(4):
            D=np.rot90(S,rot)  # CCW rot
            h,w=D.shape
            if dirn in ('top','bottom'):
                if w%2: continue
                mid=(w//2-1, w//2)
                line = D[-1] if dirn=='top' else D[0]
                if np.array_equal(np.where(line)[0], np.array(mid)):
                    found=(rot,D); break
            else:
                if h%2: continue
                mid=(h//2-1,h//2)
                line = D[:,-1] if dirn=='left' else D[:,0]
                if np.array_equal(np.where(line)[0], np.array(mid)):
                    found=(rot,D); break
        assert found is not None, (c,dirn)
        rot,D=found
        h,w=D.shape
        # arm geometry
        row_arm=ays[0]; col_arm=axs[0]
        if dirn in ('top','bottom'):
            r_arm=ays[0]  # row of horizontal arm
            c_lo=min(axs); c_hi=max(axs)
            # shape columns range
            cc_lo = c_lo-(w//2-1); 
            rr_lo = r_arm-h if dirn=='top' else r_arm+1
        else:
            c_arm=axs[0]
            r_lo=min(ays); r_hi=max(ays)
            rr_lo = r_lo-(h//2-1)
            cc_lo = c_arm-w if dirn=='left' else c_arm+1
        # stamp D at (rr_lo, cc_lo)
        ys,xs=np.where(D)
        for dy,dx in zip(ys,xs):
            out[rr_lo+dy, cc_lo+dx]=c
    return out

# submitted:
answer=solve(test_input)
answer
