# task b10624e5 pair 0 attempt 0 run tools_persistent_grid_lean2sub_deepseek-flash_low_e011_full_sub80s40v (median-length of 15 function programs)
def quads(g):
    g=np.array(g)
    return {'TL':g[0:13,0:13],'TR':g[0:13,14:27],'BL':g[14:27,0:13],'BR':g[14:27,14:27]}

def find_anchor(cells, color=2):
    pts=[(r,c) for (r,c),v in cells.items() if v==color]
    rs=[p[0] for p in pts]; cs=[p[1] for p in pts]
    r0,c0=min(rs),min(cs)
    h=max(rs)-r0+1; w=max(cs)-c0+1
    return (r0,c0,h,w)

def quadrant_cells(g, qr, qc):
    # qr,qc in {0,1}: 0=top/left (rows0-12/cols0-12),1=bottom/right(rows14-26/cols14-26)
    r0 = 0 if qr==0 else 14
    c0 = 0 if qc==0 else 14
    sub = g[r0:r0+13, c0:c0+13]
    cells={}
    for r in range(13):
        for c in range(13):
            if sub[r,c]!=4:
                cells[(r,c)]=int(sub[r,c])
    return cells,(r0,c0)

def solve(g):
    g=np.array(g)
    quads={}
    for qr in (0,1):
        for qc in (0,1):
            quads[(qr,qc)]=quadrant_cells(g,qr,qc)
    # reference = quadrant with most distinct colors
    def ndist(cells):
        return len(set(cells.values()))
    ref_key=max(quads, key=lambda k: (ndist(quads[k][0]), len(quads[k][0])))
    ref_cells,_ = quads[ref_key]
    ar0,ac0,u,_ = find_anchor(ref_cells)
    # reflected reference relative to anchor
    def reflected(hflip,vflip):
        d={}
        for (r,c),v in ref_cells.items():
            dr=r-ar0; dc=c-ac0
            if hflip: dc=(u-1)-dc
            if vflip: dr=(u-1)-dr
            d[(dr,dc)]=v
        return d
    out=[row[:] for row in g.tolist()]
    orient={(0,1):(True,False),   # TR hflip
            (1,0):(False,True),   # BL vflip
            (1,1):(True,True)}    # BR 180
    for key,(hflip,vflip) in orient.items():
        cells,(r0,c0)=quads[key]
        hr0,hc0,sh,sw=find_anchor(cells)
        refd=reflected(hflip,vflip)
        drs=[k[0] for k in refd]; dcs=[k[1] for k in refd]
        f=sh/u
        qr_lo=int(np.floor(min(drs)*f)); qr_hi=int(np.ceil((max(drs)+1)*f))-1
        qc_lo=int(np.floor(min(dcs)*f)); qc_hi=int(np.ceil((max(dcs)+1)*f))-1
        for qr in range(qr_lo,qr_hi+1):
            for qc in range(qc_lo,qc_hi+1):
                sdr=int(np.floor(qr/f)); sdc=int(np.floor(qc/f))
                if (sdr,sdc) in refd:
                    rr=r0+hr0+qr; cc=c0+hc0+qc
                    if 0<=rr<27 and 0<=cc<27:
                        out[rr][cc]=refd[(sdr,sdc)]
    return out

# submitted:
solve(test_input)
