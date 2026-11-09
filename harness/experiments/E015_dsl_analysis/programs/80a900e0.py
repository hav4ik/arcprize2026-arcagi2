# task 80a900e0 pair 0 attempt 0 run tools_persistent_grid_lean2sub_deepseek-flash_low_e010_full_s25x80 (median-length of 7 function programs)
def find_diamonds(a):
    H,W=a.shape
    centers=[]; cells=set()
    offs=[(-2,0),(-1,-1),(-1,1),(0,-2),(0,2),(1,-1),(1,1),(2,0)]
    for r in range(H):
        for c in range(W):
            cols=[]
            ok=True
            for dr,dc in offs:
                rr,cc=r+dr,c+dc
                if not(0<=rr<H and 0<=cc<W): ok=False;break
                cols.append(a[rr,cc])
            if not ok: continue
            col=cols[0]
            if col in (0,1): continue
            if all(x==col for x in cols):
                centers.append((r,c,col))
                cells.update((r+dr,c+dc) for dr,dc in offs)
    return centers,cells

def solve(a):
    a=np.array(a)
    H,W=a.shape
    centers,dcells=find_diamonds(a)
    S=set()
    for r in range(H):
        for c in range(W):
            if a[r,c] not in (0,1) and (r,c) not in dcells:
                S.add((r,c))
    seen=set(); comps=[]
    for cell in S:
        if cell in seen: continue
        stack=[cell]; comp=[]; seen.add(cell)
        while stack:
            x=stack.pop(); comp.append(x)
            for dx in (-1,0,1):
                for dy in (-1,0,1):
                    if dx==0 and dy==0: continue
                    n=(x[0]+dx,x[1]+dy)
                    if n in S and n not in seen:
                        seen.add(n); stack.append(n)
        comps.append(comp)
    out=a.copy()
    for comp in comps:
        comp=sorted(comp)
        col=int(a[comp[0][0],comp[0][1]])
        d=(comp[-1][0]-comp[0][0], comp[-1][1]-comp[0][1])
        g=math.gcd(abs(d[0]),abs(d[1])) or 1
        d=(d[0]//g,d[1]//g)
        if d in ((1,1),(-1,-1)): cands=[(-1,1),(1,-1)]
        elif d in ((1,-1),(-1,1)): cands=[(1,1),(-1,-1)]
        else: cands=[]
        mr=sum(p[0] for p in comp)/len(comp); mc=sum(p[1] for p in comp)/len(comp)
        best=None;bd=1e9
        for (cr,cc,ccol) in centers:
            dd=(cr-mr)**2+(cc-mc)**2
            if dd<bd: bd=dd;best=(cr,cc,ccol)
        if best is None:
            print("no diamond for",comp);continue
        outv=(mr-best[0],mc-best[1])
        beam=None
        for cand in cands:
            if cand[0]*outv[0]+cand[1]*outv[1]>0: beam=cand
        if beam is None:
            print("NO BEAM",comp,d,cands,outv);continue
        for end in (comp[0],comp[-1]):
            k=1
            while True:
                rr=end[0]+beam[0]*k; cc=end[1]+beam[1]*k
                if not(0<=rr<H and 0<=cc<W): break
                out[rr,cc]=col; k+=1
    return out

# submitted:
solve(np.array(test_input))
