# task c4d067a0 pair 0 attempt 1 run tools_persistent_grid_lean2subv_deepseek-flash_low_e011_full_subv80s40 (median-length of 12 function programs)
def solve(grid):
    g = np.array(grid)
    h, w = g.shape
    bg = Counter(g.flatten().tolist()).most_common(1)[0][0]
    nonbg = (g != bg)

    key_mask = np.zeros_like(nonbg)
    for r in range(h):
        for c in range(w):
            if nonbg[r, c]:
                block = nonbg[max(0,r-1):r+2, max(0,c-1):c+2]
                if block.sum() == 1:
                    key_mask[r, c] = True
    target_mask = nonbg & (~key_mask)

    krows = sorted(set(np.where(key_mask)[0].tolist()))
    kcols = sorted(set(np.where(key_mask)[1].tolist()))
    def step(lst):
        ds = [lst[i+1]-lst[i] for i in range(len(lst)-1)]
        return min(ds) if ds else 2
    sr, sc = step(krows), step(kcols)
    nr, nc = len(krows), len(kcols)
    key = np.full((nr, nc), bg, dtype=int)
    for r in krows:
        for c in kcols:
            key[(r-krows[0])//sr, (c-kcols[0])//sc] = g[r, c]

    lab, n = label(target_mask)
    comps = []
    for k in range(1, n+1):
        ys, xs = np.where(lab == k)
        comps.append((ys.min(), xs.min(), ys.max()-ys.min()+1, xs.max()-xs.min()+1, int(g[ys[0], xs[0]])))
    bs_r = Counter(c[2] for c in comps).most_common(1)[0][0]
    bs_c = Counter(c[3] for c in comps).most_common(1)[0][0]
    tops = sorted(set(c[0] for c in comps))
    given = sorted([c for c in comps if c[0] == tops[0]], key=lambda x: x[1])
    lefts = [c[1] for c in given]; colors = [c[4] for c in given]; m = len(given)
    stepv = min(lefts[i+1]-lefts[i] for i in range(len(lefts)-1)) if len(lefts)>1 else bs_r+1
    found = None
    for i in range(nr):
        for j0 in range(0, nc-m+1):
            if list(key[i, j0:j0+m]) == colors:
                found=(i,j0); break
        if found: break
    i, j0 = found
    R0 = tops[0]-stepv*i; C0 = lefts[0]-stepv*j0
    out = g.copy()
    for a in range(nr):
        for b in range(nc):
            out[R0+stepv*a:R0+stepv*a+bs_r, C0+stepv*b:C0+stepv*b+bs_c] = key[a,b]
    return out

# submitted:
solve(test_input)
