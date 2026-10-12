# task c7f57c3e pair 0 attempt 0 run tools_persistent_grid_lean2sub_deepseek-flash_high_e011_full_sub80s40h (median-length of 9 function programs)
def group_objects(g):
    bg = Counter(g.flatten()).most_common(1)[0][0]
    mask = g != bg
    lab, n = ndimage.label(mask, structure=STRUCT)
    comps = []
    for i in range(1, n+1):
        ys, xs = np.where(lab == i)
        comps.append(dict(r0=int(ys.min()), r1=int(ys.max()), c0=int(xs.min()), c1=int(xs.max()),
                          cells=[(int(y), int(x), int(g[y, x])) for y, x in zip(ys, xs)]))
    # union-find merge if expanded bboxes (by 1) overlap
    parent = list(range(n))
    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb: parent[ra] = rb
    for i in range(n):
        for j in range(i+1, n):
            a, b = comps[i], comps[j]
            if a['r0']-1 <= b['r1'] and b['r0']-1 <= a['r1'] and a['c0']-1 <= b['c1'] and b['c0']-1 <= a['c1']:
                union(i, j)
    groups = {}
    for i, c in enumerate(comps):
        groups.setdefault(find(i), []).extend(c['cells'])
    objs = []
    for cells in groups.values():
        ys = [y for y, x, c in cells]; xs = [x for y, x, c in cells]
        ones = [(y, x) for y, x, c in cells if c == 1]
        ar, ac = min(ones)
        content = tuple(sorted((y-ar, x-ac, c) for y, x, c in cells if c != 1))
        objs.append(dict(cells=cells, ones=ones, anchor=(ar, ac), content=content,
                         h=max(ys)-min(ys)+1, w=max(xs)-min(xs)+1))
    return bg, objs

def solve(grid):
    g = np.array(grid)
    bg, objs = group_objects(g)
    dims = Counter((o['h'], o['w']) for o in objs)
    base = dims.most_common(1)[0][0]
    for o in objs: o['k'] = o['h'] // base[0]
    def canon(o):
        cs = o['content'] if o['k']==1 else tuple((dr//o['k'], dc//o['k'], c) for dr,dc,c in o['content'])
        return tuple(sorted(set(cs)))
    for o in objs: o['canon'] = canon(o)
    units = [o for o in objs if o['k'] == 1]
    types = []
    for o in units:
        if o['canon'] not in types: types.append(o['canon'])
    out = np.full_like(g, bg)
    for o in objs:
        for y, x in o['ones']: out[y, x] = 1
    for o in objs:
        others = [t for t in types if t != o['canon']]
        target = others[0] if others else o['canon']
        ar, ac = o['anchor']; k = o['k']
        for dr, dc, c in target:
            for i in range(k):
                for j in range(k):
                    out[ar + k*dr + i, ac + k*dc + j] = c
    return out

# submitted:
answer = solve(test_input)
answer
