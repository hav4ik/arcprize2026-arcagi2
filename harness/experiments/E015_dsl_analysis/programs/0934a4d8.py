# task 0934a4d8 pair 0 attempt 1 run tools_persistent_grid_lean2sub_deepseek-flash_low_e009_full_sub80s40 (median-length of 3 function programs)
def f(x):  # map index to fundamental
    return x if x <= 15 else 31 - x

def solve(grid):
    a = np.array(grid)
    H, W = a.shape
    hole = (a == 8)
    ys, xs = np.where(hole)
    r0, r1, c0, c1 = ys.min(), ys.max(), xs.min(), xs.max()
    out = np.zeros((r1-r0+1, c1-c0+1), int)
    for i in range(r0, r1+1):
        for j in range(c0, c1+1):
            src = None
            cands = [(31-i, j), (i, 31-j), (31-i, 31-j)]
            # block transpose: fundamental coords
            R, C = f(i), f(j)
            if 8 <= R <= 15 and 0 <= C <= 7:
                # F[R,C]=F[C,R]; map source fundamental (C,R) back to grid
                for rr in ({C} if C < 16 else set()) | ({31-C} if 0 <= 31-C < H else set()):
                    for cc in ({R} if R < 16 else set()) | ({31-R} if 0 <= 31-R < W else set()):
                        if 0 <= rr < H and 0 <= cc < W and not hole[rr, cc]:
                            cands.append((rr, cc))
            for (u, v) in cands:
                if 0 <= u < H and 0 <= v < W and not hole[u, v]:
                    src = a[u, v]; break
            out[i-r0, j-c0] = src
    return out

# submitted:
answer = solve(test_input)
answer
