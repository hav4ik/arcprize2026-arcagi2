"""Spot-check of "yes" verdicts: hand-written arc-dsl solvers for ARC-AGI-2 eval tasks that DeepSeek solved.

    .venv/bin/python -I experiments/E015_dsl_analysis/dsl_solvers.py
Each solver is in Hodel's straight-line style; a few use a one-line lambda where Hodel would chain combinators.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import as_tuple, import_dsl, load_arc2_eval  # noqa: E402

dsl, _, _ = import_dsl()
from dsl import *  # noqa: E402,F401,F403
from constants import *  # noqa: E402,F401,F403


def solve_1818057f(I):
    # plus-shapes of 4 -> 8
    x1 = insert(UNITY, dneighbors(UNITY))
    x2 = recolor(FOUR, x1)
    x3 = occurrences(I, x2)
    x4 = lbind(shift, x1)
    x5 = mapply(x4, x3)
    O = fill(I, EIGHT, x5)
    return O


def solve_bf45cf4b(I):
    # one-colour mask x multicolour tile -> mask cells replaced by the tile
    x1 = objects(I, F, T, T)
    x2 = matcher(numcolors, ONE)
    x3 = extract(x1, x2)
    x4 = other(x1, x3)
    x5 = subgrid(x4, I)
    x6 = shape(x5)
    x7 = normalize(toindices(x3))
    x8 = apply(rbind(multiply, x6), x7)
    x9 = mapply(lbind(shift, asobject(x5)), x8)
    x10 = canvas(mostcolor(I), multiply(shape(x3), x6))
    O = paint(x10, x9)
    return O


def solve_62593bfd(I):
    # objects whose size is shared move to the top edge, unique-size objects to the bottom edge
    x1 = objects(I, T, T, T)
    x2 = chain(size, lbind(sizefilter, x1), size)
    x3 = sfilter(x1, compose(rbind(greater, ONE), x2))
    x4 = difference(x1, x3)
    x5 = fork(shift, identity, chain(toivec, invert, uppermost))
    x6 = fork(shift, identity, chain(toivec, lbind(subtract, decrement(height(I))), lowermost))
    x7 = canvas(mostcolor(I), shape(I))
    x8 = paint(x7, mapply(x5, x3))
    O = paint(x8, mapply(x6, x4))
    return O


def solve_58490d8a(I):
    # legend box of 0: for each legend colour, one dot per (size>1) object of that colour, every 2nd column
    x1 = ofcolor(I, ZERO)
    x2 = subgrid(x1, I)
    x3 = objects(I, T, T, T)
    x4 = sfilter(x3, compose(rbind(greater, ONE), size))
    x5 = objects(x2, T, F, T)
    x6 = lambda o: size(colorfilter(x4, color(o)))  # count objects of the legend colour
    x7 = lambda o: recolor(color(o), apply(lambda k: add(ulcorner(o), (0, double(k))), interval(0, x6(o), 1)))
    x8 = mapply(x7, x5)
    O = paint(canvas(ZERO, shape(x2)), x8)
    return O


def solve_aa4ec2a5(I):
    # 1-shapes on 4: shapes with holes -> 8, holes -> 6, 8-neighbourhood outline -> 2
    x1 = objects(I, T, F, F)
    x2 = colorfilter(x1, FOUR)
    x3 = toindices(mfilter(x2, compose(flip, rbind(bordering, I))))
    x4 = colorfilter(x1, ONE)
    x5 = sfilter(x4, lambda o: size(intersection(delta(o), x3)) > 0)
    x6 = ofcolor(I, ONE)
    x7 = difference(mapply(neighbors, x6), combine(x6, x3))
    x8 = fill(I, TWO, x7)
    x9 = fill(x8, SIX, x3)
    O = fill(x9, EIGHT, merge(x5))
    return O


def solve_8e5c0c38(I):
    # per colour: delete the fewest cells that make the shape left-right symmetric
    x1 = fgpartition(I)
    x2 = lambda o: apply(lambda k: intersection(o, shift(vmirror(o), (0, k))), interval(invert(width(o)), width(o), 1))
    x3 = lambda o: difference(o, argmax(x2(o), size))
    x4 = mapply(x3, x1)
    O = cover(I, x4)
    return O


def main():
    ev = load_arc2_eval()
    import inspect
    for name, fn in sorted(globals().items()):
        if not name.startswith("solve_"):
            continue
        t = ev[name[6:]]
        res = []
        for p in t["train"] + t["test"]:
            try:
                res.append(fn(as_tuple(p["input"])) == as_tuple(p["output"]))
            except Exception as e:  # noqa: BLE001
                res.append(type(e).__name__)
        n_lines = len(inspect.getsource(fn).splitlines()) - 3  # minus def, comment, return
        print(f"{name[6:]}: {sum(r is True for r in res)}/{len(res)} pairs correct, {n_lines} DSL lines  {res}")


if __name__ == "__main__":
    main()
