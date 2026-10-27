#!/usr/bin/env python3
"""E055 toolkit: a hidden-set-sized replica workload from the 120 ARC-AGI-2 evaluation tasks.

Each eval task appears once as is and K times (default 1) as a TRANSFORMED copy under a new id: one random D8
transform (7 non-identity rotations / flips) and one random permutation of colours 1-9 (0 stays 0), applied
consistently to every grid of the task (train inputs / outputs, test inputs) and to its solutions. The copy
then shares no prompt prefix with the original beyond the instruction text (vLLM's prefix cache cannot reuse the
original's KV for the copy, as it could with the untransformed `eval_x2` set, where copies run next to their
originals because the solver orders tasks by size).

Per task the transform comes from random.Random(f"{seed}:{task}:{k}") (independent of task order), drawn again
(up to 64 draws, the earliest divergence wins) until:
- the colour permutation moves every non-zero colour the task uses (a derangement on the used colours);
- the copy's prompt diverges from the original's inside the first row of the first grid (the train-0 input).
  Grids whose four borders are background in every D8 image (square, all-zero edges) cannot diverge before their
  first non-background row; the manifest lists them with the shared length.

Validation (always run; the build fails on any error):
- every copy grid inverts to the original grid (inverse D8, inverse permutation), solutions included;
- shapes: (h, w) for id / r180 / flips, (w, h) for r90 / r270 / transposes;
- the solutions file has one grid per test input of every task; ids are unique and unused by any ARC file in data/;
- prompt divergence offset per copy (characters into the first grid), written to the manifest.

Outputs (in --out-dir, default toolkit/workloads/): <name>_challenges.json, <name>_solutions.json,
<name>_idmap.json ({id: {orig, kind, copy, d8, perm, shared_chars}}), <name>_manifest.json (seed, sources and outputs
with sha256, validation summary).

Usage: make_workload.py [--seed 2026] [--copies 1] [--name evalx2t_s2026] [--no-original] [--check <name>]
"""
import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
TOOLKIT = Path(__file__).resolve().parent
DATA = ROOT / "data"

# ---------------------------------------------------------------------------------------------- D8 on list grids


def _t(g):
    return [list(r) for r in zip(*g)]


D8 = {
    "id": lambda g: [list(r) for r in g],
    "r90": lambda g: [list(r) for r in zip(*g[::-1])],  # clockwise
    "r180": lambda g: [list(r[::-1]) for r in g[::-1]],
    "r270": lambda g: [list(r) for r in zip(*g)][::-1],  # counter-clockwise
    "fh": lambda g: [list(r[::-1]) for r in g],  # mirror left-right
    "fv": lambda g: [list(r) for r in g[::-1]],  # upside down
    "t": _t,  # transpose (main diagonal)
    "at": lambda g: [list(r[::-1]) for r in _t(g)[::-1]],  # anti-transpose
}
D8_INV = {"id": "id", "r90": "r270", "r270": "r90", "r180": "r180", "fh": "fh", "fv": "fv", "t": "t", "at": "at"}
SWAPS_SHAPE = {"r90", "r270", "t", "at"}
NON_ID = ["r90", "r180", "r270", "fh", "fv", "t", "at"]


def recolour(g, perm):
    return [[perm[v] for v in r] for r in g]


def transform(g, d8, perm):
    return recolour(D8[d8](g), perm)


def invert(g, d8, perm):
    inv = [0] * 10
    for c, p in enumerate(perm):
        inv[p] = c
    return D8[D8_INV[d8]](recolour(g, inv))


def task_grids(task):
    for x in task["train"]:
        yield x["input"]
        yield x["output"]
    for x in task["test"]:
        yield x["input"]


def used_colours(task, sols):
    cs = set()
    for g in list(task_grids(task)) + list(sols):
        for r in g:
            cs.update(r)
    return sorted(c for c in cs if c)


def transform_task(task, sols, d8, perm):
    t = {"train": [{"input": transform(x["input"], d8, perm), "output": transform(x["output"], d8, perm)}
                   for x in task["train"]],
         "test": [{"input": transform(x["input"], d8, perm)} for x in task["test"]]}
    return t, [transform(g, d8, perm) for g in sols]


# ---------------------------------------------------------------------------------------------- prompt divergence

def prompt_builder():
    """The harness's own ARC Prize prompt (evals/arc_harness/arcprize.py); a local copy of its format otherwise."""
    sys.path.insert(0, str(ROOT / "evals"))
    try:
        from arc_harness.arcprize import build_prompt
        return build_prompt, "evals/arc_harness/arcprize.build_prompt"
    except Exception:  # pragma: no cover - same text as the harness, without its import chain
        tpl = (ROOT / "notes/prompts/arcprize_official_system_prompt.txt").read_text()

        def build_prompt(train, test_input):
            ex = ""
            for i, p in enumerate(train):
                ex += f"--Example {i}-- \n\n INPUT: \n\n{json.dumps(p['input'])}\n\nOUTPUT: \n\n{json.dumps(p['output'])}\n\n"
            return tpl.format(training_examples=ex, test_input=json.dumps(test_input))
        return build_prompt, "local copy of the ARC Prize template"


def common_prefix(a, b):
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def divergence(build_prompt, task, copy):
    """(characters of the first grid shared before the copy's prompt diverges, length of the copy's first grid row
    in characters, offset of the first grid in the prompt)."""
    a = build_prompt(task["train"], task["test"][0]["input"])
    b = build_prompt(copy["train"], copy["test"][0]["input"])
    g0 = json.dumps(task["train"][0]["input"])
    start = a.index(g0)
    assert b.index(json.dumps(copy["train"][0]["input"])) == start, "instruction text differs before the first grid"
    shared = common_prefix(a, b) - start
    row = len(json.dumps(copy["train"][0]["input"][0])) + 2  # "[[" ... "], "
    return shared, row, start


# ---------------------------------------------------------------------------------------------- build

def new_id(seed, tid, k, taken):
    for salt in range(1000):
        h = hashlib.sha1(f"e055:{seed}:{tid}:{k}:{salt}".encode()).hexdigest()[:8]
        if h not in taken:
            return h
    raise RuntimeError("no free id")


def known_ids():
    ids = set()
    for f in DATA.glob("arc-agi_*_challenges.json"):
        ids |= set(json.loads(f.read_text()))
    return ids


def draw(rng, colours):
    """One (d8, perm) draw; perm moves every used colour (redrawn until it does, at most 1000 times)."""
    d8 = rng.choice(NON_ID)
    for _ in range(1000):
        p = list(range(1, 10))
        rng.shuffle(p)
        perm = [0] + p
        if all(perm[c] != c for c in colours):
            return d8, perm
    return d8, perm


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build(args):
    ch_path, so_path = Path(args.challenges), Path(args.solutions)
    chal, sols = json.loads(ch_path.read_text()), json.loads(so_path.read_text())
    assert set(chal) == set(sols), "challenges and solutions cover different tasks"
    build_prompt, prompt_src = prompt_builder()
    taken = known_ids()
    out_c, out_s, idmap, worst = {}, {}, {}, []
    for tid in sorted(chal):
        task, sol = chal[tid], sols[tid]
        assert len(sol) == len(task["test"]), f"{tid}: {len(sol)} solutions for {len(task['test'])} test inputs"
        if not args.no_original:
            out_c[tid], out_s[tid] = task, sol
            idmap[tid] = {"orig": tid, "kind": "orig", "copy": 0, "d8": "id", "perm": list(range(10)),
                          "shared_chars": None}
        colours = used_colours(task, sol)
        for k in range(1, args.copies + 1):
            rng = random.Random(f"{args.seed}:{tid}:{k}")
            best = None
            used = set()
            for n_draw in range(64):
                d8, perm = draw(rng, colours)
                if (d8, tuple(perm)) in used:
                    continue
                used.add((d8, tuple(perm)))
                ct, cs = transform_task(task, sol, d8, perm)
                shared, row, _ = divergence(build_prompt, task, ct)
                if best is None or shared < best[0]:
                    best = (shared, row, d8, perm, ct, cs, n_draw + 1)
                if shared < row:
                    break
            shared, row, d8, perm, ct, cs, n_draws = best
            cid = new_id(args.seed, tid, k, taken)
            taken.add(cid)
            out_c[cid], out_s[cid] = ct, cs
            idmap[cid] = {"orig": tid, "kind": "copy", "copy": k, "d8": d8, "perm": perm, "shared_chars": shared,
                          "first_row_chars": row, "draws": n_draws}
            if shared >= row:
                worst.append((tid, cid, shared, row))
    return out_c, out_s, idmap, worst, prompt_src


def validate(chal, sols, idmap, src_chal, src_sols, build_prompt):
    """Raises AssertionError on the first inconsistency; returns summary counts."""
    assert set(chal) == set(sols) == set(idmap), "challenges / solutions / idmap ids differ"
    n = {"tasks": len(chal), "pairs": 0, "copies": 0, "grids_checked": 0, "shape_swaps": 0}
    for tid, m in idmap.items():
        orig, d8, perm = m["orig"], m["d8"], m["perm"]
        assert sorted(perm) == list(range(10)) and perm[0] == 0, f"{tid}: bad permutation {perm}"
        src, ssol, task, sol = src_chal[orig], src_sols[orig], chal[tid], sols[tid]
        assert len(task["train"]) == len(src["train"]) and len(task["test"]) == len(src["test"]), f"{tid}: pair counts"
        assert len(sol) == len(task["test"]), f"{tid}: solutions vs test inputs"
        n["pairs"] += len(task["test"])
        pairs = [(a["input"], b["input"]) for a, b in zip(task["train"], src["train"])]
        pairs += [(a["output"], b["output"]) for a, b in zip(task["train"], src["train"])]
        pairs += [(a["input"], b["input"]) for a, b in zip(task["test"], src["test"])]
        pairs += list(zip(sol, ssol))
        for g, o in pairs:
            h, w = len(o), len(o[0])
            want = (w, h) if d8 in SWAPS_SHAPE else (h, w)
            assert (len(g), len(g[0])) == want, f"{tid}: shape {(len(g), len(g[0]))} != {want} under {d8}"
            assert all(len(r) == len(g[0]) for r in g), f"{tid}: ragged grid"
            assert invert(g, d8, perm) == o, f"{tid}: grid does not invert to the original under {d8} / {perm}"
            assert transform(o, d8, perm) == g, f"{tid}: transform mismatch"
            n["grids_checked"] += 1
            n["shape_swaps"] += d8 in SWAPS_SHAPE and h != w
        if m["kind"] == "copy":
            n["copies"] += 1
            assert task != src, f"{tid}: copy identical to its original"
            shared, row, _ = divergence(build_prompt, src, task)
            assert shared == m["shared_chars"], f"{tid}: recorded divergence {m['shared_chars']} != {shared}"
        else:
            assert d8 == "id" and perm == list(range(10)) and task == src, f"{tid}: original altered"
    # no copy may equal any other task's content (e.g. a symmetric task whose transform maps it onto itself)
    seen = {}
    for tid, task in chal.items():
        key = json.dumps(task, sort_keys=True)
        assert key not in seen, f"{tid} duplicates {seen.get(key)}"
        seen[key] = tid
    # self-test of the D8 table on non-square grids: inverses and group closure
    g = [[1, 2, 3], [4, 5, 6]]
    for k in D8:
        assert D8[D8_INV[k]](D8[k](g)) == g, f"D8 inverse of {k}"
    assert len({json.dumps(D8[k](g)) for k in D8}) == 8, "D8 images not distinct"
    return n


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--challenges", default=str(DATA / "arc-agi_evaluation_challenges.json"))
    ap.add_argument("--solutions", default=str(DATA / "arc-agi_evaluation_solutions.json"))
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--copies", type=int, default=1, help="transformed copies per task (default 1: 240 tasks)")
    ap.add_argument("--no-original", action="store_true", help="only transformed copies")
    ap.add_argument("--name", default="", help="default evalx2t_s<seed> (copies 1) / evalx<1+copies>t_s<seed>")
    ap.add_argument("--out-dir", default=str(TOOLKIT / "workloads"))
    ap.add_argument("--check", default="", help="only validate an existing workload by name")
    a = ap.parse_args()
    out = Path(a.out_dir)
    src_c, src_s = json.loads(Path(a.challenges).read_text()), json.loads(Path(a.solutions).read_text())
    build_prompt, prompt_src = prompt_builder()
    if a.check:
        chal, sols, idmap = (json.loads((out / f"{a.check}_{x}.json").read_text())
                             for x in ("challenges", "solutions", "idmap"))
        n = validate(chal, sols, idmap, src_c, src_s, build_prompt)
        print(f"{a.check}: OK {n}")
        return
    name = a.name or (f"evalx{(0 if a.no_original else 1) + a.copies}t_s{a.seed}")
    chal, sols, idmap, worst, prompt_src = build(a)
    n = validate(chal, sols, idmap, src_c, src_s, build_prompt)
    out.mkdir(parents=True, exist_ok=True)
    files = {}
    for kind, obj in (("challenges", chal), ("solutions", sols), ("idmap", idmap)):
        p = out / f"{name}_{kind}.json"
        p.write_text(json.dumps(obj, separators=(",", ":")) if kind != "idmap" else json.dumps(obj, indent=0))
        files[p.name] = sha256(p)
    copies = [m for m in idmap.values() if m["kind"] == "copy"]
    sh = sorted(m["shared_chars"] for m in copies)
    manifest = {
        "name": name, "seed": a.seed, "copies": a.copies, "original_included": not a.no_original,
        "sources": {Path(a.challenges).name: sha256(a.challenges), Path(a.solutions).name: sha256(a.solutions)},
        "outputs": files, "prompt_format": prompt_src, "validation": n,
        "d8_counts": {k: sum(m["d8"] == k for m in copies) for k in NON_ID},
        "divergence": {
            "rule": "characters of the first grid shared by copy and original before their prompts differ "
                    "(target: inside the first row)",
            "within_first_row": sum(m["shared_chars"] < m["first_row_chars"] for m in copies),
            "copies": len(copies), "shared_chars_median": sh[len(sh) // 2] if sh else None,
            "shared_chars_max": sh[-1] if sh else None,
            "beyond_first_row": [{"orig": t, "copy": c, "shared_chars": s, "first_row_chars": r}
                                 for t, c, s, r in worst]},
    }
    (out / f"{name}_manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"{name}: {n['tasks']} tasks, {n['pairs']} pairs ({n['copies']} transformed copies), "
          f"{n['grids_checked']} grids inverted OK, {n['shape_swaps']} with swapped shapes")
    print(f"  divergence inside the first row of the first grid: {manifest['divergence']['within_first_row']}"
          f"/{len(copies)}; shared chars median {manifest['divergence']['shared_chars_median']}, "
          f"max {manifest['divergence']['shared_chars_max']}")
    for w in manifest["divergence"]["beyond_first_row"]:
        print(f"  beyond the first row: {w}")
    print(f"  wrote {', '.join(files)} + {name}_manifest.json to {out}")


if __name__ == "__main__":
    main()
