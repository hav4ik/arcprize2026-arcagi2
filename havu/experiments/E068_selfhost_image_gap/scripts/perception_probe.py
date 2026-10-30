#!/usr/bin/env python3
"""E068: grid perception from the image alone, DeepSeek API vs a self-hosted server, same pictures and question.

Each probe is one ARC training grid drawn exactly as the E058 harness draws it (arc_harness.grid_image: standard
palette, cell = aligned_cell(grid, "deepseek")), with no JSON and no size label. The model is asked for the grid's size,
the colours present, the four corner colours and a full transcription. Thinking on, effort low, no tools.
Scores: size exact, colour set exact, corners exact, transcription exact, and cell accuracy on size-correct answers.

  perception_probe.py --out api.jsonl [--n 60 --samples 2]                       # DeepSeek API (deepseek-flash)
  perception_probe.py --out self.jsonl --base-url http://127.0.0.1:18001/v1     # self-hosted (model dsv41, effort 50)
  perception_probe.py --score api.jsonl self.jsonl
Re-running resumes (keyed by grid id and sample).
"""
import argparse
import json
import os
import random
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "evals"))
from arc_harness.grid_image import COLOUR_NAMES, aligned_cell, data_uri  # noqa: E402

LEGEND = ", ".join(f"{k} = {v}" for k, v in COLOUR_NAMES.items())
QUESTION = (
    "The image shows one grid from a puzzle. Every cell is a solid square of one colour; there are no grid lines. "
    f"Colour key: {LEGEND}.\n\nLook at the image and answer with a JSON object only, no other text:\n"
    '{"rows": <number of rows>, "cols": <number of columns>, "colors": <sorted list of the colour numbers present>, '
    '"corners": [<top-left>, <top-right>, <bottom-left>, <bottom-right>], "grid": <the grid as a list of rows of colour '
    "numbers>}")


def probes(n, seed):
    tasks = json.load(open(ROOT / "data/arc-agi_training_challenges.json"))
    grids = []
    for tid in sorted(tasks):
        t = tasks[tid]
        for i, p in enumerate(t["train"]):
            for side in ("input", "output"):
                grids.append((f"{tid}:{i}:{side}", p[side]))
    rng = random.Random(seed)
    rng.shuffle(grids)
    # spread over sizes: up to n/4 each from <=36, <=144, <=400, >400 cells
    bins, out = {0: [], 1: [], 2: [], 3: []}, []
    for gid, g in grids:
        c = len(g) * len(g[0])
        b = 0 if c <= 36 else 1 if c <= 144 else 2 if c <= 400 else 3
        if len(bins[b]) < n // 4:
            bins[b].append((gid, g))
    for b in range(4):
        out += bins[b]
    return out


def request(args, client, grid):
    content = [{"type": "image_url", "image_url": {"url": data_uri(grid, aligned_cell(grid, "deepseek"))}},
               {"type": "text", "text": QUESTION}]
    msgs = [{"role": "user", "content": content}]
    if args.base_url:
        extra = {"chat_template_kwargs": {"thinking": True, "reasoning_effort": 50}, "temperature": 1.0, "top_p": 1.0}
        model = args.model or "dsv41"
    else:
        extra = {"reasoning_effort": "low", "thinking": {"type": "enabled"}}
        model = args.model or "deepseek-flash"
    r = client.chat.completions.create(model=model, messages=msgs, max_tokens=args.max_tokens, extra_body=extra)
    m = r.choices[0].message
    reasoning = getattr(m, "reasoning_content", None) or (m.model_extra or {}).get("reasoning_content") or ""
    return m.content or "", reasoning, r.usage.model_dump() if r.usage else None


def parse(text):
    s = text.strip()
    s = re.sub(r"^```(?:json)?|```$", "", s, flags=re.M).strip()
    i, j = s.find("{"), s.rfind("}")
    if i < 0 or j < 0:
        return None
    try:
        return json.loads(s[i:j + 1])
    except json.JSONDecodeError:
        return None


def score_one(grid, ans):
    h, w = len(grid), len(grid[0])
    colors = sorted({v for row in grid for v in row})
    corners = [grid[0][0], grid[0][-1], grid[-1][0], grid[-1][-1]]
    a = ans or {}
    size_ok = a.get("rows") == h and a.get("cols") == w
    g = a.get("grid")
    shape_ok = isinstance(g, list) and len(g) == h and all(isinstance(r, list) and len(r) == w for r in g)
    cells = (sum(g[i][j] == grid[i][j] for i in range(h) for j in range(w)) / (h * w)) if shape_ok else None
    return {"parsed": ans is not None, "size": size_ok, "colors": a.get("colors") == colors,
            "corners": a.get("corners") == corners, "exact": shape_ok and g == grid, "shape": shape_ok, "cells": cells}


def run(args):
    from openai import OpenAI
    if args.base_url:
        client = OpenAI(base_url=args.base_url, api_key="none", timeout=3600)
    else:
        client = OpenAI(base_url="https://api.deepseek.com", api_key=os.environ["DEEPSEEK_API_KEY"], timeout=3600)
    done = set()
    if os.path.exists(args.out):
        for l in open(args.out):
            r = json.loads(l)
            if "error" not in r:
                done.add((r["gid"], r["sample"]))
    jobs = [(gid, g, s) for gid, g in probes(args.n, args.seed) for s in range(args.samples) if (gid, s) not in done]
    print(f"{len(jobs)} probe calls -> {args.out}", flush=True)
    lock = threading.Lock()

    def job(j):
        gid, g, s = j
        try:
            content, reasoning, usage = request(args, client, g)
            rec = {"gid": gid, "sample": s, "content": content, "reasoning": reasoning, "usage": usage,
                   **score_one(g, parse(content))}
        except Exception as e:  # recorded, retried on resume
            rec = {"gid": gid, "sample": s, "error": f"{type(e).__name__}: {e}"[:300]}
        with lock, open(args.out, "a") as f:
            f.write(json.dumps(rec) + "\n")

    with ThreadPoolExecutor(args.workers) as ex:
        list(ex.map(job, jobs))


def report(paths):
    keys = ["parsed", "size", "colors", "corners", "exact"]
    for p in paths:
        recs = {}
        for l in open(p):
            r = json.loads(l)
            if "error" not in r:
                recs[(r["gid"], r["sample"])] = r
        rs = list(recs.values())
        cells = [r["cells"] for r in rs if r.get("cells") is not None]
        out = [f"{k} {100 * sum(bool(r[k]) for r in rs) / len(rs):5.1f}%" for k in keys]
        toks = [r["usage"]["completion_tokens"] for r in rs if r.get("usage")]
        print(f"{Path(p).name:28} n={len(rs):3d}  " + "  ".join(out) +
              f"  cell acc (shape ok, n={len(cells)}) {100 * sum(cells) / max(1, len(cells)):5.1f}%"
              f"  out tok {sum(toks) / max(1, len(toks)) / 1e3:.1f}k")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    ap.add_argument("--score", nargs="+")
    ap.add_argument("--base-url")
    ap.add_argument("--model")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--samples", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=16000)
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    if a.score:
        report(a.score)
    else:
        run(a)
