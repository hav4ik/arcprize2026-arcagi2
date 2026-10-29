#!/usr/bin/env python3
"""E066: free-generation probe of an expert mask on held-out dev prompts (image harness and its text twin).

Teacher-forced NLL missed E043's free-generation failures (broken DSML tool calls at E72, identifier typos such as
`coms` for `comps`). This samples the first assistant turn of each dev rollout's prompt (system + first user message,
images included, the harness tools, effort 50, temperature 1.0) through /v1/chat/completions and checks it:
  valid_call   the server's tool parser returned run_python / submit_answer with a JSON `code` argument
  compiles     that code compiles
  undefined    names the code reads but never binds (assignment, def, import, comprehension, argument), excluding
               builtins and the harness preloads (train, test_input, test_inputs, np): on a first turn the session
               holds nothing else, so these are mostly typos
One unique cache_salt per request. Output: one JSON line per prompt, then a summary per tag.

  gen_probe_mm.py --tok DIR [...] --parquet P [...] --n-seq 25 --seed 0 --tag f72_120 --out probe.jsonl
"""
import argparse
import ast
import builtins
import json
import random
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

import requests

from mm_common import KEY, load_index, load_rows

PRELOADED = {"train", "test_input", "test_inputs", "np", "numpy", "scipy"}
BUILTINS = set(dir(builtins))


def undefined_names(code):
    tree = ast.parse(code)
    bound, loaded = set(PRELOADED), []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            if isinstance(node.ctx, ast.Load):
                loaded.append(node.id)
            else:
                bound.add(node.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound.add(node.name)
        elif isinstance(node, ast.arg):
            bound.add(node.arg)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for a in node.names:
                bound.add((a.asname or a.name).split(".")[0])
        elif isinstance(node, ast.ExceptHandler) and node.name:
            bound.add(node.name)
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            bound.update(node.names)
    return sorted({n for n in loaded if n not in bound and n not in BUILTINS})


def first_turn_body(row, model, effort, max_tokens):
    messages = json.loads(row["messages"])
    first_user = next(i for i, m in enumerate(messages) if m["role"] == "user")
    return {"model": model, "messages": messages[: first_user + 1], "tools": json.loads(row["tools"]),
            "max_tokens": max_tokens, "temperature": 1.0, "top_p": 1.0,
            "chat_template_kwargs": {"thinking": True, "reasoning_effort": effort}, "cache_salt": uuid.uuid4().hex}


def probe(server, body):
    r = requests.post(f"{server}/v1/chat/completions", json=body, timeout=7200)
    r.raise_for_status()
    j = r.json()
    ch = j["choices"][0]
    msg = ch.get("message") or {}
    calls = msg.get("tool_calls") or []
    res = {"finish": ch.get("finish_reason"), "completion_tokens": j.get("usage", {}).get("completion_tokens"),
           "n_calls": len(calls), "valid_call": False, "compiles": None, "undefined": None, "tool": None}
    if calls:
        fn = calls[0].get("function") or {}
        res["tool"] = fn.get("name")
        try:
            code = json.loads(fn.get("arguments") or "{}").get("code")
        except json.JSONDecodeError:
            code = None
        if fn.get("name") in ("run_python", "submit_answer") and isinstance(code, str):
            res["valid_call"] = True
            try:
                res["undefined"] = undefined_names(code)
                res["compiles"] = True
            except SyntaxError:
                res["compiles"] = False
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tok", nargs="+", required=True)
    ap.add_argument("--parquet", nargs="+", required=True)
    ap.add_argument("--n-seq", type=int, default=25, help="prompts, one per held-out family")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=16000)
    ap.add_argument("--server", default="http://127.0.0.1:18001")
    ap.add_argument("--model", default="dsv41")
    ap.add_argument("--effort", type=int, default=50)
    ap.add_argument("--workers", type=int, default=25)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    recs = load_index(a.tok)
    recs.sort(key=lambda r: (r["task_id"], r["pair"], r["attempt"], r.get("batch") or "", r["dir"]))
    rng = random.Random(a.seed)
    by_fam = {}
    for r in recs:
        by_fam.setdefault(r["parent"], []).append(r)
    fams = sorted(by_fam)
    rng.shuffle(fams)
    recs = [rng.choice(by_fam[f]) for f in fams[: a.n_seq]]
    rows = load_rows(a.parquet)
    t0 = time.time()

    def job(r):
        res = probe(a.server, first_turn_body(rows[tuple(r[k] for k in KEY)], a.model, a.effort, a.max_tokens))
        res.update({k: r.get(k) for k in ("task_id", "pair", "attempt", "batch")})
        return res

    with ThreadPoolExecutor(a.workers) as ex:
        out = list(ex.map(job, recs))
    with open(a.out, "a") as f:
        for r in out:
            f.write(json.dumps({"tag": a.tag, **r}) + "\n")
    n = len(out)
    valid = [r for r in out if r["valid_call"]]
    comp = [r for r in valid if r["compiles"]]
    print(json.dumps({"tag": a.tag, "prompts": n, "valid_call": round(len(valid) / n, 3),
                      "compiles_of_valid": round(len(comp) / max(1, len(valid)), 3),
                      "with_undefined_names": round(sum(bool(r["undefined"]) for r in comp) / max(1, len(comp)), 3),
                      "length_cut": sum(r["finish"] == "length" for r in out),
                      "mean_tokens": round(sum(r["completion_tokens"] or 0 for r in out) / n),
                      "seconds": round(time.time() - t0, 1)}))


if __name__ == "__main__":
    main()
