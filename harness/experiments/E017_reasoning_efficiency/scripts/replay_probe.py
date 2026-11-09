"""Replay every run_python cell of an attempt in a fresh sandbox session and probe the session after each cell.

No API calls. Uses the local code sandbox (evals/sandbox_docker/start.sh). Probe after each cell:
  - every global callable named solve*/transform*/apply*/predict* (deep copies of inputs, stdout muted):
    number of train pairs reproduced exactly, and whether its test output equals the truth;
  - whether `answer` / `ans` equals the truth.
Truth never enters the session: the probe returns md5 hashes, compared on the host.
Output: experiments/E017_reasoning_efficiency/replay.jsonl (one row per attempt, per-cell timeline).

Usage: .venv/bin/python -I experiments/E017_reasoning_efficiency/scripts/replay_probe.py --url http://172.18.0.2:6000 [--workers 12]
"""
import argparse
import gzip
import hashlib
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "evals"))
from sandbox_client import SandboxSession, tool_output  # noqa: E402

OUT = ROOT / "experiments/E017_reasoning_efficiency"
TASKS = ROOT / "data/arc-agi-2-github/data/evaluation"

PROBE = r'''
def _e17_probe():
    import contextlib, io, json, copy, hashlib, re, time
    import numpy as _np
    def _h(v):
        try:
            if hasattr(v, 'tolist'):
                v = v.tolist()
            v = [list(map(int, (r.tolist() if hasattr(r, 'tolist') else r))) for r in v]
            return hashlib.md5(json.dumps(v).encode()).hexdigest()
        except Exception:
            return None
    g = globals()
    _ins = json.loads(%(ins)r)
    _test = json.loads(%(test)r)
    keep = {k: g.get(k) for k in ('answer', 'ans', 'train', 'test_input')}
    res = {'cands': {}, 'answer': None, 'ans': None}
    for k in ('answer', 'ans'):
        if k in g:
            res[k] = _h(g[k])
    names = [k for k, v in list(g.items()) if callable(v) and not k.startswith('_')
             and re.match(r'^(solve|transform|apply|predict)', k) and getattr(v, '__module__', None) == '__main__']
    t0 = time.time()
    for k in names[:6]:
        if time.time() - t0 > 20:
            break
        outs = []
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            for x in _ins + [_test]:
                try:
                    outs.append(_h(g[k](copy.deepcopy(x))))
                except BaseException as e:
                    if isinstance(e, KeyboardInterrupt):
                        raise
                    outs.append(None)
        res['cands'][k] = outs
    for k, v in keep.items():
        if v is not None:
            g[k] = v
    print('E17PROBE:' + json.dumps(res))
try:
    _e17_probe()
finally:
    del _e17_probe
'''


def h(grid):
    return hashlib.md5(json.dumps([list(map(int, r)) for r in grid]).encode()).hexdigest()


def cells_of(path):
    d = json.load(gzip.open(path))
    msgs = d["messages"]
    usage = [e for e in d["events"] if "usage" in e]
    tool_by_id = {m["tool_call_id"]: m["content"] for m in msgs if m["role"] == "tool"}
    cum, ai, cells = 0, 0, []
    for m in msgs:
        if m["role"] != "assistant":
            continue
        cum += usage[ai]["usage"].get("completion_tokens", 0) if ai < len(usage) else 0
        ai += 1
        for c in m.get("tool_calls") or []:
            try:
                code = json.loads(c["function"]["arguments"]).get("code", "")
            except Exception:
                code = None
            cells.append({"turn": ai, "cum_tok": cum, "name": c["function"]["name"], "code": code,
                          "orig": tool_by_id.get(c["id"], "")})
    return cells


def replay(row, url):
    task = json.loads((TASKS / f"{row['task_id']}.json").read_text())
    test = task["test"][row["pair"]]
    truth = h(test["output"])
    train_truth = [h(p["output"]) for p in task["train"]]
    ins = json.dumps([p["input"] for p in task["train"]])
    probe = PROBE % {"ins": ins, "test": json.dumps(test["input"])}
    timeline = []
    with SandboxSession(url, task["train"], test["input"], 30, 16000) as s:
        for i, c in enumerate(cells_of(row["trace_path"])):
            if c["name"] != "run_python" or c["code"] is None:
                continue
            r = s.execute(c["code"])
            out = tool_output(r, 16000)
            orig = c["orig"].split("\n\n[budget:")[0]
            p = s.execute(probe)
            pres = None
            for line in reversed(p.get("stdout", "").splitlines()):
                if line.startswith("E17PROBE:"):
                    pres = json.loads(line[9:])
                    break
            ent = {"turn": c["turn"], "cum_tok": c["cum_tok"], "status": r["process_status"],
                   "same_output": out.strip() == orig.strip(), "probe": pres is not None}
            if pres:
                best, full, full_ok, any_ok = 0, [], False, False
                for k, outs in pres["cands"].items():
                    n = sum(o == t for o, t in zip(outs[:-1], train_truth))
                    best = max(best, n)
                    ok = outs[-1] == truth
                    any_ok |= ok
                    if n == len(train_truth):
                        full.append(k)
                        full_ok |= ok
                ent.update(best_train=best, n_train=len(train_truth), full_train=full, full_train_test_ok=full_ok,
                           any_cand_test_ok=any_ok, answer_ok=pres["answer"] == truth or pres["ans"] == truth,
                           answer_set=pres["answer"] is not None)
            timeline.append(ent)
    return {k: row[k] for k in ("run", "task_id", "pair", "attempt", "outcome", "tokens")} | {"timeline": timeline}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--sample-only", action="store_true", help="only attempts in sample.jsonl, written to replay_sample.jsonl")
    args = ap.parse_args()
    rows = [json.loads(l) for l in open(OUT / "attempts_index.jsonl")]
    outp = OUT / ("replay_sample.jsonl" if args.sample_only else "replay.jsonl")
    if args.sample_only:
        want = {(s["run"], s["task_id"], s["pair"], s["attempt"]) for s in map(json.loads, open(OUT / "sample.jsonl"))}
        rows = [r for r in rows if (r["run"], r["task_id"], r["pair"], r["attempt"]) in want]
    done = set()
    if outp.exists():
        for l in open(outp):
            r = json.loads(l)
            done.add((r["run"], r["task_id"], r["pair"], r["attempt"]))
    todo = [r for r in rows if (r["run"], r["task_id"], r["pair"], r["attempt"]) not in done]
    if args.limit:
        todo = todo[:args.limit]
    lock = threading.Lock()
    with ThreadPoolExecutor(args.workers) as ex, open(outp, "a") as f:
        futs = {ex.submit(replay, r, args.url): r for r in todo}
        for n, fu in enumerate(as_completed(futs), 1):
            r = futs[fu]
            try:
                res = fu.result()
            except Exception as e:
                print("ERR", r["task_id"], r["pair"], r["attempt"], type(e).__name__, str(e)[:200], flush=True)
                continue
            with lock:
                f.write(json.dumps(res) + "\n")
                f.flush()
            if n % 25 == 0:
                print(n, "/", len(todo), flush=True)


if __name__ == "__main__":
    main()
