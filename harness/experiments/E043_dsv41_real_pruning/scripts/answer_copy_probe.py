#!/usr/bin/env python3
"""E043 no-tool round: answer-writing probe on DEV traces (never the eval set).

Feeds the teacher's prompt + reasoning up to and including `</think>` (token-exact from the tokenized dev split) and
lets the model under each mask write the answer itself (T=1, n samples, plus one greedy). The answer is parsed with
the harness parser (evals/arc_harness/arcprize.py parse) and compared with the teacher's parsed answer, on
teacher-correct traces only. This isolates the final grid-writing step from the reasoning.

  answer_copy_probe.py --tok data/tok_nt2_dev --n 48 --samples 4 --out probe.jsonl tag:engram|none:experts|none [...]
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import requests
from tokenizers import Tokenizer

sys.path.insert(0, "/workspace/dsv4/code/arcprize")
from arcprize_parse import parse  # noqa: E402

THINK_END, EOS = 128822, 1


def set_ctrl(ctrl, server, eng, exp):
    c = {"engram_keep": None if eng == "none" else os.path.realpath(eng),
         "expert_keep": None if exp == "none" else os.path.realpath(exp), "observe": None}
    with open(ctrl + ".tmp", "w") as f:
        json.dump(c, f)
    os.replace(ctrl + ".tmp", ctrl)
    requests.post(f"{server}/v1/completions", json={"model": "dsv41", "prompt": [0, 5, 6, 7], "max_tokens": 1})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default="http://127.0.0.1:18001")
    ap.add_argument("--ctrl", default="/dev/shm/e043/ctrlB.json")
    ap.add_argument("--tok", required=True)
    ap.add_argument("--tokenizer", default="/dev/shm/dsv4/DeepSeek-V4.1-Flash-1017/tokenizer.json")
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--samples", type=int, default=4)
    ap.add_argument("--out", required=True)
    ap.add_argument("configs", nargs="+")
    a = ap.parse_args()
    tk = Tokenizer.from_file(a.tokenizer)
    d = Path(a.tok)
    toks = np.memmap(d / "tokens.u32", dtype=np.uint32, mode="r")
    roles = np.memmap(d / "roles.u8", dtype=np.uint8, mode="r")
    items = []
    for line in open(d / "index.jsonl"):
        r = json.loads(line)
        if not r.get("correct"):
            continue
        ids = np.asarray(toks[r["offset"]:r["offset"] + r["length"]], np.int64)
        ro = np.asarray(roles[r["offset"]:r["offset"] + r["length"]])
        ends = np.where((ids == THINK_END) & (ro == 1))[0]
        if len(ends) == 0:
            continue
        p = int(ends[-1]) + 1
        teacher = parse(tk.decode(ids[p:].tolist(), skip_special_tokens=True))
        if teacher is None:
            continue
        items.append({"key": f"{r['task_id']}:{r['pair']}:{r.get('attempt', 0)}", "prompt": ids[:p].tolist(),
                      "teacher": teacher, "answer_tokens": int(len(ids) - p)})
        if len(items) >= a.n:
            break
    print(f"{len(items)} teacher-correct dev traces, mean prompt {np.mean([len(x['prompt']) for x in items]):.0f} "
          f"tokens, answer {np.mean([x['answer_tokens'] for x in items]):.0f} tokens", flush=True)
    for spec in a.configs:
        tag, eng, exp = spec.split(":")
        set_ctrl(a.ctrl, a.server, eng, exp)
        salt = f"copy-{tag}-{time.time()}"

        def run(it):
            out = []
            for temp, n in ((0.0, 1), (1.0, a.samples)):
                body = {"model": "dsv41", "prompt": it["prompt"], "max_tokens": 4000, "temperature": temp, "top_p": 1.0,
                        "n": n, "cache_salt": salt}
                r = requests.post(f"{a.server}/v1/completions", json=body, timeout=1800).json()
                for c in r["choices"]:
                    out.append({"temperature": temp, "exact": parse(c["text"]) == it["teacher"],
                                "parsed": parse(c["text"]) is not None, "finish": c["finish_reason"]})
            return {"config": tag, "key": it["key"], "samples": out}
        t0 = time.time()
        with ThreadPoolExecutor(16) as ex:
            res = list(ex.map(run, items))
        with open(a.out, "a") as f:
            for x in res:
                f.write(json.dumps(x) + "\n")
        g = [s for x in res for s in x["samples"] if s["temperature"] == 0.0]
        t = [s for x in res for s in x["samples"] if s["temperature"] == 1.0]
        print(f"== {tag} ({time.time() - t0:.0f}s): greedy exact {sum(s['exact'] for s in g)}/{len(g)}, "
              f"T=1 exact {sum(s['exact'] for s in t)}/{len(t)} = {sum(s['exact'] for s in t) / len(t):.3f}, "
              f"T=1 unparsed {sum(not s['parsed'] for s in t)}", flush=True)
    set_ctrl(a.ctrl, a.server, "none", "none")


if __name__ == "__main__":
    main()
