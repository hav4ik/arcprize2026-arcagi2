#!/usr/bin/env python3
"""E042 RW: does each tool turn's prompt extend the previous prompt + generated tokens token-for-token?

Runs the real harness (run_deepseek_tools.solve, eval_rw.sh flags) for a few attempts with a short turn cap, asks
the server for token IDs (`return_token_ids`), and records per turn the prompt IDs the server built and the IDs it
generated. For turns k, k+1 it reports the first position where prompt[k+1] differs from prompt[k] + generated[k],
with the decoded text around it, so the prefix-cache miss can be traced to the exact characters.

Usage (from evals/): prefix_probe.py --out <json> [--tasks N] [--max-turns T]
"""
import argparse
import copy
import json
import sys
import threading
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "evals"))
import run_deepseek_tools as h  # noqa: E402
from openai import OpenAI  # noqa: E402

TAG = "prefix_probe"
HARNESS = [
    "--mode", "grid", "--effort", "low", "--prompt", "lean2sub", "--answer-tool", "--final-mode", "session",
    "--lenient-parse", "--max-tokens", "64000", "--max-total-output", "80000", "--budget-soft", "40000",
    "--budget-notes", "0.01", "--budget-warn", "0.75", "--effort-header", "minimal", "--turns-note", "5",
    "--train-check", "--rescue-code", "--attempts", "1", "--model", "dsv41",
]


class Recorder:
    """OpenAI client stand-in: forwards chat.completions.create and keeps prompt / generated token IDs per turn."""

    def __init__(self, real):
        self.real, self.turns = real, []
        self.chat = self
        self.completions = self

    def create(self, **kw):
        rec = {"n_messages": len(kw["messages"]), "messages": copy.deepcopy(kw["messages"]),
               "prompt_ids": None, "gen_ids": []}
        self.turns.append(rec)
        return _Stream(self.real.chat.completions.create(**kw), rec)


class _Stream:
    def __init__(self, stream, rec):
        self.stream, self.rec = stream, rec

    def __iter__(self):
        for chunk in self.stream:
            extra = chunk.model_extra or {}
            if extra.get("prompt_token_ids") is not None:
                self.rec["prompt_ids"] = list(extra["prompt_token_ids"])
            for ch in chunk.choices or []:
                ids = (ch.model_extra or {}).get("token_ids")
                if ids:
                    self.rec["gen_ids"].extend(ids)
            yield chunk

    def close(self):
        self.stream.close()


def detok(base, ids):
    req = urllib.request.Request(base.rsplit("/v1", 1)[0] + "/detokenize",
                                 data=json.dumps({"model": "dsv41", "tokens": ids}).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=60))["prompt"]


def compare(base, turns):
    out = []
    for k in range(len(turns) - 1):
        a = turns[k]["prompt_ids"] + turns[k]["gen_ids"]
        b = turns[k + 1]["prompt_ids"]
        if a is None or b is None:
            continue
        i = next((j for j, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
        row = {"turn": k + 1, "prompt_k": len(turns[k]["prompt_ids"]), "gen_k": len(turns[k]["gen_ids"]),
               "prompt_k1": len(b), "first_diff": i, "exact_extension": i == len(a),
               "lost_tokens_after_diff": len(a) - i}
        if i < len(a):
            lo = max(0, i - 12)
            row["ids_expected"] = a[lo:i + 12]
            row["ids_actual"] = b[lo:i + 12]
            row["text_expected"] = detok(base, a[lo:i + 12])
            row["text_actual"] = detok(base, b[lo:i + 12])
            row["diff_offset_in_gen"] = i - len(turns[k]["prompt_ids"])
        out.append(row)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--tasks", type=int, default=8)
    ap.add_argument("--max-turns", type=int, default=3)
    ap.add_argument("--base-url", default="http://127.0.0.1:18100/v1")
    ap.add_argument("--sandbox-url", default="http://172.18.0.6:6000")
    a = ap.parse_args()
    extra = {"chat_template_kwargs": {"thinking": True, "reasoning_effort": 50}, "temperature": 1.0, "top_p": 1.0,
             "cache_salt": TAG, "return_token_ids": True}
    args = h.parser().parse_args(HARNESS + ["--max-turns", str(a.max_turns), "--sandbox-url", a.sandbox_url,
                                            "--base-url", a.base_url, "--extra-body", json.dumps(extra),
                                            "--challenges", str(ROOT / "data/arc-agi-2-github/data/evaluation")])
    tasks = h.load_tasks(args.challenges)
    names = sorted(tasks)[:a.tasks]
    results, lock = {}, threading.Lock()

    def work(t):
        client = Recorder(OpenAI(api_key="none", base_url=a.base_url, timeout=7200, max_retries=0))
        try:
            h.solve(client, args, tasks[t], tasks[t]["test"][0]["input"])
            err = None
        except Exception as exc:  # keep whatever turns were recorded
            err = f"{type(exc).__name__}: {exc}"
        rows = compare(a.base_url, [x for x in client.turns if x["prompt_ids"] is not None])
        with lock:
            results[t] = {"error": err, "comparisons": rows,
                          "turns": [{"messages": x["messages"], "prompt_ids": x["prompt_ids"], "gen_ids": x["gen_ids"]}
                                    for x in client.turns]}
            for r in rows:
                print(t, json.dumps({k: v for k, v in r.items() if not k.startswith("ids_")}, ensure_ascii=False),
                      flush=True)

    with ThreadPoolExecutor(len(names)) as ex:
        list(ex.map(work, names))
    Path(a.out).write_text(json.dumps(results))


if __name__ == "__main__":
    main()
