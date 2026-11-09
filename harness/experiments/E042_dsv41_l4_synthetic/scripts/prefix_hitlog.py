#!/usr/bin/env python3
"""E042 RW: per-turn prefix-cache hits under real ARC tool load.

Runs the real harness (run_deepseek_tools.solve, eval_rw.sh flags) on W attempts at once and writes one JSONL line
per turn: wall times, prompt / generated token counts, the server's cached-token count (needs the server flag
--enable-prompt-tokens-details), and how many tokens could have been reused (the previous prompt + generated
tokens, after checking with `return_token_ids` that the new prompt extends them token for token).
Time-boxed with --minutes; lines are flushed as turns finish.

Usage (from evals/): prefix_hitlog.py --out <jsonl> [--workers 40] [--minutes 30] [--salt TAG]
"""
import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "evals"))
import run_deepseek_tools as h  # noqa: E402
from openai import OpenAI  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from prefix_probe import HARNESS  # noqa: E402

T0 = time.time()
LOCK = threading.Lock()


class Recorder:
    def __init__(self, real, name, sink):
        self.real, self.name, self.sink = real, name, sink
        self.prev_ids, self.prev_end, self.k = None, None, 0
        self.chat = self
        self.completions = self

    def create(self, **kw):
        return _Stream(self, self.real.chat.completions.create(**kw), time.time())


class _Stream:
    def __init__(self, rec, stream, t_start):
        self.rec, self.stream, self.t_start = rec, stream, t_start
        self.prompt_ids, self.gen_ids, self.usage, self.t_first = None, [], None, None

    def __iter__(self):
        for chunk in self.stream:
            extra = chunk.model_extra or {}
            if extra.get("prompt_token_ids") is not None:
                self.prompt_ids = list(extra["prompt_token_ids"])
            for ch in chunk.choices or []:
                ids = (ch.model_extra or {}).get("token_ids")
                if ids:
                    if self.t_first is None:
                        self.t_first = time.time()
                    self.gen_ids.extend(ids)
            if chunk.usage is not None:
                self.usage = chunk.usage.model_dump()
            yield chunk
        self._log()

    def close(self):
        self.stream.close()

    def _log(self):
        r, now = self.rec, time.time()
        p = self.prompt_ids or []
        prev = r.prev_ids
        if prev is not None:
            d = next((i for i, (x, y) in enumerate(zip(prev, p)) if x != y), min(len(prev), len(p)))
        else:
            d = None
        cached = ((self.usage or {}).get("prompt_tokens_details") or {}).get("cached_tokens")
        row = {"attempt": r.name, "turn": r.k, "t_start": round(self.t_start - T0, 1),
               "ttft": round(self.t_first - self.t_start, 2) if self.t_first else None,
               "dur": round(now - self.t_start, 1),
               "gap_since_prev": round(self.t_start - r.prev_end, 1) if r.prev_end else None,
               "prompt": len(p), "gen": len(self.gen_ids), "cached": cached,
               "reusable": len(prev) if prev is not None else 0,
               "exact_extension": (d == len(prev)) if prev is not None else None, "first_diff": d}
        r.sink(row)
        r.prev_ids, r.prev_end, r.k = p + self.gen_ids, now, r.k + 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=40)
    ap.add_argument("--minutes", type=float, default=30)
    ap.add_argument("--salt", default="hitlog")
    ap.add_argument("--offset", type=int, default=0, help="skip the first N (task, pair) jobs")
    ap.add_argument("--base-url", default="http://127.0.0.1:18100/v1")
    ap.add_argument("--sandbox-url", default="http://172.18.0.6:6000")
    a = ap.parse_args()
    extra = {"chat_template_kwargs": {"thinking": True, "reasoning_effort": 50}, "temperature": 1.0, "top_p": 1.0,
             "cache_salt": a.salt, "return_token_ids": True}
    args = h.parser().parse_args(HARNESS + ["--max-turns", "40", "--sandbox-url", a.sandbox_url,
                                            "--base-url", a.base_url, "--extra-body", json.dumps(extra),
                                            "--challenges", str(ROOT / "data/arc-agi-2-github/data/evaluation")])
    tasks = h.load_tasks(args.challenges)
    jobs = [(t, p) for t in sorted(tasks) for p in range(len(tasks[t]["test"]))][a.offset:]
    out = open(a.out, "a")

    def sink(row):
        with LOCK:
            out.write(json.dumps(row) + "\n")
            out.flush()

    def work(job):
        t, p = job
        client = Recorder(OpenAI(api_key="none", base_url=a.base_url, timeout=7200, max_retries=0), f"{t}.{p}", sink)
        try:
            h.solve(client, args, tasks[t], tasks[t]["test"][p]["input"])
        except Exception as exc:
            sink({"attempt": f"{t}.{p}", "error": f"{type(exc).__name__}: {exc}"[:300]})

    ex = ThreadPoolExecutor(a.workers)
    for j in jobs:
        ex.submit(work, j)
    time.sleep(a.minutes * 60)
    sink({"stop": round(time.time() - T0, 1)})
    out.close()
    os._exit(0)


if __name__ == "__main__":
    main()
