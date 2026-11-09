#!/usr/bin/env python3
"""E042: teacher-forced NLL of real tool transcripts on the served model, to compare KV-cache formats (fp8 vs nvfp4).

Renders N saved harness transcripts (`*.messages.json.gz`) with the server's /tokenize (tools + V4.1 template), sends
each as a /v1/completions prompt with `prompt_logprobs: 0` and max_tokens 1, and reports the mean NLL of the tokens
after --skip (the model's own turns dominate there). A per-run cache_salt keeps prefix-cache hits out of it.
Usage: kv_nll_probe.py <trace dir> --tag fp8 --out <json> [--n 3] [--max-tokens 24000] [--base-url URL]
"""
import argparse
import gzip
import json
import math
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "evals"))
import run_deepseek_tools as h  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace_dir")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--skip", type=int, default=2000)
    ap.add_argument("--max-tokens", type=int, default=24000)
    ap.add_argument("--base-url", default="http://127.0.0.1:18100")
    a = ap.parse_args()
    tools = h.tools_for(h.parser().parse_args(["--mode", "grid", "--answer-tool", "--model", "dsv41"]))
    kw = {"thinking": True, "reasoning_effort": 50}
    files = sorted(Path(a.trace_dir).glob("*.messages.json.gz"))
    rows = []
    with httpx.Client(timeout=1800) as c:
        for f in files:
            if len(rows) >= a.n:
                break
            m = json.loads(gzip.open(f).read())
            m = m.get("messages", m) if isinstance(m, dict) else m
            ids = c.post(f"{a.base_url}/tokenize", json={"model": "dsv41", "messages": m, "tools": tools,
                                                       "chat_template_kwargs": kw, "add_generation_prompt": False}
                         ).json()["tokens"]
            if len(ids) < a.skip + 4000:
                continue
            ids = ids[:a.max_tokens]
            r = c.post(f"{a.base_url}/v1/completions", json={
                "model": "dsv41", "prompt": ids, "max_tokens": 1, "temperature": 0, "prompt_logprobs": 0,
                "cache_salt": f"nll_{a.tag}_{f.name}"}).json()
            plp = r["choices"][0].get("prompt_logprobs") or r.get("prompt_logprobs")
            nll = []
            for pos in range(a.skip, len(ids)):
                d = plp[pos]
                if d is None:
                    continue
                v = d.get(str(ids[pos]))
                if v is not None:
                    nll.append(-v["logprob"])
            rows.append({"file": f.name, "tokens": len(ids), "scored": len(nll), "nll": sum(nll) / len(nll),
                         "ppl": math.exp(sum(nll) / len(nll))})
            print(a.tag, rows[-1], flush=True)
    tot = sum(r["nll"] * r["scored"] for r in rows) / sum(r["scored"] for r in rows)
    Path(a.out).write_text(json.dumps({"tag": a.tag, "mean_nll": tot, "rows": rows}, indent=1))
    print(a.tag, "mean NLL", round(tot, 5))


if __name__ == "__main__":
    main()
