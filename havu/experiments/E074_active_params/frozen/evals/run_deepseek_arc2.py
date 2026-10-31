"""Score a DeepSeek model (via the DeepSeek API) on ARC-AGI-2 eval problems.

Problems and prompts come from the ARC-AGI-2 part of NVIDIA's eval set (HF dataset
nvidia/Nemotron-RL-ARC-AGI-v1, validation split): 120 problems, one test input each.

Uses NVIDIA's rendered prompts (`responses_create_params.input`) as-is, one call per problem
(pass@1 per sample; --samples N gives N independent samples per problem -> pass@N and mean accuracy).
Results are appended to a JSONL file so interrupted runs resume where they left off.

    .venv/bin/python evals/run_deepseek_arc2.py --model deepseek-flash --effort max --limit 5
"""
import argparse
import gzip
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI

ROOT = Path(__file__).resolve().parents[1]
VAL = ROOT / "data/nemotron_rl/transductive_validation.jsonl"
V2_EVAL = ROOT / "data/arc-agi_evaluation_challenges.json"


def api_key():
    for line in (ROOT / "SECRETS.yaml").read_text().splitlines():
        if line.startswith("deepseek_api_key:"):
            return line.split(":", 1)[1].strip().strip("'\"")
    raise KeyError("deepseek_api_key not found in SECRETS.yaml")


def last_boxed(text):
    """Content of the last \\boxed{...}, handling nested braces."""
    start = text.rfind("\\boxed{")
    if start < 0:
        return None
    i, depth = start + len("\\boxed{"), 1
    for j in range(i, len(text)):
        depth += {"{": 1, "}": -1}.get(text[j], 0)
        if depth == 0:
            return text[i:j]
    return None


def parse_grid(text):
    """Grid from the final answer: rows on separate lines, values separated by spaces (or packed digits)."""
    if not text:
        return None
    box = last_boxed(text)
    if box is None:
        return None
    box = box.strip()
    if box.startswith("["):
        try:
            g = json.loads(box)
            if isinstance(g, list) and all(isinstance(r, list) for r in g):
                return g
        except json.JSONDecodeError:
            pass
    box = re.sub(r"\\text\{([^}]*)\}", r"\1", box).replace("\\\\", "\n")
    rows = []
    for line in box.splitlines():
        line = line.strip().strip("`")
        if not line:
            continue
        toks = line.split() if " " in line else list(line)
        if not all(t.isdigit() and len(t) == 1 for t in toks):
            return None
        rows.append([int(t) for t in toks])
    return rows or None


def save_trace(out_jsonl, name, reasoning):
    """Store the full chain of thought gzipped next to the results file; returns its repo-relative path."""
    d = out_jsonl.with_suffix("")  # evals/results/<run>/
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{name}.reasoning.txt.gz"
    with gzip.open(path, "wt") as f:
        f.write(reasoning)
    return str(path.relative_to(ROOT))


def call(client, model, effort, messages, max_tokens):
    """Streamed call (long reasoning would otherwise risk idle-connection timeouts).
    Returns the full reasoning under "reasoning"; callers save it with save_trace and drop it from the record."""
    t0 = time.time()
    stream = client.chat.completions.create(
        model=model, messages=messages, max_tokens=max_tokens, stream=True,
        stream_options={"include_usage": True}, extra_body={"reasoning_effort": effort, "thinking": {"type": "enabled"}},
    )
    content, reasoning, usage, finish = [], [], None, None
    for chunk in stream:
        if chunk.usage:
            usage = chunk.usage.model_dump()
        if chunk.choices:
            d = chunk.choices[0].delta
            content.append(d.content or "")
            reasoning.append(getattr(d, "reasoning_content", None) or "")
            finish = chunk.choices[0].finish_reason or finish
    reasoning = "".join(reasoning)
    return {"content": "".join(content), "reasoning": reasoning, "reasoning_chars": len(reasoning),
            "usage": usage, "finish_reason": finish, "seconds": round(time.time() - t0, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="deepseek-flash")
    ap.add_argument("--effort", default="max", choices=["low", "high", "max"])
    ap.add_argument("--samples", type=int, default=1)
    ap.add_argument("--max-tokens", type=int, default=200_000)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None, help="only the first N problems (pilot runs)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    v2_ids = set(json.load(open(V2_EVAL)))
    rows = [r for r in map(json.loads, open(VAL)) if r["task_id"] in v2_ids]
    rows.sort(key=lambda r: r["task_id"])
    rows = rows[: args.limit] if args.limit else rows
    out = Path(args.out or ROOT / f"evals/results/arc2_{args.model}_{args.effort}.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)

    done = set()
    if out.exists():
        done = {(d["task_id"], d["sample"]) for d in map(json.loads, open(out)) if "error" not in d}
    jobs = [(r, s) for r in rows for s in range(args.samples) if (r["task_id"], s) not in done]
    print(f"{len(rows)} problems x {args.samples} samples, {len(jobs)} calls to run -> {out}", flush=True)

    client = OpenAI(api_key=api_key(), base_url="https://api.deepseek.com", timeout=3600, max_retries=3)
    lock = threading.Lock()

    def work(r, s):
        rec = {"task_id": r["task_id"], "sample": s, "model": args.model, "effort": args.effort,
               "difficulty": r["difficulty"]}
        try:
            res = call(client, args.model, args.effort, r["responses_create_params"]["input"], args.max_tokens)
            pred = parse_grid(res["content"])
            res["reasoning_path"] = save_trace(out, f'{r["task_id"]}_s{s}', res.pop("reasoning"))
            rec.update(res, pred=pred, correct=pred == r["expected_output"])
        except Exception as e:  # noqa: BLE001 - logged and retried on resume
            rec["error"] = f"{type(e).__name__}: {e}"[:500]
        with lock, open(out, "a") as f:
            f.write(json.dumps(rec) + "\n")
        return rec

    with ThreadPoolExecutor(args.workers) as ex:
        for i, fut in enumerate(as_completed([ex.submit(work, r, s) for r, s in jobs]), 1):
            rec = fut.result()
            status = rec.get("error") or f"correct={rec['correct']} {rec['seconds']}s " \
                f"{(rec['usage'] or {}).get('completion_tokens')} tok {rec['finish_reason']}"
            print(f"[{i}/{len(jobs)}] {rec['task_id']}#{rec['sample']} {status}", flush=True)

    summarize(out, {r["task_id"] for r in rows})


def summarize(path, task_ids):
    recs = [d for d in map(json.loads, open(path)) if "error" not in d and d["task_id"] in task_ids]
    by_task = {}
    for d in recs:
        by_task.setdefault(d["task_id"], []).append(d["correct"])
    n = len(task_ids)
    mean = sum(sum(v) / len(v) for v in by_task.values()) / n
    passk = sum(any(v) for v in by_task.values()) / n
    toks = [d["usage"]["completion_tokens"] for d in recs if d.get("usage")]
    print(f"\n{path.name}: {len(by_task)}/{n} problems scored, {len(recs)} samples")
    print(f"  mean accuracy (pass@1): {mean:.1%}   pass@{max(map(len, by_task.values()), default=0)}: {passk:.1%}")
    print(f"  unparsed answers: {sum(d['pred'] is None for d in recs)}   "
          f"truncated (length): {sum(d['finish_reason'] == 'length' for d in recs)}")
    if toks:
        print(f"  completion tokens: mean {sum(toks) / len(toks):,.0f}, max {max(toks):,}, total {sum(toks):,}")


if __name__ == "__main__":
    main()
