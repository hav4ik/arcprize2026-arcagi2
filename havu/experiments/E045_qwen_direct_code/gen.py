"""E045: Qwen3.8-27B without thinking writes a solve(grid) program directly; sample many per task.

Arms (32 samples per task each by default):
  code         all training pairs in the prompt; answer = one ```python block
  rulecode     all training pairs; answer = <rule>...</rule> then <code>...</code>
  rulecode_loo like rulecode, but one training pair is held out of the prompt (rotating over the pairs); the
               checker (validate.py) still requires every training pair, so the held-out pair is a real test
Each request asks for `n` samples of one prompt, so the prompt is prefilled once. Re-running resumes: requests
already in the output file are skipped.

Usage: gen.py --base-url http://127.0.0.1:18045/v1 --out samples.jsonl [--arms code,rulecode,rulecode_loo] [--n 32]
"""
import argparse
import asyncio
import json
import time
from pathlib import Path

from openai import AsyncOpenAI

ROOT = Path(__file__).resolve().parents[2]

INTRO = ("You are solving an ARC-AGI puzzle. Each example maps an input grid to an output grid with the same "
         "hidden rule. Grids are lists of rows; each cell is an integer 0-9 (a colour).\n\n")
TAIL = {
    "code": ("Write a Python function `solve(grid)` that implements the rule: it takes one input grid (a list of "
             "lists of ints) and returns the output grid as a list of lists of ints. It must reproduce every "
             "training example exactly and generalise to the test input(s). numpy is available as np; use only "
             "the standard library and numpy.\n\n"
             "Reply with only the final code, in a single ```python block. No explanation, and do not reason or "
             "explore inside the code: keep comments to a few short lines."),
    "rulecode": ("First state the rule, then implement it. Reply in exactly this format and nothing else:\n"
                 "<rule>\nthe transformation rule, in a few precise sentences\n</rule>\n"
                 "<code>\nPython code defining solve(grid): it takes one input grid (a list of lists of ints) "
                 "and returns the output grid as a list of lists of ints\n</code>\n\n"
                 "The code must reproduce every training example exactly and generalise to the test input(s). "
                 "numpy is available as np; use only the standard library and numpy. Do not reason or explore inside "
                 "the code: keep comments to a few short lines."),
}


def grid_text(g):
    return "[" + ",\n ".join(json.dumps(row, separators=(",", ":")) for row in g) + "]"


def build_prompt(task, arm, held_out=None):
    parts = [INTRO, "Training examples:\n"]
    shown = [p for i, p in enumerate(task["train"]) if i != held_out]
    for i, p in enumerate(shown, 1):
        parts.append(f"\nExample {i}\nInput:\n{grid_text(p['input'])}\nOutput:\n{grid_text(p['output'])}\n")
    parts.append("\nTest input(s) (your solve must handle these too):\n")
    for i, t in enumerate(task["test"], 1):
        parts.append(f"\nTest {i}\nInput:\n{grid_text(t['input'])}\n")
    parts.append("\n" + TAIL["code" if arm == "code" else "rulecode"])
    return "".join(parts)


def plan(tasks, arms, n):
    """[(request_key, task_id, arm, held_out, n_samples)]"""
    out = []
    for tid in sorted(tasks):
        k = len(tasks[tid]["train"])
        for arm in arms:
            if arm == "rulecode_loo":
                if k < 2:  # nothing to hold out; skip rather than duplicate the rulecode arm
                    continue
                per = [n // k + (1 if i < n % k else 0) for i in range(k)]
                out += [(f"{tid}|{arm}|{i}", tid, arm, i, m) for i, m in enumerate(per) if m]
            else:
                out.append((f"{tid}|{arm}|-", tid, arm, None, n))
    return out


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", default="qwen38")
    ap.add_argument("--challenges", default=str(ROOT / "data/arc-agi_evaluation_challenges.json"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--arms", default="code,rulecode,rulecode_loo")
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--max-tokens", type=int, default=3000)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--concurrency", type=int, default=8, help="requests in flight (each carries n samples)")
    ap.add_argument("--tasks", help="comma-separated task ids (default: all)")
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()

    tasks = json.loads(Path(args.challenges).read_text())
    if args.tasks:
        tasks = {t: tasks[t] for t in args.tasks.split(",")}
    reqs = plan(tasks, args.arms.split(","), args.n)
    done = set()
    if args.out.exists():
        done = {json.loads(l)["key"] for l in args.out.open() if l.strip()}
    reqs = [r for r in reqs if r[0] not in done][: args.limit]
    print(f"{len(reqs)} requests to run ({len(done)} already done)", flush=True)

    client = AsyncOpenAI(base_url=args.base_url, api_key="none", timeout=3600, max_retries=2)
    sem, lock, t0 = asyncio.Semaphore(args.concurrency), asyncio.Lock(), time.time()
    stats = {"req": 0, "samples": 0, "tokens": 0}

    async def one(key, tid, arm, held_out, n):
        async with sem:
            prompt = build_prompt(tasks[tid], arm, held_out)
            t = time.time()
            try:
                r = await client.chat.completions.create(
                    model=args.model, messages=[{"role": "user", "content": prompt}], n=n,
                    max_tokens=args.max_tokens, temperature=args.temperature, top_p=args.top_p,
                    extra_body={"top_k": args.top_k, "chat_template_kwargs": {"enable_thinking": False}})
            except Exception as exc:  # keep going; a re-run retries it
                print(f"ERROR {key}: {type(exc).__name__}: {str(exc)[:200]}", flush=True)
                return
            rec = {"key": key, "task": tid, "arm": arm, "held_out": held_out, "n": n,
                   "prompt_tokens": r.usage.prompt_tokens, "completion_tokens": r.usage.completion_tokens,
                   "seconds": round(time.time() - t, 1),
                   "samples": [{"text": c.message.content, "finish": c.finish_reason} for c in r.choices]}
            async with lock:
                with args.out.open("a") as f:
                    f.write(json.dumps(rec) + "\n")
                stats["req"] += 1
                stats["samples"] += n
                stats["tokens"] += r.usage.completion_tokens
                el = time.time() - t0
                if stats["req"] % 10 == 0 or stats["req"] == len(reqs):
                    print(f"{stats['req']}/{len(reqs)} requests, {stats['samples']} samples, "
                          f"{stats['tokens'] / max(el, 1):.0f} out tok/s, {el / 60:.1f} min", flush=True)

    await asyncio.gather(*(one(*r) for r in reqs))


if __name__ == "__main__":
    asyncio.run(main())
