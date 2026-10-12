"""Replicate ARC Prize's leaderboard setup for a DeepSeek model on the official ARC-AGI-2 public eval.

Follows github.com/arcprize/arc-agi-benchmarking as closely as is public:
- the 120 tasks / 167 test pairs from github.com/arcprize/ARC-AGI-2 (data/arc-agi-2-github);
- their prompt verbatim (notes/prompts/arcprize_official_system_prompt.txt) as a single user message,
  with grids as json.dumps;
- their parser: \\boxed{json}, else the last balanced bracket block that parses as a list of lists;
- 2 independent attempts per test pair; a pair counts if either attempt matches exactly; a task's
  score is the fraction of its test pairs solved; the final score is the mean over the 120 tasks.
ARC Prize's provider, max_tokens and temperature for V4 are unpublished; we use the DeepSeek API defaults.

    .venv/bin/python evals/run_deepseek_arcprize.py --model deepseek-flash --effort max
"""
import argparse
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI

from run_deepseek_arc2 import api_key, call, save_trace

ROOT = Path(__file__).resolve().parents[1]
# Prompt and parser live in arc_harness so the tool-use harness shares them; re-exported here.
from arc_harness.arcprize import TASKS, TEMPLATE, build_prompt, parse  # noqa: E402,F401


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="deepseek-flash")
    ap.add_argument("--effort", default="max", choices=["low", "high", "max"])
    ap.add_argument("--attempts", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=200_000)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--tasks", default=None, help="comma-separated task ids to run (default: all)")
    ap.add_argument("--tag", default="", help="suffix for the results file, e.g. 'traces'")
    args = ap.parse_args()

    tasks = {p.stem: json.load(open(p)) for p in sorted(TASKS.glob("*.json"))}
    if args.tasks:
        tasks = {k: v for k, v in tasks.items() if k in set(args.tasks.split(","))}
    if args.limit:
        tasks = dict(list(tasks.items())[: args.limit])
    tag = f"_{args.tag}" if args.tag else ""
    out = ROOT / f"evals/results/arcprize_{args.model}_{args.effort}{tag}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        done = {(d["task_id"], d["pair"], d["attempt"]) for d in map(json.loads, open(out)) if "error" not in d}
    jobs = [(t, p, a) for t, task in tasks.items() for p in range(len(task["test"]))
            for a in range(args.attempts) if (t, p, a) not in done]
    print(f"{len(tasks)} tasks, {sum(len(t['test']) for t in tasks.values())} test pairs x {args.attempts} "
          f"attempts, {len(jobs)} calls to run -> {out}", flush=True)

    client = OpenAI(api_key=api_key(), base_url="https://api.deepseek.com", timeout=3600, max_retries=3)
    lock = threading.Lock()

    def work(t, p, a):
        task = tasks[t]
        rec = {"task_id": t, "pair": p, "attempt": a, "model": args.model, "effort": args.effort}
        try:
            msgs = [{"role": "user", "content": build_prompt(task["train"], task["test"][p]["input"])}]
            res = call(client, args.model, args.effort, msgs, args.max_tokens)
            pred = parse(res["content"])
            res["reasoning_path"] = save_trace(out, f"{t}_p{p}_a{a}", res.pop("reasoning"))
            rec.update(res, pred=pred, correct=pred == task["test"][p]["output"])
        except Exception as e:  # noqa: BLE001 - logged and retried on resume
            rec["error"] = f"{type(e).__name__}: {e}"[:500]
        with lock, open(out, "a") as f:
            f.write(json.dumps(rec) + "\n")
        return rec

    with ThreadPoolExecutor(args.workers) as ex:
        for i, fut in enumerate(as_completed([ex.submit(work, *j) for j in jobs]), 1):
            r = fut.result()
            status = r.get("error") or f"correct={r['correct']} {r['seconds']}s " \
                f"{(r['usage'] or {}).get('completion_tokens')} tok {r['finish_reason']}"
            print(f"[{i}/{len(jobs)}] {r['task_id']}.{r['pair']}#{r['attempt']} {status}", flush=True)

    summarize(out, tasks)


def summarize(path, tasks):
    recs = [d for d in map(json.loads, open(path)) if "error" not in d and d["task_id"] in tasks]
    ok = {}
    for d in recs:
        ok.setdefault((d["task_id"], d["pair"]), {})[d["attempt"]] = d["correct"]

    def score(k):  # mean over tasks of the fraction of test pairs solved within the first k attempts
        return sum(sum(any(ok.get((t, p), {}).get(a, False) for a in range(k)) for p in range(len(task["test"])))
                   / len(task["test"]) for t, task in tasks.items()) / len(tasks)

    toks = [d["usage"]["completion_tokens"] for d in recs if d.get("usage")]
    print(f"\n{path.name}: {len(recs)} attempts over {len(ok)} test pairs")
    print(f"  pass@1 (attempt 1 only): {score(1):.1%}   pass@2: {score(2):.1%}")
    print(f"  unparsed: {sum(d['pred'] is None for d in recs)}   "
          f"truncated: {sum(d['finish_reason'] == 'length' for d in recs)}")
    if toks:
        print(f"  completion tokens: mean {sum(toks) / len(toks):,.0f}, total {sum(toks):,}")


if __name__ == "__main__":
    main()
