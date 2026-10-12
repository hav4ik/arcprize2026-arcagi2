"""E012: Qwen3.8-27B with the E009 Python tool harness on a self-hosted OpenAI-compatible server (raw completions).

Agent loop: evals/arc_harness/agent_qwen.py. Executors: evals/arc_harness/executor.py (Docker sandbox via
--sandbox-url / ARC_SANDBOX_URL, or --local-exec). ARC Prize prompt + lean2sub suffix, task-weighted scoring.
Re-running resumes (finished (task, pair, attempt) are skipped; errored ones are retried).

    .venv/bin/python evals/run_qwen_tools.py --verify-template
    .venv/bin/python evals/run_qwen_tools.py --pilot --tier xhigh --hard 80000 --soft 40000 --attempts 2 --tag x80s40
"""
import argparse
import asyncio
import gzip
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evals"))
from arc_harness import agent_qwen as aq  # noqa: E402
from arc_harness.arcprize import TASKS  # noqa: E402
from arc_harness.executor import local_factory, sandbox_factory  # noqa: E402

MODEL = "nvidia/Qwen3.8-27B-NVFP4"
ENDPOINT = "vastai-4xrtx6000s-qwen27b-b"
PILOT_TASKS = 15


def endpoint(name):
    import yaml
    for e in yaml.safe_load((ROOT / "ENDPOINTS.yaml").read_text()):
        if e.get("name") == name:
            return f"http://{e['addr']}/v1", e["token"]
    raise KeyError(name)


def load_tasks(args):
    tasks = {p.stem: json.loads(p.read_text()) for p in sorted(TASKS.glob("*.json"))}
    if args.pilot:
        tasks = dict(list(tasks.items())[:PILOT_TASKS])
    if args.tasks:
        tasks = {t: tasks[t] for t in args.tasks.split(",")}
    if args.limit_tasks:
        tasks = dict(list(tasks.items())[:args.limit_tasks])
    return tasks


def selected_pairs(args):
    text = args.pairs or (Path(args.pairs_file).read_text() if args.pairs_file else "")
    items = [x.strip() for x in text.replace("\n", ",").split(",") if x.strip()]
    return {(t, int(p)) for t, p in (x.rsplit(":", 1) for x in items)} or None


async def verify_template(args, tasks):
    """Token count of the server's chat template vs our rendering, for a 2-turn tool conversation, per tier."""
    from openai import AsyncOpenAI
    base, token = endpoint(args.endpoint)
    t = next(iter(tasks))
    task = tasks[t]
    ok = True
    async with AsyncOpenAI(api_key=token, base_url=base, timeout=600, max_retries=2) as client:
        for tier in ("xhigh", "medium", "low"):
            cfg = aq.AgentConfig(tier=tier)
            user = aq.user_content(task["train"], task["test"][0]["input"], cfg.prompt)
            code = "import numpy as np\nprint(np.array(train[0]['input']).shape)"
            messages = [
                {"role": "user", "content": user},
                {"role": "assistant", "content": "Let me look.", "reasoning_content": "I should inspect shapes.",
                 "tool_calls": [{"id": "c1", "type": "function", "function": {
                     "name": "run_python", "arguments": json.dumps({"code": code})}}]},
                {"role": "tool", "tool_call_id": "c1", "content": "[completed]\n(3, 3)"},
            ]
            chat = await client.chat.completions.create(
                model=args.model, messages=messages, tools=aq.tool_specs(cfg), max_tokens=1,
                extra_body={"chat_template_kwargs": {"reasoning_effort": tier}})
            # SGLang re-serializes tools through pydantic (adds "strict": false, "defer_loading": null and puts
            # description first); render that form for the comparison. The agent itself uses the plain dicts
            # (what transformers' apply_chat_template renders), which differ only in those keys.
            served = [{"type": "function", "function": {"description": t["function"]["description"],
                       "name": t["function"]["name"], "parameters": t["function"]["parameters"], "strict": False},
                       "defer_loading": None} for t in aq.tool_specs(cfg)]
            raw_prompt = (aq.render_system(tier, served) + aq.render_user(user)
                          + aq.render_assistant("I should inspect shapes.", "Let me look.",
                                                [("run_python", {"code": code})])
                          + aq.render_tool_responses(["[completed]\n(3, 3)"]) + aq.GEN_PROMPT)
            raw = await client.completions.create(model=args.model, prompt=raw_prompt, max_tokens=1)
            a, b = chat.usage.prompt_tokens, raw.usage.prompt_tokens
            ok &= a == b
            print(f"{t} {tier:6s} chat={a} raw={b} {'OK' if a == b else 'MISMATCH'}", flush=True)
    print("template verified" if ok else "TEMPLATE MISMATCH", flush=True)
    return ok


def summarize(records, tasks, attempts, pairs=None):
    latest = {(r["task_id"], r["pair"], r["attempt"]): r for r in records if "error" not in r}
    per_task = {t: [p for p in range(len(task["test"])) if pairs is None or (t, p) in pairs] for t, task in tasks.items()}

    def score(k):
        return sum(sum(any(latest.get((t, p, a), {}).get("correct", False) for a in range(k))
                       for p in ps) / len(ps) for t, ps in per_task.items()) / len(per_task)
    if not latest:
        print("no finished attempts", flush=True)
        return
    toks = [r["usage"]["completion"] for r in latest.values()]
    src = {}
    for r in latest.values():
        src[r.get("answer_source")] = src.get(r.get("answer_source"), 0) + 1
    print(f"pass@1 {score(1):.1%}  pass@{attempts} {score(attempts):.1%}  ({len(tasks)} tasks, {len(latest)} attempts)"
          f"; tokens mean {statistics.mean(toks) / 1000:.1f}k median {statistics.median(toks) / 1000:.1f}k; "
          f"sources {src}; parse failures {sum(r['parse_failures'] for r in latest.values())}", flush=True)


async def run(args, cfg, tasks, jobs, out, config):
    base, token = endpoint(args.endpoint)
    llm = aq.OpenAICompletionsLLM(base, token, args.model, cfg.temperature, cfg.top_p, cfg.top_k)
    if args.local_exec:
        make = local_factory(cfg.tool_timeout, cfg.max_tool_output)
    else:
        make = sandbox_factory(args.sandbox_url, cfg.tool_timeout, cfg.max_tool_output)
    trace_dir = out.parent / (out.stem + "_traces")
    trace_dir.mkdir(parents=True, exist_ok=True)
    queue = asyncio.Queue()
    for j in jobs:
        queue.put_nowait(j)
    stats = {"done": 0, "correct": 0, "tokens": 0}
    t_start = time.monotonic()

    async def worker():
        while True:
            try:
                t, p, a = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            task = tasks[t]
            transcript, result = await aq.solve(llm, make, task["train"], task["test"][p]["input"], cfg)
            if "error" in result:
                result["error"] = result["error"].replace(token, "[redacted]")
            path = trace_dir / f"{t}_p{p}_a{a}.txt.gz"
            with gzip.open(path, "wt") as f:
                f.write(transcript)
            rec = {"task_id": t, "pair": p, "attempt": a, "config": config, **result,
                   "correct": result["pred"] == task["test"][p]["output"], "trace": str(path.relative_to(ROOT))}
            with out.open("a") as f:
                f.write(json.dumps(rec) + "\n")
            stats["done"] += 1
            stats["correct"] += rec["correct"]
            stats["tokens"] += result["usage"]["completion"]
            el = time.monotonic() - t_start
            print(f"[{stats['done']}/{len(jobs)}] {t}.{p}#{a} correct={rec['correct']} src={rec['answer_source']} "
                  f"tok={result['usage']['completion']} turns={result['turns']} calls={result['tool_calls']} "
                  f"perr={result['parse_failures']} stop={result['stop_reason']} {result['seconds']}s "
                  f"| {stats['correct']} correct, {stats['tokens'] / el:.0f} tok/s"
                  f"{' ERROR ' + result['error'] if 'error' in result else ''}", flush=True)

    import signal
    from concurrent.futures import ThreadPoolExecutor
    main_task = asyncio.current_task()
    loop = asyncio.get_running_loop()
    # Executor calls are blocking and run via asyncio.to_thread: the default pool (min(32, cpus + 4) threads) would
    # queue tool calls behind each other at high concurrency, so give every attempt its own thread.
    loop.set_default_executor(ThreadPoolExecutor(args.concurrency + 4))
    for sig in (signal.SIGTERM, signal.SIGINT):  # cancel cleanly so every attempt closes its sandbox session
        loop.add_signal_handler(sig, main_task.cancel)
    try:
        await asyncio.gather(*(worker() for _ in range(min(args.concurrency, len(jobs)))))
    finally:
        await llm.close()
    el = time.monotonic() - t_start
    print(f"finished {stats['done']} attempts, {stats['correct']} correct, {stats['tokens']} tokens in "
          f"{el / 60:.1f} min ({stats['tokens'] / el:.0f} tok/s)", flush=True)


def main():
    import os
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tier", default="xhigh", choices=list(aq.TIER_SYSTEM))
    ap.add_argument("--hard", type=int, default=80000, help="hard cap: generated tokens per attempt")
    ap.add_argument("--soft", type=int, default=40000, help="soft target in budget notes (0 = none)")
    ap.add_argument("--budget-notes", type=float, default=0.01)
    ap.add_argument("--budget-warn", type=float, default=0.75)
    ap.add_argument("--chunk-tokens", type=int, default=16384)
    ap.add_argument("--max-turns", type=int, default=60)
    ap.add_argument("--verify-submit", action="store_true")
    ap.add_argument("--soft-wording", default="gentle", choices=["gentle", "strong"])
    ap.add_argument("--force-submit-tokens", type=int, default=256)
    ap.add_argument("--attempts", type=int, default=2)
    ap.add_argument("--concurrency", type=int, default=12, help="concurrent attempts = sandbox sessions")
    ap.add_argument("--pilot", action="store_true", help=f"first {PILOT_TASKS} sorted eval tasks (26 pairs)")
    ap.add_argument("--tasks")
    ap.add_argument("--pairs", help="comma-separated task_id:pair")
    ap.add_argument("--pairs-file", help="file with comma/newline-separated task_id:pair (e.g. E009 dev_pairs.txt)")
    ap.add_argument("--limit-tasks", type=int)
    ap.add_argument("--tag", default="")
    ap.add_argument("--endpoint", default=ENDPOINT)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--sandbox-url", default=os.environ.get("ARC_SANDBOX_URL"))
    ap.add_argument("--local-exec", action="store_true", help="LocalExecutor subprocesses instead of the sandbox")
    ap.add_argument("--skip-done-in", action="append", default=[],
                    help="also skip (task, pair, attempt) finished without error in this results file (repeatable)")
    ap.add_argument("--verify-template", action="store_true")
    ap.add_argument("--summary", action="store_true", help="only print the summary of the results file")
    args = ap.parse_args()
    tasks = load_tasks(args)
    pairs = selected_pairs(args)
    if pairs:
        tasks = {t: task for t, task in tasks.items() if any(t == u for u, _ in pairs)}
    if args.verify_template:
        sys.exit(0 if asyncio.run(verify_template(args, tasks)) else 1)
    cfg = aq.AgentConfig(tier=args.tier, max_total_output=args.hard, budget_soft=args.soft,
                         budget_notes=args.budget_notes, budget_warn=args.budget_warn, chunk_tokens=args.chunk_tokens,
                         max_turns=args.max_turns, verify_submit=args.verify_submit, soft_wording=args.soft_wording,
                         force_submit_tokens=args.force_submit_tokens)
    if cfg.budget_soft and cfg.budget_soft >= cfg.max_total_output:
        ap.error("--soft must be below --hard")
    config = aq.config_dict(cfg, args.model)
    out = ROOT / f"evals/results/qwen_tools_{args.tier}_{args.hard // 1000}k{'_' + args.tag if args.tag else ''}.jsonl"
    records = [json.loads(line) for line in out.read_text().splitlines() if line.strip()] if out.exists() else []
    if any(r.get("config") != config for r in records):
        ap.error("existing results use a different configuration; choose a new --tag")
    done = {(r["task_id"], r["pair"], r["attempt"]) for r in records if "error" not in r}
    for other in args.skip_done_in:
        done |= {(r["task_id"], r["pair"], r["attempt"]) for r in map(json.loads, Path(other).read_text().splitlines())
                 if "error" not in r}
    jobs = [(t, p, a) for a in range(args.attempts) for t, task in tasks.items() for p in range(len(task["test"]))
            if (t, p, a) not in done and (pairs is None or (t, p) in pairs)]
    print(f"{len(tasks)} tasks, {len(jobs)} attempts to run -> {out}", flush=True)
    if args.summary:
        summarize(records, tasks, args.attempts, pairs)
        return
    if not args.local_exec and not args.sandbox_url:
        ap.error("pass --sandbox-url / ARC_SANDBOX_URL or --local-exec")
    if jobs:
        asyncio.run(run(args, cfg, tasks, jobs, out, config))
    records = [json.loads(line) for line in out.read_text().splitlines() if line.strip()]
    summarize(records, tasks, args.attempts, pairs)


if __name__ == "__main__":
    main()
