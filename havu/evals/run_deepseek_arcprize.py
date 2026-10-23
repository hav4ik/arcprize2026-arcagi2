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

E039 (opt-in): --challenges/--solutions/--pairs/--out run Kaggle-format task files (e.g. training rollouts for
distillation); --openrouter-provider sends the same request through OpenRouter pinned to one provider.
E043 (opt-in): --base-url/--extra-body send the same request to a self-hosted OpenAI-compatible server.
"""
import argparse
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI

from run_deepseek_arc2 import api_key, call, save_trace

ROOT = Path(__file__).resolve().parents[1]
# Prompt and parser live in arc_harness so the tool-use harness shares them; re-exported here.
from arc_harness.arcprize import TASKS, TEMPLATE, build_prompt, parse  # noqa: E402,F401
from task_summary import INTRO as SUMMARY_INTRO, task_summary  # noqa: E402

# E035 (opt-in): no-tool versions of the E030 effort header and the E017 task summary. The tool-harness wording
# mentions code, so both are reworded here; defaults reproduce E001d/E034.
EFFORT_HEADERS = {
    "minimal": ("Reasoning Effort: Minimal. Keep your thinking short and terse: write brief notes, not full sentences. "
                "Do not restate the task, do not re-read or re-derive what is already given, and do not re-verify "
                "what has already been checked. Commit to the simplest rule that reproduces every training pair, "
                "then write the answer."),
}
PROMPT_TOKENS_PER_CHAR = 0.9  # E039: upper end of DeepSeek's tokens/char on these prompts (0.56-0.88 measured)
NOTOOL_SUMMARY_INTRO = ("A computed summary of the grids follows; it is exact, so trust it over reading or counting "
                        "the grids by hand.")


def call_openrouter(client, args, messages, max_tokens):
    """E039: call() through OpenRouter, one pinned provider, no fallbacks. Same return shape plus OpenRouter's cost."""
    t0 = time.time()
    reason = json.loads(args.or_reasoning) if args.or_reasoning else {"effort": args.effort}
    stream = client.chat.completions.create(
        model=args.model, messages=messages, max_tokens=max_tokens, stream=True, stream_options={"include_usage": True},
        extra_body={"provider": {"order": [args.openrouter_provider], "allow_fallbacks": False},
                    "reasoning": reason, "usage": {"include": True}})
    content, reasoning, usage, finish = [], [], None, None
    try:
        for chunk in stream:
            if chunk.usage:
                usage = chunk.usage.model_dump()
            if chunk.choices:
                d = chunk.choices[0].delta
                if d is not None:
                    content.append(d.content or "")
                    reasoning.append(getattr(d, "reasoning_content", None) or (d.model_extra or {}).get("reasoning") or "")
                finish = chunk.choices[0].finish_reason or finish
    finally:
        stream.close()
    if usage is None or finish is None:
        raise RuntimeError("Incomplete stream: missing finish reason or usage")
    reasoning = "".join(reasoning)
    return {"content": "".join(content), "reasoning": reasoning, "reasoning_chars": len(reasoning),
            "usage": usage, "finish_reason": finish, "seconds": round(time.time() - t0, 1)}


def call_selfhosted(client, args, messages, max_tokens):
    """E043: call() against a self-hosted OpenAI-compatible server (--base-url); request extras from --extra-body."""
    t0 = time.time()
    extra = json.loads(args.extra_body) if args.extra_body else {}
    stream = client.chat.completions.create(
        model=args.model, messages=messages, max_tokens=max_tokens, stream=True, stream_options={"include_usage": True},
        extra_body=extra)
    content, reasoning, usage, finish = [], [], None, None
    try:
        for chunk in stream:
            if chunk.usage:
                usage = chunk.usage.model_dump()
            if chunk.choices:
                d = chunk.choices[0].delta
                if d is not None:
                    content.append(d.content or "")
                    reasoning.append(getattr(d, "reasoning_content", None) or (d.model_extra or {}).get("reasoning") or "")
                finish = chunk.choices[0].finish_reason or finish
    finally:
        stream.close()
    if usage is None or finish is None:
        raise RuntimeError("Incomplete stream: missing finish reason or usage")
    reasoning = "".join(reasoning)
    return {"content": "".join(content), "reasoning": reasoning, "reasoning_chars": len(reasoning),
            "usage": usage, "finish_reason": finish, "seconds": round(time.time() - t0, 1)}


def load_kaggle(challenges, solutions):
    """Kaggle challenges/solutions files -> {task_id: {train, test: [{input, output}]}}."""
    ch, so = json.load(open(challenges)), json.load(open(solutions))
    return {t: {"train": v["train"], "test": [{"input": x["input"], "output": so[t][i]} for i, x in enumerate(v["test"])]}
            for t, v in ch.items()}


def cost_of(usage, model):
    if usage.get("cost") is not None:
        return float(usage["cost"])
    from run_deepseek_tools import PRICES  # lazy: run_deepseek_tools imports this module
    hit, miss, output = PRICES.get(model, (0.0, 0.0, 0.0))
    h = usage.get("prompt_cache_hit_tokens") or 0
    return (h * hit + max(0, usage.get("prompt_tokens", 0) - h) * miss + usage.get("completion_tokens", 0) * output) / 1e6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="deepseek-flash")
    ap.add_argument("--effort", default="max", choices=["low", "medium", "high", "xhigh", "max"])
    ap.add_argument("--attempts", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=200_000)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--tasks", default=None, help="comma-separated task ids to run (default: all)")
    ap.add_argument("--tag", default="", help="suffix for the results file, e.g. 'traces'")
    ap.add_argument("--effort-header", default="", choices=["", *EFFORT_HEADERS],
                    help="E035: system message in DeepSeek's effort-prefix style (minimal = terse thinking)")
    ap.add_argument("--task-summary", action="store_true",
                    help="E035: append the E017 host-computed grid summary (no-tool intro) to the prompt")
    ap.add_argument("--challenges", default=None, help="E039: Kaggle-format challenges file instead of the public eval")
    ap.add_argument("--solutions", default=None)
    ap.add_argument("--pairs", default=None, help="E039: comma-separated task:pair list to run (with --challenges)")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--openrouter-provider", default="", help="E039: OpenRouter provider to pin (e.g. DeepInfra)")
    ap.add_argument("--or-reasoning", default="", help='OpenRouter reasoning JSON (default {"effort": --effort})')
    ap.add_argument("--context", type=int, default=0,
                    help="E039: provider context limit; max_tokens is cut to fit the estimated prompt")
    ap.add_argument("--base-url", default="", help="E043: self-hosted OpenAI-compatible server (no balance checks)")
    ap.add_argument("--extra-body", default="", help="E043: JSON request extras for --base-url (chat_template_kwargs, "
                    "sampling, cache_salt)")
    ap.add_argument("--max-balance-stop", type=float, default=20.0,
                    help="stop starting calls once the API balance is below this (checked every ~30 s)")
    args = ap.parse_args()

    if args.challenges:
        tasks = load_kaggle(args.challenges, args.solutions)
    else:
        tasks = {p.stem: json.load(open(p)) for p in sorted(TASKS.glob("*.json"))}
    want = None
    if args.pairs is not None:  # an empty list means no pairs, not all pairs
        want = {(t, int(p)) for t, p in (x.split(":") for x in args.pairs.split(",") if x)}
        tasks = {k: v for k, v in tasks.items() if k in {t for t, _ in want}}
    if args.tasks:
        tasks = {k: v for k, v in tasks.items() if k in set(args.tasks.split(","))}
    if args.limit:
        tasks = dict(list(tasks.items())[: args.limit])
    tag = f"_{args.tag}" if args.tag else ""
    out = args.out.resolve() if args.out else ROOT / f"evals/results/arcprize_{args.model.replace('/', '-')}_{args.effort}{tag}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        done = {(d["task_id"], d["pair"], d["attempt"]) for d in map(json.loads, open(out)) if "error" not in d}
    jobs = [(t, p, a) for t, task in tasks.items() for p in range(len(task["test"]))
            for a in range(args.attempts) if (t, p, a) not in done and (want is None or (t, p) in want)]
    if want is not None:  # E039: keep the --pairs order so a partial run is a prefix of the list (e.g. parent-balanced)
        rank = {(t, int(p)): i for i, (t, p) in enumerate(x.split(":") for x in args.pairs.split(",") if x)}
        jobs.sort(key=lambda j: (rank[j[0], j[1]], j[2]))
    print(f"{len(tasks)} tasks, {sum(len(t['test']) for t in tasks.values())} test pairs x {args.attempts} "
          f"attempts, {len(jobs)} calls to run -> {out}", flush=True)

    from run_deepseek_tools import OPENROUTER_URL, deepseek_balance, openrouter_balance, openrouter_key  # lazy (cycle)
    if args.base_url:  # E043: self-hosted, nothing to pay
        key, base, balance_of, who = "none", args.base_url, (lambda _k: float("inf")), "Self-hosted"
    elif args.openrouter_provider:
        key, base, balance_of, who = openrouter_key(), OPENROUTER_URL, openrouter_balance, "OpenRouter"
    else:
        key, base, balance_of, who = api_key(), "https://api.deepseek.com", deepseek_balance, "DeepSeek"
    balance = balance_of(key)
    print(f"{who} balance ${balance:.2f}; stop launching below ${args.max_balance_stop:.2f}", flush=True)
    if args.base_url:
        args.max_balance_stop = float("-inf")
    if balance < args.max_balance_stop:
        ap.error("Balance below --max-balance-stop")
    client = OpenAI(api_key=key, base_url=base, timeout=3600, max_retries=3)
    lock = threading.Lock()
    guard = {"t": time.time(), "stop": False}

    def low_balance():
        with lock:
            if not guard["stop"] and time.time() - guard["t"] > 30:
                guard["t"] = time.time()
                try:
                    guard["stop"] = balance_of(key) < args.max_balance_stop
                except Exception:  # noqa: BLE001 - a failed balance read does not stop the run
                    pass
                if guard["stop"]:
                    print("Balance below --max-balance-stop: not starting new calls", flush=True)
            return guard["stop"]

    def work(t, p, a):
        task = tasks[t]
        rec = {"task_id": t, "pair": p, "attempt": a, "model": args.model, "effort": args.effort,
               "effort_header": args.effort_header, "task_summary": args.task_summary}
        if args.openrouter_provider:
            rec["openrouter_provider"] = args.openrouter_provider
        if low_balance():
            return None
        try:
            prompt = build_prompt(task["train"], task["test"][p]["input"])
            if args.task_summary:
                prompt += task_summary(task["train"], task["test"][p]["input"]).replace(
                    SUMMARY_INTRO, NOTOOL_SUMMARY_INTRO, 1)
            msgs = [{"role": "user", "content": prompt}]
            if args.effort_header:
                msgs.insert(0, {"role": "system", "content": EFFORT_HEADERS[args.effort_header]})
            max_tokens = args.max_tokens
            if args.context:
                est = int(PROMPT_TOKENS_PER_CHAR * sum(len(m["content"]) for m in msgs)) + 1000
                max_tokens = min(max_tokens, args.context - est)
            if args.base_url:
                res = call_selfhosted(client, args, msgs, max_tokens)
            elif args.openrouter_provider:
                res = call_openrouter(client, args, msgs, max_tokens)
            else:
                res = call(client, args.model, args.effort, msgs, max_tokens)
            res["est_cost_usd"] = round(cost_of(res["usage"], args.model), 6)
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
            if r is None:
                continue
            status = r.get("error") or f"correct={r['correct']} {r['seconds']}s " \
                f"{(r['usage'] or {}).get('completion_tokens')} tok {r['finish_reason']}"
            print(f"[{i}/{len(jobs)}] {r['task_id']}.{r['pair']}#{r['attempt']} {status}", flush=True)

    summarize(out, tasks, want)


def summarize(path, tasks, want=None):
    recs = [d for d in map(json.loads, open(path)) if "error" not in d and d["task_id"] in tasks]
    if want is not None:  # E039 --pairs: score the listed pairs only, each pair weighted equally
        recs = [d for d in recs if (d["task_id"], d["pair"]) in want]
        c = [d["correct"] for d in recs]
        print(f"\n{path.name}: {len(recs)} attempts over {len({(d['task_id'], d['pair']) for d in recs})} of "
              f"{len(want)} pairs; per-attempt correct {sum(c) / max(1, len(c)):.1%}; "
              f"cost ${sum(d.get('est_cost_usd') or 0 for d in recs):.2f}")
        return
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
