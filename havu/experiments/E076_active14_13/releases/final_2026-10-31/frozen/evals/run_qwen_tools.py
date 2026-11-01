"""E012: Qwen3.8-27B with the E009 Python tool harness on a self-hosted OpenAI-compatible server (raw completions).

Agent loop: evals/arc_harness/agent_qwen.py. Executors: evals/arc_harness/executor.py (Docker sandbox via
--sandbox-url / ARC_SANDBOX_URL, or --local-exec). ARC Prize prompt + lean2sub suffix, task-weighted scoring.
Re-running resumes (finished (task, pair, attempt) are skipped; errored ones are retried).

    .venv/bin/python evals/run_qwen_tools.py --verify-template
    .venv/bin/python evals/run_qwen_tools.py --pilot --tier xhigh --hard 80000 --soft 40000 --attempts 2 --tag x80s40

E019 program-only answers (qwen-tools-4): --answer-mode program (prompt lean2prog, tool submit_program).
Local server: --base-url http://localhost:30000/v1 --model qwen3.8-27b --context 98304.
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


def endpoint(name, base_url=None):
    if base_url:
        return base_url, None
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
    base, token = endpoint(args.endpoint, args.base_url)
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


AUG_GEOMS = ("id", "transpose", "rot90", "fliplr", "rot180", "flipud", "rot270", "antitranspose")


def augmentation(spec, t, p, a):
    """E020 --augment: (geom, colour map) for attempt a of (t, p). Attempt 0 is always the identity.
    geom cycles through AUG_GEOMS when spec has 'd8'; with 'colour', colours 1-9 are permuted (0 kept),
    seeded by (t, p, a)."""
    import random
    geom, cmap = "id", None
    if a == 0 or spec == "none":
        return geom, cmap
    if "d8" in spec:
        geom = AUG_GEOMS[a % len(AUG_GEOMS)]
    if "colour" in spec:
        rng = random.Random(f"{t}:{p}:{a}")
        perm = list(range(1, 10))
        rng.shuffle(perm)
        cmap = [0] + perm
    return geom, cmap


def _geom(g, name, inverse=False):
    import numpy as np
    a = np.array(g)
    if name == "rot90":
        a = np.rot90(a, -1 if inverse else 1)
    elif name == "rot270":
        a = np.rot90(a, 1 if inverse else -1)
    elif name == "rot180":
        a = np.rot90(a, 2)
    elif name == "transpose":
        a = a.T
    elif name == "antitranspose":
        a = np.rot90(a, 2).T
    elif name == "fliplr":
        a = a[:, ::-1]
    elif name == "flipud":
        a = a[::-1]
    return a.tolist()


def apply_aug(g, aug, inverse=False):
    geom, cmap = aug
    if g is None or (geom == "id" and cmap is None):
        return g
    if inverse:
        if cmap is not None:
            inv = [0] * 10
            for i, c in enumerate(cmap):
                inv[c] = i
            g = [[inv[v] for v in row] for row in g]
        return _geom(g, geom, inverse=True)
    g = _geom(g, geom)
    return [[cmap[v] for v in row] for row in g] if cmap is not None else g


SESSION_MEM_CAP = """
def _arc_memcap():
    import resource
    try:
        with open('/proc/self/status') as f:
            vm = next(int(l.split()[1]) * 1024 for l in f if l.startswith('VmSize:'))
        cap = vm + {mb} * 2 ** 20
        soft, hard = resource.getrlimit(resource.RLIMIT_AS)
        if hard == resource.RLIM_INFINITY or cap < hard:
            resource.setrlimit(resource.RLIMIT_AS, (cap, hard if hard != resource.RLIM_INFINITY else cap))
    except Exception:
        pass
_arc_memcap()
del _arc_memcap
"""


def note_settle(agree, rec):
    """Count a finished attempt's test output toward its pair's agreement if its program passed every train pair."""
    if rec.get("train_pass") and rec.get("pred") is not None and "error" not in rec:
        key = json.dumps(rec["pred"])
        c = agree.setdefault((rec["task_id"], int(rec["pair"])), {})
        c[key] = c.get(key, 0) + 1


MEMORY_HEAD = ("\n\nEarlier attempts at this task by other solvers did not reproduce every training pair. Their "
               "programs are below so that you do not repeat them: look for a different rule, or find exactly what "
               "they got wrong.")


def note_memory(memory, rec):
    """Keep a finished attempt whose answer program failed the train check (or that wrote no program)."""
    if "error" in rec or rec.get("train_pass"):
        return
    memory.setdefault((rec["task_id"], int(rec["pair"])), []).append(
        {"attempt": int(rec["attempt"]), "program": rec.get("program"), "train_frac": rec.get("train_frac")})


def memory_state(args):
    memory = {}
    if args.memory:
        for path in [args.out_path] + args.pool_in:
            if Path(path).exists():
                for line in Path(path).read_text().splitlines():
                    if line.strip():
                        note_memory(memory, json.loads(line))
    return memory


def memory_note(items, attempt, n, n_train, max_chars=1500):
    """E020 --memory: text for the first message of `attempt` from at most n earlier failed attempts (programs
    truncated to max_chars). Attempts without a program are summarised in one line."""
    items = [m for m in items if m["attempt"] != attempt]
    if not items:
        return None
    progs = [m for m in items if m["program"]][-n:]
    lines = [MEMORY_HEAD]
    for m in progs:
        k = round((m["train_frac"] or 0) * n_train)
        code = m["program"] if len(m["program"]) <= max_chars else m["program"][:max_chars] + "\n# ... (truncated)"
        lines.append(f"\nA program that reproduced {k} of {n_train} training pairs:\n```python\n{code.strip()}\n```")
    empty = len(items) - len([m for m in items if m["program"]])
    if empty:
        lines.append(f"\n{empty} earlier attempt(s) ran out of budget without writing a solve: draft solve early and "
                     "test it with code.")
    return "\n".join(lines)


def settle_state(args):
    """E020 --settle: agreement counts from this run's results file and every --pool-in file (labels unused)."""
    agree = {}
    if not args.settle:
        return agree
    for path in [args.out_path] + args.pool_in:
        if Path(path).exists():
            for line in Path(path).read_text().splitlines():
                if line.strip():
                    note_settle(agree, json.loads(line))
    return agree


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
        n, c = src.get(r.get("answer_source"), (0, 0))
        src[r.get("answer_source")] = (n + 1, c + r["correct"])
    print(f"pass@1 {score(1):.1%}  pass@{attempts} {score(attempts):.1%}  ({len(tasks)} tasks, {len(latest)} attempts)"
          f"; tokens mean {statistics.mean(toks) / 1000:.1f}k median {statistics.median(toks) / 1000:.1f}k; "
          f"sources {src}; parse failures {sum(r['parse_failures'] for r in latest.values())}", flush=True)


async def run(args, cfg, tasks, jobs, out, config):
    base, token = endpoint(args.endpoint, args.base_url)
    llm = aq.OpenAICompletionsLLM(base, token, args.model, cfg.temperature, cfg.top_p, cfg.top_k,
                                  timeout=args.request_timeout)
    extra = aq.HELPERS_SRC if cfg.helpers else ""
    if args.session_mem_mb:  # cap the session kernel's address space at its current size + N MB (MemoryError)
        extra = SESSION_MEM_CAP.format(mb=args.session_mem_mb) + extra
        aq.PROGRAM_MEMORY_MB = min(aq.PROGRAM_MEMORY_MB, args.session_mem_mb)
    if args.local_exec:
        make = local_factory(cfg.tool_timeout, cfg.max_tool_output, **({"prelude_extra": extra} if extra else {}))
    else:
        make = sandbox_factory(args.sandbox_url, cfg.tool_timeout, cfg.max_tool_output, prelude_extra=extra)
    trace_dir = out.parent / (out.stem + "_traces")
    trace_dir.mkdir(parents=True, exist_ok=True)
    queue = asyncio.Queue()
    for j in jobs:
        queue.put_nowait(j)
    stats = {"done": 0, "correct": 0, "tokens": 0, "settled_skips": 0}
    t_start = time.monotonic()
    pool, direct = None, {}
    if args.worker_slots:
        from arc_harness.slots import SlotPool, parse_spec
        pool = SlotPool(args.slot_state, parse_spec(args.worker_slots), args.run_name or out.stem)
        print(f"worker slots: {pool.show()}", flush=True)
    agree = settle_state(args)  # E020 --settle: (task, pair) -> Counter of train-passing test outputs
    memory = memory_state(args)  # E020 --memory: (task, pair) -> earlier attempts that failed the train check

    async def worker():
        while True:
            try:
                t, p, a = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            if args.settle and max(agree.get((t, p), {}).values(), default=0) >= args.settle:
                stats["settled_skips"] += 1
                continue
            task = tasks[t]
            base_llm, lease = llm, None
            if pool is not None:  # E020 worker slots: hold one slot on one worker for the whole attempt
                lease, wurl = await pool.acquire()
                if wurl not in direct:
                    direct[wurl] = aq.OpenAICompletionsLLM(wurl.rstrip("/") + "/v1", "none", args.model,
                                                           cfg.temperature, cfg.top_p, cfg.top_k,
                                                           timeout=args.request_timeout)
                base_llm = direct[wurl]
            attempt_llm = (aq.RoutedLLM(base_llm, args.routing_key_header, f"{args.run_name or out.stem}:{t}:{p}:{a}")
                           if args.routing_key_header else base_llm)
            try:
                await attempt(t, p, a, task, attempt_llm, lease and wurl)
            finally:
                if lease:
                    pool.release(lease)

    async def attempt(t, p, a, task, attempt_llm, worker_url):
            aug = augmentation(args.augment, t, p, a)
            train = [{"input": apply_aug(x["input"], aug), "output": apply_aug(x["output"], aug)} for x in task["train"]]
            note = memory_note(memory.get((t, p), []), a, args.memory, len(train)) if args.memory else None
            transcript, result = await aq.solve(attempt_llm, make, train, apply_aug(task["test"][p]["input"], aug), cfg,
                                                note)
            if worker_url:
                result["worker"] = worker_url
            if note:
                result["memory_from"] = [m["attempt"] for m in memory.get((t, p), []) if m["attempt"] != a][-args.memory:]
            if aug != ("id", None):
                result["augment"] = {"geom": aug[0], "colour_map": aug[1]}
                for key in ("pred", "first_tp_pred"):
                    if result.get(key) is not None:
                        try:
                            result[key] = apply_aug(result[key], aug, inverse=True)
                        except Exception:  # a ragged grid cannot be inverted: no answer
                            result[key] = None
            if "error" in result and token:
                result["error"] = result["error"].replace(token, "[redacted]")
            path = trace_dir / f"{t}_p{p}_a{a}.txt.gz"
            with gzip.open(path, "wt") as f:
                f.write(transcript)
            rec = {"task_id": t, "pair": p, "attempt": a, "config": config, **result,
                   "correct": result["pred"] == task["test"][p]["output"], "trace": str(path.relative_to(ROOT))}
            with out.open("a") as f:
                f.write(json.dumps(rec) + "\n")
            note_settle(agree, rec)
            note_memory(memory, rec)
            stats["done"] += 1
            stats["correct"] += rec["correct"]
            stats["tokens"] += result["usage"]["completion"]
            el = time.monotonic() - t_start
            prog = (f" subs={rec['n_submissions']} train={rec['train_frac']}" if "n_submissions" in rec else "")
            u = result["usage"]
            hit = (f" hit={u['cached'] / max(1, u['prompt_total']):.2f}/{u['ideal_cached'] / max(1, u['prompt_total']):.2f}"
                   if "cached" in u else "")
            print(f"[{stats['done']}/{len(jobs)}] {t}.{p}#{a} correct={rec['correct']} src={rec['answer_source']}{prog}{hit} "
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
        for d in direct.values():
            await d.close()
    el = time.monotonic() - t_start
    print(f"finished {stats['done']} attempts, {stats['correct']} correct, {stats['tokens']} tokens in "
          f"{el / 60:.1f} min ({stats['tokens'] / el:.0f} tok/s)"
          + (f"; {stats['settled_skips']} attempts skipped (settled)" if args.settle else ""), flush=True)


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
    ap.add_argument("--answer-mode", default="submit", choices=["submit", "program"],
                    help="program (E019): answers only via submit_program(code defining solve), train-checked")
    ap.add_argument("--max-submissions", type=int, default=6, help="program mode: submit_program calls per attempt")
    ap.add_argument("--program-timeout", type=float, default=10, help="program mode: seconds per grid")
    ap.add_argument("--record-train-check", action="store_true",
                    help="submit mode: record train_pass of the answer expression after the attempt (no tokens)")
    ap.add_argument("--context", type=int, default=aq.CONTEXT, help="the server's context length")
    ap.add_argument("--prompt", choices=list(aq.PROMPTS),
                    help="prompt suffix (default lean2prog in program mode, else lean2sub; E020: lean2progroles)")
    ap.add_argument("--task-summary", action="store_true", help="E020/E017 P1: append the computed perception pack")
    ap.add_argument("--helpers", action="store_true",
                    help="E020/E017 P3: preload evals/arc_helpers.py in the session and in submitted programs")
    ap.add_argument("--turn-think-cap", type=int, default=0,
                    help="E020: thinking tokens per turn before a 'test it with code now' note (0 = off)")
    ap.add_argument("--turn-think-grace", type=int, default=2048,
                    help="E020: tokens after that note before the thinking is closed into a run_python call")
    ap.add_argument("--confirm-test", action="store_true",
                    help="E020 program mode: show the first train-passing program's test output once; the model "
                         "re-submits (same or fixed code) to answer")
    ap.add_argument("--augment", default="none", choices=["none", "d8", "colour", "d8colour"],
                    help="E020 diversity: attempts >= 1 see the task under a D8 transform and/or colour permutation; "
                         "the answer grid is mapped back (attempt 0 is never transformed)")
    ap.add_argument("--test-audit", action="store_true",
                    help="E020: before accepting the first train-passing program, show its test output with host-side "
                         "audit flags (new colours, size rule, identity/empty, outside train range, slow); one review turn")
    ap.add_argument("--worked-example", action="store_true",
                    help="E020: append a short workflow demo on ARC training task 3aa6fb7a (use with --helpers)")
    ap.add_argument("--think-call-rescue", action="store_true",
                    help="E020/E021: after a think nudge, execute a complete <tool_call> written inside open thinking")
    ap.add_argument("--record-alternatives", action="store_true",
                    help="E020 bookkeeping (no tokens): record other test grids of each attempt (free attempt_2)")
    ap.add_argument("--lint", action="store_true",
                    help="E020: flag hardcoded constants (tables, cell indices, literal grids, exact-size branches)")
    ap.add_argument("--feedback", default="basic", choices=["basic", "rich"],
                    help="program mode: rich adds colour confusions and a small-grid diff (E020)")
    ap.add_argument("--base-url", help="OpenAI-compatible base URL without auth (instead of --endpoint)")
    ap.add_argument("--routing-key-header", help="send this header (e.g. x-smg-routing-key) with value "
                    "<run_name>:<task>:<pair>:<attempt> on every request of an attempt (gateway manual routing)")
    ap.add_argument("--run-name", help="run name in the routing key (default: the results file stem)")
    ap.add_argument("--out", help="results file (default evals/results/qwen_tools_<tier>_<hard>k[_tag].jsonl)")
    ap.add_argument("--attempts", type=int, default=2)
    ap.add_argument("--only-attempt", type=int, help="run only this attempt index (e.g. a 2nd sample in its own file)")
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
    ap.add_argument("--session-mem-mb", type=int, default=0,
                    help="E020 safety (not in the config): cap each session kernel at its size + N MB of address "
                         "space and submitted programs at min(3072, N) MB, so a runaway cell raises MemoryError")
    ap.add_argument("--worker-slots",
                    help="E020: per-worker attempt slots shared by all processes using --slot-state, e.g. auto:14 "
                         "(every gateway worker) or http://127.0.0.1:30101=14,...; each attempt holds a slot on the "
                         "worker with most free slots and sends every turn straight to that worker's URL")
    ap.add_argument("--request-timeout", type=float, default=3600,
                    help="seconds before a completion request is abandoned and retried (E020: 1200 guards against "
                         "half-open tunnel connections; a 16k-token chunk at ~20 tok/s takes ~14 min)")
    ap.add_argument("--slot-state", default=str(ROOT / "experiments/E019_qwen_program_answers/router/slots.json"),
                    help="shared slot state file (caps editable live: python -m arc_harness.slots --set)")
    ap.add_argument("--settle", type=int, default=0,
                    help="E020 adaptive sampling: skip an attempt (at dispatch) once this many finished attempts of "
                         "its pair passed train with the same test output (this file plus --pool-in files; 0 = off)")
    ap.add_argument("--memory", type=int, default=0,
                    help="E020 cross-attempt memory: show attempt i up to N programs of earlier attempts of its pair "
                         "(this file and --pool-in) that failed the train check (0 = off; do not combine with --augment)")
    ap.add_argument("--pool-in", action="append", default=[],
                    help="results file whose train-passing answers count toward --settle (repeatable)")
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
                         force_submit_tokens=args.force_submit_tokens, answer_mode=args.answer_mode,
                         max_submissions=args.max_submissions, program_timeout=args.program_timeout,
                         record_train_check=args.record_train_check, context=args.context,
                         prompt=args.prompt or ("lean2prog" if args.answer_mode == "program" else "lean2sub"),
                         task_summary=args.task_summary, helpers=args.helpers, turn_think_cap=args.turn_think_cap,
                         turn_think_grace=args.turn_think_grace, feedback=args.feedback,
                         confirm_test=args.confirm_test, test_audit=args.test_audit, lint=args.lint,
                         worked_example=args.worked_example, record_alternatives=args.record_alternatives,
                         think_call_rescue=args.think_call_rescue)
    if cfg.budget_soft and cfg.budget_soft >= cfg.max_total_output:
        ap.error("--soft must be below --hard")
    config = aq.config_dict(cfg, args.model)
    if args.routing_key_header:
        config["routing_key_header"] = args.routing_key_header
    if args.augment != "none":
        config["augment"] = args.augment
    if args.memory:
        config["memory"] = args.memory
    out = Path(args.out).resolve() if args.out else \
        ROOT / f"evals/results/qwen_tools_{args.tier}_{args.hard // 1000}k{'_' + args.tag if args.tag else ''}.jsonl"
    args.out_path = out
    records = [json.loads(line) for line in out.read_text().splitlines() if line.strip()] if out.exists() else []
    if any(r.get("config") != config for r in records):
        ap.error("existing results use a different configuration; choose a new --tag")
    done = {(r["task_id"], r["pair"], r["attempt"]) for r in records if "error" not in r}
    for other in args.skip_done_in:
        done |= {(r["task_id"], r["pair"], r["attempt"]) for r in map(json.loads, Path(other).read_text().splitlines())
                 if "error" not in r}
    jobs = [(t, p, a) for a in range(args.attempts) for t, task in tasks.items() for p in range(len(task["test"]))
            if (t, p, a) not in done and (pairs is None or (t, p) in pairs)
            and (args.only_attempt is None or a == args.only_attempt)]
    print(f"{len(tasks)} tasks, {len(jobs)} attempts to run -> {out}", flush=True)
    if args.summary:
        summarize(records, tasks, args.attempts, pairs)
        return
    if args.worked_example and not args.helpers:
        ap.error("--worked-example uses objects()/check(): pass --helpers")
    if not args.local_exec and not args.sandbox_url:
        ap.error("pass --sandbox-url / ARC_SANDBOX_URL or --local-exec")
    if jobs:
        asyncio.run(run(args, cfg, tasks, jobs, out, config))
    records = [json.loads(line) for line in out.read_text().splitlines() if line.strip()]
    summarize(records, tasks, args.attempts, pairs)


if __name__ == "__main__":
    main()
