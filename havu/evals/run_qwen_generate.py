"""ARC Prize-style no-tool eval of Qwen3.8-27B on a self-hosted SGLang server, through the raw generation API.

The chat template is rendered by hand and sent to /v1/completions, so a cut-off generation can be continued
verbatim (the text so far is appended to the prompt):

- Each attempt has a token budget (--budget, thinking + answer, summed over chunks). Chunks are at most
  --chunk-tokens long, so a lost request costs little and progress is never discarded.
- The model sometimes emits EOS in the middle of its thinking. Then the attempt is continued (`continuations`);
  after two continuations in a row that add < 20 tokens each, it is forced to answer.
- If the budget runs out while still thinking, "I have to give my final answer now.\\n</think>" is appended and up
  to --final-tokens more are generated (`forced_final`).
- The answer is parsed (ARC Prize parser) from the text after the last </think>.

Same prompt, parser and task-weighted scoring as E001d/E008. Records carry `pred` and `usage.completion_tokens`,
so evals/sample_stats.py works on them. Re-running resumes (finished (task, pair, attempt) are skipped).

    .venv/bin/python evals/run_qwen_generate.py --verify-template      # chat-template check against the server
    .venv/bin/python evals/run_qwen_generate.py --dry-run
    .venv/bin/python evals/run_qwen_generate.py --tier medium --budget 48000 --attempts 2 --concurrency 64
"""
import argparse
import asyncio
import gzip
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evals"))
from arc_harness.arcprize import TASKS, build_prompt, parse  # noqa: E402

HARNESS_VERSION = "qwen-gen-1"
MODEL = "nvidia/Qwen3.8-27B-NVFP4"
ENDPOINT_NAME = "vastai-4xrtx6000-qwen27b"
CONTEXT = 262144
CLOSE = "</think>"
FORCE_TEXT = "\n\nI have to give my final answer now.\n</think>\n\n"
LOW_YIELD = 20
PILOT_TASKS = 15

TIER_SYSTEM = {
    "xhigh": "Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, "
             "consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final "
             "answer.",
    "medium": None,
    "none": None,  # models without effort tiers (e.g. Qwen3.6): no system message
    "low": "Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the conclusion "
           "without unnecessary elaboration.",
}


# ---------------------------------------------------------------- pure functions (unit-tested)

def render_prompt(user_content, tier):
    """Qwen chat template, single user message, no tools, generation prompt with thinking opened."""
    s = ""
    if TIER_SYSTEM[tier]:
        s += f"<|im_start|>system\n{TIER_SYSTEM[tier]}<|im_end|>\n"
    return s + f"<|im_start|>user\n{user_content.strip()}<|im_end|>\n<|im_start|>assistant\n<think>\n"


def answer_section(gen):
    """Text after the last </think>, or None while still thinking."""
    i = gen.rfind(CLOSE)
    return None if i < 0 else gen[i + len(CLOSE):]


def next_action(gen, finish, budget_left, low_yield_streak, forced):
    """What to do after a chunk: 'done', 'continue', 'eos_continue' or 'force_final'.

    finish is the chunk's finish reason ('stop' = EOS/stop string, 'length' = max_tokens reached).
    """
    if forced:
        return "done"
    if answer_section(gen) is not None:
        # Answering: finished on EOS, or out of budget; a 'length' cut with budget left is our chunk cap.
        return "continue" if finish == "length" and budget_left > 0 else "done"
    if budget_left <= 0:
        return "force_final"
    if finish == "stop":  # EOS mid-thought
        return "force_final" if low_yield_streak >= 2 else "eos_continue"
    return "continue"


def chunk_max_tokens(budget_left, chunk_tokens, prompt_tokens, generated_tokens):
    room = CONTEXT - prompt_tokens - generated_tokens - 64
    return max(0, min(budget_left, chunk_tokens, room))


def job_order(task_ids, n_pairs, attempts, pilot=PILOT_TASKS):
    """(task, pair, attempt) jobs, the first `pilot` tasks (sorted ids) first."""
    tids = sorted(task_ids)
    return [(t, p, a) for group in (tids[:pilot], tids[pilot:]) for t in group
            for p in range(n_pairs[t]) for a in range(attempts)]


# ---------------------------------------------------------------- endpoint

def endpoint():
    import yaml
    for e in yaml.safe_load((ROOT / "ENDPOINTS.yaml").read_text()):
        if e.get("name") == ENDPOINT_NAME:
            return f"http://{e['addr']}/v1", e["token"]
    raise KeyError(ENDPOINT_NAME)


def load_tasks(args):
    tasks = {p.stem: json.loads(p.read_text()) for p in sorted(TASKS.glob("*.json"))}
    if args.tasks:
        tasks = {t: tasks[t] for t in args.tasks.split(",")}
    if args.limit_tasks:
        tasks = dict(list(tasks.items())[:args.limit_tasks])
    return tasks


def user_message(task, p):
    return build_prompt(task["train"], task["test"][p]["input"])


async def verify_template(tasks, tiers):
    """prompt_tokens of the server's chat template vs our rendering, per tier, for two tasks."""
    from openai import AsyncOpenAI
    base, token = endpoint()
    ok = True
    async with AsyncOpenAI(api_key=token, base_url=base, timeout=600, max_retries=2) as client:
        for t in list(tasks)[:2]:
            user = user_message(tasks[t], 0)
            for tier in tiers:
                kwargs = {} if tier == "none" else {"reasoning_effort": tier}
                chat = await client.chat.completions.create(
                    model=MODEL, messages=[{"role": "user", "content": user}], max_tokens=1,
                    extra_body={"chat_template_kwargs": kwargs})
                raw = await client.completions.create(model=MODEL, prompt=render_prompt(user, tier), max_tokens=1)
                a, b = chat.usage.prompt_tokens, raw.usage.prompt_tokens
                ok &= a == b
                print(f"{t} {tier:6s} chat={a} raw={b} {'OK' if a == b else 'MISMATCH'}", flush=True)
    print("template verified" if ok else "TEMPLATE MISMATCH", flush=True)
    return ok


# ---------------------------------------------------------------- run

class EndpointDown(Exception):
    pass


class Health:
    def __init__(self, max_down):
        self.max_down = max_down
        self.down_since = None
        self.stop = False

    def ok(self):
        self.down_since = None

    def failed(self):
        now = time.monotonic()
        if self.down_since is None:
            self.down_since = now
        if now - self.down_since > self.max_down:
            self.stop = True
        return self.stop


def config_of(args):
    return {"model": MODEL, "tier": args.tier, "budget": args.budget, "chunk_tokens": args.chunk_tokens,
            "final_tokens": args.final_tokens, "temperature": args.temperature, "top_p": 0.95, "top_k": 20,
            "prompt": "arcprize", "harness_version": HARNESS_VERSION,
            **({"presence_penalty": args.presence_penalty} if args.presence_penalty is not None else {})}


async def run(args, tasks, jobs, out, config):
    import openai
    from openai import AsyncOpenAI
    base, token = endpoint()
    trace_dir = out.parent / (out.stem + "_traces")
    trace_dir.mkdir(parents=True, exist_ok=True)
    health = Health(args.max_down)
    queue = asyncio.Queue()
    for j in jobs:
        queue.put_nowait(j)
    stats = {"done": 0, "correct": 0, "tokens": 0, "in_flight": 0, "errors": 0}
    t_start = time.monotonic()

    async def complete(client, prompt, max_tokens):
        """One chunk with retries. Returns (text, completion_tokens, prompt_tokens, finish, n_errors)."""
        errs = 0
        while True:
            if health.stop:
                raise EndpointDown("endpoint down")
            try:
                r = await client.completions.create(
                    model=MODEL, prompt=prompt, max_tokens=max_tokens, temperature=args.temperature, top_p=0.95,
                    extra_body={"top_k": 20},
                    **({"presence_penalty": args.presence_penalty} if args.presence_penalty is not None else {}))
                health.ok()
                c = r.choices[0]
                return c.text, r.usage.completion_tokens, r.usage.prompt_tokens, c.finish_reason, errs
            except (openai.APIConnectionError, openai.APITimeoutError, openai.InternalServerError,
                    openai.RateLimitError, openai.APIStatusError) as exc:
                errs += 1
                status = getattr(exc, "status_code", None)
                transient = status is None or status >= 500 or status in (408, 429)
                if not transient:
                    raise
                if health.failed():
                    raise EndpointDown(f"endpoint failing for > {args.max_down}s: {type(exc).__name__}") from exc
                # Connection-level failures keep retrying until the endpoint counts as down; others max 5.
                if errs > 5 and status is not None:
                    raise
                await asyncio.sleep(min(60, 5 * 2 ** min(errs - 1, 4)))

    async def attempt(client, t, p, a):
        task = tasks[t]
        raw = render_prompt(user_message(task, p), args.tier)
        rec = {"task_id": t, "pair": p, "attempt": a, "model": MODEL, "tier": args.tier, "budget": args.budget,
               "config": config, "harness_version": HARNESS_VERSION}
        gen, used, chunks, conts, streak, forced, n_err = "", 0, [], 0, 0, False, 0
        prompt_tokens, after_eos = None, False
        force_reason, error = None, None
        t0 = time.monotonic()
        try:
            while True:
                if forced:
                    mt = args.final_tokens
                else:
                    mt = chunk_max_tokens(args.budget - used, args.chunk_tokens, prompt_tokens or 0, used)
                text, ntok, ptok, finish, errs = await complete(client, raw + gen, mt)
                n_err += errs
                if prompt_tokens is None:
                    prompt_tokens = ptok
                gen += text
                used += ntok
                stats["tokens"] += ntok
                chunks.append({"tokens": ntok, "finish": finish, "end_char": len(gen)})
                if after_eos:
                    streak = streak + 1 if ntok < LOW_YIELD else 0
                act = next_action(gen, finish, args.budget - used, streak, forced)
                after_eos = act == "eos_continue"
                if act == "done":
                    break
                if act == "eos_continue":
                    conts += 1
                elif act == "force_final":
                    force_reason = "budget" if args.budget - used <= 0 else "low_yield_eos"
                    forced = True
                    gen += FORCE_TEXT
        except EndpointDown:
            raise
        except Exception as exc:  # non-transient API error, or retries exhausted
            error = f"{type(exc).__name__}: {exc}"[:500].replace(token, "[redacted]")
        content = answer_section(gen)
        pred = parse(content) if content is not None else None
        path = trace_dir / f"{t}_p{p}_a{a}.txt.gz"
        with gzip.open(path, "wt") as f:
            f.write(gen)
        rec.update(correct=pred == task["test"][p]["output"] if pred is not None else False,
                   answered=pred is not None, content=content, pred=pred, completion_tokens=used,
                   usage={"completion_tokens": used, "prompt_tokens": prompt_tokens}, chunks=chunks,
                   continuations=conts, forced_final=forced, force_reason=force_reason,
                   closed_think=CLOSE in (gen.replace(FORCE_TEXT, "") if forced else gen),
                   prompt_tokens=prompt_tokens, elapsed_s=round(time.monotonic() - t0, 1), transient_errors=n_err,
                   trace=str(path.relative_to(ROOT)))
        if error:
            rec["error"] = error
        return rec

    async def worker(client):
        while not health.stop:
            try:
                job = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            stats["in_flight"] += 1
            try:
                rec = await attempt(client, *job)
            except EndpointDown as exc:
                print(f"stopping worker: {exc}", flush=True)
                return
            finally:
                stats["in_flight"] -= 1
            with out.open("a") as f:
                f.write(json.dumps(rec) + "\n")
            stats["done"] += 1
            stats["correct"] += bool(rec["correct"])
            stats["errors"] += "error" in rec
            print(f"[{stats['done']}/{len(jobs)}] {rec['task_id']}.{rec['pair']}#{rec['attempt']} "
                  f"correct={rec['correct']} answered={rec['answered']} tok={rec['completion_tokens']} "
                  f"chunks={len(rec['chunks'])} conts={rec['continuations']} forced={rec['forced_final']} "
                  f"{rec['elapsed_s']}s{' ERROR ' + rec['error'] if 'error' in rec else ''}", flush=True)

    async def monitor():
        last_tok, last_t = 0, time.monotonic()
        while True:
            await asyncio.sleep(args.log_every)
            now = time.monotonic()
            rate = (stats["tokens"] - last_tok) / (now - last_t)
            last_tok, last_t = stats["tokens"], now
            el = now - t_start
            print(f"progress {time.strftime('%H:%M:%S')}: done {stats['done']}/{len(jobs)}, correct "
                  f"{stats['correct']}, errors {stats['errors']}, in flight {stats['in_flight']}, "
                  f"{rate:.0f} tok/s (window), {stats['tokens'] / el:.0f} tok/s (avg), elapsed {el / 60:.1f} min",
                  flush=True)

    async with AsyncOpenAI(api_key=token, base_url=base, timeout=3 * 3600, max_retries=0) as client:
        mon = asyncio.create_task(monitor())
        await asyncio.gather(*(worker(client) for _ in range(min(args.concurrency, len(jobs)))))
        mon.cancel()
    el = time.monotonic() - t_start
    print(f"finished: {stats['done']}/{len(jobs)} attempts, {stats['correct']} correct, {stats['errors']} errors, "
          f"{stats['tokens']} tokens in {el / 60:.1f} min ({stats['tokens'] / el:.0f} tok/s)"
          f"{'; STOPPED: endpoint down' if health.stop else ''}", flush=True)


def main():
    global MODEL, ENDPOINT_NAME
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tier", default="medium", choices=list(TIER_SYSTEM))
    ap.add_argument("--budget", type=int, default=48000, help="generated tokens per attempt (thinking + answer)")
    ap.add_argument("--chunk-tokens", type=int, default=16384, help="max tokens per request")
    ap.add_argument("--final-tokens", type=int, default=4096, help="tokens for the forced final answer")
    ap.add_argument("--attempts", type=int, default=2)
    ap.add_argument("--temperature", type=float, default=0.6, help="model card recommends 1.0 for thinking mode")
    ap.add_argument("--concurrency", type=int, default=64)
    ap.add_argument("--tasks", help="comma-separated task ids")
    ap.add_argument("--limit-tasks", type=int)
    ap.add_argument("--tag", default="")
    ap.add_argument("--max-down", type=float, default=600, help="seconds of failures before stopping")
    ap.add_argument("--log-every", type=float, default=120)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--endpoint", default=ENDPOINT_NAME, help="entry name in ENDPOINTS.yaml")
    ap.add_argument("--model", default=MODEL, help="served model id")
    ap.add_argument("--presence-penalty", type=float, help="Qwen3.6 card recommends 1.5 (default: not sent)")
    ap.add_argument("--verify-template", action="store_true")
    args = ap.parse_args()
    MODEL, ENDPOINT_NAME = args.model, args.endpoint

    tasks = load_tasks(args)
    if args.verify_template:
        tiers = ["none"] if args.tier == "none" else [t for t in TIER_SYSTEM if t != "none"]
        sys.exit(0 if asyncio.run(verify_template(tasks, tiers)) else 1)
    prefix = "qwen_nvfp4" if MODEL == "nvidia/Qwen3.8-27B-NVFP4" else re.sub(r"[^a-z0-9]+", "-", MODEL.lower()).strip("-")
    out = ROOT / (f"evals/results/{prefix}_gen_{args.tier}_{args.budget // 1000}k"
                  f"{'_' + args.tag if args.tag else ''}.jsonl")
    config = config_of(args)
    records = [json.loads(line) for line in out.read_text().splitlines() if line.strip()] if out.exists() else []
    if any(r.get("config") != config for r in records):
        ap.error("existing results use a different configuration; choose a new --tag")
    done = {(r["task_id"], r["pair"], r["attempt"]) for r in records if "error" not in r}
    jobs = [j for j in job_order(tasks, {t: len(v["test"]) for t, v in tasks.items()}, args.attempts)
            if j not in done]
    print(f"{len(tasks)} tasks, {sum(len(v['test']) for v in tasks.values())} pairs, {len(jobs)} attempts to run "
          f"(tier {args.tier}, budget {args.budget}) -> {out}", flush=True)
    if args.dry_run:
        t = next(iter(tasks))
        print(f"--- rendered prompt for {t} pair 0 ---")
        print(render_prompt(user_message(tasks[t], 0), args.tier))
        return
    if jobs:
        asyncio.run(run(args, tasks, jobs, out, config))


if __name__ == "__main__":
    main()
