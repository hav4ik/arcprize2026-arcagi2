"""ARC Prize-style no-tool eval on OpenRouter, routing between a free and a paid endpoint of one model.

Same prompt, parser, scoring and record format as E001d (run_deepseek_arcprize.py), so evals/sample_stats.py
and evals/report.py work on the results.

Router (to save money):
- A request goes to the free endpoint whenever it has an open slot and is not cooling down; otherwise to
  the paid endpoint; otherwise it waits. Workers never idle while either endpoint has capacity
  (continuous batching: up to --free-max + --paid-concurrency requests in flight).
- Capacity adapts (AIMD) because provider limits are unknown: a 429 cuts an endpoint's slot limit by a
  quarter (never below its floor) and pauses it for --cooldown seconds; each success raises the limit by
  one, up to its maximum. A daily-quota 429 turns the endpoint off for the rest of the run.
- A request refused before any token is generated costs nothing, so it is re-routed at once.
- Paid spend is capped by --max-paid-usd using the per-call cost OpenRouter reports in usage.
- Both endpoints get the same max_tokens (the free endpoint's cap), so attempts are comparable; every record
  stores the endpoint that served it. The paid endpoint can be pinned to one provider (--paid-provider).

    .venv/bin/python evals/run_openrouter_arcprize.py --model nvidia/nemotron-3.5-lightning \
        --free-model nvidia/nemotron-3.5-lightning:free --paid-provider CoreWeave --dry-run
"""
import argparse
import asyncio
import gzip
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import openai
from openai import AsyncOpenAI

from arc_harness.arcprize import TASKS, build_prompt, parse

ROOT = Path(__file__).resolve().parents[1]
BASE_URL = "https://openrouter.ai/api/v1"


def openrouter_key():
    for line in (ROOT / "SECRETS.yaml").read_text().splitlines():
        if line.startswith("openrouter_api_key:"):
            return line.split(":", 1)[1].strip().strip("'\"")
    raise KeyError("openrouter_api_key not found in SECRETS.yaml")


class Refused(Exception):
    """The endpoint refused the request before generating anything (free to retry elsewhere)."""

    def __init__(self, kind, message):
        super().__init__(message)
        self.kind = kind  # "rate_limited" | "quota" | "unavailable"


@dataclass
class Endpoint:
    name: str
    model: str
    extra: dict
    max_limit: int
    limit: int
    min_limit: int = 1
    in_flight: int = 0
    cooldown_until: float = 0.0
    disabled: bool = False
    served: int = 0
    refused: int = 0
    spent: float = 0.0
    output_tokens: int = 0

    def open(self, now):
        return not self.disabled and now >= self.cooldown_until and self.in_flight < self.limit


@dataclass
class Router:
    free: Endpoint | None
    paid: Endpoint | None
    max_paid_usd: float
    cooldown: float
    cond: asyncio.Condition = field(default_factory=asyncio.Condition)

    def endpoints(self):
        return [e for e in (self.free, self.paid) if e is not None]

    def _pick(self, now):
        if self.paid is not None and self.paid.spent >= self.max_paid_usd:
            self.paid.disabled = True
        for ep in self.endpoints():  # free first
            if ep.open(now):
                return ep
        return None

    async def acquire(self):
        async with self.cond:
            while True:
                if all(e.disabled for e in self.endpoints()):
                    raise RuntimeError("no endpoint left (free quota exhausted and/or paid budget reached)")
                now = time.monotonic()
                ep = self._pick(now)
                if ep is not None:
                    ep.in_flight += 1
                    return ep
                waits = [e.cooldown_until - now for e in self.endpoints() if not e.disabled and e.cooldown_until > now]
                try:
                    await asyncio.wait_for(self.cond.wait(), timeout=min(waits) if waits else None)
                except asyncio.TimeoutError:
                    pass

    async def release(self, ep, outcome, cost=0.0, output_tokens=0):
        async with self.cond:
            ep.in_flight -= 1
            ep.spent += cost
            ep.output_tokens += output_tokens
            if outcome == "ok":
                ep.served += 1
                ep.limit = min(ep.max_limit, ep.limit + 1)
            elif outcome in ("rate_limited", "unavailable"):
                ep.refused += 1
                # Gentle multiplicative decrease with a floor: providers often throttle briefly, and halving
                # on every 429 collapsed paid capacity from 24 to 1 in the first Nemotron run.
                ep.limit = max(ep.min_limit, (ep.limit * 3) // 4)
                ep.cooldown_until = time.monotonic() + self.cooldown
            elif outcome == "quota":
                ep.refused += 1
                ep.disabled = True
            self.cond.notify_all()

    def status(self):
        return " | ".join(f"{e.name}: {e.in_flight}/{e.limit} in flight, {e.served} done, {e.refused} refused, "
                          f"${e.spent:.3f}{' OFF' if e.disabled else ''}" for e in self.endpoints())


def classify_refusal(exc):
    if isinstance(exc, openai.RateLimitError):
        text = str(exc).lower()
        return "quota" if ("per-day" in text or "per day" in text or "daily" in text) else "rate_limited"
    if isinstance(exc, openai.APIStatusError) and exc.status_code in (408, 502, 503, 504, 529):
        return "unavailable"
    return None


async def stream_call(client, ep, messages, max_tokens, reasoning, sampling=None):
    """One streamed completion. Raises Refused if nothing was generated; returns a result dict otherwise."""
    extra = {"reasoning": reasoning, "usage": {"include": True}, **ep.extra}
    content, reasoning, usage, finish, served_by, error = [], [], None, None, None, None
    got_tokens = False
    try:
        stream = await client.chat.completions.create(
            model=ep.model, messages=messages, max_tokens=max_tokens, stream=True,
            stream_options={"include_usage": True}, extra_body=extra, **(sampling or {}))
    except openai.APIStatusError as exc:
        kind = classify_refusal(exc)
        if kind:
            raise Refused(kind, f"{type(exc).__name__}: {exc}") from exc
        raise
    try:
        async for chunk in stream:
            extra_fields = chunk.model_extra or {}
            served_by = extra_fields.get("provider") or served_by
            if chunk.usage is not None:
                usage = chunk.usage.model_dump()
            if extra_fields.get("error"):
                error = json.dumps(extra_fields["error"])[:500]
            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            delta = choice.delta
            if delta is not None:
                piece_r = (delta.model_extra or {}).get("reasoning") or ""
                if delta.content or piece_r:
                    got_tokens = True
                content.append(delta.content or "")
                reasoning.append(piece_r)
            finish = choice.finish_reason or finish
    except (openai.APIConnectionError, openai.APIStatusError) as exc:
        if not got_tokens and isinstance(exc, openai.APIStatusError) and classify_refusal(exc):
            raise Refused(classify_refusal(exc), str(exc)) from exc
        error = f"{type(exc).__name__}: {exc}"[:500]
    finally:
        await stream.close()
    if not got_tokens and finish in (None, "error") and usage is None:
        raise Refused("unavailable", error or "stream ended before any token")
    return {"content": "".join(content), "reasoning": "".join(reasoning), "usage": usage, "finish_reason": finish,
            "provider": served_by, "error": error if (error or finish == "error") else None}


def config_of(args):
    return {"model": args.model, "free_model": args.free_model, "paid_provider": args.paid_provider,
            "effort": args.effort, "reasoning_tokens": args.reasoning_tokens, "max_tokens": args.max_tokens,
            "prompt": "arcprize", **({"sampling": sampling_of(args)} if sampling_of(args) else {})}


def sampling_of(args):
    """Explicit sampling parameters; omitted ones use the provider default (older runs set none)."""
    return {k: v for k, v in (("temperature", args.temperature), ("top_p", args.top_p)) if v is not None}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="paid model id, e.g. nvidia/nemotron-3.5-lightning")
    ap.add_argument("--free-model", help="free variant, e.g. nvidia/nemotron-3.5-lightning:free (omit for paid only)")
    ap.add_argument("--no-paid", action="store_true", help="use only the free endpoint")
    ap.add_argument("--paid-provider", help="pin the paid endpoint to this OpenRouter provider (no fallbacks)")
    ap.add_argument("--effort", default="high", choices=["low", "medium", "high"])
    ap.add_argument("--max-tokens", type=int, default=65536)
    ap.add_argument("--reasoning-tokens", type=int,
                    help="explicit thinking budget (OpenRouter reasoning.max_tokens) instead of --effort; some "
                         "providers map effort to a small fixed budget (Reka: 2,047 tokens for Qwen 3.8 27B)")
    ap.add_argument("--attempts", type=int, default=1)
    ap.add_argument("--temperature", type=float, help="sampling temperature (default: provider default)")
    ap.add_argument("--top-p", type=float, help="nucleus sampling (default: provider default)")
    ap.add_argument("--free-max", type=int, default=8, help="upper bound on concurrent free requests")
    ap.add_argument("--free-start", type=int, default=4, help="initial free slot limit (adapts)")
    ap.add_argument("--cooldown", type=float, default=20.0, help="seconds an endpoint pauses after a 429")
    ap.add_argument("--paid-concurrency", type=int, default=16, help="upper bound on concurrent paid requests")
    ap.add_argument("--paid-min-concurrency", type=int, default=8, help="floor the paid limit never drops below")
    ap.add_argument("--max-paid-usd", type=float, default=5.0)
    ap.add_argument("--max-reroutes", type=int, default=20, help="per attempt, refusals before giving up")
    ap.add_argument("--tasks", help="comma-separated task ids")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--tag", default="")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if not args.free_model and args.no_paid:
        ap.error("--no-paid needs --free-model")

    tasks = {p.stem: json.loads(p.read_text()) for p in sorted(TASKS.glob("*.json"))}
    if args.tasks:
        tasks = {t: tasks[t] for t in args.tasks.split(",")}
    if args.limit:
        tasks = dict(list(tasks.items())[:args.limit])
    slug = re.sub(r"[^A-Za-z0-9.-]+", "-", args.model)
    out = ROOT / f"evals/results/openrouter_{slug}_{args.effort}{'_' + args.tag if args.tag else ''}.jsonl"
    config = config_of(args)
    records = [json.loads(l) for l in out.read_text().splitlines() if l.strip()] if out.exists() else []
    if any(r.get("config") != config for r in records):
        ap.error("existing results use a different configuration; choose a new --tag")
    done = {(r["task_id"], r["pair"], r["attempt"]) for r in records if "error" not in r}
    jobs = [(t, p, a) for t, task in tasks.items() for p in range(len(task["test"]))
            for a in range(args.attempts) if (t, p, a) not in done]
    print(f"{len(tasks)} tasks, {len(jobs)} attempts to run -> {out}")
    print(f"paid spend capped at ${args.max_paid_usd:.2f}; free endpoint: {args.free_model or 'none'}", flush=True)
    if args.dry_run or not jobs:
        return

    key = openrouter_key()
    free = Endpoint("free", args.free_model, {}, args.free_max, min(args.free_start, args.free_max)) \
        if args.free_model else None
    paid_extra = {"provider": {"order": [args.paid_provider], "allow_fallbacks": False}} if args.paid_provider else {}
    paid = None if args.no_paid else Endpoint("paid", args.model, paid_extra, args.paid_concurrency,
                                              args.paid_concurrency,
                                              min(args.paid_min_concurrency, args.paid_concurrency))
    reasoning = {"max_tokens": args.reasoning_tokens} if args.reasoning_tokens else {"effort": args.effort}
    trace_dir = out.with_suffix("")
    trace_dir.mkdir(parents=True, exist_ok=True)

    async def run():
        router = Router(free, paid, args.max_paid_usd, args.cooldown)
        queue = asyncio.Queue()
        for job in jobs:
            queue.put_nowait(job)
        n_done = 0

        async def attempt(client, t, p, a):
            task = tasks[t]
            messages = [{"role": "user", "content": build_prompt(task["train"], task["test"][p]["input"])}]
            rec = {"task_id": t, "pair": p, "attempt": a, "model": args.model, "effort": args.effort, "config": config}
            refusals = []
            for _ in range(args.max_reroutes):
                ep = await router.acquire()
                t0 = time.monotonic()
                try:
                    res = await stream_call(client, ep, messages, args.max_tokens, reasoning, sampling_of(args))
                except Refused as exc:
                    refusals.append(f"{ep.name}:{exc.kind}")
                    await router.release(ep, exc.kind)
                    continue
                except Exception as exc:
                    await router.release(ep, "error")
                    rec.update(endpoint=ep.name, error=f"{type(exc).__name__}: {exc}"[:500].replace(key, "[redacted]"))
                    return rec
                usage = res["usage"] or {}
                cost = float(usage.get("cost") or 0.0)
                await router.release(ep, "ok", cost, usage.get("completion_tokens") or 0)
                pred = parse(res["content"])
                path = trace_dir / f"{t}_p{p}_a{a}.reasoning.txt.gz"
                with gzip.open(path, "wt") as f:
                    f.write(res["reasoning"])
                rec.update(endpoint=ep.name, served_model=ep.model, provider=res["provider"], content=res["content"],
                           reasoning_chars=len(res["reasoning"]), usage=usage, cost_usd=cost,
                           finish_reason=res["finish_reason"], seconds=round(time.monotonic() - t0, 1),
                           pred=pred, refusals=refusals, trace=str(path.relative_to(ROOT)))
                if "output" in task["test"][p]:
                    rec["correct"] = pred == task["test"][p]["output"]
                if res["error"]:
                    rec["error"] = res["error"]
                return rec
            rec.update(error=f"refused {len(refusals)} times: {refusals[-3:]}")
            return rec

        async def worker(client):
            nonlocal n_done
            while True:
                try:
                    job = queue.get_nowait()
                except asyncio.QueueEmpty:
                    return
                try:
                    rec = await attempt(client, *job)
                except RuntimeError as exc:  # no endpoint left: stop this worker, keep the job for resume
                    print(f"stopping worker: {exc}", flush=True)
                    return
                with out.open("a") as f:
                    f.write(json.dumps(rec) + "\n")
                n_done += 1
                status = rec.get("error") or (f"correct={rec.get('correct')} {rec['usage'].get('completion_tokens')} "
                                              f"out-tok ${rec['cost_usd']:.4f} {rec['seconds']}s {rec['finish_reason']}")
                print(f"[{n_done}/{len(jobs)}] {rec['task_id']}.{rec['pair']}#{rec['attempt']} "
                      f"{rec.get('endpoint')} {status}", flush=True)

        async def monitor():
            while True:
                await asyncio.sleep(60)
                print(f"status: {router.status()}", flush=True)

        n_workers = (args.free_max if free else 0) + (args.paid_concurrency if paid else 0)
        async with AsyncOpenAI(api_key=key, base_url=BASE_URL, timeout=7200, max_retries=0) as client:
            mon = asyncio.create_task(monitor())
            await asyncio.gather(*(worker(client) for _ in range(min(n_workers, len(jobs)))))
            mon.cancel()
        print(f"final: {router.status()}", flush=True)

    asyncio.run(run())


if __name__ == "__main__":
    main()
