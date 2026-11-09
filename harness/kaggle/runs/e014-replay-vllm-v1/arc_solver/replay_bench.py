"""E014: agentic replay benchmark on Kaggle 4xL4 (offline). Runs next to solver.py and uses its install / launch /
kill / stats code, so the bench exercises the same server path as the submission.

Workload (dataset chankhavu/arc2-agentic-replay, built by kaggle/datasets/agentic_replay/build.py from E012 Qwen
tool-use transcripts): every agent replays one recorded attempt turn by turn. Turn t sends prompt_t and asks for
exactly the recorded number of generated tokens (ignore_eos, temp 1.0 / top_p 0.95 / top_k 20, chunks of 16384 like
the agent); then prompt_{t+1} = prompt_t + <the server's own generated text> + <recorded tool output + next
assistant header>, which is what the real agent sends, so prefix caching sees the real reuse pattern. A finished
attempt is followed by the next transcript from turn 0. Agents that join at a level start are warm-started at a
random turn (stationary distribution truncated to contexts <= ARC_REPLAY_WARM_MAX, recorded history as a cold
prompt), so the live working set is closer to a long run than a cold start.

Plan (env ARC_REPLAY_PLAN, JSON list): [{"name", "env": {ARC_* settings for launch_server}, "max_seqs",
"levels": [[agents, seconds], ...], "fallback_env": {...}}]. Concurrency ramps up within one server start; per level
we report decode tok/s, new-prefill tok/s, prefix-cache hit rate (client: cached/prompt tokens; server counters),
hit vs ideal on continuation turns (ideal = previous request's prompt + completion), running / waiting requests,
KV usage, preemptions (vLLM) / retractions (SGLang), TTFT and turn time. Results: /kaggle/working/replay_bench.json.
Time-boxed by ARC_BENCH_STOP_MIN (minutes from notebook start).
"""
import asyncio
import glob
import gzip
import json
import os
import random
import re
import shutil
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import solver as S  # noqa: E402

STOP = S.T0 + float(os.environ.get("ARC_BENCH_STOP_MIN", 60)) * 60
OUT = os.path.join(S.WORK, "replay_bench.json")
TOOL_DELAY = float(os.environ.get("ARC_REPLAY_TOOL_DELAY", 1.0))
WARM_MAX = int(os.environ.get("ARC_REPLAY_WARM_MAX", 20000))
CHUNK = 16384
SAMPLE_S = 10.0
RESULTS = {"configs": {}, "phases": {}}


def save():
    tmp = OUT + ".tmp"
    json.dump(RESULTS, open(tmp, "w"), indent=1)
    os.replace(tmp, OUT)


def left():
    return STOP - time.time()


def load_data():
    p = S.find_one(f"{S.INPUT}/**/agentic_replay.json.gz")
    if not p:
        raise FileNotFoundError("agentic_replay.json.gz not attached")
    return json.load(gzip.open(p, "rt"))


def warm_start(x, rng):
    """(turn index, prompt) for an agent joining mid-run: turn drawn with weight = its generated tokens among turns
    whose start context <= WARM_MAX; the prompt is the recorded transcript up to that turn."""
    cand = [(i, t["gen"]) for i, t in enumerate(x["turns"]) if t["ctx"] <= WARM_MAX and t["gen"] > 0]
    if not cand:
        return 0, x["prompt0"]
    r, acc, k = rng.random() * sum(w for _, w in cand), 0.0, cand[-1][0]
    for i, w in cand:
        acc += w
        if acc >= r:
            k = i
            break
    return k, x["prompt0"] + "".join(x["hist"][i] + x["turns"][i]["suffix"] for i in range(k))


class Replay:
    def __init__(self, data, name):
        import httpx
        self.data, self.name = data, name
        self.http = httpx.AsyncClient(base_url=S.BASE, timeout=httpx.Timeout(connect=30.0, read=1500.0, write=120.0,
                                                                              pool=None),
                                      limits=httpx.Limits(max_connections=256, max_keepalive_connections=256))
        self.next_x = 0
        self.reqs = []        # one record per request
        self.samples = []     # periodic server + client state
        self.tokens = 0       # streamed completion tokens (all agents)
        self.live = {}        # agent -> current context tokens (prompt + generated so far)
        self.level = None
        self.errors = 0

    def take(self):
        x = self.data[self.next_x % len(self.data)]
        self.next_x += 1
        return x

    async def request(self, agent, prompt, max_tokens, ideal):
        body = {"model": S.SERVED, "prompt": prompt, "max_tokens": max_tokens, "temperature": 1.0, "top_p": 0.95,
                "top_k": 20, "ignore_eos": True, "stream": True, "skip_special_tokens": False,
                "stream_options": {"include_usage": True, "continuous_usage_stats": True}}
        rec = {"agent": agent, "level": self.level, "t0": time.time(), "ideal": ideal, "max_tokens": max_tokens}
        text, ntok, usage, base = [], 0, None, self.live.get(agent, 0)
        try:
            async with self.http.stream("POST", "/v1/completions", json=body) as r:
                if r.status_code != 200:
                    raise RuntimeError(f"HTTP {r.status_code}: {(await r.aread())[:300]!r}")
                async for line in r.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    d = line[6:]
                    if d.strip() == "[DONE]":
                        break
                    d = json.loads(d)
                    if d.get("error"):
                        raise RuntimeError(str(d["error"])[:300])
                    got = False
                    for c in d.get("choices") or []:
                        if c.get("text"):
                            text.append(c["text"])
                            got = True
                    if got and "t_first" not in rec:
                        rec["t_first"] = time.time()
                    u = d.get("usage")
                    if u:
                        usage = u
                    if u and u.get("completion_tokens") is not None:
                        self.tokens += u["completion_tokens"] - ntok
                        ntok = u["completion_tokens"]
                        if u.get("prompt_tokens"):
                            self.live[agent] = u["prompt_tokens"] + ntok
                    elif got:
                        self.tokens += 1
                        ntok += 1
                        self.live[agent] = base + ntok
        except asyncio.CancelledError:
            rec["cancelled"] = True
            raise
        except Exception as exc:
            rec["error"] = f"{type(exc).__name__}: {exc}"[:300]
            self.errors += 1
        finally:
            rec["t1"] = time.time()
            rec["completion"] = ntok
            if usage:
                rec["prompt"] = usage.get("prompt_tokens")
                rec["cached"] = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
            self.reqs.append(rec)
        return "".join(text), ntok, rec

    async def agent(self, idx, warm):
        rng = random.Random(1000 + idx)
        first = True
        while True:
            x = self.take()
            k, prompt = warm_start(x, rng) if (warm and first) else (0, x["prompt0"])
            first = False
            ideal, last_prompt = 0, 0
            for ti in range(k, len(x["turns"])):
                t = x["turns"][ti]
                remaining, text = t["gen"], ""
                while remaining > 0:
                    mt = min(CHUNK, remaining)
                    out, n, rec = await self.request(idx, prompt + text, mt, ideal)
                    if rec.get("error") or n <= 0:
                        await asyncio.sleep(5)
                        break
                    text += out
                    remaining -= n
                    last_prompt = rec.get("prompt") or last_prompt
                    ideal = last_prompt + n
                if ti == len(x["turns"]) - 1 or last_prompt + 2 * CHUNK > S.MAX_MODEL_LEN:
                    break
                await asyncio.sleep(TOOL_DELAY)  # tool execution
                prompt = prompt + text + t["suffix"]
            self.live[idx] = 0

    async def sampler(self):
        while True:
            st = await asyncio.to_thread(S.server_stats, 8.0)
            self.samples.append({"t": time.time(), "level": self.level, "tokens": self.tokens,
                                 "live_ctx": sum(self.live.values()), "agents_live": sum(v > 0 for v in self.live.values()),
                                 "server": st})
            await asyncio.sleep(SAMPLE_S)

    async def run(self, levels):
        tasks, n = [], 0
        smp = asyncio.create_task(self.sampler())
        marks = []
        try:
            for li, (agents, dur) in enumerate(levels):
                if left() < 90:
                    S.log(f"[{self.name}] time box: skipping level {agents}")
                    break
                dur = min(dur, left() - 60)
                self.level = agents
                t_start = time.time()
                while n < agents:
                    tasks.append(asyncio.create_task(self.agent(n, warm=True)))
                    n += 1
                S.log(f"[{self.name}] level {agents} agents for {dur:.0f}s")
                await asyncio.sleep(dur)
                marks.append((agents, t_start, time.time()))
                S.log(f"[{self.name}] level {agents} done: " + json.dumps(self.summarize(agents, t_start, time.time())))
        finally:
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            smp.cancel()
            await self.http.aclose()
        return marks

    def summarize(self, agents, t_start, t_end):
        skip = min(90.0, (t_end - t_start) / 3)
        w0 = t_start + skip
        span = max(1.0, t_end - w0)
        sm = [s for s in self.samples if w0 <= s["t"] <= t_end]
        out = {"agents": agents, "window_s": round(span)}
        if len(sm) >= 2:
            out["decode_tok_s"] = round((sm[-1]["tokens"] - sm[0]["tokens"]) / (sm[-1]["t"] - sm[0]["t"]), 1)
            out["live_ctx_mean"] = round(sum(s["live_ctx"] for s in sm) / len(sm))
            out["live_ctx_max"] = max(s["live_ctx"] for s in sm)
            srv = [s["server"] for s in sm if s["server"]]
            if srv:
                out["running_mean"] = round(sum(x.get("running", 0) for x in srv) / len(srv), 1)
                out["running_max"] = max(x.get("running", 0) for x in srv)
                out["waiting_max"] = max(x.get("waiting", 0) for x in srv)
                out["kv_usage_mean"] = round(sum(x.get("kv_usage", 0) for x in srv) / len(srv), 3)
                out["kv_usage_max"] = round(max(x.get("kv_usage", 0) for x in srv), 3)
                a, b = srv[0], srv[-1]
                if "pc_queries" in b:
                    dq = b["pc_queries"] - a.get("pc_queries", 0)
                    out["server_hit_rate"] = round((b["pc_hits"] - a.get("pc_hits", 0)) / dq, 4) if dq else None
                    out["preemptions"] = b.get("preemptions", 0) - a.get("preemptions", 0)
                    out["server_new_prefill_tok_s"] = round((dq - (b["pc_hits"] - a.get("pc_hits", 0))) / span, 1)
                if "uncached_prefill" in b:
                    out["server_new_prefill_tok_s"] = round((b["uncached_prefill"] - a["uncached_prefill"]) / span, 1)
                    out["retracted_queue_max"] = max(x.get("retracted", 0) for x in srv)
        rq = [r for r in self.reqs if w0 <= r["t0"] <= t_end and r.get("prompt")]
        if rq:
            pt = sum(r["prompt"] for r in rq)
            ct = sum(r.get("cached") or 0 for r in rq)
            out["requests"] = len(rq)
            out["client_hit_rate"] = round(ct / pt, 4) if any(r.get("cached") is not None for r in rq) else None
            out["client_new_prefill_tok_s"] = round((pt - ct) / span, 1)
            cont = [r for r in rq if r["ideal"] > 0 and r.get("cached") is not None]
            if cont:
                out["cont_hit_vs_ideal"] = round(sum(min(r["cached"], r["ideal"]) for r in cont)
                                                 / sum(r["ideal"] for r in cont), 4)
                out["cont_miss_frac"] = round(sum(r["cached"] < 0.9 * r["ideal"] for r in cont) / len(cont), 4)
            tt = sorted(r["t_first"] - r["t0"] for r in rq if r.get("t_first"))
            if tt:
                out["ttft_p50"] = round(tt[len(tt) // 2], 2)
                out["ttft_p90"] = round(tt[int(len(tt) * 0.9)], 2)
            dur = sorted(r["t1"] - r["t0"] for r in rq if not r.get("error"))
            if dur:
                out["req_s_p50"] = round(dur[len(dur) // 2], 1)
            out["errors"] = sum(1 for r in rq if r.get("error"))
        return out


def log_lines(path):
    keep = re.compile(r"Decode batch|Prefill batch|retract|evict|preempt|Engine 000|KV Cache|Mamba|capped|"
                      r"hicache|HiCache|host|error|Error", re.I)
    try:
        return [ln.rstrip()[:400] for ln in open(path, errors="ignore") if keep.search(ln)][-2500:]
    except Exception:
        return []


async def run_config(cfg, data, mdir):
    name = cfg["name"]
    res = RESULTS["configs"].setdefault(name, {"cfg": cfg})
    if cfg.get("base"):  # same server settings as the config that started successfully, plus extra_env
        base = RESULTS["configs"].get(cfg["base"], {}).get("ok_env")
        if base is None:
            res["skipped"] = f"base config {cfg['base']} did not start"
            save()
            return
        cfg = {**cfg, "env": {**base, **cfg.get("extra_env", {})}, "fallback_env": None}
    attempts = [cfg.get("env", {})] + ([cfg["fallback_env"]] if cfg.get("fallback_env") is not None else [])
    proc = path = None
    for ai, env in enumerate(attempts):
        if left() < 10 * 60:
            res["skipped"] = f"{left() / 60:.1f} min left"
            save()
            return
        for k in list(os.environ):
            if k.startswith(("ARC_SGL_", "ARC_SERVER_EXTRA")):
                del os.environ[k]
        os.environ.update({k: str(v) for k, v in env.items()})
        S.MAX_SEQS = int(cfg.get("max_seqs", S.MAX_SEQS))
        t = time.time()
        proc, path = S.launch_server(mdir)
        ok = await asyncio.to_thread(S.wait_ready, proc, path, min(1500, max(60, left() - 300)))
        res.setdefault("starts", []).append({"env": env, "ready": ok, "startup_s": round(time.time() - t)})
        if ok:
            res["ok_env"] = env
            break
        res["starts"][-1]["log_tail"] = S.server_log_tail(path, 60)
        S.kill_server(proc)
        dst = os.path.join(S.LOG_DIR, f"{name}_failed{ai}.log")
        shutil.move(path, dst)
        save()
    else:
        return
    kv, mr = S.server_capacity(path)
    res["capacity"] = {"kv_tokens": kv, "max_running": mr}
    res["server_stats_at_start"] = S.server_stats()
    save()
    rp = Replay(data, name)
    try:
        marks = await rp.run(cfg["levels"])
    except Exception as exc:
        res["replay_error"] = f"{type(exc).__name__}: {exc}"[:500]
        marks = []
    res["levels"] = [rp.summarize(a, t0, t1) for a, t0, t1 in marks]
    res["errors"] = rp.errors
    res["samples"] = [{**s, "t": round(s["t"] - S.T0, 1)} for s in rp.samples]
    res["requests"] = [{k: (round(v - S.T0, 2) if k in ("t0", "t1", "t_first") else v) for k, v in r.items()}
                       for r in rp.reqs]
    res["log_lines"] = log_lines(path)
    res["proc_alive_at_end"] = proc.poll() is None
    save()
    S.kill_server(proc)
    try:
        shutil.move(path, os.path.join(S.LOG_DIR, f"{name}_server.log"))
    except Exception:
        pass
    S.log(f"[{name}] levels: " + json.dumps(res["levels"]))


async def main_async(plan):
    data = load_data()
    RESULTS["n_transcripts"] = len(data)
    mdir = S.model_dir()
    for cfg in plan:
        await run_config(cfg, data, mdir)
        RESULTS["phases"][f"{cfg['name']}_done"] = round(time.time() - S.T0)
        save()


def main():
    os.makedirs(S.LOG_DIR, exist_ok=True)
    S.driver_env()
    plan = json.loads(os.environ["ARC_REPLAY_PLAN"])
    RESULTS["plan"] = plan
    S.log(f"replay bench start: server {S.SERVER}, stop at {(STOP - S.T0) / 60:.0f} min")
    # Hard time box independent of the event loop: final save + kill the server tree, then exit.
    def guard():
        while time.time() < STOP + 60:
            time.sleep(5)
        S.log("replay bench guard: time box exceeded; killing the server and exiting")
        try:
            RESULTS["guard_fired"] = True
            save()
        finally:
            S.kill_server()
            os._exit(0)
    threading.Thread(target=guard, daemon=True).start()
    try:
        mdir = S.model_dir()
        pre = threading.Thread(target=S.preload, args=(mdir,), daemon=True)
        pre.start()  # page-cache the 31 GB model while the wheels install
        S.install_server()
        RESULTS["phases"]["installed"] = round(time.time() - S.T0)
        pre.join(timeout=max(1, left() - 600))
        RESULTS["phases"]["preloaded"] = round(time.time() - S.T0)
        save()
        asyncio.run(main_async(plan))
    except BaseException as exc:
        import traceback
        traceback.print_exc()
        RESULTS["fatal"] = f"{type(exc).__name__}: {exc}"[:500]
    finally:
        jit = os.environ.get("ARC_SGL_JIT_DIR", "/tmp/sglang_jit")
        if S.SERVER == "sglang" and os.path.isdir(jit):
            shutil.copytree(jit, os.path.join(S.WORK, "sglang_jit_cache_gpu"), symlinks=True, dirs_exist_ok=True)
        RESULTS["phases"]["end"] = round(time.time() - S.T0)
        save()
        S.kill_server()
        S.log("replay bench done: " + json.dumps({k: v.get("levels") for k, v in RESULTS["configs"].items()})[:6000])


if __name__ == "__main__":
    main()
    os._exit(0)
