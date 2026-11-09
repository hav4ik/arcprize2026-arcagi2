#!/usr/bin/env python3
"""Deterministic serving benchmark on our real workload: replay recorded multi-turn tool-use attempts against an
OpenAI-compatible server (vLLM) at a fixed concurrency and measure serving performance per time window.

Each attempt's turns are issued in order. Turn k's prompt is the recorded conversation up to the attempt's k-th
assistant message. The request generates exactly the recorded turn's completion tokens (ignore_eos, min_tokens =
max_tokens, temperature 0), so prompt growth and the prefix-cache pattern follow the real run whatever the model writes.

Modes (--mode):
  tokens     (default) the harness's --token-exact path. Prompts are rendered to token ids once, through the server's
             /tokenize or offline with --local-render MODEL_DIR (DeepSeek-V4.1), and cached with --prep-cache.
             Turn k+1 = turn k's prompt ids + the ids the server generated in turn k + the rendered new messages (tool
             results), sent to /v1/completions. After a turn the real run cut at max_tokens or rewrote (rescue), the
             next prompt is a fresh render, as in the harness. Text traces only.
  chat       /v1/chat/completions with the recorded messages; the server renders them. The recorded assistant turns
             replace what the server generated, so each turn re-prefills the previous turn's output: prefix-cache hits
             come out lower than in the real run. Image traces work here.
  synthetic  no tokenizer: random token ids with the recorded prompt and completion lengths, in the tokens-mode
             structure. Attention and KV load are faithful; MoE routing and the vision tower are not.

Concurrency = N attempts in flight; when one finishes, the next starts. The attempt order is a seeded shuffle
(--seed, --order). Each pass over the traces gets its own cache_salt, so a repeated attempt never hits an earlier pass's
cache. --think-time adds a fixed pause per tool result between turns (the traces do not record tool latency).

Outputs in --out: windows.jsonl (one record per --window-s, written as the run goes), requests.jsonl (one per turn),
summary.json and report.md. Server-side numbers come from /metrics (vllm:* counters); --server-log tails vLLM's per-rank
"GPU memory: ..." and "Capped the PyTorch allocator at ..." lines to report allocator headroom (cap - peak) over time.

  T=experiments/E064_manh_notebook_b32_harness/l4e2e/runs/b32m_x2t/out/dsv41_traces.tar.gz
  # once, on any CPU box: render the traces to token ids
  .venv/bin/python evals/serve_replay_bench.py --prepare-only --traces $T --local-render data/dsv4_prune/model_meta \\
      --prep-cache prep/b32m_x2t.tokens.json.gz --out prep/b32m_x2t
  # per server configuration (~20 min)
  python evals/serve_replay_bench.py --base-url http://127.0.0.1:8000 --traces $T \\
      --prep-cache prep/b32m_x2t.tokens.json.gz --tools prep/b32m_x2t/tools.json --concurrency 64 \\
      --duration 1200 --warmup-s 300 --seed 0 --server-log server.log --out runs/replay_base

How to run it on a pod and do flag ablations: notes/2026-10-30_serve_replay_bench.md. Only the standard library is
needed; tool schemas for --tools auto / harness:... come from evals/run_deepseek_tools.py (or pass a JSON file).
"""
import argparse
import array
import base64
import glob
import gzip
import hashlib
import json
import math
import os
import random
import re
import sys
import tarfile
import threading
import time
import urllib.error
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

EVALS = Path(__file__).resolve().parent
PREP_VERSION = 2  # bump when plan_tokens changes: older caches are ignored
DEFAULT_TEMPLATE_KWARGS = '{"thinking": true, "reasoning_effort": 50}'  # B32 / B32-M solver (ARC_REASONING_EFFORT=50)
# --tools auto: harness flag sets tried per trace source; the one whose turn-1 render matches the recorded prompt wins
HARNESS_BASE = ["--mode", "grid"]
TOOL_CANDIDATES = [["--answer-tool"], ["--answer-tool", "--multi-test"], [], ["--multi-test"],
                   ["--answer-tool", "--helpers"], ["--answer-tool", "--multi-test", "--helpers"], ["--helpers"],
                   ["--answer-tool", "--show-tool"], ["--answer-tool", "--multi-test", "--show-tool"]]


QUIET = False  # --quiet: warnings only


def log(msg):
    if QUIET and not msg.startswith("WARNING"):
        return
    print(f"[replay {time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


def pct(values, q):
    """Linear-interpolated q-th percentile of the non-None values, or None."""
    v = sorted(x for x in values if x is not None)
    if not v:
        return None
    i = (len(v) - 1) * q / 100.0
    lo = int(math.floor(i))
    hi = min(lo + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (i - lo)


def rnd(x, n=3):
    return None if x is None else round(x, n)


def sha1(data):
    return hashlib.sha1(data if isinstance(data, bytes) else data.encode()).hexdigest()


# ----------------------------------------------------------------------------------------------------- trace loading
@dataclass
class Turn:
    index: int  # 0-based turn number
    msg_index: int  # this turn's assistant message; the prompt is messages[:msg_index]
    prompt_rec: int  # recorded usage.prompt_tokens
    completion: int  # recorded usage.completion_tokens = the forced output length
    cached_rec: object  # recorded cached prompt tokens (None if the provider did not report them)
    finish: object
    resync: bool  # the next turn's prompt is a fresh render (this turn was cut by max_tokens or rewritten)
    n_tools: int  # tool results that follow this turn (think time)


@dataclass
class Attempt:
    key: str
    source: str  # the trace dir / tarball / file it came from (unit of --tools auto)
    digest: str  # sha1 of the trace JSON: dedup and prep-cache key
    first_digest: str  # sha1 of the turn-1 prompt messages (attempts of one task share it)
    messages: object
    turns: list
    images: bool
    dropped_turns: int = 0  # assistant messages without a usage event (a turn that failed, e.g. a cut stream)

    @property
    def tokens(self):
        return sum(t.prompt_rec + t.completion for t in self.turns)


def _has_images(messages):
    return any(isinstance(m.get("content"), list) and any(isinstance(p, dict) and p.get("type") == "image_url"
                                                         for p in m["content"]) for m in messages)


def parse_trace(obj, key, source, digest=None):
    """A harness trace ({"messages": [...], "events": [...]}, as written by evals/run_deepseek_tools.py and the dsv41
    solver) -> Attempt, or None if it is not a trace or has no usable turn. The k-th event with "usage" is the k-th
    model call, whose assistant message is the k-th assistant message (each call appends exactly one)."""
    if not isinstance(obj, dict) or not isinstance(obj.get("messages"), list) \
            or not isinstance(obj.get("events"), list):
        return None
    messages, events = obj["messages"], obj["events"]
    usage_events = [e for e in events if isinstance(e, dict) and isinstance(e.get("usage"), dict)]
    asst = [i for i, m in enumerate(messages) if isinstance(m, dict) and m.get("role") == "assistant"]
    # turns the harness rewrote (code rescue, tool-name fix, cap nudge): its --token-exact path re-renders the next
    # prompt (tok["resync"]); a nudge only appends a user message, which the prefix check in plan_tokens handles
    rewritten = {e.get("turn") for e in events if isinstance(e, dict)
                 and (("rescued" in e and e["rescued"] != "nudge") or e.get("cap_nudge"))}
    n = min(len(usage_events), len(asst))
    turns = []
    for k in range(n):
        e, u = usage_events[k], usage_events[k]["usage"]
        completion, prompt = int(u.get("completion_tokens") or 0), int(u.get("prompt_tokens") or 0)
        if completion <= 0 or prompt <= 0:
            break
        cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens")
        if cached is None:
            cached = u.get("prompt_cache_hit_tokens")
        end = asst[k + 1] if k + 1 < len(asst) else len(messages)
        n_tools = sum(1 for m in messages[asst[k] + 1:end] if isinstance(m, dict) and m.get("role") == "tool")
        turns.append(Turn(k, asst[k], prompt, completion, cached, e.get("finish_reason"),
                          e.get("finish_reason") == "length" or e.get("turn") in rewritten, n_tools))
    if not turns:
        return None
    first = json.dumps(messages[:turns[0].msg_index], sort_keys=True)
    digest = digest or sha1(json.dumps(obj, sort_keys=True))
    return Attempt(key, source, digest, sha1(first), messages, turns, _has_images(messages[:turns[-1].msg_index]),
                   len(asst) - len(turns))


def _read_json_bytes(name, data):
    if name.endswith(".gz"):
        data = gzip.decompress(data)
    return data


def _iter_tar(path):
    with tarfile.open(path) as tf:
        for m in sorted(tf.getmembers(), key=lambda m: m.name):
            if m.isfile() and m.name.endswith((".json.gz", ".json")):
                f = tf.extractfile(m)
                if f is not None:
                    yield m.name, _read_json_bytes(m.name, f.read())


def iter_trace_files(specs):
    """(source, key, json bytes) for every *.json(.gz) in the given dirs (recursive, tarballs inside included),
    tarballs (.tar.gz / .tgz), files and globs."""
    for spec in specs:
        paths = sorted(glob.glob(spec, recursive=True)) or ([spec] if os.path.exists(spec) else [])
        if not paths:
            log(f"WARNING no traces match {spec}")
        for p in map(Path, paths):
            if p.is_dir():
                for f in sorted(p.rglob("*")):
                    if not f.is_file():
                        continue
                    if f.name.endswith((".tar.gz", ".tgz")):
                        for name, data in _iter_tar(f):
                            yield str(f), name, data
                    elif f.name.endswith((".json.gz", ".json")):
                        yield str(f.parent), str(f.relative_to(p)), _read_json_bytes(f.name, f.read_bytes())
            elif p.name.endswith((".tar.gz", ".tgz")):
                for name, data in _iter_tar(p):
                    yield str(p), name, data
            elif p.is_file():
                yield str(p.parent), p.name, _read_json_bytes(p.name, p.read_bytes())


def load_attempts(specs):
    """-> (attempts in canonical order, stats). Identical traces (the same file in a dir and in its tarball) are kept
    once."""
    seen, attempts = set(), []
    stats = {"files": 0, "not_traces": 0, "duplicates": 0, "no_turns": 0, "dropped_turns": 0}
    for source, key, data in iter_trace_files(specs):
        stats["files"] += 1
        digest = sha1(data)
        if digest in seen:
            stats["duplicates"] += 1
            continue
        seen.add(digest)
        try:
            obj = json.loads(data)
        except ValueError:
            stats["not_traces"] += 1
            continue
        if not isinstance(obj, dict) or "messages" not in obj or "events" not in obj:
            stats["not_traces"] += 1
            continue
        att = parse_trace(obj, key, source, digest)
        if att is None:
            stats["no_turns"] += 1
            continue
        stats["dropped_turns"] += att.dropped_turns
        attempts.append(att)
    # by file name + content, so the order (and a seed's shuffle) does not depend on whether the same traces came
    # from a dir or its tarball
    attempts.sort(key=lambda a: (Path(a.key).name, a.digest))
    return attempts, stats


def attempt_order(attempts, seed, order="shuffle", pass_index=0):
    """The deterministic launch order of one pass: a seeded shuffle, or heaviest / lightest first (recorded prompt +
    completion tokens) with seeded tie-breaks."""
    idx = list(range(len(attempts)))
    random.Random(f"{seed}|{pass_index}").shuffle(idx)
    if order == "heavy-first":
        idx.sort(key=lambda i: -attempts[i].tokens)
    elif order == "light-first":
        idx.sort(key=lambda i: attempts[i].tokens)
    return [attempts[i] for i in idx]


# --------------------------------------------------------------------------------------------------- server client
METRIC_NAMES = ("generation_tokens_total", "prompt_tokens_total", "prefix_cache_queries_total",
                "prefix_cache_hits_total", "external_prefix_cache_queries_total", "external_prefix_cache_hits_total",
                "kv_cache_usage_perc", "num_requests_running", "num_requests_waiting", "num_preemptions_total")
METRIC_ALIASES = {"prefix_cache_queries_total": "gpu_prefix_cache_queries_total",
                  "prefix_cache_hits_total": "gpu_prefix_cache_hits_total",
                  "kv_cache_usage_perc": "gpu_cache_usage_perc", "num_preemptions_total": "num_preemptions"}
COUNTERS = ("generation_tokens_total", "prompt_tokens_total", "prefix_cache_queries_total", "prefix_cache_hits_total",
            "external_prefix_cache_queries_total", "external_prefix_cache_hits_total", "num_preemptions_total")


def parse_prometheus(text):
    """The vllm:* series we track from a /metrics page. Counters and running/waiting are summed over label sets (data
    parallel engines); KV usage is the mean (kv_cache_usage_perc) and max (kv_cache_usage_max) over them. Deprecated
    names (gpu_prefix_cache_*, gpu_cache_usage_perc, num_preemptions) are used only when the current one is missing.
    Lines that do not parse and non-finite values are skipped."""
    raw = {}
    for line in text.splitlines():
        if not line.startswith("vllm:"):
            continue
        try:
            name = line.split("{", 1)[0].split(" ", 1)[0][5:]
            value = float(line.rsplit("}", 1)[-1].split()[0] if "}" in line else line.split()[1])
        except (ValueError, IndexError):
            continue
        if math.isfinite(value):
            raw.setdefault(name, []).append(value)
    out = {}
    for name in METRIC_NAMES:
        values = raw.get(name) or raw.get(METRIC_ALIASES.get(name, ""))
        if not values:
            continue
        if name == "kv_cache_usage_perc":
            out[name], out["kv_cache_usage_max"] = sum(values) / len(values), max(values)
        else:
            out[name] = sum(values)
    return out


class Server:
    def __init__(self, base_url, api_key=None, timeout=3600.0):
        self.root = base_url.rstrip("/").removesuffix("/v1")
        self.headers = {"Content-Type": "application/json"}
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"
        self.timeout = timeout
        self.max_model_len = None

    def _req(self, path, body=None, timeout=None):
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(self.root + path, data=data, headers=self.headers,
                                     method="GET" if body is None else "POST")
        return urllib.request.urlopen(req, timeout=timeout or self.timeout)

    def get_json(self, path, timeout=30):
        with self._req(path, timeout=timeout) as r:
            return json.loads(r.read())

    def models(self):
        return [m["id"] for m in self.get_json("/v1/models").get("data", [])]

    def version(self):
        try:
            return self.get_json("/version", timeout=10).get("version")
        except Exception:
            return None

    def tokenize(self, model, messages, tools, template_kwargs, gen):
        body = {"model": model, "messages": messages, "add_generation_prompt": gen}
        if tools:
            body["tools"] = tools
        if template_kwargs:
            body["chat_template_kwargs"] = template_kwargs
        with self._req("/tokenize", body, timeout=600) as r:
            d = json.loads(r.read())
        if d.get("max_model_len"):
            self.max_model_len = d["max_model_len"]
        return d["tokens"]

    def metrics(self):
        try:
            with self._req("/metrics", timeout=10) as r:
                m = parse_prometheus(r.read().decode(errors="replace"))
            return m or None
        except Exception:
            return None

    def stream(self, path, body, stop, on_tokens, retries=2):
        """POST a streaming request; on_tokens(n) per chunk that carries new output tokens. -> result dict: t_first /
        t_end (monotonic), completion / prompt / cached tokens from usage, generated ids (if returned), finish, error,
        aborted (stop was set), retries. A connection-level failure before any response (refused / reset: the server
        never started the request) is retried up to `retries` times; HTTP errors and failures mid-stream are not."""
        for attempt in range(retries + 1):
            out = self._stream_once(path, body, stop, on_tokens)
            out["retries"] = attempt
            if not out.pop("_connect_error") or attempt == retries or stop.is_set():
                return out
            stop.wait(0.5 * (attempt + 1))
        return out

    def _stream_once(self, path, body, stop, on_tokens):
        out = {"t_first": None, "t_end": None, "completion": None, "prompt": None, "cached": None, "ids": [],
               "ids_returned": False, "finish": None, "error": None, "aborted": False}
        counted, responded = 0, False
        try:
            with self._req(path, body) as resp:
                responded = True
                for raw in resp:
                    if not raw.startswith(b"data:"):
                        continue
                    data = raw[5:].strip()
                    if data == b"[DONE]":
                        break
                    d = json.loads(data)
                    if "error" in d and not d.get("choices"):
                        out["error"] = f"server error: {json.dumps(d['error'])[:300]}"
                        break
                    n_ids, n_text = 0, 0
                    for ch in d.get("choices") or []:
                        tids = ch.get("token_ids")
                        if tids is not None:
                            out["ids_returned"] = True
                            out["ids"].extend(tids)
                            n_ids += len(tids)
                        delta = ch.get("delta") or {}
                        if (ch.get("text") or delta.get("content") or delta.get("reasoning_content")
                                or delta.get("reasoning") or delta.get("tool_calls")):
                            n_text += 1
                        out["finish"] = ch.get("finish_reason") or out["finish"]
                    usage = d.get("usage")
                    if usage:
                        out["prompt"] = usage.get("prompt_tokens", out["prompt"])
                        out["completion"] = usage.get("completion_tokens", out["completion"])
                        cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
                        out["cached"] = cached if cached is not None else out["cached"]
                    # new tokens: continuous usage stats if present, else the chunk's token ids, else 1 per text chunk
                    if usage and usage.get("completion_tokens") is not None:
                        n = usage["completion_tokens"] - counted
                    else:
                        n = n_ids if out["ids_returned"] else n_text
                    if n > 0:
                        now = time.monotonic()
                        if out["t_first"] is None:
                            out["t_first"] = now
                        counted += n
                        on_tokens(n)
                    if stop.is_set():
                        out["aborted"] = True
                        break
        except urllib.error.HTTPError as exc:
            try:
                detail = exc.read()[:300].decode(errors="replace")
            except Exception:
                detail = ""
            out["error"] = f"HTTP {exc.code}: {detail}"
        except Exception as exc:
            out["error"] = f"{type(exc).__name__}: {exc}"[:300]
        out["_connect_error"] = bool(out["error"]) and not responded and not out["error"].startswith("HTTP ")
        out["t_end"] = time.monotonic()
        if out["completion"] is not None and out["completion"] > counted and not out["aborted"]:
            on_tokens(out["completion"] - counted)  # tokens the stream did not attribute (no continuous usage)
            counted = out["completion"]
        if out["completion"] is None:
            out["completion"] = counted
        return out


# ------------------------------------------------------------------------------------------------------ tool schemas
def harness_tools(flags):
    """The tool schemas evals/run_deepseek_tools.py sends with these harness flags (e.g. ["--answer-tool"])."""
    if str(EVALS) not in sys.path:
        sys.path.insert(0, str(EVALS))
    import run_deepseek_tools as H  # noqa: E402  (lazy: only --tools auto / harness:...)
    return H.tools_for(H.parser().parse_args(HARNESS_BASE + list(flags)))


class ServerRenderer:
    """Chat messages -> prompt ids through the server's /tokenize (what the harness's --token-exact path calls)."""
    label = "server /tokenize"

    def __init__(self, server, model):
        self.server, self.model = server, model

    def render_many(self, reqs, tools, template_kwargs):
        """reqs: [(messages, add_generation_prompt)] -> [ids]."""
        return [self.server.tokenize(self.model, m, tools, template_kwargs, gen) for m, gen in reqs]


class LocalRenderer:
    """DeepSeek-V4.1 /tokenize without a server: MODEL_DIR/encoding/encoding.py (the model's reference encoder) and
    MODEL_DIR/tokenizer.json, applied the way the fork's vllm/tokenizers/deepseek_v41.py renders chat messages (tools on
    the first system message, thinking / reasoning_effort / drop_thinking from chat_template_kwargs; the generation
    header is part of encode_messages). Tokenization is batched (encode_batch runs on every core). On 1,191 recorded
    turns of r1_rerun and b32m_x2t the plans match the recorded prompt_tokens exactly. Loading encoding.py runs code
    from MODEL_DIR: pass only the model's own files."""

    def __init__(self, model_dir):
        import importlib.util
        from tokenizers import Tokenizer
        d = Path(model_dir)
        spec = importlib.util.spec_from_file_location("dsv41_encoding_local", d / "encoding" / "encoding.py")
        self.enc = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.enc)
        self.tok = Tokenizer.from_file(str(d / "tokenizer.json"))
        self.label = f"local {d}"

    def text(self, messages, tools, kw):
        conv = json.loads(json.dumps(messages))
        for m in conv:
            if "reasoning" in m:
                m["reasoning_content"] = m["reasoning"]
            if isinstance(m.get("content"), list):
                if any(isinstance(p, dict) and p.get("type") != "text" for p in m["content"]):
                    raise ValueError("--local-render renders text messages only")
                m["content"] = "\n\n".join(p.get("text", "") for p in m["content"])
        if tools:
            system = next((m for m in conv if m.get("role") == "system"), None)
            if system is None:
                system = {"role": "system", "content": ""}
                conv.insert(0, system)
            system["tools"] = tools
        kw = kw or {}
        thinking = bool(kw.get("thinking") or kw.get("enable_thinking"))
        if "thinking" not in kw and "enable_thinking" not in kw:
            thinking = True
        effort = kw.get("reasoning_effort")
        if effort == "none":
            thinking, effort = False, None
        if effort is None:
            effort = "high"
        return self.enc.encode_messages(conv, thinking_mode="thinking" if thinking else "chat",
                                        drop_thinking=kw.get("drop_thinking", True), reasoning_effort=effort)

    def render_many(self, reqs, tools, template_kwargs):
        texts = [self.text(m, tools, template_kwargs) for m, _ in reqs]
        return [e.ids for e in self.tok.encode_batch(texts, add_special_tokens=False)]


def resolve_tools(spec, attempts, renderer, template_kwargs, cache):
    """--tools -> {source: (tools, label)}. auto: per trace source, the harness flag set whose turn-1 render has the
    recorded prompt length (closest if none is exact; the choice is kept in the prep cache)."""
    sources = {}
    for a in attempts:
        sources.setdefault(a.source, a)
    if spec == "none":
        return {s: (None, "none") for s in sources}
    if spec.startswith("harness:"):
        flags = spec[len("harness:"):].split()
        tools = harness_tools(flags)
        return {s: (tools, "harness:" + " ".join(flags)) for s in sources}
    if spec != "auto":
        tools = json.loads(Path(spec).read_text())
        return {s: (tools, f"file:{spec}") for s in sources}
    out, detected = {}, cache.setdefault("tools", {})
    for source, a in sources.items():
        key = sha1(f"{a.digest}|{json.dumps(template_kwargs, sort_keys=True)}")
        if key in detected:
            flags = detected[key]["flags"]
            out[source] = (harness_tools(flags), "auto:" + (" ".join(flags) or "(no flags)"))
            continue
        target, best = a.turns[0].prompt_rec, None
        msgs = a.messages[:a.turns[0].msg_index]
        for flags in TOOL_CANDIDATES:
            tools = harness_tools(flags)
            try:
                n = len(renderer.render_many([(msgs, True)], tools, template_kwargs)[0])
            except Exception as exc:
                log(f"WARNING render failed ({type(exc).__name__}: {exc}); --tools auto falls back to "
                    f"harness:--answer-tool")
                return {s: (harness_tools(["--answer-tool"]), "fallback:--answer-tool") for s in sources}
            if best is None or abs(n - target) < abs(best[0] - target):
                best = (n, flags, tools)
            if n == target:
                break
        n, flags, tools = best
        detected[key] = {"flags": flags, "diff": n - target}
        if n != target:
            log(f"WARNING {source}: no harness tool set renders turn 1 at its recorded {target} tokens; closest "
                f"{' '.join(flags) or '(no flags)'} at {n}. Check --tools / --chat-template-kwargs.")
        out[source] = (tools, "auto:" + (" ".join(flags) or "(no flags)"))
    return out


# ------------------------------------------------------------------------------------------------------ token plans
@dataclass
class Plan:
    """Token ids of one attempt: turn 1's prompt, then per later turn ("delta", ids, 0): previous prompt + generated
    ids + ids; ("full", ids, 0): ids; ("rebase", ids, keep): previous prompt[:keep] + ids."""
    p1: array.array
    segs: list = field(default_factory=list)


def _arr(ids):
    return array.array("i", ids)


def plan_tokens(att, render_many):
    """Tokens-mode plan, mirroring the harness's --token-exact path: only the messages appended since the previous
    assistant turn are new; a turn after a cut / rewritten turn, or a render that is not prefix-stable, is sent as a
    full render. render_many([(messages, add_generation_prompt)]) -> [ids]. -> (plan, predicted prompt length per
    turn, with the generated history as long as the recorded completions)."""
    t, m = att.turns, att.messages
    reqs = [(m[:t[0].msg_index], True)]
    for k in range(1, len(t)):
        reqs.append((m[:t[k].msg_index], True))
        if not t[k - 1].resync:
            reqs.append((m[:t[k - 1].msg_index + 1], False))
    ids = iter(render_many(reqs))
    p1 = next(ids)
    plan, pred = Plan(_arr(p1)), [len(p1)]
    for k in range(1, len(t)):
        full = next(ids)
        seg = None
        if not t[k - 1].resync:
            upto = next(ids)
            n = pred[-1] + t[k - 1].completion + len(full) - len(upto)
            # the harness also re-renders after a turn whose output was not canonical (parse_fallbacks), which the
            # trace does not flag: the recorded prompt then has the full render's length, not the continued history's
            if full[:len(upto)] == upto and (n == t[k].prompt_rec or len(full) != t[k].prompt_rec):
                seg = ("delta", _arr(full[len(upto):]), 0)
                pred.append(n)
        if seg is None:
            seg = ("full", _arr(full), 0)
            pred.append(len(full))
        plan.segs.append(seg)
    return plan, pred


def plan_synthetic(att, lo, hi):
    """Synthetic plan: random ids with the recorded lengths (turn 1 seeded by the prompt messages, so attempts of one
    task share it as in the real run). -> (plan, predicted prompt length per turn)."""
    t = att.turns
    rng = random.Random("p1|" + att.first_digest)
    vocab = range(lo, hi)
    plan, pred = Plan(_arr(rng.choices(vocab, k=t[0].prompt_rec))), [t[0].prompt_rec]
    rng = random.Random("seg|" + att.digest)
    for k in range(1, len(t)):
        d = t[k].prompt_rec - t[k - 1].prompt_rec - t[k - 1].completion
        if t[k - 1].resync or d < 0:
            keep = min(t[k].prompt_rec, t[k - 1].prompt_rec)
            plan.segs.append(("rebase", _arr(rng.choices(vocab, k=t[k].prompt_rec - keep)), keep))
        else:
            plan.segs.append(("delta", _arr(rng.choices(vocab, k=d)), 0))
        pred.append(t[k].prompt_rec)
    return plan, pred


def _enc(a):
    return base64.b64encode(a.tobytes()).decode()


def _dec(s):
    a = array.array("i")
    a.frombytes(base64.b64decode(s))
    return a


def plan_to_json(plan, pred):
    return {"p1": _enc(plan.p1), "segs": [[k, _enc(ids), keep] for k, ids, keep in plan.segs], "pred": pred}


def plan_from_json(d):
    return Plan(_dec(d["p1"]), [(k, _dec(ids), keep) for k, ids, keep in d["segs"]]), d["pred"]


def load_prep_cache(path):
    if path and Path(path).exists():
        with gzip.open(path, "rt") as f:
            d = json.load(f)
        if d.get("version") == PREP_VERSION:
            return d
        log(f"prep cache {path}: version {d.get('version')} != {PREP_VERSION}, ignored")
    return {"version": PREP_VERSION, "plans": {}, "tools": {}}


def save_prep_cache(path, cache):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tmp = f"{path}.tmp"
    with gzip.open(tmp, "wt") as f:
        json.dump(cache, f)
    os.replace(tmp, path)


def prepare_plans(attempts, args, renderer, tools_by_source, template_kwargs, cache):
    """-> {attempt digest: (plan, predicted lengths)} for tokens / synthetic mode, using and filling the prep cache.
    Cache entries are keyed by trace content + tools + template kwargs (not by server or model name: a cache built with
    --local-render serves a run against the server; check_server_render verifies that on the run)."""
    plans, todo = {}, []
    for a in attempts:
        tools, _ = tools_by_source.get(a.source, (None, ""))
        settings = {"mode": args.mode, "tools": tools, "kwargs": template_kwargs,
                    "vocab": args.synthetic_vocab if args.mode == "synthetic" else None}
        key = f"{a.digest}|{sha1(json.dumps(settings, sort_keys=True))}"
        if key in cache["plans"]:
            plans[a.digest] = plan_from_json(cache["plans"][key])
        else:
            todo.append((a, tools, key))
    if not todo:
        return plans
    if args.mode == "synthetic":
        lo, hi = (int(x) for x in args.synthetic_vocab.split(":"))
        for a, _, key in todo:
            plans[a.digest] = plan_synthetic(a, lo, hi)
            cache["plans"][key] = plan_to_json(*plans[a.digest])
        return plans
    local = isinstance(renderer, LocalRenderer)
    log(f"rendering {len(todo)} attempts ({renderer.label}" + ("" if local else f", {args.prep_workers} workers")
        + ")")
    done, lock, t0 = [0], threading.Lock(), time.monotonic()

    def one(item):
        a, tools, key = item
        try:
            plan, pred = plan_tokens(a, lambda reqs: renderer.render_many(reqs, tools, template_kwargs))
        except Exception as exc:  # the attempt is left out of the run (main reports how many)
            log(f"WARNING render failed for {a.key}: {type(exc).__name__}: {exc}"[:400])
            return
        with lock:
            plans[a.digest] = (plan, pred)
            cache["plans"][key] = plan_to_json(plan, pred)
            done[0] += 1
            if done[0] % 50 == 0 or done[0] == len(todo):
                log(f"  rendered {done[0]}/{len(todo)} attempts ({time.monotonic() - t0:.0f} s)")

    # local: one attempt at a time, its renders tokenized as one parallel batch; server: attempts in parallel
    with ThreadPoolExecutor(1 if local else args.prep_workers) as ex:
        list(ex.map(one, todo))
    return plans


def check_server_render(attempts, plans, server, model, tools_by_source, template_kwargs):
    """Tokens mode with cached or locally rendered plans: does the server's /tokenize give the first attempt's planned
    turn-1 ids? -> "match", "mismatch" (logged loudly) or "unavailable"."""
    a = next((a for a in attempts if a.digest in plans), None)
    if a is None:
        return "unavailable"
    tools, _ = tools_by_source.get(a.source, (None, ""))
    try:
        ids = server.tokenize(model, a.messages[:a.turns[0].msg_index], tools, template_kwargs, True)
    except Exception as exc:
        log(f"WARNING cannot check the plans against the server's /tokenize: {type(exc).__name__}: {exc}"[:300])
        return "unavailable"
    if ids == plans[a.digest][0].p1.tolist():
        return "match"
    log(f"WARNING the server renders {a.key} turn 1 differently from the plan ({len(ids)} vs "
        f"{len(plans[a.digest][0].p1)} tokens): the prep cache / --local-render does not match this server's "
        "tokenizer or template; rebuild it")
    return "mismatch"


def plan_fidelity(attempts, plans):
    """Predicted prompt lengths (generated history = recorded completion lengths) vs the recorded prompt_tokens."""
    diffs, kinds = [], {"delta": 0, "full": 0, "rebase": 0}
    for a in attempts:
        if a.digest not in plans:
            continue
        plan, pred = plans[a.digest]
        diffs += [p - t.prompt_rec for p, t in zip(pred, a.turns)]
        for k, _, _ in plan.segs:
            kinds[k] += 1
    if not diffs:
        return {}
    return {"turns": len(diffs), "prompt_len_exact_frac": rnd(sum(d == 0 for d in diffs) / len(diffs), 4),
            "prompt_len_mean_abs_diff": rnd(sum(abs(d) for d in diffs) / len(diffs), 1),
            "prompt_len_mean_diff": rnd(sum(diffs) / len(diffs), 1), "segments": kinds}


# --------------------------------------------------------------------------------------------------- server log tail
ANSI = re.compile(r"\x1b\[[0-9;]*m")
RANK_RE = re.compile(r"\(([A-Za-z][\w-]*) pid=(\d+)\)")
UNITS = {"GiB": 1.0, "MiB": 1 / 1024, "KiB": 1 / 1024 ** 2, "GB": 1e9 / 2 ** 30, "MB": 1e6 / 2 ** 30}
_Q = r"([\d.]+) (GiB|MiB|KiB|GB|MB)"
MEM_RE = re.compile(rf"GPU memory: {_Q} allocated \(peak {_Q}\), {_Q} reserved(?: \({_Q} in split segments\))?, "
                    rf"{_Q} free, (\d+) allocator retries, (\d+) OOMs")
CAP_RE = re.compile(rf"Capped the PyTorch allocator at {_Q} \({_Q} reserved\)")
KV_RE = re.compile(r"GPU KV cache size: ([\d,]+) tokens")


def _gib(value, unit):
    return float(value) * UNITS[unit]


def parse_log_line(line):
    """One vLLM server log line -> ("mem", rank, {...GiB}), ("cap", rank, {"cap", "reserved"}), ("kv", rank, tokens)
    or None. rank is the process tag, e.g. Worker_TP2 (ANSI colour codes are stripped)."""
    line = ANSI.sub("", line)
    r = RANK_RE.search(line)
    rank = r.group(1) if r else "server"
    m = MEM_RE.search(line)
    if m:
        g = m.groups()
        return ("mem", rank, {"allocated": _gib(g[0], g[1]), "peak": _gib(g[2], g[3]), "reserved": _gib(g[4], g[5]),
                              "split": _gib(g[6], g[7]) if g[6] else None, "free": _gib(g[8], g[9]),
                              "retries": int(g[10]), "ooms": int(g[11])})
    m = CAP_RE.search(line)
    if m:
        g = m.groups()
        return ("cap", rank, {"cap": _gib(g[0], g[1]), "reserved": _gib(g[2], g[3])})
    m = KV_RE.search(line)
    if m:
        return ("kv", rank, int(m.group(1).replace(",", "")))
    return None


class LogTail:
    """Follows a vLLM server log. The first read parses the whole file (the allocator cap and KV size are logged at
    startup); window() returns the memory state since the previous window and resets the per-window extremes."""

    def __init__(self, path):
        self.path, self.pos, self.buf = path, 0, b""
        self.caps, self.last, self.kv_tokens = {}, {}, None
        self.win = {}
        self.prev_totals = None
        self.read()
        self.win = {}

    def read(self):
        try:
            size = os.path.getsize(self.path)
        except OSError:
            return
        if size < self.pos:  # truncated / rotated
            self.pos, self.buf = 0, b""
        with open(self.path, "rb") as f:
            f.seek(self.pos)
            data = f.read()
            self.pos = f.tell()
        lines = (self.buf + data).split(b"\n")
        self.buf = lines.pop()
        for raw in lines:
            rec = parse_log_line(raw.decode(errors="replace"))
            if rec is None:
                continue
            kind, rank, v = rec
            if kind == "cap":
                self.caps[rank] = v["cap"]
            elif kind == "kv":
                self.kv_tokens = v
            else:
                self.last[rank] = v
                w = self.win.setdefault(rank, {"peak": v["peak"], "allocated": v["allocated"],
                                               "reserved": v["reserved"], "free": v["free"], "lines": 0})
                w["peak"], w["allocated"] = max(w["peak"], v["peak"]), max(w["allocated"], v["allocated"])
                w["reserved"], w["free"] = max(w["reserved"], v["reserved"]), min(w["free"], v["free"])
                w["lines"] += 1

    def window(self):
        self.read()
        out = memory_window(self.caps, self.last, self.win, self.prev_totals)
        self.prev_totals = (out.get("mem_retries_total"), out.get("mem_ooms_total"))
        self.win = {}
        return out


def memory_window(caps, last, win, prev_totals=None):
    """Per-window memory fields: max allocated / peak / reserved and min free over ranks (this window's lines, else the
    latest), allocator cap, headroom = min over ranks of cap - peak, and allocator retries / OOMs (totals and the
    increase since the previous window)."""
    ranks = sorted(set(last) | set(win))
    if not ranks:
        return {"mem_cap_min_gib": rnd(min(caps.values()), 3) if caps else None}
    state = {r: win.get(r) or last[r] for r in ranks}
    head = [caps[r] - state[r]["peak"] for r in ranks if r in caps]
    retries = sum(last[r]["retries"] for r in ranks if r in last)
    ooms = sum(last[r]["ooms"] for r in ranks if r in last)
    out = {"mem_ranks": len(ranks), "mem_lines": sum(w["lines"] for w in win.values()),
           "mem_allocated_max_gib": rnd(max(s["allocated"] for s in state.values())),
           "mem_peak_max_gib": rnd(max(s["peak"] for s in state.values())),
           "mem_reserved_max_gib": rnd(max(s["reserved"] for s in state.values())),
           "mem_free_min_gib": rnd(min(s["free"] for s in state.values())),
           "mem_cap_min_gib": rnd(min(caps.values())) if caps else None,
           "mem_headroom_min_gib": rnd(min(head)) if head else None,
           "mem_retries_total": retries, "mem_ooms_total": ooms}
    if prev_totals and prev_totals[0] is not None:
        out["mem_retries"], out["mem_ooms"] = retries - prev_totals[0], ooms - prev_totals[1]
    else:
        out["mem_retries"], out["mem_ooms"] = retries, ooms
    return out


# ------------------------------------------------------------------------------------------------------- windows
def new_acc():
    return {"gen": 0, "prompt": 0, "cached": 0, "cached_n": 0, "ttft": [], "latency": [], "itl_ms": [], "turns": 0,
            "attempts": 0, "errors": 0, "aborted": 0}


def make_window(t0, t1, m0, m1, acc, mem=None, gauges=None, warmup_s=0.0, tail_at=None):
    """One window record from the metric snapshots at its ends (m0, m1; None if /metrics was unreachable), the client
    accumulator of the window, the memory fields and sampled gauges. Raw counter deltas are kept (prefix_queries, ...)
    so summaries can be summed exactly."""
    dt = max(t1 - t0, 1e-9)
    w = {"t0": rnd(t0, 2), "t1": rnd(t1, 2), "dt": rnd(dt, 2), "warmup": t0 < warmup_s - 1e-6,
         "tail": tail_at is not None and t1 > tail_at + 0.05 * dt,  # > 5% of the window past it
         "client_gen_tokens": acc["gen"], "client_prompt_tokens": acc["prompt"],
         "client_gen_tok_s": rnd(acc["gen"] / dt, 1), "client_prompt_tok_s": rnd(acc["prompt"] / dt, 1),
         "turns_done": acc["turns"], "attempts_done": acc["attempts"], "errors": acc["errors"],
         "ttft_p50": rnd(pct(acc["ttft"], 50)), "ttft_p90": rnd(pct(acc["ttft"], 90)),
         "ttft_p99": rnd(pct(acc["ttft"], 99)), "latency_p50": rnd(pct(acc["latency"], 50)),
         "latency_p90": rnd(pct(acc["latency"], 90)), "latency_p99": rnd(pct(acc["latency"], 99)),
         "itl_ms_p50": rnd(pct(acc["itl_ms"], 50), 1), "itl_ms_p90": rnd(pct(acc["itl_ms"], 90), 1)}
    if acc["cached_n"]:
        w["client_cached_frac"] = rnd(acc["cached"] / max(1, acc["prompt"]), 4)
    w.update(gauges or {})
    reset = False
    if m0 is not None and m1 is not None:
        def d(k):
            if k not in m0 or k not in m1:
                return None
            return m1[k] - m0[k]
        deltas = {k: d(k) for k in COUNTERS}
        reset = any(v is not None and v < 0 for v in deltas.values())
        if not reset:
            q, h, ext = (deltas["prefix_cache_queries_total"], deltas["prefix_cache_hits_total"],
                         deltas["external_prefix_cache_hits_total"])
            w.update({"gen_tokens": deltas["generation_tokens_total"], "prompt_tokens": deltas["prompt_tokens_total"],
                      "prefix_queries": q, "prefix_hits": h, "external_hits": ext,
                      "preemptions": deltas["num_preemptions_total"]})
            gen, prm = deltas["generation_tokens_total"], deltas["prompt_tokens_total"]
            w["gen_tok_s"] = rnd(gen / dt, 1) if gen is not None else None
            w["prompt_tok_s"] = rnd(prm / dt, 1) if prm is not None else None
            if q is not None and h is not None:
                w["prefix_hit"] = rnd(h / q, 4) if q > 0 else None
                w["uncached_tok_s"] = rnd((q - h - (ext or 0)) / dt, 1)
            if ext is not None:
                w["external_hit_tok_s"] = rnd(ext / dt, 1)
    w["counter_reset"] = reset
    if m1 is not None:
        w.update({"kv_usage": rnd(m1.get("kv_cache_usage_perc"), 4),
                  "kv_usage_max": rnd(m1.get("kv_cache_usage_max"), 4),
                  "running": m1.get("num_requests_running"), "waiting": m1.get("num_requests_waiting"),
                  "preemptions_total": m1.get("num_preemptions_total")})
    w.update(mem or {})
    return w


def summarize_windows(ws):
    """Rates over a set of windows from summed raw counts (exact), dt-weighted gauge means, extremes."""
    if not ws:
        return {}
    dt = sum(w["dt"] for w in ws)

    def total(k):
        vals = [w[k] for w in ws if w.get(k) is not None]
        return sum(vals) if vals else None

    def tdt(k):  # dt of the windows that have k
        return sum(w["dt"] for w in ws if w.get(k) is not None)

    def mean(k):
        vals = [(w[k], w["dt"]) for w in ws if w.get(k) is not None]
        return sum(v * t for v, t in vals) / sum(t for _, t in vals) if vals else None

    def ext(k, f):
        vals = [w[k] for w in ws if w.get(k) is not None]
        return f(vals) if vals else None

    q, h, e = total("prefix_queries"), total("prefix_hits"), total("external_hits")
    gen, prm = total("gen_tokens"), total("prompt_tokens")
    s = {"t0": ws[0]["t0"], "t1": ws[-1]["t1"], "dt": rnd(dt, 1), "windows": len(ws),
         "gen_tok_s": rnd(gen / tdt("gen_tokens"), 1) if gen is not None and tdt("gen_tokens") else None,
         "prompt_tok_s": rnd(prm / tdt("prompt_tokens"), 1) if prm is not None and tdt("prompt_tokens") else None,
         "client_gen_tok_s": rnd(total("client_gen_tokens") / dt, 1),
         "client_prompt_tok_s": rnd(total("client_prompt_tokens") / dt, 1),
         "prefix_hit": rnd(h / q, 4) if q else None,
         "uncached_tok_s": rnd((q - h - (e or 0)) / tdt("prefix_queries"), 1) if q is not None and tdt("prefix_queries")
         else None,
         "external_hit_tok_s": rnd(e / tdt("external_hits"), 1) if e is not None and tdt("external_hits") else None,
         "kv_usage_mean": rnd(mean("kv_usage"), 4), "kv_usage_max": rnd(ext("kv_usage_max", max) or
                                                                      ext("kv_usage", max), 4),
         "running_mean": rnd(mean("running"), 1), "waiting_mean": rnd(mean("waiting"), 1),
         "attempts_in_flight_mean": rnd(mean("attempts_in_flight"), 1),
         "preemptions": total("preemptions"), "turns_done": total("turns_done"),
         "attempts_done": total("attempts_done"), "errors": total("errors"),
         "counter_resets": sum(1 for w in ws if w.get("counter_reset")),
         "mem_headroom_min_gib": ext("mem_headroom_min_gib", min), "mem_peak_max_gib": ext("mem_peak_max_gib", max),
         "mem_reserved_max_gib": ext("mem_reserved_max_gib", max), "mem_free_min_gib": ext("mem_free_min_gib", min),
         "mem_cap_min_gib": ext("mem_cap_min_gib", min), "mem_ooms": total("mem_ooms"),
         "mem_retries": total("mem_retries")}
    return s


def summarize_requests(reqs):
    ok = [r for r in reqs if not r.get("error") and not r.get("aborted")]
    s = {"requests": len(reqs), "ok": len(ok), "errors": sum(1 for r in reqs if r.get("error")),
         "aborted": sum(1 for r in reqs if r.get("aborted"))}
    for k in ("ttft", "latency"):
        for q in (50, 90, 99):
            s[f"{k}_p{q}"] = rnd(pct([r.get(k) for r in ok], q))
    for q in (50, 90):
        s[f"itl_ms_p{q}"] = rnd(pct([r.get("itl_ms") for r in ok], q), 1)
    s["output_len_mismatch"] = sum(1 for r in ok if r["completion_served"] != r["completion_requested"])
    s["connect_retries"] = sum(r.get("retries") or 0 for r in reqs)
    s["missing_token_ids"] = sum(1 for r in ok if r.get("ids_missing"))
    served = [(r["prompt_served"], r["prompt_recorded"]) for r in ok if r.get("prompt_served") is not None]
    if served:
        diffs = [a - b for a, b in served]
        s["prompt_vs_recorded_exact_frac"] = rnd(sum(d == 0 for d in diffs) / len(diffs), 4)
        s["prompt_vs_recorded_mean_abs_diff"] = rnd(sum(abs(d) for d in diffs) / len(diffs), 1)
    rec = [(r["cached_recorded"], r["prompt_recorded"]) for r in ok if r.get("cached_recorded") is not None]
    if rec and sum(p for _, p in rec):
        s["recorded_prefix_hit"] = rnd(sum(c for c, _ in rec) / sum(p for _, p in rec), 4)
    srv = [(r["cached_served"], r["prompt_served"]) for r in ok if r.get("cached_served") is not None
           and r.get("prompt_served")]
    if srv:
        s["served_prefix_hit"] = rnd(sum(c for c, _ in srv) / sum(p for _, p in srv), 4)
    return s


# -------------------------------------------------------------------------------------------------------- the bench
class Bench:
    def __init__(self, args, server, model, attempts, plans, tools_by_source, template_kwargs, out):
        self.args, self.server, self.model, self.out = args, server, model, out
        self.attempts, self.plans, self.tools = attempts, plans, tools_by_source
        self.template_kwargs = template_kwargs
        self.extra = json.loads(args.extra_body) if args.extra_body else {}
        self.salt = None if args.cache_salt == "none" else (
            args.cache_salt if args.cache_salt != "auto" else f"replay-{uuid.uuid4().hex[:12]}")
        self.lock = threading.Lock()
        self.stop = threading.Event()  # abort in-flight requests and stop launching
        self.no_new = threading.Event()  # stop launching (in-flight attempts finish: --drain)
        self.finished = threading.Event()
        self.acc = new_acc()
        self.requests, self.windows = [], []
        self.in_flight = {"attempts": 0, "requests": 0}
        self.started = self.completed = 0
        self.tail_at = None
        self.errors = 0
        self.launches = []  # (pass, order index, attempt key) in launch order
        self._iter = self._attempt_iter()
        self.req_file = open(out / "requests.jsonl", "w")
        self.win_file = open(out / "windows.jsonl", "w")
        self.logtail = LogTail(args.server_log) if args.server_log else None
        self.t0 = None

    def now(self):
        return time.monotonic() - self.t0

    def _attempt_iter(self):
        n, p = 0, 0
        limit = self.args.attempts
        cycle = bool(limit and limit > len(self.attempts)) or (self.args.duration and not limit)
        while True:
            for i, a in enumerate(attempt_order(self.attempts, self.args.seed, self.args.order, p)):
                if limit and n >= limit:
                    return
                yield p, i, a
                n += 1
            p += 1
            if not cycle:
                return

    def next_attempt(self):
        with self.lock:
            if self.no_new.is_set() or self.stop.is_set():
                return None
            item = next(self._iter, None)
            if item is None:
                if self.tail_at is None:
                    self.tail_at = self.now()
                return None
            self.started += 1
            self.launches.append((item[0], item[1], item[2].key))
            return item

    def on_tokens(self, n):
        with self.lock:
            self.acc["gen"] += n

    def worker(self, slot):
        while not self.stop.is_set():
            item = self.next_attempt()
            if item is None:
                return
            p, i, att = item
            with self.lock:
                self.in_flight["attempts"] += 1
            try:
                ok = self.run_attempt(att, p, slot)
            finally:
                with self.lock:
                    self.in_flight["attempts"] -= 1
            if ok:
                with self.lock:
                    self.completed += 1
                    self.acc["attempts"] += 1
            elif not self.stop.is_set():
                with self.lock:
                    self.errors += 1
                    too_many = self.errors > self.args.max_errors
                if too_many:
                    log(f"more than {self.args.max_errors} failed attempts: stopping the run")
                    self.stop.set()
                    return
                self.stop.wait(1.0)  # a dead server must not become a hot loop

    def _body(self, att, k, prompt):
        t = att.turns[k]
        body = {"model": self.model, "max_tokens": t.completion, "min_tokens": t.completion, "ignore_eos": True,
                "temperature": self.args.temperature, "stream": True,
                "stream_options": {"include_usage": True, "continuous_usage_stats": True}}
        if self.args.mode == "chat":
            tools, _ = self.tools.get(att.source, (None, ""))
            body["messages"] = att.messages[:t.msg_index]
            if tools:
                body["tools"], body["tool_choice"] = tools, "auto"
            if self.template_kwargs:
                body["chat_template_kwargs"] = self.template_kwargs
        else:
            body.update({"prompt": prompt, "return_token_ids": True, "skip_special_tokens": False,
                         "spaces_between_special_tokens": False})
        return body

    def run_attempt(self, att, pass_index, slot):
        """Issue the attempt's turns in order. -> True if every turn completed."""
        plan = self.plans.get(att.digest, (None, None))[0]
        prev_prompt, prev_gen = None, None
        filler = random.Random("gen|" + att.digest)
        for k, t in enumerate(att.turns):
            if self.stop.is_set():
                return False
            prompt = None
            if plan is not None:
                if k == 0:
                    prompt = plan.p1.tolist()
                else:
                    kind, ids, keep = plan.segs[k - 1]
                    if kind == "delta":
                        prompt = prev_prompt + prev_gen + ids.tolist()
                    elif kind == "full":
                        prompt = ids.tolist()
                    else:
                        prompt = prev_prompt[:keep] + ids.tolist()
            body = self._body(att, k, prompt)
            if self.salt:
                body["cache_salt"] = f"{self.salt}-p{pass_index}"
            body.update(self.extra)
            path = "/v1/chat/completions" if self.args.mode == "chat" else "/v1/completions"
            with self.lock:
                self.in_flight["requests"] += 1
            t_start = time.monotonic()
            try:
                res = self.server.stream(path, body, self.stop, self.on_tokens)
            finally:
                with self.lock:
                    self.in_flight["requests"] -= 1
            failed = bool(res["error"] or res["aborted"])
            ids_missing = plan is not None and not failed and not res["ids_returned"]
            self._record(att, pass_index, slot, k, t, t_start, res, prompt, ids_missing)
            if failed:
                if res["error"]:
                    log(f"turn error {att.key} turn {k + 1}: {res['error']}")
                return False
            if plan is not None:
                # the next prompt continues from what the server generated (cached blocks); without returned ids the
                # history gets filler ids of the same length, whose blocks the server never cached
                gen = res["ids"] if not ids_missing else filler.choices(range(1000, 30000), k=res["completion"])
                prev_prompt, prev_gen = prompt, gen
            if k + 1 < len(att.turns) and self.args.think_time > 0 and t.n_tools:
                if self.stop.wait(self.args.think_time * t.n_tools):
                    return False
        return True

    def _record(self, att, pass_index, slot, k, t, t_start, res, prompt, ids_missing=False):
        ttft = res["t_first"] - t_start if res["t_first"] is not None else None
        latency = res["t_end"] - t_start
        itl = ((res["t_end"] - res["t_first"]) * 1000 / (res["completion"] - 1)
               if res["t_first"] is not None and res["completion"] > 1 else None)
        prompt_served = res["prompt"] if res["prompt"] is not None else (len(prompt) if prompt is not None else None)
        rec = {"attempt": att.key, "source": att.source, "pass": pass_index, "turn": k + 1, "slot": slot,
               "t_start": rnd(t_start - self.t0, 3), "t_end": rnd(res["t_end"] - self.t0, 3),
               "ttft": rnd(ttft, 4), "latency": rnd(latency, 4), "itl_ms": rnd(itl, 2),
               "prompt_sent": len(prompt) if prompt is not None else None, "prompt_served": prompt_served,
               "prompt_recorded": t.prompt_rec, "cached_served": res["cached"], "cached_recorded": t.cached_rec,
               "completion_requested": t.completion, "completion_served": res["completion"],
               "finish": res["finish"], "error": res["error"], "aborted": res["aborted"]}
        if res.get("retries"):
            rec["retries"] = res["retries"]
        if ids_missing:
            rec["ids_missing"] = True
        with self.lock:
            self.requests.append(rec)
            if res["error"]:
                self.acc["errors"] += 1
            elif res["aborted"]:
                self.acc["aborted"] += 1
            else:
                self.acc["turns"] += 1
                self.acc["prompt"] += prompt_served or 0
                if res["cached"] is not None:
                    self.acc["cached"] += res["cached"]
                    self.acc["cached_n"] += 1
                self.acc["ttft"].append(ttft)
                self.acc["latency"].append(latency)
                if itl is not None:
                    self.acc["itl_ms"].append(itl)
            self.req_file.write(json.dumps(rec) + "\n")
            self.req_file.flush()
        return rec

    def tick(self, t1, m_prev, t_prev):
        m1 = self.server.metrics()
        mem = self.logtail.window() if self.logtail else None
        with self.lock:
            acc, self.acc = self.acc, new_acc()
            gauges = {"attempts_in_flight": self.in_flight["attempts"],
                      "requests_in_flight": self.in_flight["requests"]}
        # the tail: after the first slot found no attempt left to start, or after --duration
        tail_at = self.tail_at if self.tail_at is not None else self.args.duration
        w = make_window(t_prev, t1, m_prev, m1, acc, mem, gauges, self.args.warmup_s, tail_at)
        self.windows.append(w)
        self.win_file.write(json.dumps(w) + "\n")
        self.win_file.flush()
        if not self.args.quiet:
            hit = w.get("prefix_hit")
            log(f"t={t1:6.0f}s gen {w.get('gen_tok_s')} tok/s (client {w['client_gen_tok_s']}) prompt "
                f"{w.get('prompt_tok_s')} uncached {w.get('uncached_tok_s')} hit {hit} KV {w.get('kv_usage')} "
                f"run/wait {w.get('running')}/{w.get('waiting')} preempt+{w.get('preemptions')} TTFT p50 "
                f"{w['ttft_p50']} | attempts {gauges['attempts_in_flight']} in flight, {self.completed} done"
                + (f" | headroom {w.get('mem_headroom_min_gib')} GiB" if mem else "")
                + (" [warm-up]" if w["warmup"] else "") + (" [tail]" if w["tail"] else ""))
        # a failed scrape gives the next window no server deltas (rather than two windows' counts over one dt)
        return m1

    def monitor(self, m_start):
        m_prev, t_prev, k = m_start, 0.0, 1
        while True:
            target = k * self.args.window_s
            if self.finished.wait(max(0.0, target - self.now())):
                break
            t1 = self.now()
            m_prev = self.tick(t1, m_prev, t_prev)
            t_prev, k = t1, max(k + 1, int(self.now() // self.args.window_s) + 1)
        t_end = self.now()
        if t_end > t_prev or not self.windows:
            self.tick(t_end, m_prev, t_prev)

    def run(self):
        a = self.args
        m_start = self.server.metrics()
        if m_start is None:
            log("WARNING /metrics unreachable: server-side fields will be missing")
        if self.logtail:
            self.logtail.window()  # the startup state; windows report changes from here
        self.t0 = time.monotonic()
        mon = threading.Thread(target=self.monitor, args=(m_start,), daemon=True)
        mon.start()
        workers = [threading.Thread(target=self.worker, args=(i,), daemon=True) for i in range(a.concurrency)]
        for w in workers:
            w.start()
        deadline = a.duration if a.duration else None
        while any(w.is_alive() for w in workers):
            if deadline is not None and self.now() >= deadline:
                if a.drain:
                    self.no_new.set()
                else:
                    self.stop.set()
                break
            time.sleep(0.05)
        if deadline is not None:
            self.tail_at = self.tail_at if self.tail_at is not None else deadline
        for w in workers:
            w.join(timeout=None if a.drain else 60)
        self.stop.set()
        self.finished.set()
        mon.join(timeout=60)
        self.req_file.close()
        self.win_file.close()
        return self.now()


# ------------------------------------------------------------------------------------------------------------ report
def _f(x, nd=1):
    if x is None:
        return "-"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def write_report(path, summary, windows):
    c, s, al, rq = summary["config"], summary.get("steady") or {}, summary.get("all") or {}, summary["requests_steady"]
    rqa = summary["requests_all"]
    L = [f"# Serve replay bench: {Path(c['out']).name}", "",
         f"- Server: {c['base_url']} model `{summary['server'].get('model')}`, "
         f"vLLM {summary['server'].get('version')}, "
         f"KV cache {summary['server'].get('kv_cache_tokens') or '-'} tokens (server log)",
         f"- Workload: {summary['workload']['attempts']} attempts, {summary['workload']['turns']} turns from "
         f"{len(c['traces'])} trace spec(s); mode `{c['mode']}`, concurrency {c['concurrency']}, seed {c['seed']}, "
         f"order {c['order']}, think time {c['think_time']} s per tool result, temperature {c['temperature']}",
         f"- Run: {summary['run']['elapsed_s']} s, {summary['run']['attempts_started']} attempts started, "
         f"{summary['run']['attempts_completed']} completed, {summary['run']['turns_completed']} turns, "
         f"{summary['run']['attempt_errors']} failed attempts; warm-up {c['warmup_s']} s",
         f"- Steady interval: t = {_f(s.get('t0'), 0)}-{_f(s.get('t1'), 0)} s (after warm-up, before the tail where "
         f"fewer than {c['concurrency']} attempts remain)" if s else "- Steady interval: none (all windows are "
         "warm-up or tail; see the All column)", "",
         "| Metric | Steady | All |", "|---|---|---|"]
    rows = [("Generated tok/s (server)", "gen_tok_s", 1), ("Generated tok/s (client)", "client_gen_tok_s", 1),
            ("Prompt tok/s (server)", "prompt_tok_s", 1), ("Uncached prompt tok/s", "uncached_tok_s", 1),
            ("Prefix-cache hit rate", "prefix_hit", 4), ("External (offload) hit tok/s", "external_hit_tok_s", 1),
            ("KV usage mean", "kv_usage_mean", 3), ("KV usage max", "kv_usage_max", 3),
            ("Running / waiting (mean)", None, 1), ("Preemptions", "preemptions", 0),
            ("Turns / attempts done", None, 0)]
    for label, key, nd in rows:
        if key is None and label.startswith("Running"):
            L.append(f"| {label} | {_f(s.get('running_mean'))} / {_f(s.get('waiting_mean'))} | "
                     f"{_f(al.get('running_mean'))} / {_f(al.get('waiting_mean'))} |")
        elif key is None:
            L.append(f"| {label} | {_f(s.get('turns_done'))} / {_f(s.get('attempts_done'))} | "
                     f"{_f(al.get('turns_done'))} / {_f(al.get('attempts_done'))} |")
        else:
            L.append(f"| {label} | {_f(s.get(key), nd)} | {_f(al.get(key), nd)} |")
    for label, k in (("TTFT p50 / p90 / p99 (s)", "ttft"), ("Turn latency p50 / p90 / p99 (s)", "latency")):
        L.append(f"| {label} | " + " / ".join(_f(rq.get(f"{k}_p{q}"), 2) for q in (50, 90, 99)) + " | "
                 + " / ".join(_f(rqa.get(f"{k}_p{q}"), 2) for q in (50, 90, 99)) + " |")
    L.append(f"| Inter-token latency p50 / p90 (ms) | {_f(rq.get('itl_ms_p50'))} / {_f(rq.get('itl_ms_p90'))} | "
             f"{_f(rqa.get('itl_ms_p50'))} / {_f(rqa.get('itl_ms_p90'))} |")
    if al.get("mem_cap_min_gib") is not None or al.get("mem_peak_max_gib") is not None:
        L += [f"| Allocator headroom min, cap - peak (GiB) | {_f(s.get('mem_headroom_min_gib'), 2)} | "
              f"{_f(al.get('mem_headroom_min_gib'), 2)} |",
              f"| Peak allocated max / cap min (GiB) | {_f(s.get('mem_peak_max_gib'), 2)} / "
              f"{_f(s.get('mem_cap_min_gib'), 2)} | {_f(al.get('mem_peak_max_gib'), 2)} / "
              f"{_f(al.get('mem_cap_min_gib'), 2)} |",
              f"| OOMs / allocator retries | {_f(s.get('mem_ooms'))} / {_f(s.get('mem_retries'))} | "
              f"{_f(al.get('mem_ooms'))} / {_f(al.get('mem_retries'))} |"]
    fid = summary.get("fidelity") or {}
    L += ["", "## Fidelity", "",
          "- Tools: " + "; ".join(f"{Path(k).name or k}: {v}" for k, v in (fid.get("tools") or {}).items()),
          f"- Planned prompt lengths vs recorded: {json.dumps(fid.get('plan') or {})}",
          f"- Served prompt tokens vs recorded: exact {_f(rqa.get('prompt_vs_recorded_exact_frac'), 4)}, mean |diff| "
          f"{_f(rqa.get('prompt_vs_recorded_mean_abs_diff'))} tokens",
          f"- Prefix-cache hit: recorded in the traces {_f(rqa.get('recorded_prefix_hit'), 4)}, served (usage) "
          f"{_f(rqa.get('served_prefix_hit'), 4)}, server metrics {_f(al.get('prefix_hit'), 4)}",
          f"- Output length mismatches: {rqa.get('output_len_mismatch')}; turns without returned token ids: "
          f"{rqa.get('missing_token_ids')}; request errors {rqa.get('errors')}, aborted at the end "
          f"{rqa.get('aborted')}; connection retries {rqa.get('connect_retries')}",
          "", "## Windows", ""]
    rows = windows
    if len(windows) > 60:  # merge consecutive windows into at most 60 rows
        k = math.ceil(len(windows) / 60)
        rows = []
        for i in range(0, len(windows), k):
            g = windows[i:i + k]
            m = summarize_windows(g)
            m.update({"warmup": g[0]["warmup"], "tail": g[-1]["tail"], "ttft_p50": g[len(g) // 2].get("ttft_p50"),
                      "ttft_p90": max((x["ttft_p90"] for x in g if x.get("ttft_p90") is not None), default=None),
                      "latency_p50": g[len(g) // 2].get("latency_p50"), "running": g[-1].get("running"),
                      "waiting": g[-1].get("waiting"), "kv_usage": g[-1].get("kv_usage"),
                      "attempts_in_flight": g[-1].get("attempts_in_flight")})
            rows.append(m)
        L.append(f"({len(windows)} windows merged by {k}; windows.jsonl has every one.)")
        L.append("")
    L += ["| t (s) | gen tok/s | client gen | prompt tok/s | uncached tok/s | hit | KV | run/wait | preempt | "
          "TTFT p50/p90 (s) | turn p50 (s) | attempts | headroom GiB |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for w in rows:
        flag = " (warm-up)" if w.get("warmup") else (" (tail)" if w.get("tail") else "")
        L.append(f"| {_f(w['t1'], 0)}{flag} | {_f(w.get('gen_tok_s'))} | {_f(w.get('client_gen_tok_s'))} | "
                 f"{_f(w.get('prompt_tok_s'))} | {_f(w.get('uncached_tok_s'))} | {_f(w.get('prefix_hit'), 3)} | "
                 f"{_f(w.get('kv_usage'), 3)} | {_f(w.get('running'), 0)}/{_f(w.get('waiting'), 0)} | "
                 f"{_f(w.get('preemptions'), 0)} | {_f(w.get('ttft_p50'), 2)}/{_f(w.get('ttft_p90'), 2)} | "
                 f"{_f(w.get('latency_p50'), 1)} | {_f(w.get('attempts_in_flight'), 0)} | "
                 f"{_f(w.get('mem_headroom_min_gib'), 2)} |")
    Path(path).write_text("\n".join(L) + "\n")


# -------------------------------------------------------------------------------------------------------------- main
def parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", help="server root or its /v1 URL, e.g. http://127.0.0.1:8000 (not needed for "
                    "--prepare-only --local-render)")
    ap.add_argument("--model", help="served model name (default: the first id of /v1/models)")
    ap.add_argument("--api-key", default=os.environ.get("SERVE_BENCH_API_KEY"),
                    help="bearer token (default $SERVE_BENCH_API_KEY; our vLLM servers need none)")
    ap.add_argument("--traces", nargs="+", required=True,
                    help="trace dirs (recursive, tarballs inside included), dsv41_traces.tar.gz files, files or globs")
    ap.add_argument("--mode", choices=("tokens", "chat", "synthetic"), default="tokens")
    ap.add_argument("--concurrency", type=int, default=64, help="attempts in flight")
    ap.add_argument("--duration", type=float, help="seconds; then stop launching and abort in-flight requests "
                    "(see --drain). Without --attempts, the traces are replayed in passes until then")
    ap.add_argument("--attempts", type=int, help="attempts to launch (cycles through the traces in passes if larger). "
                    "Default without --duration: one pass over every attempt")
    ap.add_argument("--drain", action="store_true", help="at --duration, let in-flight attempts finish")
    ap.add_argument("--seed", type=int, default=0, help="launch-order seed")
    ap.add_argument("--order", choices=("shuffle", "heavy-first", "light-first"), default="shuffle",
                    help="heavy-first loads the server like the late hours of a Kaggle run")
    ap.add_argument("--warmup-s", type=float, default=300.0, help="windows starting before this are excluded from the "
                    "steady summary")
    ap.add_argument("--window-s", type=float, default=30.0)
    ap.add_argument("--think-time", type=float, default=0.0, help="seconds per tool result between turns")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--tools", default="auto", help="auto (per trace source, the evals/run_deepseek_tools.py flag set "
                    "matching the recorded turn-1 prompt), none, harness:'<flags>' (e.g. harness:'--answer-tool "
                    "--multi-test') or a JSON file of tool schemas. tokens/chat modes only")
    ap.add_argument("--chat-template-kwargs", default=DEFAULT_TEMPLATE_KWARGS,
                    help=f"JSON for /tokenize and chat requests (default {DEFAULT_TEMPLATE_KWARGS}, the B32 solver's)")
    ap.add_argument("--extra-body", help="JSON merged into every generation request")
    ap.add_argument("--cache-salt", default="auto", help="auto: a fresh salt per run (+ pass index), so nothing cached "
                    "by an earlier run on the same server is reused; none: omit; or a fixed string")
    ap.add_argument("--prep-cache", help="tokens mode: rendered token ids are kept here (.json.gz) and reused")
    ap.add_argument("--local-render", metavar="MODEL_DIR", help="tokens mode, DeepSeek-V4.1: render with MODEL_DIR's "
                    "encoding/encoding.py + tokenizer.json instead of the server's /tokenize (build the prep cache "
                    "on a CPU box: --prepare-only --local-render DIR --prep-cache F, no server needed)")
    ap.add_argument("--prep-workers", type=int, default=8)
    ap.add_argument("--prepare-only", action="store_true", help="load traces, render / plan, report fidelity, exit")
    ap.add_argument("--synthetic-vocab", default="1000:30000", help="synthetic mode: token id range lo:hi")
    ap.add_argument("--server-log", help="vLLM server log to tail for GPU memory / allocator cap lines")
    ap.add_argument("--max-errors", type=int, default=50, help="stop the run after this many failed attempts")
    ap.add_argument("--request-timeout", type=float, default=3600.0)
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--quiet", action="store_true", help="log warnings only")
    return ap


def main(argv=None):
    global QUIET
    args = parser().parse_args(argv)
    QUIET = args.quiet
    if args.concurrency <= 0 or args.window_s <= 0:
        raise SystemExit("--concurrency and --window-s must be positive")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    template_kwargs = json.loads(args.chat_template_kwargs) if args.chat_template_kwargs else {}
    attempts, load_stats = load_attempts(args.traces)
    images = [a for a in attempts if a.images]
    if images and args.mode == "tokens":
        log(f"WARNING {len(images)} attempts carry images: token ids hold no pixels, skipped (use --mode chat)")
        attempts = [a for a in attempts if not a.images]
    if not attempts:
        raise SystemExit("no usable attempts in --traces")
    log(f"{len(attempts)} attempts, {sum(len(a.turns) for a in attempts)} turns "
        f"({load_stats['duplicates']} duplicate files, {load_stats['not_traces']} non-trace files, "
        f"{load_stats['dropped_turns']} turns without usage dropped)")

    if args.local_render and args.mode != "tokens":
        raise SystemExit("--local-render is for --mode tokens")
    offline = bool(args.prepare_only and args.local_render)
    if not args.base_url and not offline:
        raise SystemExit("--base-url is required (except for --prepare-only --local-render)")
    server, model = None, args.model
    if args.base_url:
        server = Server(args.base_url, args.api_key, args.request_timeout)
        try:
            model = model or server.models()[0]
        except Exception as exc:
            if not offline:
                raise SystemExit(f"server not reachable at {server.root}/v1/models: {type(exc).__name__}: {exc}")
            server = None
    renderer = LocalRenderer(args.local_render) if args.local_render else (
        ServerRenderer(server, model) if server else None)
    cache = load_prep_cache(args.prep_cache) if args.mode == "tokens" else {"version": PREP_VERSION, "plans": {},
                                                                              "tools": {}}
    tools_by_source = {}
    if args.mode in ("tokens", "chat"):
        tools_by_source = resolve_tools(args.tools, attempts, renderer, template_kwargs, cache)
        distinct = {json.dumps(t, sort_keys=True): t for t, _ in tools_by_source.values() if t}
        for i, t in enumerate(distinct.values()):  # for --tools FILE on a box without evals/run_deepseek_tools.py
            (out / ("tools.json" if len(distinct) == 1 else f"tools_{i}.json")).write_text(json.dumps(t, indent=1))
    plans = {}
    if args.mode in ("tokens", "synthetic"):
        plans = prepare_plans(attempts, args, renderer, tools_by_source, template_kwargs, cache)
        if args.prep_cache and args.mode == "tokens":
            save_prep_cache(args.prep_cache, cache)
        failed = [a for a in attempts if a.digest not in plans]
        if failed:
            log(f"WARNING {len(failed)} attempts could not be rendered and are left out")
            attempts = [a for a in attempts if a.digest in plans]
            if not attempts:
                raise SystemExit("no attempt could be rendered")
    fid = {"tools": {s: label for s, (_, label) in tools_by_source.items()}, "plan": plan_fidelity(attempts, plans),
           "renderer": renderer.label if renderer and args.mode == "tokens" else None}
    if args.mode == "tokens" and server is not None:
        fid["server_render_check"] = check_server_render(attempts, plans, server, model, tools_by_source,
                                                         template_kwargs)
    fid["max_model_len"] = server.max_model_len if server else None
    if fid["plan"]:
        log(f"planned prompt lengths vs recorded: {json.dumps(fid['plan'])}")
        if fid["plan"]["prompt_len_exact_frac"] < 0.95:
            log("WARNING planned prompts differ from the recorded ones on more than 5% of turns: check --tools, "
                "--chat-template-kwargs and that the server runs the traces' tokenizer / template")
    workload = {"attempts": len(attempts), "turns": sum(len(a.turns) for a in attempts),
                "prompt_tokens_recorded": sum(t.prompt_rec for a in attempts for t in a.turns),
                "completion_tokens_recorded": sum(t.completion for a in attempts for t in a.turns),
                "max_context_p50": pct([a.turns[-1].prompt_rec + a.turns[-1].completion for a in attempts], 50),
                "max_context_p90": pct([a.turns[-1].prompt_rec + a.turns[-1].completion for a in attempts], 90),
                "images_skipped": len(images) if args.mode == "tokens" else 0, **load_stats}
    config = {k: v for k, v in vars(args).items() if k != "api_key"}
    if args.prepare_only:
        summary = {"config": config, "workload": workload, "fidelity": fid}
        (out / "prepare.json").write_text(json.dumps(summary, indent=1))
        log(f"prepared; wrote {out / 'prepare.json'}")
        return summary
    if args.mode != "chat":
        for a in attempts:
            a.messages = None  # the plans hold everything the run needs
    bench = Bench(args, server, model, attempts, plans, tools_by_source, template_kwargs, out)
    kv_tokens = bench.logtail.kv_tokens if bench.logtail else None
    log(f"replaying at concurrency {args.concurrency} ({args.mode} mode, model {model}, cache salt {bench.salt})")
    elapsed = bench.run()
    ws = bench.windows
    steady = [w for w in ws if not w["warmup"] and not w["tail"]]
    t0s, t1s = (steady[0]["t0"], steady[-1]["t1"]) if steady else (None, None)
    reqs_steady = [r for r in bench.requests if t0s is not None and t0s <= r["t_end"] <= t1s]
    summary = {"config": config, "server": {"model": model, "version": server.version(), "kv_cache_tokens": kv_tokens,
                                            "cache_salt": bench.salt},
               "workload": workload, "fidelity": fid,
               "run": {"elapsed_s": rnd(elapsed, 1), "attempts_started": bench.started,
                       "attempts_completed": bench.completed, "attempt_errors": bench.errors,
                       "turns_completed": sum(1 for r in bench.requests if not r["error"] and not r["aborted"]),
                       "tail_at_s": rnd(bench.tail_at, 1)},
               "steady": summarize_windows(steady), "all": summarize_windows(ws),
               "requests_steady": summarize_requests(reqs_steady), "requests_all": summarize_requests(bench.requests),
               "launch_order": [f"{p}:{i}:{k}" for p, i, k in bench.launches[:2000]]}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    write_report(out / "report.md", summary, ws)
    s = summary["steady"] or summary["all"]
    log(f"done: {bench.completed} attempts in {elapsed:.0f} s; {'steady' if summary['steady'] else 'all'}: gen "
        f"{s.get('gen_tok_s')} tok/s (client {s.get('client_gen_tok_s')}), hit {s.get('prefix_hit')}, KV mean "
        f"{s.get('kv_usage_mean')}; wrote {out}/summary.json, report.md, windows.jsonl, requests.jsonl")
    return summary


if __name__ == "__main__":
    main()
