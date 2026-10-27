#!/usr/bin/env python3
"""E055 toolkit: one markdown report for a Kaggle-replica run of the dsv41 tool solver (pod run dir, archived run,
live mirror or Kaggle bench output).

  replica_report.py <run_dir> [--solutions F] [--challenges F] [--idmap F] [--baseline <run_dir>] [--out F]

Run dir layouts understood: pods (run.log, env.txt, mem.log, out/{submission.json, dsv41_traces.tar.gz,
dsv41_logs/...}), E050 (no out/), Kaggle bench (kernel_log.json instead of run.log), live mirrors (flat
attempts.jsonl / metrics.jsonl / solver.log / status.json). Files written by replica_run.sh (replica.json) give the
workload and the deadline shift.

Sections: (a) score (Kaggle task-weighted metric + per pair, cluster-bootstrap sd, per-attempt accuracy),
(b) coverage, (c) throughput over time, (d) time decomposition, (e) memory, (f) bottleneck verdict.
Everything is measured from the run's own logs; the few modelling constants (Kaggle prefill slowdown k, the value of
a second attempt when the run has too few to measure it) are named where they are used.
"""
import argparse
import ast
import bisect
import collections
import datetime as dt
import gzip
import io
import json
import math
import random
import re
import statistics
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
TOOLKIT = Path(__file__).resolve().parent
E055 = TOOLKIT.parent
BIG = 10 ** 9  # completion tokens the solver adds to a cut stream
KAGGLE_KV_POOL = 1_554_183  # Kaggle 4xL4, util 0.945 (V5 / V6 server logs)
KAGGLE_DEADLINES_S = {"stop_new": 39000, "force": 40200, "hard": 41400, "kill": 42300}
KAGGLE_RAM_GIB = 173  # Kaggle cgroup memory limit (memlog.sh note)
KAGGLE_SHIFT_S = 600  # Kaggle's install + slower load before the solver's attempts (replica_run.sh)
# Kaggle vs pod, measured on the one twin pair (Kaggle bench V1 = V6 settings, 3 h; pod mt36_b05_gate60, same settings):
# effective prefill rate 821 vs 1187 tok/s (k = 1.45; notes section 7 saturated rates: 1.26-1.45); pure-decode per-stream
# rate at 34-36 running 9.60 vs 10.20 tok/s (d = 0.94). Generated tokens 1 / 1.5 / 2 h after server up: Kaggle 0.889 /
# 0.893 / 0.910 x the pod; the stretch below with (k, d) = (1.45, 0.94) predicts 0.878 / 0.878 / 0.893
# (calibrate_twin.py; in-sample: k and d come from the same pair).
PREFILL_K = (1.3, 1.6)  # range of k used for the prefill-only bound
KCLOCK = {"central": (1.45, 0.94), "optimistic": (1.3, 1.0), "pessimistic": (1.6, 0.94)}  # (k, d)
DEFAULT_A1_GAIN = 0.11  # P(attempt 1 right and attempt 0 wrong) per pair (E046 run 7: pass@1 73.7 -> pass@2 84.6)
FULL_PREFILL_MIN, FULL_PREFILL_CACHED = 50_000, 640
THRASH_HIT, THRASH_UNCACHED = 0.85, 500  # thrash signature (poll_e055.sh): hits < 0.85 and uncached > 500 tok/s

# ---------------------------------------------------------------------------------------------- helpers


def first(*paths):
    for p in paths:
        if p and Path(p).exists():
            return Path(p)
    return None


def jsonl(path):
    out = []
    if path and path.exists():
        for line in path.read_text(errors="replace").splitlines():
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except ValueError:
                    pass
    return out


def valid_grid(g):
    return (isinstance(g, list) and 0 < len(g) <= 30 and all(isinstance(r, list) and 0 < len(r) <= 30 for r in g)
            and len({len(r) for r in g}) == 1
            and all(isinstance(v, int) and not isinstance(v, bool) and 0 <= v <= 9 for r in g for v in r))


def pct(a, b, nd=1):
    return f"{100 * a / b:.{nd}f}%" if b else "n/a"


def fmt_h(h):
    return "n/a" if h is None else f"{h:.2f} h"


def q(xs, f):
    xs = sorted(xs)
    if not xs:
        return None
    return xs[min(len(xs) - 1, max(0, int(round(f * (len(xs) - 1)))))]


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else None


def r1(x, nd=1):
    return "n/a" if x is None else f"{x:.{nd}f}"


def completion(rec):
    c = int(((rec.get("usage") or {}).get("completion")) or 0)
    return c % BIG if c >= BIG or (rec.get("usage") or {}).get("completion_reported_cut") else c


def table(head, rows):
    out = ["| " + " | ".join(head) + " |", "|" + "|".join("---" for _ in head) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


LINE = re.compile(r"^\[(\d\d):(\d\d):(\d\d) \+([\d.]+)h\] (.*)$")
ENGINE = re.compile(r"(\d\d)-(\d\d) (\d\d):(\d\d):(\d\d) \[loggers.py:\d+\] Engine 000: Avg prompt throughput: "
                    r"([\d.]+) tokens/s, Avg generation throughput: ([\d.]+) tokens/s, Running: (\d+) reqs, "
                    r"Waiting: (\d+) reqs, GPU KV cache usage: ([\d.]+)%")
SRV_TS = re.compile(r"(\d\d)-(\d\d) (\d\d):(\d\d):(\d\d) \[")


# ---------------------------------------------------------------------------------------------- loading


class Run:
    def __init__(self, d, a):
        self.dir = d = Path(d)
        self.name = d.name
        self.logs = first(d / "out" / "dsv41_logs", d / "dsv41_logs") or d
        L = self.logs
        self.notes = []  # caveats found while loading
        self.replica = json.loads((d / "replica.json").read_text()) if (d / "replica.json").exists() else {}
        self.env = self._env()
        self.kaggle = (d / "kernel_log.json").exists() or str(self.env.get("ARC_CHALLENGES", "")).startswith("/kaggle")
        self.records = jsonl(L / "attempts.jsonl")
        self.metrics = jsonl(L / "metrics.jsonl")
        self.status = json.loads((L / "status.json").read_text()) if (L / "status.json").exists() else {}
        self.solver_lines = self._solver_lines()
        self.runlog = self._runlog()
        sub = first(d / "out" / "submission.json", d / "submission.json")
        self.submission = json.loads(sub.read_text()) if sub else None
        self.traces_path = first(d / "out" / "dsv41_traces.tar.gz", d / "dsv41_traces.tar.gz", L / "traces")
        self.server_logs = sorted(L.glob("server*.log"), key=lambda p: (p.name != "server.log", p.name))
        self.mem_path = first(d / "mem.log")
        self._parse_solver()
        self._resolve_data(a)

    # -- env / settings
    def _env(self):
        env = {}
        p = self.dir / "env.txt"
        if p.exists():
            for line in p.read_text().splitlines():
                if "=" in line:
                    k, v = line.split("=", 1)
                    env[k.strip()] = v.strip()
        kl = self.dir / "kernel_log.json"
        if kl.exists():  # Kaggle notebook: settings cells print dicts
            for x in json.loads(kl.read_text()):
                s = x.get("data", "") if isinstance(x, dict) else ""
                for line in s.splitlines():
                    m = re.match(r"^(?:stress bench|E046 settings|settings)[^:]*: (\{.*\})\s*$", line)
                    if m:
                        try:
                            env.update({k: str(v) for k, v in ast.literal_eval(m.group(1)).items()})
                        except (ValueError, SyntaxError):
                            pass
        env.update({k: str(v) for k, v in (self.replica.get("env") or {}).items() if k not in env})
        return env

    def _runlog(self):
        p = self.dir / "run.log"
        if p.exists():
            return p.read_text(errors="replace")
        kl = self.dir / "kernel_log.json"
        if kl.exists():
            return "".join(x.get("data", "") for x in json.loads(kl.read_text()) if isinstance(x, dict))
        return ""

    def _solver_lines(self):
        p = first(self.logs / "solver.log")
        text = p.read_text(errors="replace") if p else ""
        if not text and (self.dir / "run.log").exists():
            text = (self.dir / "run.log").read_text(errors="replace")
        out, day, prev = [], 0, None
        for line in text.splitlines():
            m = LINE.match(line.strip())
            if not m:
                continue
            hh, mm, ss, eh, msg = m.groups()
            tod = int(hh) * 3600 + int(mm) * 60 + int(ss)
            if prev is not None and tod < prev - 43200:
                day += 1
            prev = tod
            out.append({"clock": day * 86400 + tod, "h": float(eh), "msg": msg})
        return out

    def _parse_solver(self):
        L = self.solver_lines
        self.t0_clock = statistics.median([x["clock"] - x["h"] * 3600 for x in L]) if L else None
        self.deadlines_h = {}
        self.events = collections.defaultdict(list)
        for x in L:
            msg, h = x["msg"], x["h"]
            m = re.search(r"deadlines: stop-new ([\d.]+)h, force ([\d.]+)h, hard stop ([\d.]+)h", msg)
            if m and not self.deadlines_h:
                self.deadlines_h = dict(zip(("stop_new", "force", "hard"), map(float, m.groups())))
            m = re.search(r"attempting (\d+) tasks, (\d+) pairs", msg)
            if m:
                self.events["attempting"].append((h, int(m.group(1)), int(m.group(2))))
            for key, pat in (("server_start", r"^server start \((first|restart \d+)\)"),
                             ("server_up", r"^server up after (\d+)s"), ("server_died", r"^server died \(code (-?\d+)\)"),
                             ("server_exit_startup", r"^server exited during startup"),
                             ("server_not_up", r"^server not up after"),
                             ("no_restart", r"^server down and no restart left"),
                             ("admission", r"^admission closed: .*?(\d+) jobs left unstarted"),
                             ("force", r"^force deadline: (\d+) open streams cut"),
                             ("gate", r"^KV gate: usage ([\d.]+) \(limit ([\d.]+)\); holding new attempts at (\d+) running"),
                             ("gate_missing", r"^WARNING KV gate: no KV usage"),
                             ("requeue", r"^re-queue (\S+) after a server-side error"),
                             ("retry", r"^retry (\S+?):"),
                             ("loop_error", r"^ERROR main loop iteration failed"),
                             ("result_failed", r"^result handling failed"),
                             ("signal", r"^signal (\d+): final write and exit"),
                             ("hard_stop", r"^hard stop$"), ("final", r"^submission written \(final\)"),
                             ("local_score", r"^local score: (\d+)/(\d+) pairs"), ("done", r"^done$"),
                             ("crashed", r"^solver crashed"), ("env_warning", r"^WARNING environment"),
                             ("write_failed", r"^submission write failed")):
                m = re.search(pat, msg)
                if m:
                    self.events[key].append((h, m.groups(), msg))
            m = re.search(r"GPU KV cache size: ([\d,]+) tokens", msg)
            if m:
                self.events["kv_pool"].append((h, int(m.group(1).replace(",", "")), msg))

    def _resolve_data(self, a):
        def find(name):
            if not name:
                return None
            base = Path(name).name
            for d in (TOOLKIT / "workloads", E055, ROOT / "data"):
                if (d / base).exists():
                    return d / base
            return None
        rep = self.replica
        sol = (a.solutions and Path(a.solutions)) or (rep.get("solutions_local") and ROOT / rep["solutions_local"]) \
            or find(self.env.get("ARC_SOLUTIONS"))
        chal = (a.challenges and Path(a.challenges)) or (rep.get("challenges_local") and ROOT / rep["challenges_local"]) \
            or find(self.env.get("ARC_CHALLENGES"))
        if chal is None and sol is not None and sol.name.endswith("_solutions.json"):
            chal = first(sol.with_name(sol.name.replace("_solutions.json", "_challenges.json")))
        if sol is None and chal is not None and chal.name.endswith("_challenges.json"):
            sol = first(chal.with_name(chal.name.replace("_challenges.json", "_solutions.json")))
        if sol is None:  # last resort: the eval set, if the run's task ids are eval ids
            ev = ROOT / "data" / "arc-agi_evaluation_solutions.json"
            ids = set(self.submission or {}) or {r.get("task_id") for r in self.records if "task_id" in r}
            if ids and ids <= set(json.loads(ev.read_text())):
                sol, chal = ev, ROOT / "data" / "arc-agi_evaluation_challenges.json"
                self.notes.append("solutions not named by the run: matched to the eval set by task ids")
        self.sol_path, self.chal_path = sol, chal
        self.solutions = json.loads(sol.read_text()) if sol else None
        self.challenges = json.loads(chal.read_text()) if chal and chal.exists() else None
        idm = (a.idmap and Path(a.idmap)) or (sol and first(sol.with_name(sol.name.replace("_solutions.json", "_idmap.json"))))
        raw = json.loads(idm.read_text()) if idm else {}
        self.idmap_path = idm
        self.orig, self.kind = {}, {}
        for t in (self.solutions or {}):
            m = raw.get(t)
            if isinstance(m, dict):
                self.orig[t], self.kind[t] = m.get("orig", t), m.get("kind", "orig" if m.get("orig", t) == t else "copy")
            elif isinstance(m, str):
                self.orig[t], self.kind[t] = m, "copy"
            else:
                self.orig[t], self.kind[t] = t, "orig"
        self.identical_copies = 0
        if self.challenges and raw:
            self.identical_copies = sum(1 for t, k in self.kind.items() if k == "copy" and self.orig[t] in self.challenges
                                        and self.challenges[t] == self.challenges[self.orig[t]])

    # -- derived settings
    def setting(self, k, default=None, cast=str):
        v = self.env.get(k)
        try:
            return cast(v) if v not in (None, "") else default
        except ValueError:
            return default

    @property
    def workers(self):
        w = self.setting("ARC_WORKERS", None, int)
        if w is None:
            w = int(max([m.get("running") or 0 for m in self.metrics] + [0])) or 48
        return w

    @property
    def multi(self):
        return self.setting("ARC_MULTI_TEST") == "1" or any(r.get("multi_test") for r in self.records)

    @property
    def attempts(self):
        return self.setting("ARC_ATTEMPTS", 2, int)


# ---------------------------------------------------------------------------------------------- server log (10 s)


def server_series(run):
    """10-s engine stats from server*.log, mapped to solver elapsed seconds; plus classified error lines (with their
    time, so errors from the final shutdown can be told apart)."""
    rows, errs, startup_oom = [], [], 0
    run.gpu_total = None
    starts = [x for x in run.solver_lines if x["msg"].startswith("server start")]
    tz = None
    for i, p in enumerate(run.server_logs):
        lines = p.read_text(errors="replace").splitlines()
        st = starts[i] if i < len(starts) else (starts[-1] if starts else None)
        base_day, up_seen, k_days = None, False, 0
        for line in lines:
            m = SRV_TS.search(line)
            el = None
            if m and run.t0_clock is not None and st is not None:
                mo, dd, hh, mi, ss = map(int, m.groups())
                tod = hh * 3600 + mi * 60 + ss
                day = dt.date(2026, mo, dd).toordinal()
                if base_day is None:
                    # this log's first line comes seconds after the solver's "server start" line: that fixes the
                    # time-zone offset (multiple of 15 min) and the day of the log's first date on the solver clock
                    base_day = day
                    if tz is None:
                        diff = ((st["clock"] % 86400) - tod + 43200) % 86400 - 43200
                        tz = round(diff / 900) * 900
                    k_days = round((st["clock"] - (tod + tz)) / 86400)
                el = (day - base_day + k_days) * 86400 + tod + tz - run.t0_clock
            if "Starting vLLM server on" in line:
                up_seen = True
            if run.gpu_total is None and ("total: " in line or "Free memory on device" in line):
                mt = re.search(r"total: (\d{10,})\)", line)
                mf = re.search(r"Free memory on device \(([\d.]+)/([\d.]+) GiB\)", line)
                if mt:
                    run.gpu_total = int(mt.group(1))
                elif mf:  # vLLM's device total in GiB (22.04 on an L4 = 23.66e9 bytes)
                    run.gpu_total = int(float(mf.group(2)) * 2 ** 30)
            if "CUDACachingAllocator" in line and "OOM" in line:
                if up_seen:
                    errs.append((el, "allocator OOM warning after server up"))
                else:
                    startup_oom += 1
                continue
            e = ENGINE.search(line)
            if e:
                if el is None:
                    continue
                prompt, gen, running, waiting, kv = map(float, e.groups()[5:])
                rows.append({"t": el, "prompt": prompt, "gen": gen, "running": running, "waiting": waiting,
                             "kv": kv / 100})
                continue
            low = line.lower()
            if "out of memory" in low or "outofmemoryerror" in low:
                errs.append((el, re.sub(r"^.*?(RuntimeError|torch\.\S*OutOfMemoryError|\S*Error)", r"\1", line.strip())[:120]))
            elif "died unexpectedly" in line:
                errs.append((el, "worker process died unexpectedly"))
            elif re.search(r"\bERROR\b", line) and "serving.py" not in line:
                msg = re.sub(r"^.*?\bERROR \S+ \S+ \[.*?\] ", "", line.strip())
                if re.match(r"^[\w.]*(Error|Exception)\b", msg):  # exception summary lines, not traceback frames
                    errs.append((el, msg[:110]))
    rows.sort(key=lambda r: r["t"])
    return rows, errs, startup_oom


def stall_share(rows, workers):
    """Share of server time in which decode stalls on prefill, per 10-s window: 1 - gen / clean-decode rate at the same
    batch size. vLLM counts a prompt's tokens only when its prefill completes, so a window with prompt throughput 0
    can still be inside a long chunked prefill (thrash: 25-30 tok/s at 44 streams with "0" prompt tokens); the
    clean-decode reference is therefore the 90th percentile of the per-step time's inverse among prompt-0 windows at
    +-3 running, within 30 min when there are >= 5 of them, never below 85% of the run-wide value."""
    rows = [r for r in rows if r["t"] > 0]
    pure = [r for r in rows if r["prompt"] < 1 and r["running"] >= 4 and r["gen"] > 0]
    pt = [r["t"] for r in pure]
    glob = collections.defaultdict(list)
    for y in pure:
        glob[int(y["running"])].append(y["gen"] / y["running"])  # per-stream tok/s

    def ref_rate(n, cand):
        c = [y["gen"] / y["running"] for y in cand if abs(y["running"] - n) <= 3]
        return q(c, 0.9) if len(c) >= 5 else None
    gcache = {}
    out = []
    for r in rows:
        n = r["running"]
        if n < 4:
            continue
        if n not in gcache:
            gcache[n] = ref_rate(n, pure)
        g = gcache[n]
        if g is None:  # no clean window at this batch size: step time ~ a + b n (E042 DS1: 58 ms + 0.9 ms / seq)
            allp = [(y["running"], y["gen"] / y["running"]) for y in pure]
            if not allp:
                continue
            n0, r0 = max(allp, key=lambda x: x[1])
            g = 1000 / ((1000 / r0) + 0.9 * (n - n0))
        lo, hi = bisect.bisect_left(pt, r["t"] - 1800), bisect.bisect_right(pt, r["t"] + 1800)
        loc = ref_rate(n, pure[lo:hi])
        per = max(loc, 0.85 * g) if loc is not None else g
        out.append((r, min(1.0, max(0.0, 1 - r["gen"] / (n * per)))))
    return out


# ---------------------------------------------------------------------------------------------- traces


def read_traces(path):
    """Per trace file: list of turn usages and tool statuses. Unreadable members (cut by a signal) are counted."""
    out, bad = {}, 0
    if path is None:
        return out, bad

    def one(name, raw):
        d = json.loads(gzip.decompress(raw))
        turns = [e for e in d.get("events", []) if "usage" in e]
        tools = collections.Counter(e.get("status") for e in d.get("events", []) if "status" in e)
        return {"turns": [{"turn": e.get("turn"), "prompt": (e.get("usage") or {}).get("prompt_tokens") or 0,
                           "cached": ((e.get("usage") or {}).get("prompt_tokens_details") or {}).get("cached_tokens") or 0,
                           "completion": (e.get("usage") or {}).get("completion_tokens") or 0,
                           "finish": e.get("finish_reason")} for e in turns], "tools": tools}
    if Path(path).is_dir():
        for f in sorted(Path(path).glob("*.json.gz")):
            try:
                out[f.name] = one(f.name, f.read_bytes())
            except Exception:
                bad += 1
        return out, bad
    try:
        with tarfile.open(path) as tf:
            for m in tf:
                if not m.isfile() or not m.name.endswith(".json.gz"):
                    continue
                try:
                    out[Path(m.name).name] = one(m.name, tf.extractfile(m).read())
                except Exception:
                    bad += 1
    except Exception:
        bad += 1
    return out, bad


# ---------------------------------------------------------------------------------------------- analysis


def records_by_key(run):
    """Final attempt records: the last record of each (task, pair, attempt); error records that were re-run are
    dropped (the solver re-queued them), except as provisional answers when no later record brought a grid."""
    recs = [r for r in run.records if "task_id" in r]
    last = {}
    for r in recs:
        last[(r["task_id"], r.get("pair", 0), r["attempt"])] = r
    return recs, last


def split(rec, n_test):
    n = rec.get("multi_test")
    if n:
        preds = rec["pred"] if isinstance(rec.get("pred"), list) and len(rec["pred"]) == n else [None] * n
        return [(q, preds[q]) for q in range(n)]
    return [(rec.get("pair", 0), rec.get("pred"))]


def reconstruct_submission(run, last, keep=None):
    """Rebuild submission.json from the attempt records like the solver: per pair, the final record of each attempt,
    or, when that brought no grid, the latest earlier error record that did (the solver keeps a re-queued attempt's
    harvested session answer as provisional until a later record of the same attempt brings a grid); then
    train-passing first, attempt order, two distinct grids. `last` is unused beyond its keys' order (kept for callers)."""
    hist = collections.defaultdict(list)
    for r in run.records:
        if "task_id" in r:
            hist[(r["task_id"], r.get("pair", 0), r["attempt"])].append(r)
    by = collections.defaultdict(list)
    for (t, p, a), rs in hist.items():
        n_t = len(run.solutions.get(t, [])) if run.solutions else 1
        per_pair = collections.defaultdict(list)  # q -> [(record, grid)] in order
        for r in rs:
            for q, g in split(r, n_t):
                per_pair[q].append((r, g))
        for q, seq in per_pair.items():
            r, g = seq[-1]
            if not valid_grid(g):
                prov = [(x, y) for x, y in seq[:-1] if x.get("error") and valid_grid(y)]
                if prov:
                    r, g = prov[-1]
            if not valid_grid(g) or (keep is not None and not keep(r)):
                continue
            by[(t, q)].append({"pred": g, "train_pass": r.get("train_pass"), "attempt": a})
    sub = {}
    for t, sol in (run.solutions or {}).items():
        rows = []
        for q in range(len(sol)):
            order = sorted(by.get((t, q), []), key=lambda r: (r["train_pass"] is not True, r["attempt"]))
            got = []
            for r in order:
                if r["pred"] not in got:
                    got.append(r["pred"])
            fb = (run.challenges or {}).get(t, {}).get("test", [{}] * (q + 1))[q].get("input", [[0]]) if run.challenges else [[0]]
            rows.append({"attempt_1": got[0] if got else fb, "attempt_2": got[1] if len(got) > 1 else fb})
        sub[t] = rows
    return sub


def cluster_bootstrap(task_scores, clusters, reps=2000, seed=55):
    """sd of the task-weighted score, resampling original tasks (a task and its transformed copies together)."""
    groups = collections.defaultdict(list)
    for t, s in task_scores.items():
        groups[clusters.get(t, t)].append(s)
    keys = sorted(groups)
    if len(keys) < 2:
        return None
    rng = random.Random(seed)
    vals = []
    for _ in range(reps):
        tot = n = 0
        for _ in keys:
            g = groups[keys[rng.randrange(len(keys))]]
            tot += sum(g)
            n += len(g)
        vals.append(100 * tot / n)
    return statistics.pstdev(vals)


def analyse(run, a):
    R = {}
    sol = run.solutions or {}
    recs, last = records_by_key(run)
    R["n_records"] = len(recs)
    R["sub_source"] = "submission.json"
    sub = run.submission
    if sub is None and sol:
        sub = reconstruct_submission(run, last)
        R["sub_source"] = "reconstructed from attempts.jsonl (no submission.json)"
    R["sub"] = sub
    tasks = sorted(sol) if sol else sorted(sub or {})
    ntest = {t: len(sol[t]) for t in tasks} if sol else {t: len(sub[t]) for t in tasks}
    T, P = len(tasks), sum(ntest.values())
    R["T"], R["P"], R["ntest"] = T, P, ntest

    # ---- (a) score
    pair_ok, task_score, a1_only = {}, {}, {}
    fallback_rows = 0
    if sub and sol:
        for t in tasks:
            rows = sub.get(t, [])
            hits = 0
            for qi in range(ntest[t]):
                row = rows[qi] if qi < len(rows) else {}
                ok = sol[t][qi] in (row.get("attempt_1"), row.get("attempt_2"))
                pair_ok[(t, qi)] = ok
                a1_only[(t, qi)] = sol[t][qi] == row.get("attempt_1")
                hits += ok
                if run.challenges and t in run.challenges:
                    ti = run.challenges[t]["test"][qi]["input"]
                    if row.get("attempt_1") == ti and row.get("attempt_2") == ti:
                        fallback_rows += 1
            task_score[t] = hits / ntest[t]
    R["pair_ok"], R["task_score"], R["fallback_rows"] = pair_ok, task_score, fallback_rows
    R["score_task"] = 100 * sum(task_score.values()) / T if task_score else None
    R["score_pair"] = 100 * sum(pair_ok.values()) / P if pair_ok else None
    R["score_a1_task"] = 100 * sum(sum(a1_only[(t, q)] for q in range(ntest[t])) / ntest[t] for t in tasks) / T if a1_only else None
    R["sd_task"] = cluster_bootstrap(task_score, run.orig) if task_score else None
    by_kind = collections.defaultdict(list)
    for t, s in task_score.items():
        by_kind[run.kind.get(t, "orig")].append(s)
    R["by_kind"] = {k: (100 * sum(v) / len(v), len(v)) for k, v in by_kind.items()}
    by_n = collections.defaultdict(list)
    for t, s in task_score.items():
        by_n["single-test" if ntest[t] == 1 else "multi-test"].append(s)
    R["by_ntest"] = {k: (100 * sum(v) / len(v), len(v)) for k, v in by_n.items()}

    # ---- per-attempt accuracy (finished = final record without error)
    att_rows = []  # one per (record, pair)
    for (t, p, at), r in last.items():
        if t not in ntest:
            continue
        err = bool(r.get("error"))
        for qi, g in split(r, ntest[t]):
            if qi >= ntest[t]:
                continue
            ok = bool(sol) and valid_grid(g) and g == sol[t][qi]
            att_rows.append({"t": t, "q": qi, "a": at, "ok": ok, "pred": valid_grid(g), "err": err, "rec": r,
                             "cut": bool((r.get("cuts") or {}).get("deadline")), "loop": bool((r.get("cuts") or {}).get("loop")),
                             "tp": r.get("train_pass"), "stop": r.get("stop_reason"), "src": r.get("answer_source"),
                             "start": (r.get("t_end_h") or 0) - (r.get("wall_s") or 0) / 3600, "end": r.get("t_end_h") or 0})
    fin = [x for x in att_rows if not x["err"]]
    R["att_rows"], R["fin"] = att_rows, fin
    R["acc"] = {}
    for label, sel in (("all finished", fin), ("attempt 0", [x for x in fin if x["a"] == 0]),
                       ("attempt 1", [x for x in fin if x["a"] == 1]),
                       ("train_pass True", [x for x in fin if x["tp"] is True]),
                       ("train_pass not True", [x for x in fin if x["tp"] is not True]),
                       ("cut by force deadline", [x for x in fin if x["cut"]]),
                       ("no answer (pred None)", [x for x in fin if not x["pred"]])):
        R["acc"][label] = (sum(x["ok"] for x in sel), len(sel))
    for k in ("orig", "copy"):
        sel = [x for x in fin if run.kind.get(x["t"], "orig") == k]
        if sel and len(set(run.kind.values())) > 1:
            R["acc"][f"kind {k}"] = (sum(x["ok"] for x in sel), len(sel))
    # answer path: a correct finished grid that is missing from the submission row
    R["answer_path_lost"] = sum(1 for x in fin if x["ok"] and not pair_ok.get((x["t"], x["q"]), True))
    ls = run.events.get("local_score")
    R["solver_local_score"] = (int(ls[-1][1][0]), int(ls[-1][1][1])) if ls else None

    # ---- (b) coverage
    jobs = {}
    for t in tasks:
        if run.multi and ntest[t] > 1:
            jobs[(t, 0)] = list(range(ntest[t]))
        else:
            for qi in range(ntest[t]):
                jobs[(t, qi)] = [qi]
    fin_job = collections.defaultdict(set)
    err_job = collections.defaultdict(set)
    for (t, p, at), r in last.items():
        if (t, p) in jobs:
            (err_job if r.get("error") else fin_job)[(t, p)].add(at)
    R["jobs"] = jobs
    R["fin_job"] = fin_job
    R["no_a0"] = [j for j in jobs if 0 not in fin_job[j]]
    R["no_a0_pairs"] = sum(len(jobs[j]) for j in R["no_a0"])
    # an error record was re-queued if the solver logged a re-queue / retry for its key; the remaining ones are final
    rq = collections.Counter(g[0] for k in ("requeue", "retry") for _, g, _ in run.events.get(k, []))
    nerr = collections.Counter(f"{r['task_id']}.{r.get('pair', 0)}#{r['attempt']}" for r in recs if r.get("error"))
    R["final_err_keys"] = {k for k, n in nerr.items() if n > rq.get(k, 0)}
    R["no_a0_err"] = [j for j in R["no_a0"] if f"{j[0]}.{j[1]}#0" in R["final_err_keys"]]
    R["job_dist"] = collections.Counter(len(fin_job[j]) for j in jobs)
    per_task = collections.Counter()
    for (t, p), s in fin_job.items():
        per_task[t] += len(s)
    R["task_dist"] = collections.Counter(per_task[t] for t in tasks)
    R["a1_fin"] = sum(1 for j in jobs if 1 in fin_job[j])
    R["cut_deadline"] = sum(1 for r in last.values() if (r.get("cuts") or {}).get("deadline"))
    R["cut_loop"] = sum(1 for r in last.values() if (r.get("cuts") or {}).get("loop"))
    R["err_records"] = sum(1 for r in recs if r.get("error"))
    R["err_final"] = sum(1 for r in last.values() if r.get("error"))
    R["last_end_h"] = max([r.get("t_end_h") or 0 for r in recs if not r.get("error")] + [0]) or None
    a1s = [x["start"] for x in att_rows if x["a"] == 1]
    R["a1_first_start"] = min(a1s) if a1s else None
    # how the run ended
    ev = run.events
    end = "in progress / log cut" if not (ev.get("final") or ev.get("done")) else "solver finished"
    if ev.get("signal"):
        end = f"signal {ev['signal'][-1][1][0]} at {fmt_h(ev['signal'][-1][0])} (operator stop or notebook end)"
    elif ev.get("hard_stop"):
        end = f"hard stop at {fmt_h(ev['hard_stop'][-1][0])}"
    elif ev.get("crashed"):
        end = "solver crashed"
    if "kill switch" in run.runlog:
        end += "; kill switch fired"
    R["end"] = end
    last_h = max([x["h"] for x in run.solver_lines] + [m.get("elapsed_h") or 0 for m in run.metrics] + [0])
    R["last_h"] = last_h
    fh = run.deadlines_h.get("force")
    # a run that emptied its queue before the deadline (e.g. one attempt per job) is complete, not truncated
    R["completed_all"] = bool((ev.get("final") or ev.get("done")) and not ev.get("signal")
                              and run.status.get("queued") == 0 and run.status.get("running") == 0)
    R["truncated"] = bool(fh and last_h < fh - 0.05 and not R["completed_all"])
    if R["completed_all"] and fh and last_h < fh - 0.05:
        R["end"] += f"; every job ran before the deadline (queue empty at {fmt_h(last_h)})"

    # ---- (c) throughput windows
    W = []
    prev_pre = None
    for m in run.metrics:
        pre = m.get("preemptions") or 0
        d_pre = pre - prev_pre if prev_pre is not None and pre >= prev_pre else (pre if prev_pre is not None else 0)
        prev_pre = pre
        w = dict(m, d_pre=d_pre, t1=m.get("elapsed_h") or 0, t0=(m.get("elapsed_h") or 0) - (m.get("window_s") or 0) / 3600)
        # a server restart resets the counters: the first window after it has negative rates; count it as 0
        w["reset"] = (w.get("gen_tok_s") or 0) < 0 or (w.get("uncached_tok_s") or 0) < 0
        if w["reset"]:
            w["gen_tok_s"], w["uncached_tok_s"], w["prefix_hit"] = 0.0, 0.0, None
        W.append(w)
    sn = run.deadlines_h.get("stop_new", 1e9)
    died_h = [h for h, _, _ in ev.get("server_died", [])]
    for w in W:
        after_restart = any(h <= w["t1"] <= h + 0.25 for h in died_h)  # re-queued attempts re-prefill from scratch
        w["thrash"] = (w["t1"] > 0.4 and not after_restart and (w.get("prefix_hit") or 1) < THRASH_HIT
                       and (w.get("uncached_tok_s") or 0) > THRASH_UNCACHED)
    healthy = [w["gen_tok_s"] for w in W if w["t1"] >= 0.4 and w["t1"] <= sn and (w.get("prefix_hit") or 0) >= 0.9
               and (w.get("uncached_tok_s") or 0) <= THRASH_UNCACHED and (w.get("kv") or 0) < 0.8
               and (w.get("running") or 0) >= 0.95 * run.workers]
    if len(healthy) < 3:
        healthy = [w["gen_tok_s"] for w in W if w["t1"] >= 0.4]
        R["ref_note"] = "few full-concurrency healthy windows: reference = 75th percentile of all windows"
        R["ref_gen"] = q(healthy, 0.75)
    else:
        R["ref_note"] = (f"median of {len(healthy)} healthy windows at >= 95% of {run.workers} workers "
                         "(hits >= 0.9, uncached <= 500, KV < 0.8)")
        R["ref_gen"] = statistics.median(healthy)
    R["W"] = W
    R["gen_total"] = sum((w.get("gen_tok_s") or 0) * (w.get("window_s") or 0) for w in W)
    R["rec_tokens"] = sum(completion(r) for r in recs)
    R["fin_tokens"] = sum(completion(r) for r in last.values() if not r.get("error"))

    # ---- server 10 s series, prefill share
    rows, errs, startup_oom = server_series(run)
    stops = [h for k in ("signal", "hard_stop", "final") for h, _, _ in ev.get(k, [])]
    R["stop_h"] = min(stops) if stops else None
    cut = (R["stop_h"] - 60 / 3600) * 3600 if R["stop_h"] is not None else None
    R["srv_rows"], R["startup_oom"] = rows, startup_oom
    gt = getattr(run, "gpu_total", None)
    R["gpu_kind"] = ("L4" if gt and abs(gt - 23.66e9) < 0.05e9 else
                     "RTX 4090" if gt and abs(gt - 25.25e9) < 0.05e9 else
                     (f"unknown GPU ({gt / 2 ** 30:.1f} GiB)" if gt else "unknown GPU"))
    kvp = ev.get("kv_pool")
    if gt is None and kvp and 1.30e6 <= kvp[0][1] <= 1.75e6 and 0.92 <= run.setting("ARC_GPU_UTIL", 0.93, float) <= 0.95:
        R["gpu_kind"] = "L4"  # no server.log: E128's KV pool on 4x L4 at util 0.92-0.95 is 1.40-1.66M tokens
        run.notes.append("no server.log: GPUs taken as L4 from the KV pool; prefill share from 5-min windows "
                         "(uncached tok/s / 1,200), so the Kaggle-clock correction is approximate")
    R["srv_errs"] = collections.Counter(m for el, m in errs if cut is None or el is None or el < cut)
    R["srv_errs_shutdown"] = sum(1 for el, m in errs if cut is not None and el is not None and el >= cut)
    sh = stall_share(rows, run.workers)
    R["stall"] = sh
    if sh:
        s = mean(x[1] for x in sh)
        ptoks = sum(x[0]["prompt"] * 10 for x in sh)
        R["prefill_share"] = s
        R["implied_prefill_rate"] = ptoks / (s * 10 * len(sh)) if s > 0 else None
        R["kaggle_factor"] = {k: 1 / (1 + (k - 1) * s) for k in PREFILL_K}
    else:
        R["prefill_share"] = None

    # ---- (d) slot accounting from metrics windows (gauges at the window end, weighted by window length)
    slot = collections.Counter()
    for w in W:
        if w["t1"] < 0.15 or w["t1"] > sn:
            continue
        ws = w.get("window_s") or 0
        run_n = w.get("running") or 0
        in_srv = min(run_n, (w.get("srv_running") or 0) + (w.get("srv_waiting") or 0))
        slot["slots"] += run.workers * ws
        slot["in_server"] += in_srv * ws
        slot["outside"] += (run_n - in_srv) * ws
        slot["idle"] += max(0, run.workers - run_n) * ws
    R["slot"] = slot
    # finer: 10-s server samples vs the solver's running count rebuilt from attempt records (each record spans
    # [t_end - wall, t_end]); limited to times before the last record end (later in-flight attempts have no record)
    starts = sorted(((r.get("t_end_h") or 0) * 3600 - (r.get("wall_s") or 0)) for r in recs if r.get("wall_s"))
    ends_ = sorted((r.get("t_end_h") or 0) * 3600 for r in recs if r.get("wall_s"))
    tmax = sn * 3600 if not R["truncated"] else \
        max(ends_ or [0]) - (q([r.get("wall_s") or 0 for r in recs], 0.9) or 0)
    slot10 = collections.Counter()
    gauge_t = [w["t1"] * 3600 for w in W]
    gauge = [w.get("running") or 0 for w in W]
    if gauge:
        tmax = min(sn, max(w["t1"] for w in W)) * 3600
    for x in rows:
        if x["t"] < 0.15 * 3600 or x["t"] > tmax:
            continue
        n_run = (gauge[min(len(gauge) - 1, bisect.bisect_left(gauge_t, x["t"]))] if gauge
                 else bisect.bisect_right(starts, x["t"]) - bisect.bisect_right(ends_, x["t"]))
        if n_run <= 0:
            continue
        slot10["running"] += n_run
        slot10["in_server"] += min(n_run, x["running"] + x["waiting"])
        slot10["idle"] += max(0, run.workers - n_run)
        slot10["slots"] += run.workers
    R["slot10"], R["slot10_tmax"] = slot10, tmax
    # per attempt: decode-equivalent time at the prevailing per-stream rate
    rate_t = [(r["t"], r["gen"] / r["running"]) for r in rows if r["running"] > 0 and r["t"] > 0]
    if not rate_t:
        rate_t = [((w["t1"]) * 3600, w["gen_tok_s"] / w["srv_running"]) for w in W if (w.get("srv_running") or 0) > 0]
    tt = [x[0] for x in rate_t]
    dec = []
    for r in last.values():
        if r.get("error") or not r.get("wall_s"):
            continue
        e = (r.get("t_end_h") or 0) * 3600
        s0 = e - r["wall_s"]
        lo, hi = bisect.bisect_left(tt, s0), bisect.bisect_right(tt, e)
        rs = [x[1] for x in rate_t[lo:hi]] or [x[1] for x in rate_t[max(0, lo - 3):lo + 3]]
        if not rs:
            continue
        rbar = mean(rs)
        if not rbar:
            continue
        d = completion(r) / rbar
        dec.append({"wall": r["wall_s"], "decode": min(d, r["wall_s"]), "other": max(0.0, r["wall_s"] - d),
                    "tools": r.get("tool_calls") or 0, "turns": r.get("turns") or 0, "tok": completion(r),
                    "tc": r.get("train_check_seconds") or 0})
    R["dec"] = dec

    # ---- gate holds
    g = ev.get("gate", [])
    episodes = []
    for h, grp, _ in g:
        if episodes and h - episodes[-1][1] <= 330 / 3600:
            episodes[-1][1] = h
            episodes[-1][2].append(int(grp[2]))
        else:
            episodes.append([h, h, [int(grp[2])]])
    R["gate_lines"], R["gate_episodes"] = len(g), episodes
    R["gate_limit"] = float(g[0][1][1]) if g else run.setting("ARC_KV_ADMIT_MAX", 0.0, float)
    # ---- restarts and downtime
    downs = []
    died = [h for h, _, _ in ev.get("server_died", [])]
    ups = [h for h, _, _ in ev.get("server_up", [])]
    for h in died:
        nxt = [u for u in ups if u >= h]
        downs.append((h, (nxt[0] if nxt else last_h) - h))
    R["downs"] = downs

    # ---- traces: full re-prefills, tool statuses
    traces, bad = read_traces(run.traces_path) if not a.no_traces else ({}, 0)
    R["traces_n"], R["traces_bad"] = len(traces), bad
    rec_by_file = {}
    for r in recs:
        f = r.get("trace_file") or f"{r['task_id']}_p{r.get('pair', 0)}_a{r['attempt']}.json.gz"
        rec_by_file[f] = r
    full, tools, unc_total, turns_total = [], collections.Counter(), 0, 0
    for f, tr in traces.items():
        tools.update(tr["tools"])
        for x in tr["turns"]:
            turns_total += 1
            unc_total += max(0, x["prompt"] - x["cached"])
            if (x["turn"] or 0) > 1 and x["prompt"] >= FULL_PREFILL_MIN and x["cached"] <= FULL_PREFILL_CACHED:
                full.append((f, x["turn"], x["prompt"], x["cached"], (rec_by_file.get(f) or {}).get("t_end_h")))
    R["full_prefills"], R["tool_status"], R["trace_uncached"], R["trace_turns"] = full, tools, unc_total, turns_total
    te = collections.Counter()
    for r in last.values():
        for k, v in (r.get("token_exact") or {}).items():
            te[k] += v or 0
    R["token_exact"] = te

    # ---- (e) memory
    mem = []
    if run.mem_path and run.t0_clock is not None:
        prev, day = None, 0
        for line in run.mem_path.read_text(errors="replace").splitlines():
            m = re.match(r"(\d\d):(\d\d):(\d\d) used_gib=(\d+) rss_gib=([\d.]+)", line)
            if not m:
                continue
            cg = re.search(r"cg_gib=([\d.]+)", line)
            gp = re.search(r"gpu_w_mhz=(\S+)", line)
            tod = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + int(m.group(3))
            if prev is not None and tod < prev - 43200:
                day += 1
            prev = tod
            el = day * 86400 + tod - run.t0_clock
            if el < -3600:  # the memlog's first line is after midnight, T0 before it
                el += 86400
            if gp:
                vals = [x.split(",") for x in gp.group(1).split(";") if x.count(",") == 1]
                try:
                    R.setdefault("gpu", []).append((el / 3600, [float(a_) for a_, _ in vals],
                                                    [float(b_) for _, b_ in vals]))
                except ValueError:
                    pass
            mem.append((el / 3600, int(m.group(4)), float(m.group(5)),
                        float(cg.group(1)) if cg and float(cg.group(1)) > 0 else None))
    R["mem"] = mem

    # ---- (f) verdict inputs
    a0 = [x for x in fin if x["a"] == 0 and not x["cut"]]
    a0.sort(key=lambda x: x["start"])
    late = a0[len(a0) // 2:] if len(a0) >= 8 else a0
    R["acc0"] = mean(x["ok"] for x in a0)
    # value of a missing first attempt: accuracy of the latest-started half, shrunk toward 70% (typical E128
    # per-attempt accuracy) with 20 pseudo-attempts so that a handful of early attempts does not decide it
    R["acc0_late_raw"], R["acc0_late_n"] = mean(x["ok"] for x in late), len(late)
    R["acc0_late"] = (sum(x["ok"] for x in late) + 0.70 * 20) / (len(late) + 20)
    late_recs = [r for r in last.values() if not r.get("error") and r["attempt"] == 0]
    late_recs.sort(key=lambda r: (r.get("t_end_h") or 0) - (r.get("wall_s") or 0) / 3600)
    lr = late_recs[len(late_recs) // 2:] if len(late_recs) >= 8 else late_recs
    R["tok_per_att_late"] = mean(completion(r) for r in lr)
    R["pairs_per_job"] = P / len(jobs) if jobs else 1
    both = collections.defaultdict(dict)
    for x in fin:
        both[(x["t"], x["q"])][x["a"]] = x["ok"]
    pb = [v for v in both.values() if 0 in v and 1 in v]
    R["a1_pairs_both"] = len(pb)
    R["a1_gain_measured"] = mean((v[1] and not v[0]) for v in pb) if pb else None
    R["a1_gain"] = R["a1_gain_measured"] if len(pb) >= 20 else DEFAULT_A1_GAIN
    R["kclock"] = kaggle_corrected(run, R)
    return R


# ---------------------------------------------------------------------------------------------- Kaggle clock


def kaggle_clock(run, R, k, d):
    """Pod elapsed seconds -> Kaggle-equivalent elapsed seconds. Before the first server-up: + KAGGLE_SHIFT_S (Kaggle's
    install and slower FUSE load). After it, every 10-s server interval lasts (1 + (k - 1) s) / d times longer on
    Kaggle, s = that interval's prefill (decode-stall) share; intervals without server stats (outages) are not
    stretched. Assumes the same sequence of server states, only slower: tools (~0.5% of attempt time) do not stretch."""
    ups = run.events.get("server_up")
    t_up = ups[0][0] * 3600 if ups else 0.0
    st = sorted((r["t"], sh) for r, sh in R["stall"])
    if not st:  # no server.log: s from the 5-min windows, uncached tok/s / 1,200 (approximate)
        st = sorted(((w["t1"] * 3600, min(1.0, (w.get("uncached_tok_s") or 0) / 1200)) for w in R["W"]))
        step = 300.0
    else:
        step = 10.0
    ts, cum = [t_up], [t_up + KAGGLE_SHIFT_S]
    for t, sh in st:
        if t <= ts[-1]:
            continue
        dt_ = t - ts[-1]
        stretched = min(dt_, step * 1.5)
        cum.append(cum[-1] + stretched * (1 + (k - 1) * sh) / d + (dt_ - stretched))
        ts.append(t)

    def f(t):
        if t <= t_up:
            return t + KAGGLE_SHIFT_S
        i = bisect.bisect_right(ts, t) - 1
        if i >= len(ts) - 1:
            return cum[-1] + (t - ts[-1]) * (cum[-1] - cum[0]) / max(1.0, ts[-1] - ts[0])  # extrapolate at the mean
        return cum[i] + (t - ts[i]) * (cum[i + 1] - cum[i]) / (ts[i + 1] - ts[i])
    return f


def kaggle_corrected(run, R):
    """Scores the run would have had on Kaggle's clock: attempts are dropped if they would have started after
    Kaggle's stop-new or ended after its force deadline (their session answers at a force cut are rarely right:
    0/11 on the Kaggle bench). Kaggle deadlines = this run's deadlines on Kaggle's clock (+ the launcher's shift)."""
    if run.kaggle or not run.solutions or not run.deadlines_h:
        return None
    if R.get("gpu_kind") != "L4":
        return {"skip": f"the run's GPUs are {R.get('gpu_kind')}, not L4: the Kaggle-clock calibration (k, d) does not apply"}
    shift = run.replica.get("shift_s") or 0
    force_k = run.deadlines_h["force"] * 3600 + shift
    stop_k = run.deadlines_h["stop_new"] * 3600 + shift
    _, last = records_by_key(run)
    out = {}
    for name, (k, d) in KCLOCK.items():
        f = kaggle_clock(run, R, k, d)
        # pod time at which Kaggle's force deadline falls
        lo, hi = 0.0, 48 * 3600.0
        for _ in range(60):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if f(mid) < force_k else (lo, mid)
        pod_force = lo

        def keep(r, f=f):
            e = (r.get("t_end_h") or 0) * 3600
            s0 = e - (r.get("wall_s") or 0)
            return f(e) <= force_k and f(s0) <= stop_k
        sub = reconstruct_submission(run, last, keep)
        hits, tw, per_task = 0, 0.0, {}
        for t, sol in run.solutions.items():
            h = sum(sol[qi] in (sub[t][qi]["attempt_1"], sub[t][qi]["attempt_2"]) for qi in range(len(sol)))
            hits += h
            tw += h / len(sol)
            per_task[t] = h / len(sol)
        kept = sum(1 for r in last.values() if not r.get("error") and keep(r))
        out[name] = {"k": k, "d": d, "task": 100 * tw / len(run.solutions), "pairs": hits, "kept": kept,
                     "per_task": per_task,
                     "pod_force_h": pod_force / 3600,
                     "reached": R["last_h"] * 3600 >= pod_force or R["completed_all"],
                     "kclock_last_h": f(R["last_h"] * 3600) / 3600}
    # the same reconstruction without any cut, to separate the clock effect from reconstruction differences
    sub = reconstruct_submission(run, last)
    out["uncorrected (records)"] = {"task": 100 * sum(sum(sol[qi] in (sub[t][qi]["attempt_1"], sub[t][qi]["attempt_2"])
                                                          for qi in range(len(sol))) / len(sol)
                                                      for t, sol in run.solutions.items()) / len(run.solutions)}
    return out


# ---------------------------------------------------------------------------------------------- verdict


def verdict(run, R):
    """Score points (task-weighted %, and pairs) attributed to each bottleneck."""
    T, P = R["T"], R["P"]
    w_pair = {(t, q): 1 / R["ntest"][t] for t in R["ntest"] for q in range(R["ntest"][t])}
    acc_l = R["acc0_late"] if R["acc0_late"] is not None else (R["acc0"] or 0)
    out = {}
    # coverage: pairs whose job has no finished attempt 0
    cov_pairs = R["no_a0_pairs"] * acc_l
    cov_tw = sum(w_pair[(t, q)] for j in R["no_a0"] for (t, q) in [(j[0], qq) for qq in R["jobs"][j]]) * acc_l
    no_a1 = [j for j in R["jobs"] if 0 in R["fin_job"][j] and 1 not in R["fin_job"][j]] if run.attempts > 1 else []
    a1_pairs = sum(len(R["jobs"][j]) for j in no_a1) * R["a1_gain"]
    a1_tw = sum(w_pair[(j[0], qq)] for j in no_a1 for qq in R["jobs"][j]) * R["a1_gain"]
    p1, p2 = (100 * cov_tw / T, 100 * a1_tw / T) if T else (0, 0)
    out["coverage (deadline)"] = {
        "pairs": cov_pairs + a1_pairs, "points": p1 + p2, "first": p1, "second": p2,
        "why": f"first attempts {p1:.1f}: {R['no_a0_pairs']} pairs in {len(R['no_a0'])} jobs without a finished "
               f"attempt 0 x {100 * acc_l:.0f}% (latest-started half of uncut attempt 0s, {R['acc0_late_n']} pair-attempts at "
               f"{100 * (R['acc0_late_raw'] or 0):.0f}%, shrunk toward 70%); second attempts {p2:.1f}: "
               f"{len(no_a1)} jobs without a finished attempt 1 x {100 * R['a1_gain']:.0f}% gain"}
    # value of one more attempt: unstarted attempt-0 jobs first, then second attempts
    tok = R["tok_per_att_late"] or 40000
    mean_w = (sum(w_pair.values()) / len(w_pair)) if w_pair else 1

    def points_for_tokens(tokens):
        n_att = tokens / tok
        n0 = min(n_att, len(R["no_a0"])) if not R["truncated"] else 0
        n1 = n_att - n0
        pairs = n0 * R["pairs_per_job"] * acc_l + n1 * R["pairs_per_job"] * R["a1_gain"]
        return n_att, pairs, 100 * pairs * mean_w / T if T else 0
    # throughput / KV: tokens below the healthy reference rate, by cause
    ref = R["ref_gen"] or 0
    sn = run.deadlines_h.get("stop_new", 1e9)
    gate_iv = [(e[0] - 300 / 3600, e[1]) for e in R["gate_episodes"]]
    down_iv = [(h, h + d) for h, d in R["downs"]]
    causes = collections.Counter()
    for w in R["W"]:
        if w["t1"] < 0.4 or w["t0"] > sn:
            continue
        lost = max(0.0, ref - (w.get("gen_tok_s") or 0)) * (w.get("window_s") or 0)
        if lost <= 0:
            continue
        mid = (w["t0"] + w["t1"]) / 2
        if any(a <= mid <= b + 0.05 for a, b in down_iv):
            causes["crash downtime"] += lost
        elif w["thrash"]:
            causes["KV thrash"] += lost
        elif any(a <= mid <= b for a, b in gate_iv):
            causes["KV gate hold"] += lost
        elif (w.get("uncached_tok_s") or 0) > THRASH_UNCACHED:
            causes["prefill burst"] += lost
        elif (w.get("running") or 0) < 0.9 * run.workers:
            causes["low concurrency"] += lost
        else:
            causes["other"] += lost
    tp_tokens = sum(v for k, v in causes.items() if k != "crash downtime")
    n_att, pairs, pts = points_for_tokens(tp_tokens)
    out["throughput/KV"] = {"pairs": pairs, "points": pts, "tokens": tp_tokens, "causes": causes, "attempts": n_att,
                            "why": f"{tp_tokens / 1e6:.2f}M tokens below the healthy {ref:.0f} tok/s before stop-new "
                                   f"= {n_att:.1f} attempts of {tok / 1e3:.0f}k tokens"}
    # crashes: downtime + tokens of errored attempts + jobs left without an answer by errors
    crash_tok = causes.get("crash downtime", 0) + sum(completion(r) for r in run.records if r.get("error"))
    n_att_c, pairs_c, pts_c = points_for_tokens(crash_tok)
    err_pairs = sum(len(R["jobs"][j]) for j in R["no_a0_err"]) * acc_l
    out["crashes"] = {"pairs": pairs_c + err_pairs, "points": pts_c + 100 * err_pairs * mean_w / T if T else 0,
                      "tokens": crash_tok,
                      "why": f"{len(R['downs'])} server deaths ({sum(d for _, d in R['downs']) * 60:.0f} min down), "
                             f"{R['err_records']} errored attempt records ({crash_tok / 1e6:.2f}M tokens incl. downtime), "
                             f"{len(R['no_a0_err'])} jobs whose attempt 0 ended in a final (not re-queued) error"}
    # accuracy: attempted pairs without a correct answer in the submission
    attempted = {(t, q) for j in R["jobs"] if R["fin_job"][j] for (t, q) in [(j[0], qq) for qq in R["jobs"][j]]}
    wrong = [k for k in attempted if not R["pair_ok"].get(k, False)]
    out["accuracy"] = {"pairs": len(wrong), "points": 100 * sum(w_pair[k] for k in wrong) / T if T else 0,
                       "why": f"{len(wrong)} of {len(attempted)} attempted pairs without a correct answer "
                              f"(no answer: {R['acc']['no answer (pred None)'][1]} finished pair-attempts)"}
    return out


# ---------------------------------------------------------------------------------------------- report


def report(run, R, a, base=None):
    o = []
    env = run.env
    T, P = R["T"], R["P"]
    dl = run.deadlines_h
    shift = run.replica.get("shift_s")
    if shift is None:
        shift = 0
    o.append(f"# Replica report: {run.name}\n")
    o.append(f"Run dir `{run.dir.relative_to(ROOT) if run.dir.is_relative_to(ROOT) else run.dir}`; "
             f"{'Kaggle notebook output' if run.kaggle else 'pod run'}; generated by "
             f"`experiments/E055_kaggle_replica/toolkit/replica_report.py`.\n")
    cfg = [f"workers {run.workers}", f"multi-test {'on' if run.multi else 'off'}"]
    for k, lab in (("ARC_MULTI_BUDGET", "multi budget"), ("ARC_KV_ADMIT_MAX", "KV gate"), ("ARC_GPU_UTIL", "util"),
                   ("ARC_ATTEMPTS", "attempts"), ("ARC_KIT", "kit"), ("ARC_AUTO_REPAIR", "auto-repair"),
                   ("ARC_CONTEXT_LIMIT", "context limit"), ("ARC_REASONING_EFFORT", "effort")):
        if env.get(k):
            cfg.append(f"{lab} {env[k]}")
    extra = env.get("ARC_SERVER_EXTRA_ARGS", "")
    if "retention-interval" in extra:
        m = re.search(r'retention-interval", "(\d+)', extra)
        cfg.append(f"retention interval {m.group(1) if m else '?'}")
    kvp = run.events.get("kv_pool")
    pool = kvp[0][1] if kvp else None
    o.append("- **Config:** " + ", ".join(cfg))
    o.append(f"- **Workload:** {T} tasks / {P} pairs; solutions `{run.sol_path.name if run.sol_path else 'none'}`"
             + (f"; idmap `{run.idmap_path.name}`" if run.idmap_path else "")
             + (f"; **{run.identical_copies} copies identical to their originals (shared prompt prefixes: prefill and KV "
                f"optimistic; use a transformed workload from make_workload.py)**" if run.identical_copies else ""))
    o.append(f"- **Deadlines (pod clock):** stop-new {fmt_h(dl.get('stop_new'))}, force {fmt_h(dl.get('force'))}, "
             f"hard stop {fmt_h(dl.get('hard'))}, kill {fmt_h(run.setting('ARC_KILL_AT_S', 42300, float) / 3600)}; "
             + (f"shifted {shift} s earlier than Kaggle's (replica_run.sh)" if shift else
                ("Kaggle notebook clock" if run.kaggle else
                 ("**not shifted: this run had ~600 s more solver time than Kaggle**"
                  if abs((dl.get('stop_new') or 0) * 3600 - KAGGLE_DEADLINES_S['stop_new']) < 60 else
                  "custom deadlines (not a 12 h replica)"))))
    o.append(f"- **GPUs:** {R.get('gpu_kind')}")
    if pool:
        o.append(f"- **KV pool:** {pool:,} tokens ({100 * (pool / KAGGLE_KV_POOL - 1):+.1f}% vs Kaggle's {KAGGLE_KV_POOL:,})")
    o.append(f"- **End:** {R['end']}; last log line at {fmt_h(R['last_h'])}"
             + ("; **truncated before the force deadline: coverage is not measured**" if R["truncated"] else ""))
    for n in run.notes:
        o.append(f"- Note: {n}")

    # ---- summary
    V = verdict(run, R)
    infra = sorted(((k, v) for k, v in V.items() if k != "accuracy" and not (R["truncated"] and k.startswith("coverage"))),
                   key=lambda kv: -kv[1]["points"])
    o.append("\n## Summary\n")
    if R["score_task"] is not None:
        sd = R["sd_task"]
        o.append(f"- **Kaggle score (task-weighted): {R['score_task']:.2f}**" + (f" (cluster-bootstrap sd {sd:.1f})" if sd else "")
                 + f"; per pair {R['score_pair']:.2f}% ({sum(R['pair_ok'].values())}/{P}).")
        K = R.get("kclock")
        if K and "skip" in K:
            o.append(f"- Kaggle-clock correction skipped: {K['skip']}.")
        elif K and K["central"]["reached"]:
            o.append(f"- **On Kaggle's clock: {K['central']['task']:.2f}** (optimistic {K['optimistic']['task']:.2f}, "
                     f"pessimistic {K['pessimistic']['task']:.2f}): the pod's extra speed and the ~600 s staging gap "
                     f"removed (section (a), Kaggle-clock correction).")
        elif K:
            o.append(f"- Kaggle-clock correction not applicable: the run stopped at Kaggle-clock "
                     f"{K['central']['kclock_last_h']:.2f} h, before the force deadline.")
    o.append(f"- Largest infrastructure loss: **{infra[0][0]}** ~{infra[0][1]['points']:.1f} points"
             f" ({infra[0][1]['why']}); then " + ", ".join(f"{k} ~{v['points']:.1f}" for k, v in infra[1:]) + ".")
    o.append(f"- Accuracy (wrong or missing answers on attempted pairs): {V['accuracy']['points']:.1f} points; "
             f"per-attempt accuracy of finished attempts {pct(*R['acc']['all finished'])} "
             f"(attempt 0 {pct(*R['acc']['attempt 0'])}).")

    # ---- (a)
    o.append("\n## (a) Score\n")
    o.append("Kaggle's leaderboard metric is task-weighted: each task scores the fraction of its test pairs with a "
             "correct attempt_1 or attempt_2, averaged over tasks (V5 58.19 = 139.67/240 and v3 50.97 = 122.33/240 are "
             "exact thirds; no per-pair denominator between 100 and 500 reproduces both). Per-pair numbers are shown "
             "for comparison with older runs.\n")
    rows = []
    if R["score_task"] is not None:
        rows.append(["task-weighted (Kaggle)", f"**{R['score_task']:.2f}**", f"{T} tasks",
                     f"sd {R['sd_task']:.1f}" if R["sd_task"] else ""])
        rows.append(["per pair", f"{R['score_pair']:.2f}", f"{sum(R['pair_ok'].values())}/{P}", ""])
        rows.append(["attempt_1 only (task-weighted)", f"{R['score_a1_task']:.2f}", "", "what attempt_2 adds: "
                     f"{R['score_task'] - R['score_a1_task']:+.2f}"])
        for k, (s, n) in sorted(R["by_kind"].items()):
            if len(R["by_kind"]) > 1:
                rows.append([f"{k} tasks", f"{s:.2f}", f"{n} tasks", ""])
        for k, (s, n) in sorted(R["by_ntest"].items()):
            rows.append([f"{k} tasks", f"{s:.2f}", f"{n} tasks", ""])
    o.append(table(["score", "value", "n", "note"], rows) if rows else "No solutions found: score not computed.")
    o.append(f"\nSubmission: {R['sub_source']}; rows still at the fallback (test input twice): {R['fallback_rows']}"
             + (f"; solver's own local score {R['solver_local_score'][0]}/{R['solver_local_score'][1]} pairs "
                f"({'matches' if R['solver_local_score'][0] == sum(R['pair_ok'].values()) else '**differs from**'} this scorer)"
                if R["solver_local_score"] else "")
             + f"; correct finished grids missing from the submission (answer-path losses): {R['answer_path_lost']}.")
    if R["sd_task"]:
        o.append(f"The sd resamples the {len(set(run.orig.get(t, t) for t in R['task_score']))} original tasks (a task "
                 "and its copy together) and so covers attempt noise and task sampling; two replicas of one config "
                 "differ by about sqrt(2) x sd.")
    o.append("\nPer-attempt accuracy of finished attempts (one row per pair; multi-test attempts count once per test input):\n")
    o.append(table(["attempts", "correct", "n", "accuracy"],
                   [[k, c, n, pct(c, n)] for k, (c, n) in R["acc"].items() if n]))
    if R["a1_pairs_both"]:
        o.append(f"\nPairs with both attempts finished: {R['a1_pairs_both']}; attempt 1 right where attempt 0 was wrong: "
                 f"{100 * R['a1_gain_measured']:.1f}%.")

    K = R.get("kclock")
    if K and "skip" in K:
        o.append(f"\nKaggle-clock correction skipped: {K['skip']}.")
    elif K:
        o.append("\n### Kaggle-clock correction\n")
        o.append("The pod is faster than Kaggle's 50 W L4s, and on Kaggle the server comes up ~600 s later. The pod "
                 "timeline is stretched onto Kaggle's clock (prefill k x slower, decode d x the pod's rate; see "
                 "`kaggle_clock`), and attempts that would have started after stop-new or ended after the force deadline "
                 "on that clock are dropped; the submission is then rebuilt from the remaining records like the "
                 "solver does.\n")
        rows = []
        for name in ("central", "optimistic", "pessimistic"):
            x = K[name]
            rows.append([name, f"k {x['k']}, d {x['d']}", f"**{x['task']:.2f}**" if name == "central" else f"{x['task']:.2f}",
                         x["pairs"], x["kept"], fmt_h(x["pod_force_h"]),
                         "yes" if x["reached"] else f"**no** (stopped at Kaggle-clock {x['kclock_last_h']:.2f} h)"])
        o.append(table(["clock", "Kaggle vs pod", "task-weighted score", "pairs right", "finished attempts kept",
                        "pod time of Kaggle's force deadline", "run reached it"], rows))
        o.append(f"\nSame records without the clock cut: {K['uncorrected (records)']['task']:.2f} (vs {R['score_task']:.2f} "
                 "from submission.json; a difference means the record rebuild and the solver's own selection disagree).")
        if not K["central"]["reached"]:
            o.append("The run stopped before Kaggle's force deadline on the stretched clock, so the corrected score is "
                     "a lower bound of an unfinished run, not a 12 h estimate.")

    # ---- (b)
    o.append("\n## (b) Coverage\n")
    J = R["jobs"]
    rows = [["jobs (attempt units)", f"{len(J)} ({'one per multi-test task' if run.multi else 'one per pair'})"],
            ["jobs without a finished attempt 0", f"{len(R['no_a0'])} ({R['no_a0_pairs']} pairs, "
             f"{len({j[0] for j in R['no_a0']})} tasks; {len(R['no_a0_err'])} of them ended in an error)"],
            ["finished attempts per job", ", ".join(f"{k}: {v}" for k, v in sorted(R["job_dist"].items()))],
            ["finished attempts per task", ", ".join(f"{k}: {v}" for k, v in sorted(R["task_dist"].items()))],
            ["second attempts finished", f"{R['a1_fin']} of {len(J)} jobs ({pct(R['a1_fin'], len(J))})"],
            ["first attempt-1 start", fmt_h(R["a1_first_start"])],
            ["attempts cut by the force deadline", f"{R['cut_deadline']}"
             + (f" ({pct(*R['acc']['cut by force deadline'])} of their pairs right)" if R['acc']['cut by force deadline'][1] else "")],
            ["attempts cut by the loop guard", R["cut_loop"]],
            ["errored attempt records (re-queued or final)", f"{R['err_records']} ({R['err_final']} final)"],
            ["admission closed", "; ".join(f"{fmt_h(h)} ({g[0]} jobs left unstarted)" for h, g, _ in run.events.get("admission", [])) or "no"],
            ["force deadline", "; ".join(f"{fmt_h(h)} ({g[0]} open streams cut)" for h, g, _ in run.events.get("force", [])) or "not reached"],
            ["last finished attempt", fmt_h(R["last_end_h"])],
            ["records / status", f"{R['n_records']} records; status: done {run.status.get('done')}, running "
             f"{run.status.get('running')}, queued {run.status.get('queued')}, pairs with an answer "
             f"{run.status.get('pairs_with_answer')}/{run.status.get('pairs')}"]]
    o.append(table(["coverage", "value"], rows))
    if R["truncated"] and R["last_end_h"]:
        t1 = R["last_end_h"]
        recent = [r for r in records_by_key(run)[1].values() if not r.get("error") and (r.get("t_end_h") or 0) >= t1 - 2]
        rate_h = len(recent) / min(2, t1) if t1 else 0
        left = (run.deadlines_h.get("force") or 0) - R["last_h"]
        if rate_h:
            o.append(f"\nTruncated run: {rate_h:.1f} finished attempts/h over the last 2 h; the {len(R['no_a0'])} jobs "
                     f"without attempt 0 would take ~{len(R['no_a0']) / rate_h:.1f} h at that rate, and "
                     f"{left:.1f} h were left to the force deadline (a rate extrapolation, not a measurement; later "
                     "jobs are larger).")
    if R["no_a0"] and len(R["no_a0"]) <= 30:
        o.append("\nJobs without a finished attempt 0: " + ", ".join(f"`{t}.{p}`" for t, p in sorted(R["no_a0"])))

    # ---- (c)
    o.append("\n## (c) Throughput over time\n")
    W = R["W"]
    if W:
        hours = collections.defaultdict(list)
        for w in W:
            hours[int(w["t1"])].append(w)
        ends = collections.Counter(int(r.get("t_end_h") or 0) for r in records_by_key(run)[1].values() if not r.get("error"))
        srv_kv = collections.defaultdict(float)
        for r in R["srv_rows"]:
            srv_kv[int(r["t"] // 3600)] = max(srv_kv[int(r["t"] // 3600)], r["kv"])
        stall_h = collections.defaultdict(list)
        for r, s in R["stall"]:
            stall_h[int(r["t"] // 3600)].append(s)
        rows, cum = [], 0
        for h in sorted(hours):
            ws = hours[h]
            tot = sum(w["window_s"] for w in ws) or 1
            g = sum(w["gen_tok_s"] * w["window_s"] for w in ws) / tot
            hit = mean(w["prefix_hit"] for w in ws if w.get("prefix_hit") is not None)
            unc = sum(w["uncached_tok_s"] * w["window_s"] for w in ws) / tot
            cum += ends.get(h, 0)
            sh = mean(stall_h.get(h, []))
            rows.append([f"{h}-{h + 1}", len(ws), f"{g:.0f}", r1(hit, 3), f"{unc:.0f}",
                         f"{mean(w['kv'] for w in ws):.2f} / {max(max(w['kv'] for w in ws), srv_kv.get(h, 0)):.2f}",
                         f"{mean(w['running'] for w in ws):.0f}", f"{mean(w.get('srv_running') or 0 for w in ws):.0f}",
                         f"{max(w.get('srv_waiting') or 0 for w in ws):.0f}", f"{sum(w['d_pre'] for w in ws):.0f}",
                         ends.get(h, 0), cum, sum(w["thrash"] for w in ws), r1(None if sh is None else 100 * sh, 0)])
        o.append(table(["hour", "win", "gen tok/s", "prefix hits", "uncached tok/s", "KV mean / max", "running",
                        "srv running", "srv waiting max", "preempt", "attempts done", "cum.", "thrash win",
                        "prefill % of server time"], rows))
        o.append(f"\nGenerated tokens (server counters): {R['gen_total'] / 1e6:.2f}M; in attempt records: "
                 f"{R['rec_tokens'] / 1e6:.2f}M (finished {R['fin_tokens'] / 1e6:.2f}M); the rest was in attempts still "
                 f"running at the end or in failed turns. Healthy reference rate: {r1(R['ref_gen'], 0)} tok/s "
                 f"({R['ref_note']}).")
        kvs = [(w["t1"], w["kv"]) for w in W] + [(r["t"] / 3600, r["kv"]) for r in R["srv_rows"]]
        if kvs:
            mx = max(kvs, key=lambda x: x[1])
            over = [t for t, k in sorted(kvs) if k >= 0.8]
            o.append(f"KV: max {mx[1]:.3f} at {fmt_h(mx[0])}; first >= 0.80 at {fmt_h(over[0]) if over else 'never'}; "
                     f"preemptions total {sum(w['d_pre'] for w in W):.0f}.")
        th = [w for w in W if w["thrash"]]
        o.append(f"Thrash-signature windows (after 0.4 h, prefix hits < {THRASH_HIT} and uncached > {THRASH_UNCACHED} "
                 f"tok/s): {len(th)}" + (": " + ", ".join(f"{w['t1']:.2f} h ({w['gen_tok_s']:.0f} tok/s, hits "
                                                            f"{w['prefix_hit']}, KV {w['kv']:.2f})" for w in th[:12])
                                         + (" ..." if len(th) > 12 else "") if th else "."))
    else:
        o.append("No metrics.jsonl windows.")
    if R["prefill_share"] is not None:
        kf = R["kaggle_factor"]
        o.append(f"\nPrefill share of server time (decode stall vs pure-decode windows at the same batch size, "
                 f"server.log 10-s stats): **{100 * R['prefill_share']:.1f}%**; implied effective prefill rate "
                 f"{r1(R['implied_prefill_rate'], 0)} tok/s."
                 + ("" if run.kaggle else
                    f" Kaggle prefills ~{PREFILL_K[0]}-{PREFILL_K[1]}x slower (50 W vs 72 W), so the same work would run "
                    f"at x{kf[PREFILL_K[1]]:.3f}-x{kf[PREFILL_K[0]]:.3f} of this throughput on Kaggle from prefill alone "
                    f"(1 / (1 + (k - 1) s)); pure decode is another ~0-6% slower there (twin pair: 9.60 vs 10.20 tok/s "
                    f"per stream). The Kaggle-clock correction in (a) applies both."))

    # ---- (d)
    o.append("\n## (d) Time decomposition\n")
    sl, s10 = R["slot"], R["slot10"]
    if s10.get("slots"):
        o.append(f"Worker-slot time ({run.workers} slots; 10-s server samples up to {fmt_h(R['slot10_tmax'] / 3600)} "
                 f"against the solver's running count from metrics.jsonl): slots holding an attempt "
                 f"{pct(s10['running'], s10['slots'])} (idle: gate, refill, empty queue {pct(s10['idle'], s10['slots'])}); "
                 f"of the attempt time, {pct(s10['in_server'], s10['running'])} has a request in the server (running "
                 f"or waiting) and {pct(s10['running'] - s10['in_server'], s10['running'])} is outside it (tool "
                 f"execution, host train check, harness, tokenization).")
    elif sl.get("slots"):
        o.append(f"Worker-slot time from metrics windows (5-min gauges, attempt phase): in the server "
                 f"{pct(sl['in_server'], sl['slots'])}, outside the server {pct(sl['outside'], sl['slots'])}, idle "
                 f"slots {pct(sl['idle'], sl['slots'])}.")
    dec = R["dec"]
    if dec:
        tw = sum(d["wall"] for d in dec)
        o.append(f"\nPer finished attempt ({len(dec)}): decode-equivalent time = completion tokens / the server's "
                 f"per-stream rate (gen tok/s / running, includes prefill stalls) over the attempt's lifetime; the rest "
                 f"is tools, train check, harness, own prefill and queueing.\n")
        rows = []
        for lab, key in (("wall s", "wall"), ("decode-equivalent s", "decode"), ("other s", "other"),
                         ("completion tokens", "tok"), ("turns", "turns"), ("tool calls", "tools")):
            xs = [d[key] for d in dec]
            rows.append([lab, f"{q(xs, .5):.0f}", f"{q(xs, .9):.0f}", f"{max(xs):.0f}", f"{sum(xs) / len(xs):.0f}"])
        o.append(table(["per attempt", "median", "p90", "max", "mean"], rows))
        per_call = [d["other"] / d["tools"] for d in dec if d["tools"]]
        o.append(f"\nDecode share of attempt wall time: {pct(sum(d['decode'] for d in dec), tw)}; other per tool call: "
                 f"median {r1(q(per_call, .5), 1)} s; host train checks {sum(d['tc'] for d in dec) / 3600:.2f} h total.")
    ts = R["tool_status"]
    if ts:
        o.append(f"Tool calls in traces ({R['traces_n']} traces, {R['traces_bad']} unreadable): "
                 + ", ".join(f"{k} {v}" for k, v in ts.most_common()) + ".")
    ep = R["gate_episodes"]
    o.append(f"\nKV gate (limit {R['gate_limit'] or 'off'}): {R['gate_lines']} hold lines in {len(ep)} episodes"
             + (": " + "; ".join(f"{a_:.2f}-{b_:.2f} h at {min(n)}-{max(n)} running" for a_, b_, n in ep[:12])
                + (" ..." if len(ep) > 12 else "") if ep else "") + ". Lines are logged at most every 5 min while held, "
             "so the hold time is about lines x 5 min.")
    srv = [f"{len(run.events.get('server_start', []))} server start(s)"]
    for h, d in R["downs"]:
        srv.append(f"died at {fmt_h(h)}, down {d * 60:.1f} min")
    o.append("Server: " + "; ".join(srv) + f"; startup allocator OOM warnings (benign retries while loading): "
             f"{R['startup_oom']}.")
    if R["srv_errs"]:
        o.append("Server errors during the run: " + "; ".join(f"{v}x {k}" for k, v in R["srv_errs"].most_common(6)) + ".")
    if R["srv_errs_shutdown"]:
        o.append(f"({R['srv_errs_shutdown']} error lines from the final shutdown ignored.)")
    se = [(k, len(run.events.get(k, []))) for k in ("requeue", "retry", "loop_error", "result_failed", "env_warning",
                                                     "write_failed", "gate_missing", "crashed", "no_restart")]
    o.append("Solver: " + ", ".join(f"{k} {n}" for k, n in se) + ".")
    full = R["full_prefills"]
    rate = R.get("implied_prefill_rate") or 1500
    if R["traces_n"]:
        ftok = sum(x[2] - x[3] for x in full)
        o.append(f"\nFull re-prefills (turn > 1, prompt >= {FULL_PREFILL_MIN // 1000}k, cached <= {FULL_PREFILL_CACHED}): "
                 f"**{len(full)}** of {R['trace_turns']} turns, {ftok / 1e6:.2f}M tokens = {pct(ftok, R['trace_uncached'])} of "
                 f"all uncached prompt tokens; at {rate:.0f} tok/s ~{ftok / rate / 60:.0f} min of stalled decode."
                 + (" Worst: " + ", ".join(f"`{f}` turn {t} ({p // 1000}k, end {fmt_h(e)})"
                                         for f, t, p, c, e in sorted(full, key=lambda x: -x[2])[:5]) if full else ""))
        te = R["token_exact"]
        if te:
            o.append("Token-exact counters (finished attempts): " + ", ".join(f"{k} {v}" for k, v in sorted(te.items())) + ".")

    # ---- (e)
    o.append("\n## (e) Memory\n")
    mem = R["mem"]
    if mem:
        has_cg = any(x[3] is not None for x in mem)
        hours = collections.defaultdict(list)
        for x in mem:
            hours[int(max(0, x[0]))].append(x)
        o.append(table(["hour", "host used GiB max", "RSS GiB max"] + (["cgroup GiB max"] if has_cg else []),
                       [[f"{h}-{h + 1}", max(x[1] for x in v), f"{max(x[2] for x in v):.1f}"]
                        + ([f"{max(x[3] or 0 for x in v):.1f}"] if has_cg else []) for h, v in sorted(hours.items())]))
        key = 3 if has_cg else 2  # the cgroup figure is what Kaggle's limit applies to; RSS otherwise
        late = [(x[0], x[key]) for x in mem if x[0] >= max(0.5, mem[-1][0] - 2) and x[key] is not None]
        slope = None
        if len(late) > 10 and mem[-1][0] - mem[0][0] >= 3:
            mh, mu = mean(x[0] for x in late), mean(x[1] for x in late)
            den = sum((x[0] - mh) ** 2 for x in late)
            slope = sum((x[0] - mh) * (x[1] - mu) for x in late) / den if den else None
        peak = max(x[key] or 0 for x in mem)
        lab = "cgroup" if has_cg else "RSS"
        o.append(f"\nPeak {lab} {peak:.1f} GiB vs Kaggle's cgroup limit {KAGGLE_RAM_GIB} GiB (headroom "
                 f"{KAGGLE_RAM_GIB - peak:.0f} GiB)" + (f"; {lab} slope over the last 2 h {slope:+.2f} GiB/h -> "
                                                     f"~{mem[-1][key] + slope * max(0, 11.75 - mem[-1][0]):.0f} GiB at 11.75 h"
                                                     if slope is not None and slope > 0.05 else "")
                 + ". `host used` is `free -g` inside the container, i.e. the whole host's memory (other tenants "
                 "included), so it is not a limit check; the RSS sum double-counts shared pages (pinned Engram tables)"
                 + ("" if has_cg else "; the cgroup's memory.current (what Kaggle's OOM killer sees) was not logged "
                    "(replica_run.sh's memlog adds it)") + ".")
        gpu = R.get("gpu")
        if gpu:
            hrs = collections.defaultdict(list)
            for h, w, mhz in gpu:
                if w:
                    hrs[int(max(0, h))].append((sum(w) / len(w), max(w), sum(mhz) / len(mhz)))
            o.append("\nGPU power and SM clock (memlog, per-GPU mean / max W, mean MHz): " + "; ".join(
                f"{h}-{h + 1} h {mean(x[0] for x in v):.0f} / {max(x[1] for x in v):.0f} W, {mean(x[2] for x in v):.0f} MHz"
                for h, v in sorted(hrs.items())) + ". Kaggle's L4s are capped at 50 W.")
    else:
        o.append("No mem.log.")

    # ---- (f)
    o.append("\n## (f) Bottleneck verdict\n")
    cov = [] if not R["truncated"] else [("coverage (deadline)", dict(V["coverage (deadline)"], why="not measured: "
                                           "the run stopped early (at stop: " + V["coverage (deadline)"]["why"] + ")"))]
    o.append(table(["bottleneck", "task-weighted points", "pairs", "basis"],
                   [[k, f"{v['points']:.1f}" if not k.startswith("coverage") or not R["truncated"] else "n/a",
                     f"{v['pairs']:.1f}" if not k.startswith("coverage") or not R["truncated"] else "n/a", v["why"]]
                    for k, v in infra + cov + [("accuracy", V["accuracy"])]]))
    c = V["throughput/KV"]["causes"]
    if c:
        o.append("\nThroughput/KV tokens lost by cause (windows below the healthy rate): "
                 + ", ".join(f"{k} {v / 1e6:.2f}M" for k, v in c.most_common()) + ".")
    o.append(f"\nConversion: a lost attempt is worth {R['pairs_per_job']:.2f} pairs x "
             + (f"{100 * (R['acc0_late'] or 0):.0f}% while jobs without attempt 0 remain, then " if R["no_a0"] and not R["truncated"] else "")
             + f"{100 * R['a1_gain']:.0f}% (second-attempt gain, "
             + ("measured on " + str(R['a1_pairs_both']) + " pairs" if R['a1_pairs_both'] >= 20 else
                f"default {DEFAULT_A1_GAIN:.2f}: too few pairs with both attempts") + f"); {r1((R['tok_per_att_late'] or 0) / 1e3, 0)}k "
             "tokens per attempt (latest-started half). Coverage and second attempts not run are what the deadline "
             "left undone; throughput/KV and crashes are the parts of that time budget lost to degraded windows and "
             "outages (causes, not additive with them). Accuracy points are the ceiling for model/harness work, not a "
             "fixable infrastructure loss.")
    if R["truncated"]:
        o.append("\n**This run stopped before its force deadline, so coverage is not measured and throughput losses "
                 "are valued at second-attempt rates.**")
    lead = infra[0]
    o.append(f"\n**Verdict:** the largest infrastructure cost is **{lead[0]}** (~{lead[1]['points']:.1f} points); "
             + ("accuracy dominates overall" if V["accuracy"]["points"] > lead[1]["points"] else "it exceeds the accuracy loss")
             + f" ({V['accuracy']['points']:.1f} points of wrong or missing answers on attempted pairs).")

    if base is not None:
        o.append("\n## Paired comparison\n")
        Rb = base[1]
        ka, kb = R.get("kclock") or {}, Rb.get("kclock") or {}
        use_k = all(x.get("central", {}).get("reached") for x in (ka, kb))
        sa = ka["central"]["per_task"] if use_k else R["task_score"]
        sb = kb["central"]["per_task"] if use_k else Rb["task_score"]
        common = sorted(set(sa) & set(sb))
        if common:
            d = {t: sa[t] - sb[t] for t in common}
            sd = cluster_bootstrap(d, run.orig)
            delta = 100 * sum(d.values()) / len(d)
            o.append(f"{run.name} - {base[0].name} on {len(common)} common tasks "
                     f"({'Kaggle-clock central scores' if use_k else 'pod-clock submission scores'}): **{delta:+.2f}** "
                     f"task-weighted points" + (f" (paired cluster-bootstrap sd {sd:.1f})" if sd else "") + ".")
            if not use_k and abs(R["last_h"] - Rb["last_h"]) > 0.25:
                o.append(f"Warning: the runs ended at {R['last_h']:.2f} h and {Rb['last_h']:.2f} h; compare finished 12 h "
                         "replicas.")
    return "\n".join(o) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("run_dir")
    ap.add_argument("--solutions", default="")
    ap.add_argument("--challenges", default="")
    ap.add_argument("--idmap", default="")
    ap.add_argument("--baseline", default="", help="another run dir on the same workload: paired score difference")
    ap.add_argument("--no-traces", action="store_true", help="skip dsv41_traces.tar.gz")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    run = Run(a.run_dir, a)
    R = analyse(run, a)
    base = None
    if a.baseline:
        b = argparse.Namespace(**{**vars(a), "solutions": a.solutions, "no_traces": True})
        rb = Run(a.baseline, b)
        base = (rb, analyse(rb, b))
    text = report(run, R, a, base)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(text)
        print(f"wrote {a.out}")
    else:
        print(text)


if __name__ == "__main__":
    main()
