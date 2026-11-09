#!/usr/bin/env python3
"""Threshold alerts over sampler.py's JSONL (live file on the pod, or a local copy).

Each alert is one line "<UTC> <LEVEL> <code>: <message>" on stdout and appended to --alerts. Levels: INFO < WARN <
ALERT < CRIT. Level crossings fire once (state in --state), so the watcher can run in a loop (--follow) or be
polled (--once, e.g. over ssh: the exit code is the highest new level, 0 = nothing new, 1 INFO, 2 WARN, 3 ALERT,
4 CRIT).

Alerts (defaults; each threshold also reads an env var, e.g. RELI_RAM_WARN=150):
  ram          Kaggle-eq host RAM (cgroup anon + shmem + 7 GiB) >= 155 WARN, >= 165 ALERT, >= 173 CRIT (OOM-killed on
               Kaggle). ram_source: WARN once if no cgroup accounting is readable and the figure falls back to the
               host-wide /proc/meminfo (it then includes other tenants' memory).
  cap_peak     worst rank's allocator cap - peak allocated < 0.5 GiB WARN, < 0.26 GiB ALERT (OOM band); once per
               level per server instance.
  retries      a rank's "allocator retries" grew: ALERT (the allocator hit its cap and flushed its cache).
  vllm_oom     a rank's vLLM OOM count grew, or a "CUDA out of memory" line appeared: CRIT.
  engine_dead  new EngineDeadError / worker-died / execute_model-timeout lines: ALERT. traceback: WARN.
  two_engines  more than one EngineCore process: CRIT (a restart holding two engines' host RAM).
  server_died  the solver logged "server died": ALERT. supervisor_restart: the supervisor's start count changed.
  not_up       the supervisor state is not "up" for > 5 min: ALERT.
  stop_error   an attempt was FINALIZED as error (solver "stop=error"): ALERT. Per-try error records are normal.
  error_burst  >= 5 new error records within 10 min: WARN.
  timeout      new APITimeoutError / "Request timed out" lines: WARN (a try spent without a server death).
  trend        a slope (after warm-up) reaches a limit before the run's end (--end-h): WARN, re-sent at most hourly.
  stale        no new sample for > 300 s: ALERT (the sampler or the host is stuck).
  notebook_end the runner logged the notebook's end: INFO (--follow exits after it).
  smi_free     nvidia-smi free < N MiB: off by default (board free is not a pressure signal; vLLM caps under it).

Usage: watcher.py --samples mon.jsonl --alerts alerts.log --state watch_state.json --once
       watcher.py --samples mon.jsonl --alerts alerts.log --follow [--exit-on ALERT] [--until 04:00]"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import reli_common as rc  # noqa: E402

LEVELS = {"INFO": 1, "WARN": 2, "ALERT": 3, "CRIT": 4}
NAMES = {v: k for k, v in LEVELS.items()}


def env_float(name, default):
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return float(default)


def build_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--samples", required=True)
    ap.add_argument("--alerts", help="append alert lines here (default: <samples>.alerts)")
    ap.add_argument("--state", help="state file (default: <samples>.watch_state.json)")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--once", action="store_true", help="evaluate new samples once; exit code = max new level")
    mode.add_argument("--follow", action="store_true", help="poll until the notebook ends (or --exit-on/--until)")
    ap.add_argument("--poll-s", type=float, default=60.0)
    ap.add_argument("--exit-on", choices=sorted(LEVELS, key=LEVELS.get),
                    help="--follow: exit at the first alert at or above this level (prints FLAG)")
    ap.add_argument("--until", help="--follow: at this UTC HH:MM print HOURLY_DUE and a status summary, then exit")
    ap.add_argument("--summary", action="store_true", help="print the status summary of the newest sample")
    ap.add_argument("--no-stale", action="store_true", help="skip the stale check (offline copies)")
    t = ap.add_argument_group("thresholds (env var in brackets)")
    t.add_argument("--ram-warn", type=float, default=env_float("RELI_RAM_WARN", 155), help="[RELI_RAM_WARN] GiB")
    t.add_argument("--ram-alert", type=float, default=env_float("RELI_RAM_ALERT", 165), help="[RELI_RAM_ALERT]")
    t.add_argument("--ram-crit", type=float, default=env_float("RELI_RAM_CRIT", rc.KAGGLE_RAM_LIMIT_GIB),
                   help="[RELI_RAM_CRIT] Kaggle's memory.max")
    t.add_argument("--cap-peak-warn", type=float, default=env_float("RELI_CAP_PEAK_WARN", 0.5),
                   help="[RELI_CAP_PEAK_WARN] GiB")
    t.add_argument("--cap-peak-alert", type=float, default=env_float("RELI_CAP_PEAK_ALERT", rc.CAP_PEAK_LIMIT_GIB),
                   help="[RELI_CAP_PEAK_ALERT] GiB")
    t.add_argument("--error-burst", type=int, default=int(env_float("RELI_ERROR_BURST", 5)),
                   help="[RELI_ERROR_BURST] new error records ...")
    t.add_argument("--error-burst-min", type=float, default=env_float("RELI_ERROR_BURST_MIN", 10),
                   help="[RELI_ERROR_BURST_MIN] ... within this many minutes")
    t.add_argument("--not-up-min", type=float, default=env_float("RELI_NOT_UP_MIN", 5), help="[RELI_NOT_UP_MIN]")
    t.add_argument("--stale-s", type=float, default=env_float("RELI_STALE_S", 300), help="[RELI_STALE_S]")
    t.add_argument("--smi-free-warn-mib", type=float, default=env_float("RELI_SMI_FREE_WARN_MIB", 0),
                   help="[RELI_SMI_FREE_WARN_MIB] 0 = off")
    t.add_argument("--end-h", type=float, default=env_float("RELI_END_H", 11.5),
                   help="[RELI_END_H] run end on the Kaggle clock (the solver's hard stop)")
    t.add_argument("--warmup-min", type=float, default=env_float("RELI_WARMUP_MIN", 20), help="[RELI_WARMUP_MIN]")
    t.add_argument("--window-h", type=float, default=env_float("RELI_WINDOW_H", 2), help="[RELI_WINDOW_H]")
    t.add_argument("--trend-every-min", type=float, default=env_float("RELI_TREND_EVERY_MIN", 30))
    a = ap.parse_args(argv)
    a.alerts = a.alerts or a.samples + ".alerts"
    a.state = a.state or a.samples + ".watch_state.json"
    return a


def trends(records, a):
    """[(code, status, text)] for the RAM and VRAM trends of the current server segment."""
    if not records:
        return []
    hours, _ = rc.clock(records)
    seg = rc.segments(records)[-1]
    w = rc.post_warmup(seg, a.warmup_min * 60, a.window_h * 3600)
    now_h = hours(records[-1]["t"])
    out = []
    for code, name, path, limit, direction in (
            ("ram", "Kaggle-eq RAM", "kaggle_eq_gib", a.ram_crit, "up"),
            ("cap_peak", "worst cap - peak", "vllm.worst.cap_minus_peak", a.cap_peak_alert, "down")):
        pts = [(hours(r["t"]), rc.get_path(r, path)) for r in w]
        cur = rc.get_path(records[-1], path)
        slope = rc.linfit(pts) if len(pts) >= 10 else None
        st, txt = rc.trend_verdict(name, cur, slope, limit, direction, now_h, a.end_h)
        out.append((code, st, txt))
    return out


class Watcher:
    def __init__(self, a, state=None):
        self.a = a
        self.s = state or {}
        self.s.setdefault("n", 0)
        self.s.setdefault("ram_level", 0)
        self.s.setdefault("cap_level", {})
        self.s.setdefault("retries", {})
        self.s.setdefault("ooms", {})
        self.s.setdefault("counts", {})
        self.s.setdefault("err_hist", [])
        self.s.setdefault("smi_level", {})

    def _bump(self, key, value):
        """True when a cumulative counter grew since the last sample seen (the first sample counts from 0)."""
        old = self.s["counts"].get(key, 0)
        self.s["counts"][key] = value
        return value > old

    def evaluate(self, r, recs, i, clk=None):
        """Alerts for recs[i] (recs = every sample so far; clk = rc.clock(recs), computed once per poll)."""
        a, s, out = self.a, self.s, []
        t = r["t"]
        hours, label = clk or rc.clock(recs[:i + 1])
        tag = " [%s %.2f]" % (label, hours(t))

        def emit(level, code, msg):
            out.append((t, level, code, msg + tag))

        if r.get("sampler_error"):
            emit("WARN", "sampler_error", r["sampler_error"])
        keq, src = r.get("kaggle_eq_gib"), r.get("kaggle_eq_source")
        if src == "meminfo_hostwide" and not s.get("src_warned"):
            s["src_warned"] = True
            emit("WARN", "ram_source", "no cgroup memory accounting found; Kaggle-eq falls back to the host-wide "
                                       "/proc/meminfo (includes other tenants' memory)")
        if keq is not None:
            lvl = 4 if keq >= a.ram_crit else 3 if keq >= a.ram_alert else 2 if keq >= a.ram_warn else 0
            if lvl > s["ram_level"]:
                cg = r.get("cgroup") or {}
                parts = ("anon %s + shmem %s GiB + Kaggle overhead" % (cg.get("anon_gib"), cg.get("shmem_gib"))
                         if cg.get("version") else "host-wide meminfo form")
                emit(NAMES[lvl], "ram", "Kaggle-eq host RAM %.1f GiB (%s: %s)" % (keq, src, parts)
                     + (" -- Kaggle would OOM-kill the notebook (%.1f GiB)" % a.ram_crit if lvl == 4 else ""))
            s["ram_level"] = max(s["ram_level"], lvl)
        v = r.get("vllm") or {}
        key = v.get("server_key") or "-"
        ranks = v.get("ranks") or {}
        cmps = {rk: d["cap_minus_peak"] for rk, d in ranks.items() if d.get("cap_minus_peak") is not None}
        if cmps:
            rank = min(cmps, key=cmps.get)
            cmp, d = cmps[rank], ranks[rank]
            lvl = 3 if cmp < a.cap_peak_alert else 2 if cmp < a.cap_peak_warn else 0
            if lvl > s["cap_level"].get(key, 0):
                emit(NAMES[lvl], "cap_peak", "worst rank %s: cap - peak %.2f GiB < %.2f (cap %s, peak %s, reserved %s, "
                     "retries %s; all ranks %s)" % (rank, cmp, a.cap_peak_alert if lvl == 3 else a.cap_peak_warn,
                                                    d.get("cap"), d.get("peak"), d.get("reserved"), d.get("retries"),
                                                    {k: cmps[k] for k in sorted(cmps)}))
                s["cap_level"][key] = lvl
        for field, store, level, code, what in (("retries", "retries", "ALERT", "retries",
                                                 "allocator retries (the allocator hit its cap and flushed its cache)"),
                                                ("ooms", "ooms", "CRIT", "vllm_oom", "vLLM OOM count")):
            grown = []
            for rk, d in sorted(ranks.items()):
                x = d.get(field)
                if x is not None and x > s[store].get(key + "/" + rk, 0):
                    grown.append("%s %d (was %d)" % (rk, x, s[store].get(key + "/" + rk, 0)))
                    s[store][key + "/" + rk] = x
            if grown:
                w = rc.worst_ranks(ranks)
                emit(level, code, "%s: %s; worst alloc %s, peak %s, reserved %s" % (
                    what, ", ".join(grown), w.get("alloc"), w.get("peak"), w.get("reserved")))
        c = v.get("counts") or {}
        if self._bump("oom_lines", c.get("oom", 0)):
            emit("CRIT", "vllm_oom", "CUDA out-of-memory line(s) in the server logs: %d" % c["oom"])
        if self._bump("dead_lines", c.get("dead", 0)):
            emit("ALERT", "engine_dead", "engine-death lines in the server logs: %d" % c["dead"])
        if self._bump("server_tracebacks", c.get("traceback", 0)):
            emit("WARN", "traceback", "Traceback lines in the server logs: %d" % c["traceback"])
        eng = r.get("engines") or 0
        if eng > 1 and not s.get("two_engines"):
            emit("CRIT", "two_engines", "%d EngineCore processes: two engines' host RAM at once (pids %s)"
                 % (eng, [p[0] for p in r.get("vllm_procs", []) if str(p[1]).startswith("VLLM::EngineCore")]))
        s["two_engines"] = eng > 1
        sc = (r.get("solver") or {}).get("counts") or {}
        if self._bump("server_died", sc.get("server_died", 0)):
            emit("ALERT", "server_died", "solver: server died (x%d)" % sc["server_died"])
        if self._bump("stop_error", sc.get("stop_error", 0)):
            emit("ALERT", "stop_error", "solver: %d attempt(s) finalized as error (stop=error)" % sc["stop_error"])
        if self._bump("timeout", sc.get("timeout", 0)):
            emit("WARN", "timeout", "solver: %d APITimeoutError / timed-out line(s)" % sc["timeout"])
        if self._bump("solver_tracebacks", sc.get("traceback", 0)):
            emit("WARN", "traceback", "Traceback lines in solver.log: %d" % sc["traceback"])
        sup = r.get("supervisor") or {}
        if sup.get("start") is not None:
            if s.get("sup_start") is not None and sup["start"] != s["sup_start"]:
                emit("ALERT", "supervisor_restart", "supervisor start %s -> %s (state %s, error %s)"
                     % (s["sup_start"], sup["start"], sup.get("state"), sup.get("error")))
            s["sup_start"] = sup["start"]
        if sup.get("state") not in (None, "up"):
            s.setdefault("not_up_since", t)
            if t - s["not_up_since"] > a.not_up_min * 60 and not s.get("not_up_alerted"):
                emit("ALERT", "not_up", "supervisor state %r for %.0f s" % (sup.get("state"), t - s["not_up_since"]))
                s["not_up_alerted"] = True
        else:
            s.pop("not_up_since", None)
            s.pop("not_up_alerted", None)
        er = (r.get("attempts") or {}).get("error_records")
        if er is not None:
            hist = [h for h in s["err_hist"] if h[0] >= t - a.error_burst_min * 60]
            if hist and er - hist[0][1] >= a.error_burst and not s.get("burst_alerted_at", 0) >= hist[0][0]:
                emit("WARN", "error_burst", "%d new error records in %.0f min (%d total)"
                     % (er - hist[0][1], (t - hist[0][0]) / 60, er))
                s["burst_alerted_at"] = t
            hist.append([t, er])
            s["err_hist"] = hist
        if a.smi_free_warn_mib > 0:
            for g in r.get("gpus") or []:
                fr = g.get("free_mib")
                if fr is not None and fr < a.smi_free_warn_mib and not s["smi_level"].get(str(g["index"])):
                    emit("INFO", "smi_free", "GPU %d nvidia-smi free %.0f MiB (board level; not a pressure signal)"
                         % (g["index"], fr))
                    s["smi_level"][str(g["index"])] = 1
        if r.get("notebook_end") and not s.get("nb_end"):
            emit("INFO", "notebook_end", r["notebook_end"])
            s["nb_end"] = True
        if t - s.get("last_trend_t", 0) >= a.trend_every_min * 60:
            s["last_trend_t"] = t
            for code, st, txt in trends(recs[:i + 1], a):
                prev = s.setdefault("trend", {}).get(code)
                if st in ("RISK", "FAIL") and (prev is None or prev[0] != st or t - prev[1] >= 3600):
                    emit("WARN", "trend", txt)
                    s["trend"][code] = [st, t]
        return out

    def stale(self, records, now):
        if self.a.no_stale or not records:
            return []
        age = now - records[-1]["t"]
        if age > self.a.stale_s:
            if not self.s.get("stale_alerted"):
                self.s["stale_alerted"] = True
                return [(now, "ALERT", "stale", "no new sample for %.0f s (sampler stopped or host stuck)" % age)]
        else:
            self.s.pop("stale_alerted", None)
        return []

    def run_once(self, now=None):
        recs = rc.load_records(self.a.samples)
        out = []
        clk = rc.clock(recs)
        for i in range(self.s["n"], len(recs)):
            out.extend(self.evaluate(recs[i], recs, i, clk))
        self.s["n"] = len(recs)
        out.extend(self.stale(recs, now or time.time()))
        return out, recs


def summary(records, a):
    if not records:
        return "no samples"
    r = records[-1]
    hours, label = rc.clock(records)
    v = r.get("vllm") or {}
    w = v.get("worst") or {}
    sc = (r.get("solver") or {}).get("counts") or {}
    m = (r.get("solver") or {}).get("metrics") or {}
    lines = ["status at %s (%s %.2f): Kaggle-eq %s GiB (%s), engines %s, sandboxes %s; worst cap-peak %s GiB, "
             "cap-reserved %s, retries %s, OOMs %s; KV pool %s; solver died %s, re-queues %s, retries %s, "
             "stop=error %s; gen %s tok/s, KV %s, running %s, waiting %s; supervisor %s"
             % (r["utc"], label, hours(r["t"]), r.get("kaggle_eq_gib"), r.get("kaggle_eq_source"), r.get("engines"),
                (r.get("sandboxes") or {}).get("n"), w.get("cap_minus_peak"), w.get("cap_minus_reserved"),
                w.get("retries"), w.get("ooms"), v.get("kv_tokens"), sc.get("server_died"), sc.get("requeue"),
                sc.get("retry"), sc.get("stop_error"), m.get("gen_tok_s"), m.get("kv"), m.get("running"),
                m.get("waiting"), (r.get("supervisor") or {}).get("state"))]
    for code, st, txt in trends(records, a):
        lines.append("  trend %s: %s %s" % (code, st, txt))
    return "\n".join(lines)


def load_state(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def save_state(path, s):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(s, f)
    os.replace(tmp, path)


def write_alerts(path, alerts):
    lines = ["%s %s %s: %s" % (rc.utc(t), lvl, code, msg) for t, lvl, code, msg in alerts]
    if lines:
        with open(path, "a") as f:
            f.write("\n".join(lines) + "\n")
        print("\n".join(lines), flush=True)


def until_epoch(hhmm, now):
    h, m = (int(x) for x in hhmm.split(":"))
    t = now - now % 86400 + h * 3600 + m * 60
    return t if t > now else t + 86400


def main(argv=None):
    a = build_args(argv)
    w = Watcher(a, load_state(a.state))
    if a.summary and not (a.once or a.follow):
        print(summary(rc.load_records(a.samples), a))
        return 0
    deadline = until_epoch(a.until, time.time()) if a.until else None
    while True:
        alerts, recs = w.run_once()
        save_state(a.state, w.s)
        write_alerts(a.alerts, alerts)
        top = max([LEVELS[x[1]] for x in alerts] or [0])
        if a.once or not a.follow:
            if a.summary:
                print(summary(recs, a))
            return top
        if a.exit_on and top >= LEVELS[a.exit_on]:
            print("FLAG " + "; ".join("%s %s" % (x[1], x[2]) for x in alerts if LEVELS[x[1]] >= LEVELS[a.exit_on]))
            print(summary(recs, a))
            return top
        if w.s.get("nb_end"):
            print(summary(recs, a))
            return 0
        if deadline and time.time() >= deadline:
            print("HOURLY_DUE")
            print(summary(recs, a))
            return 0
        time.sleep(a.poll_s if not deadline else max(1.0, min(a.poll_s, deadline - time.time())))


if __name__ == "__main__":
    sys.exit(main())
