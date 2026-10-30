#!/usr/bin/env python3
"""Replica report from sampler.py's JSONL (a live file or a local copy): dynamics, not snapshots.

Sections:
- VERDICT: one line, the worst of: "stable" / "growing toward X, ETA ~Y h" / "blew up at Z" for host RAM
  (Kaggle-eq vs 173 GiB) and GPU memory (worst rank's allocator cap - peak vs 0.26 GiB), plus unplanned server
  deaths, finalized error attempts and two-engine episodes.
- snapshots every --bucket-min minutes (host RAM, VRAM per worst rank, retries, OOMs, throughput);
- per-hour dynamics: the change within each hour of the clock (normalized per hour), marked at restarts;
- trends: least-squares slope per hour over the last --window-h hours of the current server, after --warmup-min
  minutes of warm-up, the current value, the limit and the time to reach it at that slope;
- events: server instances, deaths, OOM lines, finalized errors, two engines, the notebook's end;
- fault injections (with --fault-log): the recovery checks of recovery.py;
- attempts (with --logs-dir).
The clock is the Kaggle clock (solver elapsed hours) when the solver log gave its zero, else hours since the first
sample.

Usage: report.py --samples mon.jsonl [--logs-dir dsv41_logs] [--fault-log fault_inject.jsonl] [--json out.json]"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import reli_common as rc  # noqa: E402
import recovery  # noqa: E402


def build_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--samples", required=True)
    ap.add_argument("--logs-dir", help="the solver's log dir copy (solver.log, attempts.jsonl)")
    ap.add_argument("--fault-log", help="fault_inject.py log")
    ap.add_argument("--bucket-min", type=float, default=30)
    ap.add_argument("--warmup-min", type=float, default=20, help="skip this long after each server start")
    ap.add_argument("--window-h", type=float, default=2.0, help="slope window (end of the current server)")
    ap.add_argument("--end-h", type=float, default=11.5, help="run end on the clock (the solver's hard stop)")
    ap.add_argument("--ram-limit", type=float, default=rc.KAGGLE_RAM_LIMIT_GIB)
    ap.add_argument("--cap-peak-limit", type=float, default=rc.CAP_PEAK_LIMIT_GIB)
    ap.add_argument("--fault-window-s", type=float, default=3600)
    ap.add_argument("--max-restart-s", type=float, default=900)
    ap.add_argument("--json", help="also write the report as JSON")
    return ap.parse_args(argv)


def smi_free_min(r):
    v = [g["free_mib"] / 1024 for g in r.get("gpus") or [] if g.get("free_mib") is not None]
    return min(v) if v else None


def gen_tok(r):
    return ((r.get("solver") or {}).get("metrics") or {}).get("gen_tok_s")


# (key, label, getter, limit, direction, in the verdict)
METRICS = (
    ("ram", "Kaggle-eq RAM", "kaggle_eq_gib", "ram", "up", True),
    ("cap_peak", "worst cap - peak", "vllm.worst.cap_minus_peak", "cap", "down", True),
    ("cap_reserved", "worst cap - reserved", "vllm.worst.cap_minus_reserved", 0.0, "down", False),
    ("peak", "worst peak allocated", "vllm.worst.peak", None, "up", False),
    ("reserved", "worst reserved", "vllm.worst.reserved", None, "up", False),
    ("tp_rss", "TP workers RSS", "groups.tp_workers.rss_gib", None, "up", False),
    ("engine_rss", "EngineCore RSS", "groups.engine_core.rss_gib", None, "up", False),
    ("sandboxes", "sandboxes RSS", "sandboxes.total_gib", None, "up", False),
    ("dev_shm", "/dev/shm used", "dev_shm_used_gib", None, "up", False),
    ("smi_free", "nvidia-smi free min (board; info only)", smi_free_min, None, "down", False),
)


def get(r, g):
    return g(r) if callable(g) else rc.get_path(r, g)


def fmt(x, nd=2):
    if x is None:
        return "-"
    if isinstance(x, float):
        return ("%." + str(nd) + "f") % x
    return str(x)


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(fmt(c) for c in row) + " |" for row in rows]
    return "\n".join(out)


def fault_windows(faults, window_s):
    return [(f["t"], f["t"] + window_s + float(f.get("seconds") or 0)) for f in faults]


def build(a):
    recs = rc.load_records(a.samples)
    recs = [r for r in recs if "t" in r]
    if not recs:
        return {"verdict": ("N/A", "no samples in %s" % a.samples)}, "no samples in %s" % a.samples
    hours, label = rc.clock(recs)
    faults = recovery.load_faults(a.fault_log) if a.fault_log else []
    fw = fault_windows(faults, a.fault_window_s)
    limits = {"ram": a.ram_limit, "cap": a.cap_peak_limit}
    lines = []
    t0 = rc.kaggle_t0(recs)
    nfast = sum(1 for r in recs if r.get("fast"))
    lines.append("# Reliability report: %s" % a.samples)
    lines.append("samples %d (%d fast), %s .. %s; clock: %s%s" % (
        len(recs), nfast, rc.utc(recs[0]["t"]), rc.utc(recs[-1]["t"]), label,
        (" (solver t0 %s)" % rc.utc(t0)) if t0 else ""))
    srcs = sorted({r.get("kaggle_eq_source") for r in recs if r.get("kaggle_eq_source")})
    last = recs[-1]
    foreign = (None if last.get("kaggle_eq_meminfo_gib") is None or last.get("kaggle_eq_gib") is None
               else last["kaggle_eq_meminfo_gib"] - last["kaggle_eq_gib"])
    lines.append("host RAM source: %s%s" % (", ".join(srcs) or "-", "" if not foreign or "meminfo_hostwide" in srcs
                 else "; /proc/meminfo is host-wide here (+%.1f GiB not ours at the last sample)" % foreign))
    if any(s == "meminfo_hostwide" for s in srcs):
        lines.append("WARNING: no cgroup accounting: Kaggle-eq used the host-wide /proc/meminfo and overstates RAM")

    # ---- events
    events, verdicts = [], []
    segs = rc.segments(recs)
    for s in segs:
        v = s[-1].get("vllm") or {}
        if v.get("server_key"):
            caps = {k: d.get("cap_minus_reserved0") for k, d in (v.get("ranks") or {}).items()}
            events.append((s[0]["t"], "server instance %s: KV pool %s tokens, cap - reserved at start %s"
                           % (v.get("server_key"), v.get("kv_tokens"), caps)))
    prev = None
    for r in recs:
        sc = ((r.get("solver") or {}).get("counts") or {})
        if prev is not None:
            pc = ((prev.get("solver") or {}).get("counts") or {})
            if sc.get("server_died", 0) > pc.get("server_died", 0):
                planned = any(lo <= r["t"] <= hi for lo, hi in fw)
                events.append((r["t"], "server died%s" % (" (fault injection)" if planned else " (UNPLANNED)")))
                if not planned:
                    verdicts.append(("FAIL", "server died at %.2f h without a fault injection" % hours(r["t"])))
            if sc.get("stop_error", 0) > pc.get("stop_error", 0):
                events.append((r["t"], "%d attempt(s) finalized as error" % (sc["stop_error"] - pc.get("stop_error", 0))))
            if (r.get("engines") or 0) > 1 and (prev.get("engines") or 0) <= 1:
                events.append((r["t"], "%d EngineCore processes at once" % r["engines"]))
                verdicts.append(("FAIL", "two engines' host RAM at once at %.2f h" % hours(r["t"])))
        prev = r
    sc_last = ((last.get("solver") or {}).get("counts") or {})
    if sc_last.get("stop_error"):
        verdicts.append(("FAIL", "%d attempt(s) finalized as error" % sc_last["stop_error"]))
    oom_t = next((r["t"] for r in recs if ((r.get("vllm") or {}).get("counts") or {}).get("oom")
                  or ((r.get("vllm") or {}).get("worst") or {}).get("ooms")), None)
    if oom_t:
        events.append((oom_t, "first GPU OOM (vLLM OOM count or a CUDA out-of-memory line)"))
    ram_t = next((r["t"] for r in recs if (r.get("kaggle_eq_gib") or 0) >= a.ram_limit), None)
    if ram_t:
        events.append((ram_t, "Kaggle-eq RAM reached %.1f GiB: Kaggle would OOM-kill the notebook" % a.ram_limit))
    rt = next((r for r in recs if (((r.get("vllm") or {}).get("worst") or {}).get("retries") or 0) > 0), None)
    if rt:
        events.append((rt["t"], "first allocator retry (the allocator hit its cap)"))
    if last.get("notebook_end"):
        events.append((last["t"], "notebook end: %s" % last["notebook_end"]))
    for f in faults:
        events.append((f["t"], "fault %s on %s pid %s" % (f.get("action"), f.get("target"), f.get("pid"))))
    events.sort(key=lambda e: e[0])

    # ---- trends (current server segment, server running, after warm-up)
    run_segs = [s for s in segs if (s[-1].get("vllm") or {}).get("server_key")]
    cur = [r for r in (run_segs[-1] if run_segs else segs[-1]) if (r.get("engines") or 0) >= 1] or \
        (run_segs[-1] if run_segs else segs[-1])
    w = rc.post_warmup(cur, a.warmup_min * 60, a.window_h * 3600)
    now_h = hours(cur[-1]["t"])
    trend_rows, trend_json = [], {}
    for key, name, g, lim, direction, in_verdict in METRICS:
        limit = limits.get(lim, lim) if isinstance(lim, str) else lim
        pts = [(hours(r["t"]), get(r, g)) for r in w]
        pts = [p for p in pts if p[1] is not None]
        span = (pts[-1][0] - pts[0][0]) if len(pts) > 1 else 0
        slope = rc.linfit(pts) if len(pts) >= 5 and span >= 0.25 else None
        current = get(cur[-1], g)
        blew = None
        if key == "cap_peak" and oom_t:
            blew = hours(oom_t)
        if key == "ram" and ram_t:
            blew = hours(ram_t)
        if limit is not None:
            nm = name if blew is None else {"cap_peak": "GPU memory (first OOM)",
                                            "ram": "host RAM (Kaggle's limit)"}.get(key, name)
            st, txt = rc.trend_verdict(nm, current, slope, limit, direction, now_h, a.end_h, blew)
            eta = rc.eta_hours(current, slope, limit, direction) if slope is not None else None
        else:
            st, txt, eta = "-", "", None
        trend_rows.append([name, current, None if slope is None else round(slope, 3), limit,
                           "-" if eta is None else "%.1f h" % eta, (st + ": " + txt) if txt else "-"])
        trend_json[key] = {"now": current, "slope_per_h": slope, "limit": limit, "eta_h": eta, "status": st,
                           "text": txt, "points": len(pts)}
        if in_verdict and limit is not None:
            verdicts.append((st, txt))
    if (((last.get("vllm") or {}).get("worst") or {}).get("retries") or 0) > 0:
        verdicts.append(("WATCH", "allocator retries %s (the allocator hit its cap)"
                         % last["vllm"]["worst"]["retries"]))

    # ---- fault recovery
    fault_res = []
    if faults:
        slog, att = recovery.default_paths(a.logs_dir)
        for f in faults:
            res = recovery.analyze(f, recs, slog, att, a.fault_window_s, a.max_restart_s)
            fault_res.append(res)
            if res["verdict"] == "FAIL":
                verdicts.append(("FAIL", "fault %s at %.2f h: recovery FAILED (%s)" % (
                    f.get("action"), hours(f["t"]), ", ".join(c[0] for c in res["checks"] if c[1] == "FAIL"))))

    ranked = sorted(verdicts, key=lambda v: -rc.STATUS_ORDER.get(v[0], 0))
    top = ranked[0] if ranked else ("N/A", "no data")
    others = [v[1] for v in ranked[1:] if v[1]]
    verdict_line = "VERDICT: %s: %s%s" % (top[0], top[1], ("; " + "; ".join(others)) if others else "")
    lines.insert(1, verdict_line)

    # ---- snapshots
    rows, seen = [], set()
    for r in recs:
        k = int(hours(r["t"]) * 60 // a.bucket_min)
        if k in seen and r is not last:
            continue
        seen.add(k)
        wv = (r.get("vllm") or {}).get("worst") or {}
        cg = r.get("cgroup") or {}
        m = (r.get("solver") or {}).get("metrics") or {}
        rows.append([hours(r["t"]), r.get("utc"), r.get("kaggle_eq_gib"), r.get("kaggle_eq_meminfo_gib"),
                     cg.get("anon_gib"), cg.get("shmem_gib"), r.get("dev_shm_used_gib"),
                     "%s / %s" % ((r.get("sandboxes") or {}).get("n"), fmt((r.get("sandboxes") or {}).get("total_gib"))),
                     r.get("engines"), wv.get("peak"), wv.get("reserved"), wv.get("cap_minus_peak"),
                     wv.get("cap_minus_reserved"), wv.get("retries"), wv.get("ooms"),
                     None if smi_free_min(r) is None else round(smi_free_min(r), 2), m.get("gen_tok_s"), m.get("kv")])
    lines.append("\n## Snapshots (every %g min)" % a.bucket_min)
    lines.append(table([label, "UTC", "Kaggle-eq", "meminfo form", "cg anon", "cg shmem", "/dev/shm",
                        "sandboxes n / GiB", "engines", "worst peak", "worst reserved", "cap-peak", "cap-reserved",
                        "retries", "OOMs", "smi free min", "gen tok/s", "KV"], rows))

    # ---- per-hour dynamics
    lines.append("\n## Per-hour dynamics (change within each hour, per hour; '-' = under 15 min of data)")
    by_h = {}
    for r in recs:
        by_h.setdefault(int(hours(r["t"]) // 1), []).append(r)
    hrows = []
    for h in sorted(by_h):
        b = by_h[h]
        f0, f1 = b[0], b[-1]
        span = (f1["t"] - f0["t"]) / 3600
        restart = len({(x.get("vllm") or {}).get("server_key") for x in b}) > 1
        if restart:  # measure within the newest server only
            k1 = (f1.get("vllm") or {}).get("server_key")
            b2 = [x for x in b if (x.get("vllm") or {}).get("server_key") == k1]
            f0, span = b2[0], (f1["t"] - b2[0]["t"]) / 3600

        def d(g):
            x0, x1 = get(f0, g), get(f1, g)
            return None if (x0 is None or x1 is None or span < 0.25) else round((x1 - x0) / span, 3)
        gens = [gen_tok(x) for x in b if gen_tok(x) is not None]
        hrows.append(["%d-%d" % (h, h + 1), get(f1, "kaggle_eq_gib"), d("kaggle_eq_gib"),
                      get(f1, "vllm.worst.peak"), d("vllm.worst.peak"), get(f1, "vllm.worst.reserved"),
                      d("vllm.worst.reserved"), get(f1, "vllm.worst.cap_minus_peak"), get(f1, "vllm.worst.retries"),
                      d("groups.tp_workers.rss_gib"), d("sandboxes.total_gib"),
                      round(sum(gens) / len(gens)) if gens else None,
                      "server restart" if restart else ""])
    lines.append(table([label, "Kaggle-eq end", "/h", "worst peak end", "/h", "worst reserved end", "/h",
                        "cap-peak end", "retries", "TP RSS /h", "sandboxes /h", "gen tok/s", "note"], hrows))

    lines.append("\n## Trends (slope over the last %.1f h of the current server, %g min warm-up skipped; %d points)"
                 % (a.window_h, a.warmup_min, len(w)))
    lines.append(table(["metric", "now", "slope /h", "limit", "time to limit", "verdict"], trend_rows))

    lines.append("\n## Events")
    for t, e in events:
        lines.append("- %s (%s %.2f) %s" % (rc.hms(t), label, hours(t), e))
    if not events:
        lines.append("- none")

    if fault_res:
        lines.append("\n## Fault injections")
        for res in fault_res:
            lines.append(recovery.format_result(res, hours))

    att = (last.get("attempts") or {})
    lines.append("\n## Attempts")
    lines.append("attempt records %s, per-try error records %s (re-queued jobs), finalized errors %s; solver: died %s, "
                 "re-queues %s, retries %s, timeouts %s" % (att.get("records"), att.get("error_records"),
                                                           sc_last.get("stop_error"), sc_last.get("server_died"),
                                                           sc_last.get("requeue"), sc_last.get("retry"),
                                                           sc_last.get("timeout")))
    if a.logs_dir:
        jo = recovery.job_outcomes(os.path.join(a.logs_dir, "attempts.jsonl"), t0, 0)
        if jo:
            lines.append("jobs with an error record: %d; finished normally later: %d; pending or lost: %d" % jo)
    lines.append("\n" + verdict_line)
    out = {"verdict": {"status": top[0], "line": verdict_line}, "trends": trend_json,
           "events": [[t, e] for t, e in events], "faults": fault_res, "clock": label, "t0": t0}
    return out, "\n".join(lines)


def main(argv=None):
    a = build_args(argv)
    out, text = build(a)
    print(text)
    if a.json:
        with open(a.json, "w") as f:
            json.dump(out, f, indent=1, default=str)
    return 0


if __name__ == "__main__":
    sys.exit(main())
