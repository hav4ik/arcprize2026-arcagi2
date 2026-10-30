"""Recovery checks after a fault_inject.py action, from sampler.py records, the solver log and attempts.jsonl.

Pass criteria (kill-tp, sigstop-engine, and gpu-oom-burst when it kills the engine):
  finalized_errors  no attempt FINALIZED as error after the fault (solver "stop=error")            FAIL otherwise
  requeued          the in-flight jobs were re-queued ("re-queue" lines) when the server died       FAIL otherwise
  tries_spent       no "retry" lines: a retry spends one of the job's tries (it is not a free
                    re-queue), so a second death would finalize those jobs as errors               WARN otherwise
  one_engine        never two EngineCore processes, no old and new vLLM processes at once, and
                    the Kaggle-eq RAM peak within --ram-margin of the pre-fault value             FAIL otherwise
  ram_limit         Kaggle-eq RAM stayed under Kaggle's 173 GiB                                    FAIL otherwise
  restart_time      "server up" within --max-restart-s of the fault (kill) or of the hang limit   FAIL otherwise
  server_died       kill-tp / sigstop-engine: the solver saw the death (else the fault did not
                    register or the hang was never detected)                                       FAIL otherwise
  jobs              errored jobs since the fault that finished normally later vs still pending     WARN if pending
                    at the end of the data
cpu-stress: no server death and no finalized error during the stress window; the throughput dip is reported."""
import json
import os

import reli_common as rc

DEATH_ACTIONS = ("kill-tp", "sigstop-engine")


def load_faults(path):
    out = []
    try:
        with open(path) as f:
            for line in f:
                try:
                    e = json.loads(line)
                except ValueError:
                    continue
                if e.get("phase") == "signaled" and not e.get("dry_run"):
                    out.append(e)
    except OSError:
        pass
    return out


def solver_events(path, ref_epoch):
    """[(epoch, kind, line)] from the solver log; stamps are mapped to the day closest to ref_epoch."""
    out = []
    try:
        f = open(path, errors="replace")
    except OSError:
        return out
    with f:
        for line in f:
            m = rc.STAMP.match(line)
            if not m:
                continue
            t = rc.stamp_near(int(m.group(1)), int(m.group(2)), int(m.group(3)), ref_epoch)
            kind = None
            if "server died" in line:
                kind = "server_died"
            elif rc.UP_AFTER.search(line):
                kind = "server_up"
            elif "re-queue " in line:
                kind = "requeue"
            elif rc.RETRY.search(line):
                kind = "retry"
            elif rc.FINAL_ERROR.search(line):
                kind = "stop_error"
            elif "APITimeoutError" in line or "Request timed out" in line:
                kind = "timeout"
            if kind:
                out.append((t, kind, line.strip()[:240]))
    return out


def job_outcomes(path, t0, since):
    """(errored jobs since `since`, of them finished normally later, still pending) from attempts.jsonl."""
    if not t0:
        return None
    recs = []
    try:
        with open(path) as f:
            for line in f:
                try:
                    recs.append(json.loads(line))
                except ValueError:
                    pass
    except OSError:
        return None
    key = lambda r: (r.get("task_id"), r.get("pair", 0), r.get("attempt"))
    err, ok = {}, {}
    for r in recs:
        te = r.get("t_end_h")
        if te is None:
            continue
        t = t0 + te * 3600
        if r.get("error") and t >= since - 60:
            err.setdefault(key(r), t)
        elif not r.get("error"):
            ok[key(r)] = max(ok.get(key(r), 0), t)
    done = sum(1 for k, t in err.items() if ok.get(k, 0) > t)
    return len(err), done, len(err) - done


def _pids(r):
    return {p[0] for p in r.get("vllm_procs") or []}


def analyze(entry, records, solver_log=None, attempts=None, window_s=3600, max_restart_s=900, ram_margin_gib=10.0,
            ram_limit=rc.KAGGLE_RAM_LIMIT_GIB, hang_limit_s=2100):
    t = entry["t"]
    act = entry.get("action")
    before = [r for r in records if r["t"] <= t]
    after = [r for r in records if t < r["t"] <= t + window_s]
    pre = before[-1] if before else None
    old = _pids(pre) if pre else set()
    tl, checks = [], []

    def at(x):
        return "+%.0fs" % (x - t)

    gone = next((r["t"] for r in after if not (_pids(r) & old)), None) if old else None
    new = next((r["t"] for r in after if _pids(r) - old), None)
    gpu_free = next((r["t"] for r in after if not ({a[0] for a in r.get("gpu_apps") or []} & old)), None) \
        if old else None
    coexist = [r for r in after if (_pids(r) & old) and (_pids(r) - old)
               and any(str(p[1]).startswith("VLLM::EngineCore") for p in r.get("vllm_procs") or [] if p[0] not in old)]
    max_eng = max([r.get("engines") or 0 for r in after] or [0])
    pre_keq = pre.get("kaggle_eq_gib") if pre else None
    keqs = [r.get("kaggle_eq_gib") for r in after if r.get("kaggle_eq_gib") is not None]
    peak_keq = max(keqs) if keqs else None
    if gone:
        tl.append((gone, "all pre-fault vLLM processes gone"))
    if gpu_free:
        tl.append((gpu_free, "no pre-fault vLLM process on any GPU"))
    if new:
        tl.append((new, "first new vLLM process"))
    last_state = None
    for r in after:
        st = (r.get("supervisor") or {}).get("state")
        if st != last_state and st is not None:
            tl.append((r["t"], "supervisor state %s (start %s)" % (st, (r.get("supervisor") or {}).get("start"))))
        last_state = st
    ev = [e for e in solver_events(solver_log, t) if t - 2 <= e[0] <= t + window_s] if solver_log else []
    died = next((e for e in ev if e[1] == "server_died"), None)
    up = next((e for e in ev if e[1] == "server_up" and (died is None or e[0] >= died[0])), None)
    n = {k: sum(1 for e in ev if e[1] == k) for k in ("requeue", "retry", "stop_error", "timeout")}
    if died:
        tl.append((died[0], "solver: " + died[2][:160]))
    if up:
        tl.append((up[0], "solver: " + up[2][:160]))
    tl.sort(key=lambda x: x[0])

    if act == "cpu-stress":
        end = t + float(entry.get("seconds") or 0)
        during = [e for e in ev if e[0] <= end + 60]
        d = [e for e in during if e[1] == "server_died"]
        se = [e for e in during if e[1] == "stop_error"]
        checks.append(("no_death", "PASS" if not d else "FAIL", "%d server death(s) during the stress" % len(d)))
        checks.append(("finalized_errors", "PASS" if not se else "FAIL", "%d attempt(s) finalized as error" % len(se)))
        g_before = [((r.get("solver") or {}).get("metrics") or {}).get("gen_tok_s") for r in before[-10:]]
        g_during = [((r.get("solver") or {}).get("metrics") or {}).get("gen_tok_s") for r in after if r["t"] <= end]
        g_before = [x for x in g_before if x is not None]
        g_during = [x for x in g_during if x is not None]
        if g_before and g_during:
            checks.append(("throughput", "INFO", "gen tok/s %.0f before, min %.0f during"
                           % (sum(g_before) / len(g_before), min(g_during))))
    else:
        se = [e for e in ev if e[1] == "stop_error"]
        checks.append(("finalized_errors", "PASS" if not se else "FAIL",
                       "%d attempt(s) finalized as error%s" % (len(se), (": " + se[0][2][:120]) if se else "")))
        if act in DEATH_ACTIONS:
            checks.append(("server_died", "PASS" if died else "FAIL",
                           ("solver saw the death at %s" % at(died[0])) if died else
                           "no 'server died' in the solver log within %d s" % window_s))
        if died:
            checks.append(("requeued", "PASS" if n["requeue"] else "FAIL",
                           "%d re-queue line(s) after the death" % n["requeue"]))
            limit = max_restart_s if act != "sigstop-engine" else hang_limit_s + max_restart_s
            if up:
                checks.append(("restart_time", "PASS" if up[0] - t <= limit else "FAIL",
                               "server up %.0f s after the fault (limit %d s)" % (up[0] - t, limit)))
            else:
                checks.append(("restart_time", "FAIL", "no 'server up' after the death within %d s" % window_s))
        elif act == "gpu-oom-burst":
            w0 = (pre or {}).get("vllm", {}).get("worst", {}) if pre else {}
            w1 = (after[-1].get("vllm") or {}).get("worst", {}) if after else {}
            checks.append(("absorbed", "PASS", "no server death; retries %s -> %s, OOMs %s -> %s"
                           % (w0.get("retries"), w1.get("retries"), w0.get("ooms"), w1.get("ooms"))))
        checks.append(("tries_spent", "PASS" if not n["retry"] else "WARN",
                       "%d retry line(s) (each spends one of the job's tries)" % n["retry"]))
        two = max_eng > 1 or bool(coexist) or (peak_keq is not None and pre_keq is not None
                                                and peak_keq > pre_keq + ram_margin_gib)
        checks.append(("one_engine", "FAIL" if two else "PASS",
                       "max %d EngineCore, %s; Kaggle-eq peak %s GiB vs %s before (margin %.0f)"
                       % (max_eng, "old and new vLLM processes coexisted at %s" % coexist[0]["utc"] if coexist
                          else "no old/new overlap", peak_keq, pre_keq, ram_margin_gib)))
        if peak_keq is not None:
            checks.append(("ram_limit", "PASS" if peak_keq < ram_limit else "FAIL",
                           "Kaggle-eq peak %.1f GiB (limit %.1f)" % (peak_keq, ram_limit)))
    if n.get("timeout"):
        checks.append(("timeouts", "WARN", "%d APITimeoutError line(s) after the fault" % n["timeout"]))
    jo = job_outcomes(attempts, rc.kaggle_t0(records), t) if attempts else None
    if jo:
        e, ok, pend = jo
        checks.append(("jobs", "PASS" if not pend else "WARN",
                       "%d job(s) with an error record since the fault: %d finished normally later, %d pending%s"
                       % (e, ok, pend, " (lost if the run has ended)" if pend else "")))
    order = {"PASS": 0, "INFO": 0, "WARN": 1, "FAIL": 2}
    verdict = max([c[1] for c in checks] or ["PASS"], key=lambda s: order.get(s, 0))
    if verdict == "INFO":
        verdict = "PASS"
    return {"action": act, "t": t, "utc": rc.utc(t), "target": entry.get("target"), "pid": entry.get("pid"),
            "timeline": [(x - t, s) for x, s in tl], "checks": checks, "verdict": verdict,
            "samples_after": len(after), "fast_samples": sum(1 for r in after if r.get("fast"))}


def format_result(res, hours=None):
    h = (" (%.2f h)" % hours(res["t"])) if hours else ""
    lines = ["fault %s at %s%s on %s pid %s: %s" % (res["action"], res["utc"], h, res.get("target"), res.get("pid"),
                                                   res["verdict"])]
    for dt, s in res["timeline"]:
        lines.append("    %+7.0fs  %s" % (dt, s))
    for name, st, detail in res["checks"]:
        lines.append("  %-5s %-16s %s" % (st, name, detail))
    if res["samples_after"] and not res["fast_samples"] and res["action"] in DEATH_ACTIONS:
        lines.append("  note: no 5 s samples after the fault (pass --fault-log to sampler.py): timings are coarse")
    return "\n".join(lines)


def default_paths(logs_dir):
    return (os.path.join(logs_dir, "solver.log"), os.path.join(logs_dir, "attempts.jsonl")) if logs_dir else (None,
                                                                                                             None)

