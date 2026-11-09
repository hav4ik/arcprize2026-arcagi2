#!/usr/bin/env python3
"""E059 B16 (4x4090): restart-under-load metrics from a copied-back run dir (run_b16.sh + killer.sh), restart-aware.

    .venv/bin/python experiments/E059_restart_under_load/b16_4090/analyze_b16.py RUN_DIR [--out F.json] [--md F.md]

Inputs in RUN_DIR (/root/runs/<name>): arm_t0, env.txt, run.log, killer.log, mem2s.log (killer.sh: host meminfo every
2 s, GiB), mem2s_guard.log (run_b16.sh: host AnonPages/Shmem + cgroup anon/shmem/file every 2 s, MiB), memdetail.log,
metrics10.jsonl (sampler.py), gpu.csv, out/dsv41_logs/{solver.log, attempts.jsonl, status.json}, out/dsv41_traces.tar.gz.

Server counters reset when the solver restarts the server, so counter windows are taken inside one server's life:
"pre" = server 1 up -> kill, "post" = server 2 up -> force deadline. Generated tokens across the restart come from a
cumulative series with the reset removed (each server's counter added on top of the previous one's last value).
Memory: "steady" = the 5 min before the kill, "restart" = kill -> kill + 20 min (and through server 2 up + 2 min).
Container numbers (cgroup anon + shmem) are the primary ones: the host-wide AnonPages also hold other tenants' memory.
"""
import argparse
import bisect
import collections
import datetime as dt
import json
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments/E057b_workers64_stress/scripts"))
import compare_arms as ca  # noqa: E402

TS = re.compile(r"^\[?(\d\d):(\d\d):(\d\d)")


def abs_time(hms, t0):
    """Epoch of an HH:MM:SS (UTC) stamp on T0's day (next day if it is before T0 by more than an hour)."""
    m = TS.match(hms)
    if not m:
        return None
    h, mi, s = map(int, m.groups())
    day = dt.datetime.fromtimestamp(t0, dt.timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    t = day.timestamp() + h * 3600 + mi * 60 + s
    return t + 86400 if t < t0 - 3600 else t


def kv_line(line):
    return {k: float(v) for k, v in re.findall(r"([A-Za-z_]+)=([0-9.]+)(?=[ ,]|$)", line)}


def read_mem2s(run, t0):
    """killer.sh mem2s.log: GiB host values, vllm_procs, gpu_mib list."""
    out = []
    p = run / "mem2s.log"
    for line in (p.read_text(errors="replace").splitlines() if p.exists() else []):
        t = abs_time(line, t0)
        if t is None:
            continue
        d = kv_line(line)
        g = re.search(r"gpu_mib=([0-9,]+)", line)
        d["gpu_mib"] = [int(x) for x in g.group(1).split(",") if x] if g else []
        d["t"] = t
        out.append(d)
    return out


def read_guard(run, t0):
    """run_b16.sh mem2s_guard.log: MiB values -> GiB."""
    out = []
    p = run / "mem2s_guard.log"
    for line in (p.read_text(errors="replace").splitlines() if p.exists() else []):
        t = abs_time(line, t0)
        if t is None:
            continue
        d = kv_line(line)
        r = {"t": t, "vllm_procs": d.get("vllm_procs")}
        for k in ("host_anon", "host_shmem", "cg_anon", "cg_shmem", "cg_file", "cg_current"):
            if k in d:
                r[k] = d[k] / 1024
        if "cg_anon" in r and "cg_shmem" in r:
            r["cg_as"] = r["cg_anon"] + r["cg_shmem"]
        if "host_anon" in r and "host_shmem" in r:
            r["host_as"] = r["host_anon"] + r["host_shmem"]
        out.append(r)
    return out


def solver_events(run, t0):
    p = run / "out/dsv41_logs/solver.log"
    lines = p.read_text(errors="replace").splitlines() if p.exists() else []
    ev = collections.defaultdict(list)
    pats = {"server_up": "server up after", "server_died": "server died", "server_start": "server start (",
            "exited_startup": "server exited during startup", "requeue_conn": "after a server-side error",
            "retry": "] retry ", "admission_closed": "admission closed", "local_score": "local score",
            "force": "force", "hard_stop": "hard stop", "kv_gate": "KV gate", "not_up": "server not up after",
            "no_restart": "no restart left"}
    for line in lines:
        t = abs_time(line, t0)
        for k, s in pats.items():
            if s in line:
                ev[k].append((t, line.split("] ", 1)[-1][:300]))
    return ev, lines


def cum_gen(rows):
    """[(t, cumulative generated tokens over all servers)] with counter resets removed; None rows are gaps."""
    out, off, prev = [], 0.0, None
    for r in rows:
        if "m" not in r:
            continue
        g = r["m"].get("vllm:generation_tokens_total", 0.0)
        if prev is not None and g < prev:
            off += prev
        prev = g
        out.append((r["t"], off + g))
    return out


def rate(series, a, b):
    """Mean tok/s over [a, b] from a cumulative series (linear interpolation; flat across gaps)."""
    ts = [t for t, _ in series]
    if not ts or b <= a:
        return None

    def at(t):
        i = bisect.bisect_right(ts, t)
        if i == 0:
            return series[0][1]
        if i >= len(ts):
            return series[-1][1]
        (t1, v1), (t2, v2) = series[i - 1], series[i]
        # across an outage gap (no server samples) the counter cannot be interpolated: hold the last value
        if t2 - t1 > 30:
            return v1
        return v1 + (v2 - v1) * (t - t1) / (t2 - t1)
    return (at(b) - at(a)) / (b - a)


def first_cross(series, t_from, target, t_max, width=300, step=10):
    """First t >= t_from + width where the trailing `width`-s mean rate reaches target."""
    t = t_from + width
    while t <= t_max:
        r = rate(series, t - width, t)
        if r is not None and r >= target:
            return t
        t += step
    return None


def stats(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None
    return {"n": len(xs), "median": round(statistics.median(xs), 2), "mean": round(sum(xs) / len(xs), 2),
            "min": round(min(xs), 2), "max": round(max(xs), 2)}


def memory_restart(guard, m2, kill, up2, t0):
    pre_g = [g for g in guard if kill - 300 <= g["t"] < kill]
    post_end = max(kill + 1200, (up2 or kill) + 120)
    post_g = [g for g in guard if kill <= g["t"] <= post_end]
    pre_h = [m for m in m2 if kill - 300 <= m["t"] < kill]
    post_h = [m for m in m2 if kill <= m["t"] <= post_end]
    res = {"window_steady_s": [round(kill - 300 - t0), round(kill - t0)],
           "window_restart_s": [round(kill - t0), round(post_end - t0)]}
    for name, pre, post, key in (("container_cgroup_anon+shmem_GiB", pre_g, post_g, "cg_as"),
                                 ("container_cgroup_shmem_GiB", pre_g, post_g, "cg_shmem"),
                                 ("container_cgroup_anon_GiB", pre_g, post_g, "cg_anon"),
                                 ("host_AnonPages+Shmem_GiB(guard)", pre_g, post_g, "host_as")):
        s, p = stats([g.get(key) for g in pre]), stats([g.get(key) for g in post])
        res[name] = {"steady": s, "restart": p,
                     "peak_over_steady_median": round(p["max"] / s["median"], 3) if s and p and s["median"] else None}
        if p:
            pk = max(post, key=lambda g: g.get(key) or 0)
            res[name]["peak_at_s_after_kill"] = round(pk["t"] - kill)
    hs = lambda m: (m.get("AnonPages") or 0) + (m.get("Shmem") or 0)
    s, p = stats([hs(m) for m in pre_h]), stats([hs(m) for m in post_h])
    res["host_AnonPages+Shmem_GiB(killer mem2s)"] = {
        "steady": s, "restart": p, "peak_over_steady_median": round(p["max"] / s["median"], 3) if s and p else None}
    # timeline
    tl = {}
    gone = next((g for g in guard if g["t"] > kill and (g.get("vllm_procs") or 0) == 0), None)
    tl["old_vllm_procs_gone_s"] = round(gone["t"] - kill) if gone else None
    gone_m = next((m for m in m2 if m["t"] > kill and (m.get("vllm_procs") or 0) == 0), None)
    tl["old_vllm_procs_gone_s(mem2s)"] = round(gone_m["t"] - kill) if gone_m else None
    gpu_free = next((m for m in m2 if m["t"] > kill and m["gpu_mib"] and max(m["gpu_mib"]) < 1000), None)
    tl["gpus_freed_s"] = round(gpu_free["t"] - kill) if gpu_free else None
    shm_low = next((g for g in guard if g["t"] > kill and (g.get("cg_shmem") if g.get("cg_shmem") is not None else 999) < 5), None)
    tl["old_shmem_released_lt5GiB_s"] = round(shm_low["t"] - kill) if shm_low else None
    if shm_low:
        tl["cg_anon+shmem_at_release_GiB"] = round(shm_low.get("cg_as") or 0, 2)
    t_from = shm_low["t"] if shm_low else kill
    new_procs = next((g for g in guard if g["t"] > t_from and (g.get("vllm_procs") or 0) > 0), None)
    tl["new_vllm_procs_seen_s"] = round(new_procs["t"] - kill) if new_procs else None
    alloc = next((g for g in guard if g["t"] > t_from and (g.get("cg_shmem") or 0) > 10), None)
    tl["new_shmem_gt10GiB_s"] = round(alloc["t"] - kill) if alloc else None
    full = next((g for g in guard if g["t"] > t_from and (g.get("cg_shmem") or 0) > 0.95 * ((res["container_cgroup_shmem_GiB"]["steady"] or {}).get("median") or 1e9)), None)
    tl["new_shmem_reaches_95pct_of_steady_s"] = round(full["t"] - kill) if full else None
    tl["server2_up_s"] = round(up2 - kill) if up2 else None
    mins = [g for g in guard if kill <= g["t"] <= (up2 or kill + 1200)]
    if mins:
        lo = min(mins, key=lambda g: g.get("cg_as") or 1e9)
        tl["min_cg_anon+shmem_between_GiB"] = round(lo.get("cg_as") or 0, 2)
        tl["min_at_s"] = round(lo["t"] - kill)
    # overlap check: any sample after the kill with old memory still held AND the new server allocating?
    st = (res["container_cgroup_shmem_GiB"]["steady"] or {}).get("median") or 0
    over = [g for g in post_g if (g.get("cg_shmem") or 0) > 1.1 * st]
    tl["samples_shmem_gt_1.1x_steady"] = len(over)
    res["timeline_s_after_kill"] = tl
    # a 30-s view (every 15th guard sample) of the restart for the README
    res["restart_series_30s"] = [(round(g["t"] - kill), round(g.get("cg_anon") or 0, 1), round(g.get("cg_shmem") or 0, 1),
                                  int(g.get("vllm_procs") or 0)) for g in guard[::15] if kill - 60 <= g["t"] <= post_end]
    return res


def jobs_at_kill(recs, t0, kill, up2):
    """Attempt records that ended between the kill and server 2 up (+60 s), their errors, and whether each job got a
    later record (re-run) and how that ended."""
    key = lambda r: (r.get("task_id"), r.get("pair"), r.get("attempt"))
    end = lambda r: t0 + (r.get("t_end_h") or 0) * 3600
    lo, hi = kill - 30, (up2 or kill + 1200) + 60
    hit = [r for r in recs if "task_id" in r and lo <= end(r) <= hi]
    errs = collections.Counter((r.get("error") or "")[:70] for r in hit if r.get("error"))
    by_job = collections.defaultdict(list)
    for r in recs:
        if "task_id" in r:
            by_job[key(r)].append(r)
    later = {"rerun_finished_ok": 0, "rerun_error": 0, "rerun_cut": 0, "no_rerun_record": 0}
    for r in hit:
        if not r.get("error"):
            continue
        after = [x for x in by_job[key(r)] if end(x) > end(r) + 1]
        if not after:
            later["no_rerun_record"] += 1
            continue
        last = after[-1]
        if last.get("error"):
            later["rerun_error"] += 1
        elif last.get("cuts"):
            later["rerun_cut"] += 1
        else:
            later["rerun_finished_ok"] += 1
    return {"records_ending_kill_to_up2": len(hit), "with_error": sum(1 for r in hit if r.get("error")),
            "error_kinds": dict(errs.most_common()), "harvested_session_answer": sum(1 for r in hit if r.get("error") and r.get("pred")),
            "jobs_rerun": later}


def analyse(run):
    run = Path(run)
    env = ca.env_of(run)
    t0 = float((run / "arm_t0").read_text().strip())
    W = int(env["ARC_WORKERS"])
    stop_new, force = t0 + float(env["ARC_STOP_NEW_S"]), t0 + float(env["ARC_FORCE_AT_S"])
    rows = ca.jsonl(run / "metrics10.jsonl")
    recs = ca.jsonl(run / "out/dsv41_logs/attempts.jsonl")
    sols = json.loads(Path(env["ARC_SOLUTIONS"].replace("/root/data", str(ROOT / "experiments/E055_kaggle_replica/toolkit/workloads"))).read_text())
    ev, lines = solver_events(run, t0)
    kl = (run / "killer.log").read_text(errors="replace").splitlines() if (run / "killer.log").exists() else []
    kill = next((abs_time(l, t0) for l in kl if "SIGKILL worker TP1" in l), None)
    ups = [t for t, _ in ev["server_up"]]
    up1 = ups[0] if ups else None
    up2 = next((t for t in ups if kill and t > kill), None)
    res = {"run": run.name, "workers": W, "t0_utc": dt.datetime.fromtimestamp(t0, dt.timezone.utc).isoformat(),
           "killer_log": kl, "kill_s": round(kill - t0) if kill else None,
           "server_up_s": [round(t - t0) for t in ups], "server_up_lines": [x for _, x in ev["server_up"]],
           "server_died": [(round(t - t0), x) for t, x in ev["server_died"]],
           "server_starts": [(round(t - t0), x) for t, x in ev["server_start"]],
           "exited_during_startup": [(round(t - t0), x[:200]) for t, x in ev["exited_startup"]],
           "no_restart_left": len(ev["no_restart"]),
           "requeue_conn_lines": len(ev["requeue_conn"]),
           "requeue_conn_kinds": dict(collections.Counter(re.sub(r"^re-queue \S+ after a server-side error \(\d+\): ", "", x)[:60]
                                                          for _, x in ev["requeue_conn"]).most_common()),
           "retry_lines_using_job_tries": [(round(t - t0), x[:200]) for t, x in ev["retry"]],
           "admission_closed": [(round(t - t0), x[:200]) for t, x in ev["admission_closed"]],
           "local_score": [x for _, x in ev["local_score"]]}
    try:
        st = json.loads((run / "out/dsv41_logs/status.json").read_text())
        res["final_status"] = {k: st.get(k) for k in ("done", "errors", "pairs_with_answer", "pairs", "restarts", "phase")}
    except Exception:
        res["final_status"] = None
    # attempt records with errors (whole run)
    er = [r for r in recs if r.get("error")]
    res["attempt_records"] = len([r for r in recs if "task_id" in r])
    res["attempt_records_with_error"] = len(er)
    res["error_kinds_whole_run"] = dict(collections.Counter((r.get("error") or "")[:70] for r in er).most_common())
    if kill:
        res["jobs_at_kill"] = jobs_at_kill(recs, t0, kill, up2)
        guard, m2 = read_guard(run, t0), read_mem2s(run, t0)
        res["memory_restart"] = memory_restart(guard, m2, kill, up2, t0)
        # throughput
        cg = cum_gen(rows)
        pre30 = rate(cg, kill - 1800, kill - 10)
        pre15 = rate(cg, kill - 900, kill - 10)
        thr = {"pre_kill_30min_tok_s": round(pre30, 1) if pre30 else None,
               "pre_kill_15min_tok_s": round(pre15, 1) if pre15 else None}
        for m in (5, 15, 30, 45, 60):
            r_k = rate(cg, kill + (m - 5) * 60, kill + m * 60)
            thr[f"kill+{m - 5}..{m}min_tok_s"] = round(r_k, 1) if r_k is not None else None
            if up2:
                r_u = rate(cg, up2 + (m - 5) * 60, up2 + m * 60)
                thr[f"up2+{m - 5}..{m}min_tok_s"] = round(r_u, 1) if r_u is not None else None
        for frac in (0.5, 0.9, 1.0):
            c = first_cross(cg, kill, frac * pre30, force) if pre30 else None
            thr[f"first_5min_window_ge_{frac}x_pre_s_after_kill"] = round(c - kill) if c else None
            thr[f"first_5min_window_ge_{frac}x_pre_s_after_up2"] = round(c - up2) if c and up2 else None
        lost = rate(cg, kill, (up2 or kill)) if up2 else None
        thr["outage_kill_to_up2_s"] = round(up2 - kill) if up2 else None
        thr["gen_tok_s_kill_to_up2"] = round(lost, 1) if lost is not None else None
        # tokens lost: in-flight attempts' completion tokens at the kill (from records that ended in the outage)
        hit = [r for r in recs if "task_id" in r and kill - 30 <= t0 + (r.get("t_end_h") or 0) * 3600 <= (up2 or kill) + 60]
        thr["completion_tokens_of_attempts_ended_by_kill"] = sum(((r.get("usage") or {}).get("completion") or 0) % ca.BIG for r in hit)
        # 5-min series around the kill for the README (-30 .. +60 min)
        thr["series_5min_rel_kill"] = [(m, round(rate(cg, kill + m * 60, kill + (m + 5) * 60) or 0, 1)) for m in range(-30, 61, 5)]
        res["throughput"] = thr
        res["pre"] = {"window_s": [round(up1 - t0), round(kill - t0)],
                      "server": ca.server_stats(rows, up1, kill - 5, W, stop_new),
                      "memory": ca.mem_stats(rows, up1, kill - 5)}
    lo2 = up2 or up1
    grid, tot, used = ca.inflight_series(run, recs, t0, t0, force + 300)
    res["inflight_attempts_with_trace"] = used

    def ctx(lo, hi):
        sel = [(g, v) for g, v in zip(grid, tot) if lo <= g <= hi]
        if not sel:
            return {}
        pk = max(sel, key=lambda x: x[1])
        return {"mean": round(ca.mean([v for _, v in sel])), "peak": round(pk[1]), "peak_at_s": round(pk[0] - t0),
                "min_above_4.5M": round(sum(v > 4.5e6 for _, v in sel) * 10 / 60, 1),
                "min_above_4.8M": round(sum(v > 4.8e6 for _, v in sel) * 10 / 60, 1)}
    res["inflight_ctx_whole"] = ctx(up1, force)
    if kill:
        res["inflight_ctx_pre"] = ctx(up1, kill)
        res["inflight_ctx_post"] = ctx(lo2, force)
    res["inflight_series_60s"] = [(round(g - t0), round(v)) for g, v in list(zip(grid, tot))[::6]]
    wins_pre = ca.host_ratio_windows(rows, up1, kill) if kill else []
    wins_post = ca.host_ratio_windows(rows, lo2, force)
    res["host_ratio_pre_warm"] = ca.cliff_stats([w for w in wins_pre if w["t0"] >= up1 + 600], grid, tot, t0)
    res["host_ratio_post"] = ca.cliff_stats(wins_post, grid, tot, t0)
    res["host_ratio_post_warm"] = ca.cliff_stats([w for w in wins_post if w["t0"] >= lo2 + 600], grid, tot, t0)
    res["host_ratio_windows_post"] = [{**w, "t0": round(w["t0"] - t0), "r": None if w["r"] is None else round(w["r"], 3),
                                       "gen_tok_s": round(w["gen_tok_s"], 1)} for w in wins_post]
    res["post"] = {"window_s": [round(lo2 - t0), round(force - t0)],
                   "server": ca.server_stats(rows, lo2, force, W, stop_new),
                   "attempts": ca.attempts_stats(recs, sols, t0, lo2, force, force),
                   "memory": ca.mem_stats(rows, lo2, force)}
    # memory growth after the restart (container numbers from the 2-s guard log)
    guard = read_guard(run, t0)
    g_post = [g for g in guard if lo2 + 600 <= g["t"] <= force]
    if g_post:
        a, b = g_post[:150], g_post[-150:]
        ma = ca.mean([g.get("cg_as") for g in a]); mb = ca.mean([g.get("cg_as") for g in b])
        res["memory_growth_post"] = {"cg_anon+shmem_first5min_GiB": round(ma, 2), "cg_anon+shmem_last5min_GiB": round(mb, 2),
                                     "growth_GiB": round(mb - ma, 2),
                                     "per_h_GiB": round((mb - ma) / ((g_post[-1]["t"] - g_post[0]["t"]) / 3600), 2),
                                     "peak_cg_anon+shmem_GiB": round(max(g.get("cg_as") or 0 for g in g_post), 2),
                                     "peak_host_anon+shmem_GiB": round(max(g.get("host_as") or 0 for g in g_post), 2),
                                     "peak_cg_current_GiB": round(max(g.get("cg_current") or 0 for g in g_post), 2)}
    g_all = [g for g in guard if up1 <= g["t"] <= force]
    if g_all:
        res["memory_whole"] = {"peak_cg_anon+shmem_GiB": round(max(g.get("cg_as") or 0 for g in g_all), 2),
                               "peak_host_anon+shmem_GiB": round(max(g.get("host_as") or 0 for g in g_all), 2)}
    res["whole"] = {"window_s": [round(up1 - t0), round(force - t0)],
                    "attempts": ca.attempts_stats(recs, sols, t0, up1, force, force)}
    if kill:
        cg = cum_gen(rows)
        res["whole"]["gen_tok_s_incl_outage"] = round(rate(cg, up1, force), 1)
        res["whole"]["generated_tokens"] = round(rate(cg, up1, force) * (force - up1))
    res["traces"] = ca.traces_stats(run)
    res["gpu"] = ca.gpu_stats(run)
    res["gate_log"] = ca.gate_log(run)[0]
    return res


def md(r):
    th, mr, ja = r.get("throughput", {}), r.get("memory_restart", {}), r.get("jobs_at_kill", {})
    c = mr.get("container_cgroup_anon+shmem_GiB", {})
    o = [f"# {r['run']}", "",
         f"- kill at +{r.get('kill_s')} s; server up at {r.get('server_up_s')} s; deaths {r.get('server_died')}",
         f"- memory (container anon+shmem): steady {c.get('steady')}, restart {c.get('restart')}, peak/steady {c.get('peak_over_steady_median')}",
         f"- timeline after kill: {mr.get('timeline_s_after_kill')}",
         f"- throughput: {({k: v for k, v in th.items() if k != 'series_5min_rel_kill'})}",
         f"- jobs at kill: {ja}",
         f"- re-queue lines {r.get('requeue_conn_lines')} {r.get('requeue_conn_kinds')}; retries {len(r.get('retry_lines_using_job_tries', []))}",
         f"- post: {r.get('post', {}).get('server')}",
         f"- post attempts: {r.get('post', {}).get('attempts')}",
         f"- in-flight ctx whole/pre/post: {r.get('inflight_ctx_whole')} / {r.get('inflight_ctx_pre')} / {r.get('inflight_ctx_post')}",
         f"- host ratio post warm: {r.get('host_ratio_post_warm')}",
         f"- memory growth post: {r.get('memory_growth_post')}",
         f"- local score: {r.get('local_score')}; final status {r.get('final_status')}"]
    return "\n".join(o) + "\n"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run")
    p.add_argument("--out")
    p.add_argument("--md")
    a = p.parse_args()
    r = analyse(a.run)
    if a.out:
        Path(a.out).write_text(json.dumps(r, indent=1, default=str))
    if a.md:
        Path(a.md).write_text(md(r))
    print(md(r))


if __name__ == "__main__":
    main()
