#!/usr/bin/env python3
"""E057b: per-arm stress metrics for the B+ worker-count test (64 vs 56 workers), from the copied-back run dirs.

    .venv/bin/python experiments/E057b_workers64_stress/scripts/compare_arms.py RUN_A RUN_B [--out F.json] [--md F.md]

Each run dir is /root/runs/<name> from run_arm.sh: run.log, env.txt, metrics10.jsonl (10-s /metrics + status.json +
host memory), memdetail.log, gpu.csv, out/dsv41_logs/{solver.log, attempts.jsonl, metrics.jsonl, server.log},
out/dsv41_traces.tar.gz, out/submission.json; plus arm_t0 (epoch of the podrun.sh launch = local_run's T0).

Windows (solver clock, T0 = local_run start): "full" = server up -> force deadline (ARC_FORCE_AT_S); "last40" = the
40 min before the force deadline. Counters are differenced between the samples nearest the window ends.
"""
import argparse
import bisect
import collections
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments/E055_kaggle_replica/toolkit"))
from replica_report import read_traces  # noqa: E402  (per-turn usage from the trace archive)

FULL_PREFILL_MIN, FULL_PREFILL_CACHED = 50_000, 640  # as replica_report.py
LAST_S = 2400
BIG = 10 ** 9  # the solver adds this to a cut stream's completion tokens


def jsonl(p):
    out = []
    if Path(p).exists():
        for line in Path(p).read_text(errors="replace").splitlines():
            try:
                out.append(json.loads(line))
            except ValueError:
                pass
    return out


def env_of(run):
    env = {}
    for line in (run / "env.txt").read_text().splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            env[k] = v
    return env


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def hist_quantile(b0, b1, q):
    """Upper bucket bound holding quantile q of the observations between two cumulative-bucket snapshots."""
    if not b0 and not b1:
        return None
    les = sorted(b1, key=lambda s: math.inf if s == "+Inf" else float(s))
    d = [(le, b1.get(le, 0.0) - (b0 or {}).get(le, 0.0)) for le in les]
    tot = d[-1][1] if d else 0
    if tot <= 0:
        return None
    for le, c in d:
        if c >= q * tot:
            return math.inf if le == "+Inf" else float(le)
    return None


def window(rows, t_lo, t_hi):
    """Samples with server metrics inside [t_lo, t_hi], plus the counter deltas between the first and last."""
    ws = [r for r in rows if "m" in r and t_lo <= r["t"] <= t_hi]
    if len(ws) < 2:
        return ws, None
    a, b = ws[0]["m"], ws[-1]["m"]
    d = {k: b.get(k, 0.0) - a.get(k, 0.0) for k in b if not k.startswith("_") and isinstance(b[k], (int, float))}
    d["_dt"] = ws[-1]["t"] - ws[0]["t"]
    d["_b0"], d["_b1"] = a.get("_buckets", {}), b.get("_buckets", {})
    return ws, d


def server_stats(rows, t_lo, t_hi, workers, stop_new_abs):
    ws, d = window(rows, t_lo, t_hi)
    if not d:
        return {}
    dt = d["_dt"]
    g = lambda k: d.get(k, 0.0)
    kv = [r["m"].get("vllm:kv_cache_usage_perc") for r in ws]
    kv = [x for x in kv if x is not None]
    wait = [r["m"].get("vllm:num_requests_waiting", 0.0) for r in ws]
    run = [r["m"].get("vllm:num_requests_running", 0.0) for r in ws]
    q = g("vllm:prefix_cache_queries_total")
    hits = g("vllm:prefix_cache_hits_total")
    ext = g("vllm:external_prefix_cache_hits_total")
    extq = g("vllm:external_prefix_cache_queries_total")
    ttft_n, ttft_s = g("vllm:time_to_first_token_seconds_count"), g("vllm:time_to_first_token_seconds_sum")
    qt_n, qt_s = g("vllm:request_queue_time_seconds_count"), g("vllm:request_queue_time_seconds_sum")
    b0, b1 = d["_b0"], d["_b1"]
    tt0, tt1 = b0.get("vllm:time_to_first_token_seconds"), b1.get("vllm:time_to_first_token_seconds")
    # solver side: attempts running vs the worker cap while work was queued (only before stop-new)
    st = [r["st"] for r in ws if r.get("st") and r["t"] < stop_new_abs]
    held = [s for s in st if (s.get("queued") or 0) > 0 and (s.get("running") or 0) < workers]
    out = {
        "window_min": round(dt / 60, 1),
        "gen_tok_s": round(g("vllm:generation_tokens_total") / dt, 1),
        "prompt_tok_s": round(g("vllm:prompt_tokens_total") / dt, 1),
        "kv_mean": round(mean(kv), 3) if kv else None, "kv_peak": round(max(kv), 3) if kv else None,
        "kv_share_gt_0.90": round(sum(x > 0.90 for x in kv) / len(kv), 3) if kv else None,
        "kv_share_gt_0.95": round(sum(x > 0.95 for x in kv) / len(kv), 3) if kv else None,
        "preemptions": int(g("vllm:num_preemptions_total")),
        "preemptions_per_h": round(g("vllm:num_preemptions_total") / dt * 3600, 1),
        "srv_running_mean": round(mean(run), 1), "srv_running_max": int(max(run)),
        "srv_waiting_mean": round(mean(wait), 2), "srv_waiting_max": int(max(wait)),
        "srv_waiting_share_gt0": round(sum(x > 0 for x in wait) / len(wait), 3),
        "prefix_queries_tok_s": round(q / dt, 1),
        "gpu_hit_share": round(hits / q, 3) if q else None,
        "host_restored_tokens": int(ext), "host_restored_tok_s": round(ext / dt, 1),
        "host_hit_share": round(ext / q, 3) if q else None,
        "host_queries": int(extq),
        "uncached_tok_s": round((q - hits - ext) / dt, 1),
        "true_uncached_share": round((q - hits - ext) / q, 3) if q else None,
        "offload_store_GiB": round(g("vllm:kv_offload_store_bytes_total") / 2 ** 30, 1) if "vllm:kv_offload_store_bytes_total" in d else None,
        "offload_load_GiB": round(g("vllm:kv_offload_load_bytes_total") / 2 ** 30, 1) if "vllm:kv_offload_load_bytes_total" in d else None,
        "requests_finished": int(sum(v for k, v in d.items() if k.startswith("vllm:request_success_total"))),
        "ttft_mean_s": round(ttft_s / ttft_n, 2) if ttft_n else None,
        "ttft_p50_s_le": hist_quantile(tt0, tt1, 0.5), "ttft_p90_s_le": hist_quantile(tt0, tt1, 0.9),
        "ttft_p99_s_le": hist_quantile(tt0, tt1, 0.99),
        "queue_time_mean_s": round(qt_s / qt_n, 2) if qt_n else None,
        "solver_samples_before_stop_new": len(st),
        "solver_running_mean": round(mean([s.get("running") for s in st]), 1) if st else None,
        "solver_held_share": round(len(held) / len(st), 3) if st else None,
    }
    return out


def attempts_stats(recs, sol, t0, t_lo, t_hi, force_abs):
    """Finished attempts (last record per job; requeued error records dropped as the solver does) ending in the window."""
    fin = []
    for r in recs:
        if "task_id" not in r:
            continue
        end = t0 + r.get("t_end_h", 0) * 3600
        if not (t_lo <= end <= t_hi + 60):
            continue
        fin.append(r)
    nat = [r for r in fin if not r.get("cuts") and not r.get("error")]
    cut = [r for r in fin if r.get("cuts")]
    err = [r for r in fin if r.get("error")]
    dt_h = (min(t_hi, force_abs) - t_lo) / 3600
    npairs = lambda r: int(r.get("multi_test") or 1)
    in_win = [r for r in nat if t0 + r.get("t_end_h", 0) * 3600 <= min(t_hi, force_abs)]
    out = {"finished": len(fin), "natural": len(in_win), "cut_at_force": len(cut), "errors": len(err),
           "attempts_per_h": round(len(in_win) / dt_h, 1) if dt_h > 0 else None,
           "pairs_per_h": round(sum(npairs(r) for r in in_win) / dt_h, 1) if dt_h > 0 else None,
           "no_answer": sum(1 for r in fin if not r.get("pred")),
           "no_answer_natural": sum(1 for r in in_win if not r.get("pred")),
           "completion_tokens_natural": sum((r.get("usage") or {}).get("completion", 0) for r in in_win),
           "mean_wall_s_natural": round(mean([r.get("wall_s") for r in in_win]), 0) if in_win else None,
           "per_stream_tok_s_median_natural": (sorted((r.get("usage") or {}).get("completion", 0) / r["wall_s"]
                                                      for r in in_win if r.get("wall_s"))[len(in_win) // 2]
                                               if in_win else None),
           "stop_reasons": dict(collections.Counter(r.get("stop_reason") for r in fin).most_common())}
    # accuracy over answered pairs (per attempt; a multi-test attempt answers its pairs together)
    ans = ok = 0
    for r in fin:
        pred = r.get("pred")
        if not pred:
            continue
        t, n = r["task_id"], npairs(r)
        preds = pred if (r.get("multi_test") and isinstance(pred, list) and len(pred) == n) else [pred]
        pairs = range(n) if r.get("multi_test") else [r.get("pair", 0)]
        for p, g in zip(pairs, preds):
            if g is None:
                continue
            ans += 1
            ok += int(sol.get(t) is not None and p < len(sol[t]) and sol[t][p] == g)
    out["answered_pairs"] = ans
    out["answered_pair_acc"] = round(ok / ans, 3) if ans else None
    return out


def traces_stats(run):
    tp = run / "out/dsv41_traces.tar.gz"
    tr, bad = read_traces(tp if tp.exists() else (run / "out/dsv41_logs/traces"))
    turns = [x for v in tr.values() for x in v["turns"]]
    later = [x for x in turns if (x["turn"] or 0) > 1]
    full = [x for x in later if x["prompt"] >= FULL_PREFILL_MIN and x["cached"] <= FULL_PREFILL_CACHED]
    big = [x for x in later if x["prompt"] >= FULL_PREFILL_MIN]
    re_tok = sum(max(0, x["prompt"] - x["cached"]) for x in later)
    return {"trace_files": len(tr), "unreadable": bad, "turns": len(turns), "turns_after_first": len(later),
            "turns_ge50k": len(big), "full_reprefills": len(full),
            "full_reprefill_share_of_ge50k": round(len(full) / len(big), 3) if big else None,
            "uncached_prompt_tokens_after_first_turn": re_tok,
            "uncached_share_after_first_turn": round(re_tok / sum(x["prompt"] for x in later), 3) if later else None}


def mem_stats(rows, t_lo, t_hi):
    ws = [r["mem"] for r in rows if r.get("mem") and t_lo <= r["t"] <= t_hi]
    if not ws:
        return {}
    mx = lambda k: max((m.get(k) or 0) for m in ws)
    mn = lambda k: min((m.get(k) if m.get(k) is not None else math.inf) for m in ws)
    return {"cgroup_peak_GiB": mx("cgroup_GiB"), "shmem_peak_GiB": mx("Shmem_GiB"), "anon_peak_GiB": mx("AnonPages_GiB"),
            "anon_plus_shmem_peak_GiB": round(max((m.get("Shmem_GiB") or 0) + (m.get("AnonPages_GiB") or 0) for m in ws), 2),
            "memavailable_min_GiB": mn("MemAvailable_GiB")}


def gpu_stats(run):
    p = run / "gpu.csv"
    per = collections.defaultdict(list)
    if p.exists():
        for line in p.read_text().splitlines():
            f = [x.strip() for x in line.split(",")]
            if len(f) >= 6 and f[1].isdigit():
                try:
                    per[int(f[1])].append((float(f[2]), float(f[3]), float(f[4])))
                except ValueError:
                    pass
    return {f"gpu{i}": {"mem_peak_MiB": int(max(x[0] for x in v)), "util_mean": round(mean([x[1] for x in v]), 1),
                        "power_mean_W": round(mean([x[2] for x in v]), 1)} for i, v in sorted(per.items())}


def gate_log(run):
    s = (run / "out/dsv41_logs/solver.log")
    lines = s.read_text(errors="replace").splitlines() if s.exists() else []
    c = collections.Counter()
    for l in lines:
        if "KV gate: usage" in l:
            c["kv_usage_hold_lines"] += 1
        elif "KV gate: server has" in l:
            c["queue_hold_lines"] += 1
        elif "admission closed" in l:
            c["admission_closed"] += 1
        elif "server died" in l or "server exited" in l:
            c["server_deaths"] += 1
        elif "re-queue" in l:
            c["requeues"] += 1
    up = next((l for l in lines if "server up after" in l), "")
    score = [l for l in lines if "local score" in l.lower()]
    return dict(c), up.split("] ", 1)[-1][:160], (score[-1].split("] ", 1)[-1][:200] if score else None)


def trace_turns(run):
    """trace file name -> list of (prompt_tokens, total_tokens) per model turn."""
    import gzip
    import tarfile
    out = {}
    tp = run / "out/dsv41_traces.tar.gz"

    def one(raw):
        d = json.loads(gzip.decompress(raw))
        out = []
        for e in d.get("events", []):
            if "usage" not in e:
                continue
            u = e.get("usage") or {}
            pr, co = u.get("prompt_tokens") or 0, u.get("completion_tokens") or 0
            co = co % BIG  # a stream cut at the force deadline reports BIG (10^9) extra completion tokens
            out.append((pr, pr + co))
        return out
    if tp.exists():
        with tarfile.open(tp) as tf:
            for m in tf:
                if m.isfile() and m.name.endswith(".json.gz"):
                    try:
                        out[Path(m.name).name] = one(tf.extractfile(m).read())
                    except Exception:
                        pass
    else:
        for f in sorted((run / "out/dsv41_logs/traces").glob("*.json.gz")):
            try:
                out[f.name] = one(f.read_bytes())
            except Exception:
                pass
    return out


def inflight_series(run, recs, t0, t_lo, t_hi, step=10):
    """In-flight total context (tokens) over time: each attempt's turns placed evenly over [end - wall_s, end]; point 0
    is the first prompt, point k the k-th turn's total_tokens; linear in between (coordinator's definition)."""
    turns = trace_turns(run)
    grid = [t_lo + i * step for i in range(int((t_hi - t_lo) / step) + 1)]
    tot = [0.0] * len(grid)
    used = 0
    for r in recs:
        tr = turns.get(r.get("trace_file") or "")
        if not tr or r.get("wall_s") is None or r.get("t_end_h") is None:
            continue
        end = t0 + r["t_end_h"] * 3600
        start = end - r["wall_s"]
        pts_v = [tr[0][0]] + [x[1] for x in tr]
        n = len(pts_v) - 1
        pts_t = [start + (end - start) * k / n for k in range(n + 1)]
        used += 1
        lo = bisect.bisect_left(grid, start)
        hi = bisect.bisect_right(grid, end)
        for i in range(lo, hi):
            t = grid[i]
            k = min(n - 1, max(0, bisect.bisect_right(pts_t, t) - 1))
            t1, t2 = pts_t[k], pts_t[k + 1]
            f = (t - t1) / (t2 - t1) if t2 > t1 else 1.0
            tot[i] += pts_v[k] + f * (pts_v[k + 1] - pts_v[k])
    return grid, tot, used


def host_ratio_windows(rows, t_lo, t_hi, width=180):
    """Host-hit ratio r = d external_prefix_cache_hits / d external_prefix_cache_queries over consecutive 180-s windows
    from t_lo; a window counts when num_requests_waiting > 0 in at least half of its samples and it has queries."""
    ms = [r for r in rows if "m" in r and t_lo <= r["t"] <= t_hi]
    out = []
    w0 = t_lo
    while w0 + width <= t_hi:
        ws = [r for r in ms if w0 <= r["t"] <= w0 + width]
        if len(ws) >= 3:
            a, b = ws[0]["m"], ws[-1]["m"]
            dq = b.get("vllm:external_prefix_cache_queries_total", 0) - a.get("vllm:external_prefix_cache_queries_total", 0)
            dh = b.get("vllm:external_prefix_cache_hits_total", 0) - a.get("vllm:external_prefix_cache_hits_total", 0)
            waitf = sum(r["m"].get("vllm:num_requests_waiting", 0) > 0 for r in ws) / len(ws)
            out.append({"t0": w0, "r": (dh / dq) if dq > 0 else None, "dq": dq, "wait_frac": round(waitf, 2),
                        "qual": waitf >= 0.5 and dq > 0,
                        "gen_tok_s": (b.get("vllm:generation_tokens_total", 0) - a.get("vllm:generation_tokens_total", 0)) / (ws[-1]["t"] - ws[0]["t"])})
        w0 += width
    return out


def cliff_stats(wins, grid, tot, t0=0.0, width=180):
    q = [w for w in wins if w["qual"]]
    rs = sorted(w["r"] for w in q)
    med = rs[len(rs) // 2] if rs else None
    low_min, runs, cur = 0.0, [], []
    for w in wins:
        if w["qual"] and w["r"] < 0.5:
            cur.append(w)
        else:
            if cur:
                runs.append(cur)
            cur = []
    if cur:
        runs.append(cur)
    sustained = [r for r in runs if len(r) * width >= 360]
    low_min = sum(len(r) * width for r in sustained) / 60

    def ctx_at(t):
        i = bisect.bisect_left(grid, t)
        return tot[i] if 0 <= i < len(tot) else None
    # the same over every window with host queries, whether or not the server queued (waiting > 0 is rare under the gate)
    allq = [w for w in wins if w["r"] is not None]
    ra = sorted(w["r"] for w in allq)
    runs_a, cur = [], []
    for w in wins:
        if w["r"] is not None and w["r"] < 0.5:
            cur.append(w)
        else:
            if cur:
                runs_a.append(cur)
            cur = []
    if cur:
        runs_a.append(cur)
    low_all = sum(len(r) * width for r in runs_a if len(r) * width >= 360) / 60
    ctx_low = [ctx_at(w["t0"] + width / 2) for w in q if w["r"] < 0.5]
    ctx_ok = [ctx_at(w["t0"] + width / 2) for w in q if w["r"] >= 0.5]
    return {"windows": len(wins), "qualifying_windows": len(q), "r_median_qual": round(med, 3) if med is not None else None,
            "qual_windows_r_lt_0.5": sum(1 for w in q if w["r"] < 0.5),
            "minutes_r_lt_0.5_sustained_ge_360s": round(low_min, 1),
            "sustained_low_runs_start_s_len_s": [(round(r[0]["t0"] - t0), len(r) * width) for r in sustained],
            "all_windows_with_queries": len(allq), "r_median_all": round(ra[len(ra) // 2], 3) if ra else None,
            "r_p10_all": round(ra[len(ra) // 10], 3) if ra else None,
            "minutes_r_lt_0.5_sustained_ge_360s_all": round(low_all, 1),
            "inflight_ctx_mean_when_r_lt_0.5": round(mean([c for c in ctx_low if c is not None])) if ctx_low else None,
            "inflight_ctx_mean_when_r_ge_0.5": round(mean([c for c in ctx_ok if c is not None])) if ctx_ok else None}


def admission_closed_abs(run, t0):
    """Epoch of the solver's "admission closed" line (UTC clock on T0's day), or None."""
    import datetime as dt
    s = run / "out/dsv41_logs/solver.log"
    for line in (s.read_text(errors="replace").splitlines() if s.exists() else []):
        if "admission closed" in line:
            hh, mm, ss = map(int, line[1:9].split(":"))
            day = dt.datetime.fromtimestamp(t0, dt.timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
            t = day.timestamp() + hh * 3600 + mm * 60 + ss
            return t + 86400 if t < t0 - 60 else t
    return None


def analyse(run):
    run = Path(run)
    env = env_of(run)
    W = int(env["ARC_WORKERS"])
    t0 = float((run / "arm_t0").read_text().strip())
    stop_new, force = t0 + float(env["ARC_STOP_NEW_S"]), t0 + float(env["ARC_FORCE_AT_S"])
    rows = jsonl(run / "metrics10.jsonl")
    up = next((r["t"] for r in rows if "m" in r), None)
    recs = jsonl(run / "out/dsv41_logs/attempts.jsonl")
    sols = json.loads(Path(env["ARC_SOLUTIONS"].replace("/root/data", str(ROOT / "experiments/E055_kaggle_replica/toolkit/workloads"))).read_text())
    gl, upline, score = gate_log(run)
    res = {"run": run.name, "workers": W, "server_up_s": round(up - t0) if up else None, "server_up_line": upline,
           "local_score_line": score, "gate_log": gl, "gpu": gpu_stats(run), "traces": traces_stats(run)}
    grid, tot, used = inflight_series(run, recs, t0, t0, force + 300)
    res["inflight_attempts_with_trace"] = used
    res["inflight_series_10s"] = [(round(g - t0), round(v)) for g, v in zip(grid, tot)]
    wins = host_ratio_windows(rows, up, force) if up else []
    res["host_ratio_windows"] = [{**w, "t0": round(w["t0"] - t0), "r": None if w["r"] is None else round(w["r"], 3),
                                  "gen_tok_s": round(w["gen_tok_s"], 1)} for w in wins]
    adm = admission_closed_abs(run, t0)
    res["admission_closed_s"] = round(adm - t0) if adm else None
    open_end = min(stop_new, adm) if adm else stop_new
    for name, lo, hi in (("full", up, force), ("last40", force - LAST_S, force), ("open", up, open_end)):
        if lo is None:
            continue
        sel = [v for g, v in zip(grid, tot) if lo <= g <= hi]
        res[name] = {"window_s": [round(lo - t0), round(hi - t0)],
                     "server": server_stats(rows, lo, hi, W, stop_new),
                     "attempts": attempts_stats(recs, sols, t0, lo, hi, hi),
                     "memory": mem_stats(rows, lo, hi),
                     "inflight_ctx": {"mean": round(mean(sel)) if sel else None, "peak": round(max(sel)) if sel else None,
                                      "min_above_4.5M": round(sum(v > 4.5e6 for v in sel) * 10 / 60, 1),
                                      "min_above_4.8M": round(sum(v > 4.8e6 for v in sel) * 10 / 60, 1)},
                     "host_ratio": cliff_stats([w for w in wins if lo <= w["t0"] and w["t0"] + 180 <= hi], grid, tot, t0),
                     # without the cold start (first 10 min after server up: first prompts have nothing in host yet)
                     "host_ratio_warm": cliff_stats([w for w in wins if max(lo, up + 600) <= w["t0"] and w["t0"] + 180 <= hi],
                                                    grid, tot, t0)}
    res["memory_whole_run"] = mem_stats(rows, t0, t0 + 10 ** 6)
    try:
        st = json.loads((run / "out/dsv41_logs/status.json").read_text())
        res["final_status"] = {k: st.get(k) for k in ("done", "errors", "pairs_with_answer", "pairs", "restarts")}
    except Exception:
        res["final_status"] = None
    jobs_started = len({(r.get("task_id"), r.get("pair"), r.get("attempt")) for r in recs if "task_id" in r})
    res["jobs_with_record"] = jobs_started
    return res


def md_table(a, b):
    keys = [("server", "gen_tok_s"), ("attempts", "attempts_per_h"), ("attempts", "pairs_per_h"),
            ("attempts", "natural"), ("attempts", "cut_at_force"), ("attempts", "no_answer"),
            ("attempts", "completion_tokens_natural"), ("attempts", "per_stream_tok_s_median_natural"),
            ("attempts", "mean_wall_s_natural"),
            ("attempts", "answered_pairs"), ("attempts", "answered_pair_acc"),
            ("server", "kv_mean"), ("server", "kv_peak"), ("server", "kv_share_gt_0.90"), ("server", "kv_share_gt_0.95"),
            ("server", "preemptions"), ("server", "preemptions_per_h"),
            ("server", "srv_running_mean"), ("server", "srv_waiting_mean"), ("server", "srv_waiting_max"),
            ("server", "srv_waiting_share_gt0"), ("server", "solver_running_mean"), ("server", "solver_held_share"),
            ("server", "gpu_hit_share"), ("server", "host_restored_tokens"), ("server", "host_restored_tok_s"),
            ("server", "host_hit_share"), ("server", "uncached_tok_s"), ("server", "true_uncached_share"),
            ("server", "ttft_mean_s"), ("server", "ttft_p50_s_le"), ("server", "ttft_p90_s_le"),
            ("server", "queue_time_mean_s"), ("memory", "cgroup_peak_GiB"), ("memory", "anon_peak_GiB"),
            ("memory", "shmem_peak_GiB"), ("memory", "anon_plus_shmem_peak_GiB"),
            ("inflight_ctx", "mean"), ("inflight_ctx", "peak"), ("inflight_ctx", "min_above_4.5M"),
            ("host_ratio", "qualifying_windows"), ("host_ratio", "r_median_qual"), ("host_ratio", "qual_windows_r_lt_0.5"),
            ("host_ratio", "minutes_r_lt_0.5_sustained_ge_360s"), ("host_ratio_warm", "qualifying_windows"),
            ("host_ratio_warm", "r_median_qual"), ("host_ratio_warm", "minutes_r_lt_0.5_sustained_ge_360s"),
            ("host_ratio_warm", "all_windows_with_queries"), ("host_ratio_warm", "r_median_all"),
            ("host_ratio_warm", "r_p10_all"), ("host_ratio_warm", "minutes_r_lt_0.5_sustained_ge_360s_all")]
    o = [f"| metric | {a['run']} full | {b['run']} full | {a['run']} last 40 min | {b['run']} last 40 min "
         f"| {a['run']} open | {b['run']} open |", "|---|---|---|---|---|---|---|"]
    for sec, k in keys:
        cell = lambda r, w: r.get(w, {}).get(sec, {}).get(k)
        o.append(f"| {sec}.{k} | {cell(a, 'full')} | {cell(b, 'full')} | {cell(a, 'last40')} | {cell(b, 'last40')} "
                 f"| {cell(a, 'open')} | {cell(b, 'open')} |")
    o.append(f"| window_s | {a.get('full', {}).get('window_s')} | {b.get('full', {}).get('window_s')} | "
             f"{a.get('last40', {}).get('window_s')} | {b.get('last40', {}).get('window_s')} | "
             f"{a.get('open', {}).get('window_s')} | {b.get('open', {}).get('window_s')} |")
    return "\n".join(o)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("runs", nargs="+")
    p.add_argument("--out")
    p.add_argument("--md")
    a = p.parse_args()
    res = [analyse(r) for r in a.runs]
    js = json.dumps(res, indent=1, default=str)
    if a.md and len(res) == 1:
        Path(a.md).write_text(md_table(res[0], res[0]) + "\n")
    if a.out:
        Path(a.out).write_text(js)
    if a.md and len(res) == 2:
        Path(a.md).write_text(md_table(*res) + "\n")
    print(js if not a.out else f"wrote {a.out}")
    if len(res) == 2:
        print(md_table(*res))


if __name__ == "__main__":
    main()
