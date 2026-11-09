#!/usr/bin/env python3
"""Dashboard files for a pod 4 run (read-only on the pod; stdout). Same formats as pod 3's b32m_v3/data.

  dash.py <run> gpumem   the server logs' "Capped the PyTorch allocator", "GPU memory:" and "Avg generation
                         throughput" lines, start 0 (dsv41_logs/server/server.log), then r1, r2, ... in order
  dash.py <run> ram      utc, kaggle_clock_h, kaggle_eq_gib (the kit sampler's cgroup figure + 7 GiB): 1 point per
                         10 min, every sample of a fast-sampling stretch (server down / fault) plus its neighbours
  dash.py <run> tps      utc, kaggle_h, gen_tok_s, running, waiting, kv_pct: 1 row per minute from the /metrics scrape
                         (0 / 0 / 0 and empty kv while the solver logs the server as down or the scrape is missing)
  dash.py <run> status   status.json: progress, accuracy vs the x2t solutions, errors, deaths / restarts, VRAM margins
                         (cap - reserved / cap - peak, worst rank, current server), retries / OOMs, Kaggle-eq RAM,
                         serving numbers, faults, verdict (argv[3] or automatic)
Run: /root/runs/<run>; working dir: /root/kwork_<run>. Kaggle clock zero: the solver's t0 (the notebook start)."""
import calendar, glob, json, os, re, subprocess, sys, time

RUN = sys.argv[1]; WHAT = sys.argv[2]
R = f"/root/runs/{RUN}"; W = f"/root/kwork_{RUN}"; L = f"{W}/dsv41_logs"
SOL = "/root/data/evalx2t_s2026_solutions.json"


def samples():
    out = []
    try:
        for l in open(f"{R}/mon.jsonl"):
            try:
                r = json.loads(l)
            except ValueError:
                continue
            if r.get("kind") == "sample":
                out.append(r)
    except OSError:
        pass
    return out


def t0_of(S):
    for r in reversed(S):
        t = (r.get("solver") or {}).get("t0")
        if t:
            return t
    try:
        for l in open(f"{R}/nb.timeline"):
            m = re.match(r"(\d\d):(\d\d):(\d\d) NOTEBOOK_START", l)
            if m:
                t = time.time(); d = t - t % 86400 + int(m[1]) * 3600 + int(m[2]) * 60 + int(m[3])
                return d - 86400 if d > t + 300 else d
    except OSError:
        pass
    return S[0]["t"] if S else time.time()


def server_logs():
    def key(p):
        m = re.search(r"/r(\d+)/server/", p)
        return int(m[1]) if m else 0
    return sorted(glob.glob(f"{L}/server/server.log") + glob.glob(f"{L}/r*/server/server.log"), key=key)


def down_windows(t0):
    """[(died, up)] from the solver log; an open window ends now."""
    out, died = [], None
    try:
        for line in open(f"{L}/solver.log", errors="replace"):
            m = re.match(r"\[(\d\d):(\d\d):(\d\d) \+([\d.]+)h\]", line)
            if not m:
                continue
            ref = t0 + float(m[4]) * 3600
            t = ref - ref % 86400 + int(m[1]) * 3600 + int(m[2]) * 60 + int(m[3])
            t = min((t + k * 86400 for k in (-1, 0, 1)), key=lambda x: abs(x - ref))
            if "server died" in line and died is None:
                died = t
            elif "server up after" in line and died is not None:
                out.append((died, t)); died = None
    except OSError:
        pass
    return out, died


iso = lambda t: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t))

if WHAT == "gpumem":
    pat = re.compile(r"GPU memory:|Capped the PyTorch allocator|Avg generation throughput")
    for f in server_logs():
        for line in open(f, errors="replace"):
            if pat.search(line):
                sys.stdout.write(line)

elif WHAT == "ram":
    S = samples(); t0 = t0_of(S)
    keep = set()
    for i, r in enumerate(S):
        if r.get("fast"):
            keep.update({max(0, i - 1), i, min(len(S) - 1, i + 1)})
    seen = set()
    for i, r in enumerate(S):
        k = int(r["t"] // 600)
        if k not in seen:
            seen.add(k); keep.add(i)
    if S:
        keep.add(len(S) - 1)
    print("utc\tkaggle_clock_h\tkaggle_eq_gib")
    for i in sorted(keep):
        r = S[i]
        if r.get("kaggle_eq_gib") is not None:
            print(f"{iso(r['t'])}\t{(r['t'] - t0) / 3600:.3f}\t{r['kaggle_eq_gib']:.2f}")

elif WHAT == "tps":
    S0 = samples(); t0 = t0_of(S0)
    sc, cur = [], None
    try:
        for line in open(f"{R}/metrics_scrape.txt", errors="replace"):
            line = line.strip()
            if line.startswith("T "):
                cur = {"t": int(line.split()[1])}; sc.append(cur)
            elif cur is not None and line.startswith("vllm:"):
                k, v = line.rsplit(" ", 1)
                try:
                    cur[k[5:]] = cur.get(k[5:], 0.0) + float(v)
                except ValueError:
                    pass
    except OSError:
        pass
    sc = [s for s in sc if "generation_tokens_total" in s]
    wins, open_died = down_windows(t0)
    isdown = lambda t: any(a <= t <= b for a, b in wins) or (open_died is not None and t >= open_died)
    rows = {}
    for i in range(1, len(sc)):
        p, s = sc[i - 1], sc[i]; dt = s["t"] - p["t"]
        for m in range(p["t"] // 60 + 1, s["t"] // 60 + 1):
            if dt > 180 or isdown(m * 60):
                rows[m] = None
            else:
                d = s["generation_tokens_total"] - p["generation_tokens_total"]
                rows[m] = ((d if d >= 0 else s["generation_tokens_total"]) / dt, s)
    print("utc\tkaggle_h\tgen_tok_s\trunning\twaiting\tkv_pct")
    if sc:
        last = sc[-1]["t"] // 60
        if open_died is not None:
            last = max(last, int(time.time()) // 60 - 1)
        for m in range(sc[0]["t"] // 60 + 1, last + 1):
            t = m * 60; v = rows.get(m)
            head = f"{iso(t)}\t{(t - t0) / 3600:.3f}"
            if v is None:
                print(f"{head}\t0\t0\t0\t")
            else:
                r_, s = v
                print(f"{head}\t{r_:.1f}\t{int(s.get('num_requests_running', 0))}\t{int(s.get('num_requests_waiting', 0))}\t"
                      f"{100 * s.get('kv_cache_usage_perc', 0):.1f}")

elif WHAT == "status":
    S = samples(); t0 = t0_of(S); last = S[-1] if S else {}
    sol = json.load(open(SOL))
    g = lambda x: json.loads(x) if isinstance(x, str) else x
    done = corr = recs = errrec = 0
    try:
        for l in open(f"{L}/attempts.jsonl"):
            r = json.loads(l); recs += 1
            if r.get("error"):
                errrec += 1
            if r.get("attempt") != 0 or r.get("pair", 0) != 0 or r.get("error"):
                continue
            t = sol[r["task_id"]]; pr = g(r.get("pred")); done += 1
            corr += (isinstance(pr, list) and len(pr) == len(t) and all(pr[i] == t[i] for i in range(len(t)))) \
                if r.get("multi_test") else pr == t[0]
    except OSError:
        pass
    try:
        st = json.load(open(f"{L}/status.json"))
    except (OSError, ValueError):
        st = {}
    # cumulative over every server log: max allocator retries, OOM lines, engine-dead lines
    retries = ooms = 0
    for f in server_logs():
        out = subprocess.run(["grep", "-aE", "GPU memory:|OutOfMemoryError|CUDA out of memory", f],
                             capture_output=True, text=True).stdout
        for line in out.splitlines():
            m = re.search(r"(\d+) allocator retries, (\d+) OOMs", line)
            if m:
                retries = max(retries, int(m[1])); ooms = max(ooms, int(m[2]))
            elif "OutOfMemoryError" in line or "CUDA out of memory" in line:
                ooms = max(ooms, 1)
    sv = last.get("solver") or {}; cnt = sv.get("counts") or {}; met = sv.get("metrics") or {}
    vw = ((last.get("vllm") or {}).get("worst")) or {}
    sup = last.get("supervisor") or {}
    faults = []
    try:
        for l in open(f"{R}/fault.jsonl"):
            d = json.loads(l)
            if d.get("phase") in ("signaled", "started", "allocated") or (
                    not d.get("dry_run") and d.get("action") in ("gpu-oom-burst", "cpu-stress") and "phase" not in d):
                faults.append({k: d.get(k) for k in ("action", "utc", "pid", "target", "signal") if k in d})
    except (OSError, ValueError):
        pass
    now = last.get("t") or time.time()
    out = {"utc": iso(now), "kaggle_h": round((now - t0) / 3600, 2), "first_done": done, "first_correct": corr,
           "pairs_answered": st.get("pairs_with_answer"), "attempt_records": recs, "errored": errrec,
           "finalized_errors": cnt.get("stop_error", 0), "retries_spent": cnt.get("retry", 0),
           "requeues": cnt.get("requeue", 0), "timeouts": cnt.get("timeout", 0),
           "deaths": cnt.get("server_died", 0), "restarts": sup.get("start"), "supervisor": sup.get("state"),
           "retries_max": retries, "ooms": ooms,
           "cap_minus_reserved_min": vw.get("cap_minus_reserved"), "cap_minus_peak_min": vw.get("cap_minus_peak"),
           "kaggle_eq_gib": last.get("kaggle_eq_gib"), "engines": last.get("engines"),
           "gen_tok_s": met.get("gen_tok_s"), "running": met.get("running"), "waiting": met.get("waiting"),
           "kv": met.get("kv"), "faults": faults}
    if len(sys.argv) > 3 and sys.argv[3]:
        out["verdict"] = sys.argv[3]
    else:
        bad = []
        if out["finalized_errors"]:
            bad.append(f"{out['finalized_errors']} attempt(s) finalized as error")
        if ooms:
            bad.append(f"{ooms} OOM(s)")
        if retries:
            bad.append(f"{retries} allocator retries")
        unplanned = out["deaths"] - len(faults)
        if unplanned > 0:
            bad.append(f"{unplanned} unplanned server death(s)")
        out["verdict"] = ("WATCH: " + "; ".join(bad)) if bad else (
            f"stable; cap-reserved min {vw.get('cap_minus_reserved')} GiB, Kaggle-eq {last.get('kaggle_eq_gib')} GiB"
            + (f"; {len(faults)} fault(s) injected, {out['deaths']} death(s)" if faults else ""))
    print(json.dumps(out))
