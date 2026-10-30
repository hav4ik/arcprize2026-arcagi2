"""Host-RAM / VRAM sampler for the Manh v12 replica (read-only, every 60 s) -> <out>.jsonl, alerts -> <out>.alerts.
Per sample: /proc/meminfo (MemTotal, MemAvailable, AnonPages, Shmem, Cached, Mlocked, PageTables, AnonHugePages),
Kaggle-equivalent RAM = AnonPages + Shmem + 7 GiB (limit 173.0 GiB, no swap), RSS summed by group (vLLM TP workers,
EngineCore, API server, his supervisor, solver, tool sandboxes = the solver's descendants, notebook kernel, other),
nvidia-smi per GPU, vLLM's latest per-rank "GPU memory:" line, supervisor status, solver/attempt error counts.
Alerts (once per level crossing / per new event): RAM WARN >= 155, ALERT >= 165, CRIT >= 173 GiB Kaggle-eq; VRAM
nvidia-smi free < 1.0 GiB WARN, < 0.5 ALERT; any CUDA OOM line, allocator retry/OOM count > 0, engine death,
supervisor restart, stop=error attempt (exact lines). Usage: v12_sampler.py <work dir> <run dir> <out prefix>"""
import glob, json, os, re, subprocess, sys, time
W, R, OUT = sys.argv[1:4]; L = f"{W}/dsv41_logs"; G = 2**30
ANSI = re.compile(r"\x1b\[[0-9;]*m")
GM = re.compile(r"Worker_TP(\d).*?GPU memory: ([\d.]+) GiB allocated \(peak ([\d.]+) GiB\), ([\d.]+) GiB reserved.*?, ([\d.]+) GiB free, (\d+) allocator retries, (\d+) OOMs")
CAP = re.compile(r"Worker_TP(\d).*?Capped the PyTorch allocator at ([\d.]+) GiB \(([\d.]+) GiB reserved\)")
cap_alerted = set()
CAPS_BY_LOG = {}
MEMK = ("MemTotal", "MemAvailable", "AnonPages", "Shmem", "Cached", "Mlocked", "PageTables", "AnonHugePages")
def alert(level, msg):
    with open(OUT + ".alerts", "a") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())} {level} {msg}\n")
def meminfo():
    d = {}
    for l in open("/proc/meminfo"):
        k = l.split(":")[0]
        if k in MEMK: d[k] = int(l.split()[1]) * 1024
    return d
def procs():
    P = {}
    for p in os.listdir("/proc"):
        if not p.isdigit(): continue
        try:
            st = open(f"/proc/{p}/stat").read(); ppid = int(st.rsplit(")", 1)[1].split()[1])
            cmd = open(f"/proc/{p}/cmdline", "rb").read().replace(b"\0", b" ").decode(errors="replace")
            comm = open(f"/proc/{p}/comm").read().strip()
            rss = next((int(l.split()[1]) * 1024 for l in open(f"/proc/{p}/status") if l.startswith("VmRSS:")), 0)
            P[int(p)] = (ppid, comm, cmd, rss)
        except (OSError, ValueError, StopIteration):
            pass
    return P
def groups(P):
    kids = {}
    for p, (pp, *_ ) in P.items(): kids.setdefault(pp, []).append(p)
    def desc(r):
        out, todo = [], [r]
        while todo:
            x = todo.pop(); out.append(x); todo += kids.get(x, [])
        return out
    g = dict.fromkeys(("tp_workers", "engine_core", "api_server", "supervisor", "solver", "sandboxes", "kernel", "other_kernel_tree"), 0)
    kern = [p for p, v in P.items() if "ipykernel_launcher" in v[2]]
    solv = [p for p, v in P.items() if "solver_dsv41.py" in v[2]]
    sand = set(x for s in solv for x in desc(s) if x != s)
    seen = set()
    for p, (pp, comm, cmd, rss) in P.items():
        if comm.startswith("VLLM::Worker"): k = "tp_workers"
        elif comm.startswith("VLLM::EngineCor"): k = "engine_core"
        elif "vllm.entrypoints" in cmd: k = "api_server"
        elif "dsv41_vllm.py" in cmd: k = "supervisor"
        elif p in solv: k = "solver"
        elif p in sand: k = "sandboxes"
        elif p in kern: k = "kernel"
        else: continue
        g[k] += rss; seen.add(p)
    for k_ in kern:
        g["other_kernel_tree"] += sum(P[x][3] for x in desc(k_) if x not in seen and x in P)
    return g
def nvsmi():
    r = subprocess.run("nvidia-smi --query-gpu=index,memory.used,memory.free,power.draw,clocks.sm,temperature.gpu --format=csv,noheader,nounits",
                       shell=True, capture_output=True, text=True, timeout=60)
    out = []
    for l in r.stdout.strip().splitlines():
        i, u, f, pw, sm, t = [x.strip() for x in l.split(",")]
        out.append({"gpu": int(i), "used_mib": float(u), "free_mib": float(f), "power_w": float(pw), "sm_mhz": float(sm), "temp_c": float(t)})
    return out
def tail(path, n):
    try:
        with open(path, "rb") as f:
            f.seek(max(0, os.path.getsize(path) - n)); return ANSI.sub("", f.read().decode(errors="replace"))
    except OSError:
        return ""
ram_lvl = 0; vram_lvl = {}; last_start = None; seen = {"oom_lines": 0, "dead_lines": 0, "stop_error": 0, "retry": 0}
logpos = {}
EW = {"cap_lt1": set(), "keq150": False, "keq_rate_t": 0, "wait30_t": 0, "gen0_t": 0, "err_att": 0, "t500": 0,
      "srv_err": 0, "sup": None}
hist_cap = []  # (t, worst cap - peak)
hist_keq = []  # (t, kaggle-eq)
ENG = re.compile(r"Avg generation throughput: ([\d.]+) tokens/s, Running: (\d+) reqs, Waiting: (\d+) reqs")
import urllib.request
TPS = {"t": None, "gen": None}
T_NB = os.path.getmtime(f"{R}/notebook.sha256") if os.path.exists(f"{R}/notebook.sha256") else time.time()
def tps_row(t):
    """One /metrics row per sample: gen tok/s from the generation_tokens_total delta; 0/0/0/empty when down."""
    try:
        txt_ = urllib.request.urlopen("http://127.0.0.1:18084/metrics", timeout=5).read().decode()
        val = lambda name: sum(float(l.rsplit(" ", 1)[1]) for l in txt_.splitlines() if l.startswith(name + "{"))
        gen, run_, wait_, kv_ = val("vllm:generation_tokens_total"), val("vllm:num_requests_running"), val("vllm:num_requests_waiting"), val("vllm:kv_cache_usage_perc")
        rate = (gen - TPS["gen"]) / (t - TPS["t"]) if TPS["gen"] is not None and gen >= TPS["gen"] and t > TPS["t"] else 0.0
        TPS.update(t=t, gen=gen)
        row = f"{rate:.1f}\t{int(run_)}\t{int(wait_)}\t{kv_ * 100:.1f}"
    except Exception:
        TPS.update(t=None, gen=None); row = "0\t0\t0\t"
    new = not os.path.exists(OUT + ".tps.tsv")
    with open(OUT + ".tps.tsv", "a") as f:
        if new: f.write("utc\tkaggle_h\tgen_tok_s\trunning\twaiting\tkv_pct\n")
        f.write(f"{time.strftime('%H:%M:%S', time.gmtime(t))}\t{(t - T_NB) / 3600:.3f}\t{row}\n")
def nb_alive():
    try:
        pid = int(open(f"{R}/run_nb.pid").read().strip()); os.kill(pid, 0); return True
    except Exception:
        return False
while True:
    t = time.time(); rec = {"t": round(t, 1), "utc": time.strftime("%H:%M:%S", time.gmtime(t))}
    try:
        m = meminfo(); rec["meminfo_gib"] = {k: round(v / G, 2) for k, v in m.items()}
        # Kaggle-eq from the container's cgroup (v1 memory.stat total_rss + total_shmem; v2 anon + shmem) + 7 GiB: the
        # pod's /proc/meminfo is host-wide (~11 GiB of AnonPages that is not ours, 2026-10-30 04:00). Meminfo form kept.
        cg = {}
        for f in ("/sys/fs/cgroup/memory/memory.stat", "/sys/fs/cgroup/memory.stat"):
            try:
                cg = {l.split()[0]: int(l.split()[1]) for l in open(f)}; break
            except OSError:
                pass
        cg_rss = cg.get("total_rss", cg.get("anon")); cg_sh = cg.get("total_shmem", cg.get("shmem"))
        rec["kaggle_eq_meminfo_gib"] = round((m["AnonPages"] + m["Shmem"]) / G + 7, 2)
        if cg_rss is not None and cg_sh is not None:
            rec["cgroup_gib"] = {"rss": round(cg_rss / G, 2), "shmem": round(cg_sh / G, 2)}
            keq = (cg_rss + cg_sh) / G + 7
        else:
            keq = rec["kaggle_eq_meminfo_gib"]
        rec["kaggle_eq_gib"] = round(keq, 2)
        lvl = 3 if keq >= 173 else 2 if keq >= 165 else 1 if keq >= 155 else 0
        if lvl > ram_lvl:
            alert(["", "WARN", "ALERT", "CRIT"][lvl], f"Kaggle-eq host RAM {keq:.1f} GiB (cgroup rss + shmem + 7; meminfo form {rec['kaggle_eq_meminfo_gib']})" + (" -- would be OOM-killed on Kaggle (173.0 GiB)" if lvl == 3 else ""))
        ram_lvl = max(ram_lvl, lvl)
        rec["rss_gib"] = {k: round(v / G, 2) for k, v in groups(procs()).items()}
        rec["gpus"] = nvsmi()
        for g_ in rec["gpus"]:
            fr = g_["free_mib"] / 1024; lv = 2 if fr < 0.5 else 1 if fr < 1.0 else 0
            if lv > vram_lvl.get(g_["gpu"], 0) and g_["used_mib"] > 4000:
                alert(["", "WARN", "ALERT"][lv], f"GPU {g_['gpu']} nvidia-smi free {fr:.2f} GiB (used {g_['used_mib']:.0f} MiB)")
            if g_["used_mib"] > 4000: vram_lvl[g_["gpu"]] = max(vram_lvl.get(g_["gpu"], 0), lv)
        logs = sorted(glob.glob(f"{L}/**/server.log", recursive=True), key=os.path.getmtime)
        rec["server_log"] = logs[-1] if logs else None
        txt = tail(logs[-1], 3_000_000) if logs else ""
        vr = {}
        for mm in GM.finditer(txt):
            vr[mm.group(1)] = [float(x) for x in mm.groups()[1:]]
        rec["vllm_gpu_mem"] = {f"TP{k}": {"alloc": v[0], "peak": v[1], "reserved": v[2], "free": v[3], "retries": int(v[4]), "ooms": int(v[5])} for k, v in sorted(vr.items())}
        # the startup "Capped the PyTorch allocator" lines sit at the head of the log, out of the 3 MB tail once the
        # log grows: read them from the head, once per server log (cached)
        if logs and logs[-1] not in CAPS_BY_LOG:
            with open(logs[-1], "rb") as fh:
                head = ANSI.sub("", fh.read(20_000_000).decode(errors="replace"))
            cc = {mm.group(1): (float(mm.group(2)), float(mm.group(3))) for mm in CAP.finditer(head)}
            if len(cc) == 4:
                CAPS_BY_LOG[logs[-1]] = cc
        caps = CAPS_BY_LOG.get(logs[-1], {}) if logs else {}
        rec["alloc_cap"] = {}
        for k, (cx, cy) in sorted(caps.items()):
            v = vr.get(k)
            rec["alloc_cap"][f"TP{k}"] = {"cap": cx, "reserved_at_cap": cy, "cap_minus_reserved0": round(cx - cy, 3),
                                         "cap_minus_peak": round(cx - v[1], 3) if v else None,
                                         "cap_minus_reserved_now": round(cx - v[2], 3) if v else None}
            if v:  # cap - RESERVED is the real margin (pod 3 OOM, 2026-10-30): alert at 0.30 / 0.15 / 0.05 GiB
                for lvl_, thr in (("EARLY", 0.30), ("ALERT", 0.15), ("CRIT", 0.05)):
                    if cx - v[2] < thr and (logs[-1], k, thr) not in cap_alerted:
                        alert(lvl_, f"TP{k}: cap - reserved = {cx - v[2]:.2f} GiB < {thr} (cap {cx}, reserved {v[2]}, peak {v[1]}, retries {int(v[4])})")
                        cap_alerted.add((logs[-1], k, thr))
            if v and cx - v[1] < 0.26 and (logs[-1], k) not in cap_alerted:
                alert("ALERT", f"TP{k}: cap - peak allocated = {cx - v[1]:.2f} GiB < 0.26 (cap {cx}, peak {v[1]}, reserved {v[2]}, retries {int(v[4])})")
                cap_alerted.add((logs[-1], k))
        rt = sum(v[4] + v[5] for v in vr.values())
        if rt > seen["retry"]:
            alert("ALERT", f"allocator retries/OOMs total {rt:.0f}: " + "; ".join(l for l in txt.splitlines() if "GPU memory:" in l)[-600:]); seen["retry"] = rt
        alltxt = "".join(tail(f, 50_000_000) for f in logs)
        ooml = [l for l in alltxt.splitlines() if re.search(r"CUDA out of memory|OutOfMemoryError", l)]
        deadl = [l for l in alltxt.splitlines() if re.search(r"EngineDeadError|Engine core proc .* died|EngineCore .*(died|exited)|RuntimeError: Engine", l)]
        if len(ooml) > seen["oom_lines"]: alert("ALERT", "CUDA OOM: " + " | ".join(ooml[seen["oom_lines"]:][:3])[:900]); seen["oom_lines"] = len(ooml)
        if len(deadl) > seen["dead_lines"]: alert("ALERT", "engine death: " + " | ".join(deadl[seen["dead_lines"]:][:3])[:900]); seen["dead_lines"] = len(deadl)
        st = {}
        for f in glob.glob(f"{L}/**/supervisor.json", recursive=True) + glob.glob(f"{L}/supervisor.json"):
            try: st = json.load(open(f)); break
            except Exception: pass
        rec["supervisor"] = st
        if st.get("start") is not None:
            if last_start is not None and st["start"] != last_start:
                alert("ALERT", f"supervisor restart: start {last_start} -> {st['start']}: {json.dumps(st)[:500]}")
            last_start = st["start"]
        # --- early warnings (coordinator, 04:30) ---
        open(OUT + ".heartbeat", "w").write(f"{t:.0f} {rec['utc']}\n")
        tps_row(t)
        wc = [v["cap_minus_peak"] for v in rec.get("alloc_cap", {}).values() if v.get("cap_minus_peak") is not None]
        if wc:
            w_ = min(wc); hist_cap.append((t, w_)); hist_cap[:] = [x for x in hist_cap if x[0] >= t - 1800]
            for k, v in rec["alloc_cap"].items():
                if v.get("cap_minus_peak") is not None and v["cap_minus_peak"] < 1.0 and (logs[-1], k) not in EW["cap_lt1"]:
                    alert("EARLY", f"{k}: cap - peak {v['cap_minus_peak']:.2f} GiB < 1.0"); EW["cap_lt1"].add((logs[-1], k))
            if hist_cap and hist_cap[0][1] - w_ > 0.15:
                alert("EARLY", f"cap - peak fell {hist_cap[0][1] - w_:.2f} GiB in {(t - hist_cap[0][0]) / 60:.0f} min ({hist_cap[0][1]:.2f} -> {w_:.2f})"); hist_cap[:] = [(t, w_)]
        hist_keq.append((t, keq)); hist_keq[:] = [x for x in hist_keq if x[0] >= t - 3600]
        if keq > 150 and not EW["keq150"]:
            alert("EARLY", f"Kaggle-eq RAM {keq:.1f} GiB > 150"); EW["keq150"] = True
        if len(hist_keq) > 30 and t - hist_keq[0][0] > 1800:
            rate = (keq - hist_keq[0][1]) / ((t - hist_keq[0][0]) / 3600)
            if rate > 2 and t - EW["keq_rate_t"] > 3600:
                alert("EARLY", f"Kaggle-eq RAM rising {rate:.1f} GiB/h ({hist_keq[0][1]:.1f} -> {keq:.1f})"); EW["keq_rate_t"] = t
        eng = [mm.groups() for mm in ENG.finditer(txt[-200_000:])]
        if eng:
            g_, r_, w_q = float(eng[-1][0]), int(eng[-1][1]), int(eng[-1][2])
            rec["engine_last"] = {"gen": g_, "running": r_, "waiting": w_q}
            if w_q > 30 and t - EW["wait30_t"] > 1800:
                alert("EARLY", f"server waiting {w_q} > 30 (running {r_})"); EW["wait30_t"] = t
            last18 = eng[-18:]
            if len(last18) >= 18 and all(float(x[0]) == 0 for x in last18) and all(int(x[1]) > 0 for x in last18) and t - EW["gen0_t"] > 1800:
                alert("ALERT", f"gen tok/s 0 for the last {len(last18)} engine lines while running {r_}"); EW["gen0_t"] = t
        srv_err = [l for l in alltxt.splitlines() if " ERROR " in l or "Traceback" in l or '" 500' in l]
        if len(srv_err) > EW["srv_err"]:
            alert("ALERT", "server log ERROR/Traceback/500: " + " | ".join(srv_err[EW["srv_err"]:][:3])[:900]); EW["srv_err"] = len(srv_err)
        supk = (st.get("start"), st.get("state"))
        if EW["sup"] is not None and supk != EW["sup"]:
            alert("ALERT", f"supervisor status change {EW['sup']} -> {supk}: {json.dumps(st)[:400]}")
        EW["sup"] = supk
        if not nb_alive():
            alert("ALERT", "notebook runner process (run_nb.pid) is not alive")
        sl = tail(f"{L}/solver.log", 30_000_000)
        t500 = len(re.findall(r"APITimeoutError|timed out|Error code: 500|status_code=500", sl))
        if t500 > EW["t500"]:
            alert("ALERT", "solver APITimeoutError/500: " + " | ".join([l for l in sl.splitlines() if re.search(r"APITimeoutError|timed out|Error code: 500|status_code=500", l)][-3:])[:900]); EW["t500"] = t500
        try:
            ea = [json.loads(x) for x in open(f"{L}/attempts.jsonl") if '"error": "' in x or '"stop_reason": "error"' in x]
        except OSError:
            ea = []
        if len(ea) > EW["err_att"]:
            alert("ALERT", f"errored attempts {len(ea)}: " + " | ".join(f"{a.get('task_id')}.{a.get('pair')}#{a.get('attempt')} {str(a.get('error'))[:150]}" for a in ea[EW["err_att"]:][:5]))
            EW["err_att"] = len(ea)
        se = [l for l in sl.splitlines() if "stop=error" in l]
        if len(se) > seen["stop_error"]: alert("ALERT", "stop=error attempts: " + " | ".join(se[seen["stop_error"]:][:5])[:1200]); seen["stop_error"] = len(se)
        rec["solver"] = {"stop_error": len(se), "server_died": sl.count("server died"), "requeues": sl.count("re-queue"),
                         "last_metrics": next((l.split("] server: ")[1][:220] for l in reversed(sl.splitlines()) if "] server: " in l), None)}
        nb = open(f"{R}/nb.timeline").read() if os.path.exists(f"{R}/nb.timeline") else ""
        if "NOTEBOOK_DONE" in nb:
            alert("INFO", "notebook " + [l for l in nb.splitlines() if "NOTEBOOK_DONE" in l][-1][:300])
            with open(OUT + ".jsonl", "a") as f: f.write(json.dumps(rec) + "\n")
            break
    except Exception as exc:
        rec["sampler_error"] = f"{type(exc).__name__}: {exc}"
    with open(OUT + ".jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")
    time.sleep(max(1, 60 - (time.time() - t)))
