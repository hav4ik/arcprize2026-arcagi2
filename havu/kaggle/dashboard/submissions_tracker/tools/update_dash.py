"""Rebuild the dashboard's live data: mem.json (via build_mem.py), tps.json, and runs.json's latest GPU/throughput fields."""
import json, os, re, subprocess, sys
D = os.environ.get("DASH_DIR") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # the dashboard folder (live copy: the session scratchpad)
R = "/home/vu/Workspace/ARC-AGI-2/experiments/E062_manh_vllm_fork/per_layer"
LIVE = {"Manh V12 replica": ("01:56:00", [f"{R}/phase_h/manh_v12/data/gpumem.log", f"{R}/phase_h/manh_v12/data/dsv41_logs/server/server.log"]),
        "B32-M V3 replica": ("03:30:40", [f"{R}/pod3_runs/b32m_v3/data/gpumem.log"])}
print(subprocess.run([sys.executable, f"{D}/tools/build_mem.py"], capture_output=True, text=True, check=True).stdout, end="")
mem = json.load(open(f"{D}/mem.json"))
old_tps = json.load(open(f"{D}/tps.json")) if os.path.exists(f"{D}/tps.json") else []
tps = []
OUT = []
# runs whose tps.tsv is trusted (Manh V12 since pod 2 fixed it, checked against server.log: 241 minute rows, one isolated zero blip)
TSV_TRUST = {"B32-M V3 replica", "Manh V12 replica", "Ours V1 qualification"}
def kind_of(run):
    try:
        st = json.load(open(STATUS[run]))
        if "fault injection" in str(st.get("verdict", "")).lower(): return "fault injection"
        return "OOM hang + restart" if (st.get("ooms") or 0) > 0 or (st.get("deaths") or 0) > 0 else False
    except Exception:
        return False
STATUS = {"Manh V12 replica": f"{R}/phase_h/manh_v12/data/status.json", "B32-M V3 replica": f"{R}/pod3_runs/b32m_v3/data/status.json"}
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pod4
P4 = pod4.info()
if P4:  # a third live run appears once its status.json exists
    LIVE[pod4.NAME] = (P4["t0"], [f"{P4['dir']}/gpumem.log"])
    STATUS[pod4.NAME] = f"{P4['dir']}/status.json"
FIN_PATH = f"{D}/finished.json"
try:
    FIN = json.load(open(FIN_PATH))  # runs whose solver has exited: {run: {"h": kaggle hours at exit, ...}}; kept once seen
except Exception:
    FIN = {}
FIN_RE = re.compile(r"solver (?:exit|exited|finished|done)|run (?:finished|ended|is over)|\bfinished\b", re.I)
FINAL_RE = re.compile(r"no restart left|server gone|final OOM", re.I)
FINAL_LABEL = "final OOM, no restart left (server gone)"
try:
    FINALS = set(json.load(open(f"{D}/final_oom.json")))  # runs whose open outage is the last one (all restarts used), per the coordinator
except Exception:
    FINALS = set()
def verdict_of(run):
    try:
        return str(json.load(open(STATUS[run])).get("verdict", ""))
    except Exception:
        return ""
for _run, _p in STATUS.items():
    if _run not in FIN and os.path.exists(_p):
        try:
            _st = json.load(open(_p))
            if _st.get("finished") or _st.get("solver_exit") or FIN_RE.search(str(_st.get("verdict", ""))):
                FIN[_run] = {"h": float(_st["kaggle_h"]), "utc": str(_st.get("utc")), "verdict": str(_st.get("verdict", ""))[:200]}
        except Exception:
            pass
for _run, _f in FIN.items():  # the verdict may name the exact exit time: "... finished at Kaggle 7.26 h"
    _m = re.search(r"(?:finished|exited|ended) at Kaggle (\d+(?:\.\d+)?) ?h", _f.get("verdict", ""))
    if _m: _f["h"] = float(_m[1])
json.dump(FIN, open(FIN_PATH, "w"))
def status_h(run):
    try:
        h = float(json.load(open(STATUS[run]))["kaggle_h"])
        return min(h, FIN[run]["h"]) if run in FIN else h  # a finished run is not zero-filled past its exit
    except Exception:
        return None
def read_tsv(path):
    rows = []
    for l in open(path):
        c = l.rstrip("\n").split("\t")
        try:
            rows.append({"utc": c[0], "t": float(c[1]), "g": float(c[2]), "r": float(c[3]), "w": float(c[4]),
                         "kv": float(c[5]) if len(c) > 5 and c[5].strip() not in ("", "nan", "NaN", "None") else None})
        except (ValueError, IndexError):
            continue
    # an isolated zero between two serving minutes is a failed /metrics scrape (the log shows generation), not an outage
    return [x for i, x in enumerate(rows) if not (0 < i < len(rows) - 1 and x["g"] < 5 and rows[i - 1]["g"] >= 100 and rows[i + 1]["g"] >= 100)]
def sod(u):
    m = re.search(r"(\d\d):(\d\d):(\d\d)", u)
    return int(m[1]) * 3600 + int(m[2]) * 60 + int(m[3]) if m else None
def oom_times(run):
    """UTC seconds of day of the OOMs named in the run's status verdict. A verdict forgets the OOM time once the server is declared
    dead, so every time seen is kept in oom_seen.json."""
    seen = {}
    try:
        seen = json.load(open(f"{D}/oom_seen.json"))
    except Exception:
        pass
    ts = set(seen.get(run, []))
    try:
        v = str(json.load(open(STATUS[run])).get("verdict", ""))
        for m in re.finditer(r"OOM(?: at)? \(?(\d\d):(\d\d)(?::(\d\d))? UTC", v):
            ts.add(int(m[1]) * 3600 + int(m[2]) * 60 + int(m[3] or 0))
    except Exception:
        pass
    seen[run] = sorted(ts)
    json.dump(seen, open(f"{D}/oom_seen.json", "w"))
    return sorted(ts)
def find_outages(rows, oom=None):
    """Outages in minute rows: from the first row with gen < 50 tok/s once serving has been seen (gen >= 50), until generation
    resumes (gen >= 50, or the server serving again after a row with nothing running). Returns (start_h, end_h, down rows, end row).
    With oom (a list of UTC seconds of day from status.json), a down row within 15 min after it starts the outage at the OOM time, and one
    row is enough; otherwise a single zero minute is taken for a failed scrape and needs a second row."""
    out, started, cur, prev, flag, st, used = [], False, None, None, False, None, set()
    for x in rows:
        if x["g"] >= 50:
            started = True
        if cur is None:
            xs = sod(x["utc"])
            near = [o for o in (oom or []) if o not in used and xs is not None and 0 <= xs - o <= 900]
            oo = max(near) if near else None
            by_oom = oo is not None and x["g"] < 50
            if started and (x["g"] < 5 or by_oom or (x["g"] < 50 and prev is not None and prev["g"] >= 50)):
                cur, flag = [x], by_oom
                if by_oom: used.add(oo)  # one OOM starts one outage
                st = x["t"] - (xs - oo) / 3600 if by_oom else x["t"]
        else:
            if x["g"] >= 50 or (prev["r"] == 0 and x["r"] > 0):
                out.append((st, x["t"], cur, x, flag)); cur = None
            else:
                cur.append(x)
        prev = x
    if cur:
        out.append((st, None, cur, None, flag))
    return [o[:4] for o in out if len(o[2]) >= 2 or o[4]]
def bins_from_tsv(path):
    """tps.tsv columns: utc, kaggle_h, gen_tok_s, running, waiting, kv_pct (0 rows while the server is down)."""
    rows = [(x["t"], x["g"], x["r"], x["w"], x["kv"]) for x in read_tsv(path)]
    bins = {}
    for r in rows:
        if r[0] >= 0:
            bins.setdefault(int(r[0] * 6), []).append(r)
    pts = []
    for b, v in sorted(bins.items()):
        kv = [x[4] for x in v if x[4] is not None]
        pts.append({"t_h": sum(x[0] for x in v) / len(v), "gen_tok_s": sum(x[1] for x in v) / len(v), "running": sum(x[2] for x in v) / len(v),
                    "waiting": sum(x[3] for x in v) / len(v), "kv_pct": sum(kv) / len(kv) if kv else None})
    return pts
for run, (t0, logs) in LIVE.items():
    pts = []
    tsv = os.path.join(os.path.dirname(logs[0]), "tps.tsv")
    if os.path.exists(tsv) and run in TSV_TRUST:
        pts = bins_from_tsv(tsv)
        rows = read_tsv(tsv)
        slog = os.path.join(os.path.dirname(logs[0]), "solver.log")
        ups = []  # the solver's "server up after" lines, as kaggle hours
        if os.path.exists(slog):
            for l in open(slog, errors="replace"):
                m = re.match(r"\[[\d:]+ \+([\d.]+)h\] server up after", l)
                if m: ups.append(float(m[1]))
        for st_h, en_h, down, endrow in find_outages(rows, oom_times(run)):
            if en_h is None:  # no recovery row yet (the pod sync lags): close it when the solver reports the server up
                en_h = next((u for u in ups if u > st_h), None)
            en = en_h if en_h is not None else rows[-1]["t"]
            OUT.append({"run": run, "start_h": round(st_h, 2), "end_h": round(en_h, 2) if en_h is not None else None,
                        "kind": FINAL_LABEL if (en_h is None and (run in FIN or run in FINALS or FINAL_RE.search(verdict_of(run)))) else (kind_of(run) or "no generation"),
                        "utc": f"{down[0]['utc'][11:16]}-{(endrow or rows[-1])['utc'][11:16]}",
                        "ongoing": en_h is None, "down_t": sorted({round(st_h, 3)} | {x["t"] for x in down})})
            if en_h is None and status_h(run):  # an open outage reads 0 on the GPU charts up to the latest status
                last = OUT[-1]["down_t"][-1]
                OUT[-1]["down_t"] += [round(last + k / 12, 3) for k in range(1, int((status_h(run) - last) * 12) + 1)] + [round(status_h(run), 3)]
            # bins touching the outage are replaced by the minute rows, so the dip shows at full depth
            lo, hi = int(st_h * 6), int(en * 6)
            pts = [q for q in pts if not (lo <= int(q["t_h"] * 6) <= hi)]
            pts += [{"t_h": x["t"], "gen_tok_s": x["g"], "running": x["r"], "waiting": x["w"], "kv_pct": x["kv"]}
                    for x in rows if lo <= int(x["t"] * 6) <= hi and not (x["g"] >= 50 and x["t"] > en)]
        pts.sort(key=lambda q: q["t_h"])
    else:
        for log in logs:  # the first log with throughput lines wins
            if os.path.exists(log):
                out = subprocess.run([sys.executable, f"{D}/tools/tps_series.py", log, t0, "10"], capture_output=True, text=True, check=True).stdout
                pts = [dict(zip(("t_h", "gen_tok_s", "running", "waiting", "kv_pct", "n"), map(float, l.split("\t")))) for l in out.splitlines()]
                if pts:
                    break
        if not pts:
            pts = [p for p in old_tps if p["run"] == run]
    # a missing 10-min bin means no generation: fill it with zeros, from the first bin up to status.json's kaggle_h
    sh = status_h(run)
    if pts and sh is not None:
        have = {int(p["t_h"] * 6) for p in pts}
        for b in range(int(min(p["t_h"] for p in pts) * 6), int(sh * 6) + 1):
            if b not in have and (b + 0.5) / 6 <= sh:
                pts.append({"t_h": (b + 0.5) / 6, "gen_tok_s": 0.0, "running": 0.0, "waiting": 0.0, "kv_pct": None})
        pts.sort(key=lambda p: p["t_h"])
    if pts and not (os.path.exists(tsv) and run in TSV_TRUST):
        seen, cur = False, None
        for q in pts + [{"t_h": 1e9, "gen_tok_s": 999}]:
            if q["gen_tok_s"] >= 50: seen = True
            z = seen and q["gen_tok_s"] == 0
            if z and cur is None:
                cur = [int(q["t_h"] * 6), int(q["t_h"] * 6)]
            elif z:
                cur[1] = int(q["t_h"] * 6)
            elif cur is not None:
                OUT.append({"run": run, "start_h": round(cur[0] / 6, 2), "end_h": round((cur[1] + 1) / 6, 2), "kind": "no generation",
                            "utc": "", "ongoing": False, "down_t": [(b + 0.5) / 6 for b in range(cur[0], cur[1] + 1)]}); cur = None
    if run in FIN and pts:  # the line ends at the solver exit, at 0
        pts = [q for q in pts if q["t_h"] <= FIN[run]["h"] + 0.01]
        if pts and pts[-1]["t_h"] < FIN[run]["h"] - 0.02 and pts[-1]["gen_tok_s"] > 0:
            pts.append({"t_h": FIN[run]["h"], "gen_tok_s": 0.0, "running": 0.0, "waiting": 0.0, "kv_pct": None})
    for p in pts:
        p.pop("n", None); p["run"] = run
        for k in ("t_h", "gen_tok_s", "running", "waiting", "kv_pct"):
            if p.get(k) is not None: p[k] = round(p[k], 2 if k == "t_h" else 1)
    tps += pts
REF = json.load(open(f"{D}/ref_b32.json")) if os.path.exists(f"{D}/ref_b32.json") else {"tps": [], "mem": []}
tps += REF["tps"]
json.dump(tps, open(f"{D}/tps.json", "w"), indent=1)
mem = [m for m in mem if not m["run"].startswith("B32 replica")] + REF["mem"]
for o in OUT:  # the GPU charts drop to 0 during an outage; RAM stays as measured
    last_down = max(o["down_t"])
    for m in mem:
        if m["run"] == o["run"] and o["start_h"] <= m["t_h"] <= last_down:
            m["vram_free_min_gib"] = m["cap_minus_peak_gib"] = m["cap_minus_reserved_gib"] = None
    mem += [{"run": o["run"], "t_h": round(t, 3), "ram_kaggle_eq_gib": None, "vram_free_min_gib": 0, "cap_minus_peak_gib": 0, "cap_minus_reserved_gib": 0} for t in o["down_t"]]
for m in mem: m.setdefault("cap_minus_reserved_gib", None)
mem.sort(key=lambda m: (m["run"], m["t_h"]))
mem = [m for m in mem if any(m.get(k) is not None for k in ("vram_free_min_gib", "cap_minus_peak_gib", "ram_kaggle_eq_gib"))]
json.dump([{k: v for k, v in o.items() if k not in ("down_t", "ongoing")} for o in OUT], open(f"{D}/outages.json", "w"), indent=1)
json.dump(mem, open(f"{D}/mem.json", "w"), indent=1)
runs = json.load(open(f"{D}/runs.json"))
if P4 and not any(r["run"] == pod4.NAME for r in runs):
    row = {"run": pod4.NAME, "pod": "pod 4", "started_utc": P4["start"], "elapsed_h": None, "first_done": None, "first_correct": None,
           "deaths": 0, "restarts": 0, "errored": 0, "ram_kaggle_eq_gib": None, "vram_free_min_gib": None, "updated_utc": None}
    k = max((i for i, r in enumerate(runs) if r["run"] in LIVE), default=-1)
    runs.insert(k + 1, row)
for r in runs:
    if r["run"] not in LIVE:
        continue
    g = [m for m in mem if m["run"] == r["run"] and m.get("vram_free_min_gib") is not None]
    c = [m for m in mem if m["run"] == r["run"] and m.get("cap_minus_peak_gib") is not None]
    if g: r["vram_free_min_gib"] = max(g, key=lambda m: m["t_h"])["vram_free_min_gib"]
    if c: r["cap_spare_gib"] = max(c, key=lambda m: m["t_h"])["cap_minus_peak_gib"]
    cr = [m for m in mem if m["run"] == r["run"] and m.get("cap_minus_reserved_gib") is not None]
    if cr: r["cap_reserved_gib"] = max(cr, key=lambda m: m["t_h"])["cap_minus_reserved_gib"]
    a = [m for m in mem if m["run"] == r["run"] and m.get("ram_kaggle_eq_gib") is not None]
    if a: r["ram_kaggle_eq_gib"] = max(a, key=lambda m: m["t_h"])["ram_kaggle_eq_gib"]
    p = sorted((x for x in tps if x["run"] == r["run"]), key=lambda x: x["t_h"])
    if p:
        last = [x for x in p if x["t_h"] > p[-1]["t_h"] - 0.5]
        # points mix 10-min bins with 1-min rows around outages, so weight each by the time it covers
        wts = [min(max(x["t_h"] - (p[i - 1]["t_h"] if i else x["t_h"] - 1 / 6), 1e-3), 1 / 6) for i, x in enumerate(p)]
        lw = [(w, x) for w, x in zip(wts, p) if x["t_h"] > p[-1]["t_h"] - 0.5]
        r["tps_recent"] = round(sum(w * x["gen_tok_s"] for w, x in lw) / sum(w for w, _ in lw))
        # run mean from the first serving point on (before it the server was loading); outage zeros stay in; time-weighted
        i0 = next((i for i, x in enumerate(p) if x["gen_tok_s"] >= 50), None)
        r["tps_mean"] = round(sum(w * x["gen_tok_s"] for w, x in zip(wts[i0:], p[i0:])) / sum(wts[i0:])) if i0 is not None else None
        r["tps_through_h"] = round(p[-1]["t_h"], 1)
for r in runs:
    p = STATUS.get(r["run"])
    if p and os.path.exists(p):
        st = json.load(open(p))
        for k in ("first_done", "first_correct", "errored", "deaths", "restarts", "ooms"):
            if st.get(k) is not None: r[k] = st[k]  # the pods keep their counters cumulative
        if st.get("kaggle_h") is not None: r["elapsed_h"] = round(float(st["kaggle_h"]), 2)
        if st.get("verdict"):
            # the dashboard is link-shared: keep addresses, hosts and paths out of the published text
            v = re.sub(r"\b\d{1,3}(\.\d{1,3}){3}(:\d+)?\b", "<addr>", str(st["verdict"]))
            v = re.sub(r"(/(root|home|tmp|kaggle|workspace)[^\s,;)]*)", "<path>", v)
            r["status"] = f"running ({v[:200]})"
        if st.get("utc"):
            u = str(st["utc"]).replace("T", " ").rstrip("Z")
            r["updated_utc"] = (u if "-" in u else "2026-10-30 " + u)[:16]
json.dump(runs, open(f"{D}/runs.json", "w"), indent=1)
for r in runs:
    if r["run"] in LIVE:
        print(r["run"], {k: r.get(k) for k in ("vram_free_min_gib", "cap_spare_gib", "tps_recent", "tps_mean", "tps_through_h")})

# Accuracy over time (acc.json): one series per live run from its attempts.jsonl, scored against the workload solutions.
SOLS = "/home/vu/Workspace/ARC-AGI-2/experiments/E055_kaggle_replica/toolkit/workloads/evalx2t_s2026_solutions.json"
ATT = {"Manh V12 replica": f"{R}/phase_h/manh_v12/data/dsv41_logs/attempts.jsonl",
       "B32-M V3 replica": f"{R}/pod3_runs/b32m_v3/data/attempts.jsonl",
       "Ours V1 qualification": f"{R}/pod4_runs/v1_qual/data/attempts.jsonl",
       # reference: the 68.89 submission's full pod replica (10-27), same workload
       "B32 replica (LB 68.89), 10-27": "/home/vu/Workspace/ARC-AGI-2/experiments/E055_kaggle_replica/runs/candBf_x2t/out/dsv41_logs/attempts.jsonl"}
acc = []
for run, path in ATT.items():
    if os.path.exists(path):
        acc += json.loads(subprocess.run([sys.executable, f"{D}/tools/acc_series.py", run, path, SOLS], capture_output=True, text=True, check=True).stdout)
json.dump(acc, open(f"{D}/acc.json", "w"), indent=1)
print("acc points", len(acc))
# runs whose solver has exited: mark them finished (status text with the final score from the accuracy series)
runs = json.load(open(f"{D}/runs.json"))
for r in runs:
    f = FIN.get(r["run"])
    if not f:
        continue
    sc = [a["score_so_far"] for a in acc if a["run"] == r["run"]]
    sc = sc[-1] if sc else None
    o = [x for x in OUT if x["run"] == r["run"]]
    n = r.get("ooms") or 0
    ordn = f"{n}{'th' if 10 <= n % 100 <= 20 or n % 10 > 3 or n % 10 == 0 else ['th', 'st', 'nd', 'rd'][n % 10]}"
    if o and o[-1]["end_h"] is None:
        r["status"] = f"finished early: server gone after the {ordn} OOM at {o[-1]['start_h']:.2f} h; final {f.get('final') or 'score ' + str(sc)}"
    else:
        r["status"] = f"finished; final {f.get('final') or 'score ' + str(sc)}"
    r["elapsed_h"] = round(f["h"], 2)
json.dump(runs, open(f"{D}/runs.json", "w"), indent=1)
