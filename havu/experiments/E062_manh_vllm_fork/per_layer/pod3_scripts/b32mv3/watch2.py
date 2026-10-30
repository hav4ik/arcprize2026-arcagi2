"""Tight watcher for the B32-M v3 replica (local; read-only on the pod except the sampler restart, the fast-mode flag
file and server-log copies). Every 60 s runs state_now.py on the pod, writes a heartbeat file, restarts the sampler if
its newest sample is > 240 s old, turns the sampler's 5 s mode on (fast.flag) once cap - peak < 0.35 GiB on any rank or
after the first allocator retry, and EXITS printing ALERT lines (plus a 5-min timeline and, for engine events, the
matching server-log lines; the server logs are copied to /root/e066/b32mv3/preserved/<UTC>/) on:
the first allocator retry, a retry storm (>= +10 within 10 min; the running count is in every 30-min report), an OOM, an engine-death pattern, a server death or restart; cap - peak below
0.40 / 0.30 / 0.26 GiB or a drop > 0.1 GiB within 30 min; Kaggle-eq > 155 GiB or rising > 2 GiB/h; any errored attempt,
APITimeoutError or HTTP 500; gen tok/s at 0 for > 3 min with requests running; waiting > 30; a server ERROR /
Traceback line; the notebook ending. Also exits with REPORT30 / REPORT60 at the 30-min / hourly marks of the Kaggle
clock (T0 03:30:40 UTC). Every 5 min it also refreshes the dashboard files (export_dash.sh). Alerts already reported are kept in watch2_state.json, so a restarted watcher does not repeat them.
Usage: watch2.py"""
import datetime as dt, json, os, subprocess, sys, time

S = os.path.dirname(os.path.abspath(__file__))
IP, PORT = open(f"{S}/pod3_ssh.txt").read().split()
T0 = dt.datetime(2026, 10, 30, 3, 30, 40, tzinfo=dt.timezone.utc).timestamp()
SF, HB = f"{S}/watch2_state.json", f"{S}/watch2.heartbeat"
st = json.load(open(SF)) if os.path.exists(SF) else {}
hist = st.get("hist", [])


def ssh(cmd, timeout=90):
    return subprocess.run(["timeout", str(timeout), "ssh", "-o", "BatchMode=yes", "-p", PORT, f"root@{IP}", cmd],
                          capture_output=True, text=True).stdout


def cap_peak(s):
    v = [r["cap"] - r["peak"] for r in s["ranks"].values() if "cap" in r and "peak" in r]
    return min(v) if v else None


def save():
    st["hist"] = hist[-120:]
    json.dump(st, open(SF, "w"))


def timeline():
    rows = []
    for h in hist[-6:]:
        sc = h.get("scrape") or []
        rate = (sc[-1]["generation_tokens_total"] - sc[-2]["generation_tokens_total"]) / max(1, sc[-1]["t"] - sc[-2]["t"]) if len(sc) > 1 else None
        rk = h["ranks"]
        rows.append(f"  {time.strftime('%H:%M:%S', time.gmtime(h['t']))} K{(h['t'] - T0) / 3600:.2f}h cap-peak "
                    + "/".join(f"{rk[k]['cap'] - rk[k]['peak']:.2f}" if 'peak' in rk.get(k, {}) else "-" for k in "0123")
                    + f" retries {'/'.join(str(rk.get(k, {}).get('retries', '-')) for k in '0123')} gen/s {rate if rate is None else round(rate)} "
                    + f"run/wait {sc[-1].get('num_requests_running', '-') if sc else '-'}/{sc[-1].get('num_requests_waiting', '-') if sc else '-'} "
                    + f"keq {h['keq']} died {h['died']} starts {h['starts']} att_err {h['att_err']}")
    return "\n".join(rows)


def finish(lines, engine=False):
    print(time.strftime("%H:%M:%S", time.gmtime()), "\n".join(lines))
    print("timeline (60 s rounds):\n" + timeline())
    if engine:
        stamp = time.strftime("%H%M%S", time.gmtime())
        print(ssh(f"D=/root/e066/b32mv3/preserved/{stamp}; mkdir -p $D && cp /kaggle/working/dsv41_logs/server*.log /kaggle/working/dsv41_logs/solver.log $D/ && echo preserved in $D; "
                  "for f in /kaggle/working/dsv41_logs/server*.log; do echo == $f; sed 's/\\x1b\\[[0-9;]*m//g' $f | grep -anE 'out of memory|OutOfMemoryError|EngineDead|EngineCore encountered|WorkerProc hit|execute_model timed out| ERROR |Traceback' | tail -12 | cut -c1-300; done; "
                  "grep -aE 'server died|server start|server up after|retry|re-queue' /kaggle/working/dsv41_logs/solver.log | tail -8 | cut -c1-250", 120))
    save()
    sys.exit(0)


def dash_sync():
    """Dashboard files every 5 min (export_dash.sh: gpumem.log, ram.tsv, status.json with the automatic verdict)."""
    r = subprocess.run([f"{S}/export_dash.sh"], capture_output=True, text=True, timeout=600)
    open(f"{S}/dash_sync.log", "a").write(time.strftime("%H:%M:%S ", time.gmtime()) + (r.stdout.strip().splitlines() or ["?"])[-1] + "\n")


next_sync = time.time()
# the next report mark (Kaggle clock half hours)
now = time.time()
k = int((now - T0) // 1800) + 1
mark = T0 + k * 1800
while True:
    raw = ssh("/usr/bin/python3 /root/e066/b32mv3/state_now.py")
    try:
        s = json.loads(raw.strip().splitlines()[-1])
    except Exception:
        open(HB, "w").write(f"{time.time()} ssh/parse failure\n")
        time.sleep(60)
        continue
    hist.append(s)
    open(HB, "w").write(f"{s['t']} K{(s['t'] - T0) / 3600:.2f}h ok\n")
    al, eng = [], False
    rep = st.setdefault("rep", {})
    # sampler alive?
    if s["t"] - s.get("mon_t", 0) > 240:
        print(ssh("setsid nohup /usr/bin/python3 /root/e066/b32mv3/mon_sampler.py /root/e066/b32mv3/mon.jsonl 60 > /root/e066/b32mv3/mon.out 2>&1 < /dev/null & sleep 2; pgrep -af mon_sampler | head -2"))
        al.append(f"ALERT sampler was stale ({s['t'] - s.get('mon_t', 0):.0f} s), restarted")
    cp = cap_peak(s)
    rmax = max([r.get("retries", 0) for r in s["ranks"].values()] or [0])
    if (cp is not None and cp < 0.35 or rmax > 0) and not rep.get("fast"):
        ssh("touch /root/e066/b32mv3/fast.flag"); rep["fast"] = 1
        al.append(f"INFO sampler switched to 5 s (cap-peak {cp}, retries {rmax})")
    r10 = [max([r.get("retries", 0) for r in h["ranks"].values()] or [0]) for h in hist if h["t"] >= s["t"] - 600]
    storm = r10 and rmax - min(r10) >= 10 and s["t"] - rep.get("storm_t", 0) > 1800
    if storm:
        rep["storm_t"] = s["t"]
    if (rmax > 0 and rep.get("retries", 0) == 0) or storm:  # the first retry, then only a storm (>= +10 in 10 min, at most every 30 min)
        al.append(f"ALERT allocator retries now {rmax} (was {rep.get('retries', 0)}): " + ", ".join(f"TP{k} {r.get('retries')}" for k, r in sorted(s['ranks'].items())))
        rep["retries"] = rmax
    ooms = sum(r.get("ooms", 0) for r in s["ranks"].values()) + s["srv"]["oom"]
    if ooms > rep.get("ooms", 0):
        al.append(f"ALERT OOM: vLLM OOM count {sum(r.get('ooms', 0) for r in s['ranks'].values())}, server-log OOM lines {s['srv']['oom']}"); rep["ooms"] = ooms; eng = True
    for key, label in (("dead", "engine-death pattern lines"), ("error", "server ERROR lines"), ("tb", "server Traceback lines"), ("h500", "HTTP 500 lines")):
        if s["srv"][key] > rep.get(key, 0):
            al.append(f"ALERT {label}: {s['srv'][key]} (was {rep.get(key, 0)})"); rep[key] = s["srv"][key]; eng = eng or key in ("dead", "tb")
    for key, label in (("died", "server died"), ("starts", "server starts")):
        if s[key] > rep.get(key, {"died": 0, "starts": 1}[key]):
            al.append(f"ALERT {label}: {s[key]} (logs {s['logs']})"); rep[key] = s[key]; eng = True
    if s["att_err"] > rep.get("att_err", 0):
        al.append(f"ALERT errored attempt records {s['att_err']} (was {rep.get('att_err', 0)}), timeouts {s['att_timeout']}"); rep["att_err"] = s["att_err"]
    if cp is not None:
        for lv in (0.40, 0.30, 0.26):
            if cp < lv and rep.get("cp_level", 9) > lv:
                al.append(f"ALERT cap-peak {cp:.2f} GiB below {lv:.2f} (min over ranks)"); rep["cp_level"] = lv
        win = [cap_peak(h) for h in hist if h["t"] >= s["t"] - 1800 and cap_peak(h) is not None]
        if win and max(win) - cp > 0.1 and s["t"] - rep.get("drop_t", 0) > 1800:
            al.append(f"ALERT cap-peak dropped {max(win) - cp:.2f} GiB within 30 min (now {cp:.2f})"); rep["drop_t"] = s["t"]
    if s["keq"] > 155 and rep.get("keq155") is None:
        al.append(f"ALERT Kaggle-eq {s['keq']} GiB > 155"); rep["keq155"] = 1
    old = [h for h in hist if h["t"] <= s["t"] - 1800]
    if old and s["t"] - rep.get("slope_t", 0) > 3600:
        sl = (s["keq"] - old[-1]["keq"]) / ((s["t"] - old[-1]["t"]) / 3600)
        if sl > 2:
            al.append(f"ALERT Kaggle-eq rising {sl:.1f} GiB/h (now {s['keq']})"); rep["slope_t"] = s["t"]
    sc = s.get("scrape") or []
    if len(sc) >= 4 and sc[-1]["generation_tokens_total"] == sc[-4]["generation_tokens_total"] and sc[-1].get("num_requests_running", 0) > 0 \
            and sc[-1]["t"] - sc[-4]["t"] >= 170 and s["t"] - rep.get("stall_t", 0) > 900:
        al.append(f"ALERT gen tok/s 0 for {sc[-1]['t'] - sc[-4]['t']} s with {sc[-1]['num_requests_running']} running"); rep["stall_t"] = s["t"]
    if sc and sc[-1].get("num_requests_waiting", 0) > 30 and s["t"] - rep.get("wait_t", 0) > 1800:
        al.append(f"ALERT waiting {sc[-1]['num_requests_waiting']} > 30"); rep["wait_t"] = s["t"]
    if s["nb_end"] or not s["papermill"]:
        al.append(f"ALERT notebook ended (nb_end {s['nb_end']}, papermill procs {s['papermill']})")
    if any(a.startswith("ALERT") for a in al):
        finish(al, eng)
    if time.time() >= mark:
        finish([("REPORT60" if round((mark - T0) / 1800) % 2 == 0 else "REPORT30") + f" due (Kaggle {(mark - T0) / 3600:.1f} h)"] + al)
    if time.time() >= next_sync:
        try:
            dash_sync()
        except Exception as exc:
            open(f"{S}/dash_sync.log", "a").write(f"sync failed: {exc}\n")
        next_sync = time.time() + 300
    save()
    time.sleep(max(5, 60 - (time.time() - s["t"])))
