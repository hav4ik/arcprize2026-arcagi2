"""Replica watcher (read-only): every 60 s reads the latest server log (the supervisor's current output root), the
solver log, attempts.jsonl and the supervisor status; every 10 min appends a summary line to <out>.log; appends FLAG
lines to <out>.flags at once on: VRAM free < 1 GiB, allocator retries or OOMs > 0, a new "server died", a supervisor
state other than "up" for > 5 min, an attempt finalized as error (solver "stop=error"), a Traceback in solver.log,
the notebook's end. Usage: r1_watch.py <work dir> <run dir> <out prefix>"""
import glob, json, os, re, subprocess, sys, time
W, R, OUT = sys.argv[1:4]
L = f"{W}/dsv41_logs"; ANSI = re.compile(r"\x1b\[[0-9;]*m")
GM = re.compile(r"Worker_TP(\d).*GPU memory: ([\d.]+) GiB allocated \(peak ([\d.]+) GiB\), ([\d.]+) GiB reserved.*?, ([\d.]+) GiB free, (\d+) allocator retries, (\d+) OOMs")
def flag(msg):
    with open(OUT + ".flags", "a") as f:
        f.write(f"{time.strftime('%H:%M:%S', time.gmtime())} FLAG {msg}\n")
def tail(path, n=400000):
    try:
        with open(path, "rb") as f:
            f.seek(max(0, os.path.getsize(path) - n)); return ANSI.sub("", f.read().decode(errors="replace"))
    except OSError:
        return ""
seen_died = seen_err = seen_tb = 0; not_up_since = None; last_sum = 0; done_flag = False
while True:
    st = {}
    try:
        st = json.load(open(f"{L}/e066_supervisor.json"))
    except Exception:
        pass
    logs = sorted(glob.glob(f"{L}/**/server/server.log", recursive=True) + glob.glob(f"{L}/server/server.log"), key=os.path.getmtime)
    vram = {}
    for m in GM.finditer(tail(logs[-1]) if logs else ""):
        vram[m.group(1)] = tuple(float(x) for x in m.groups()[1:])
    for r, (al, pk, rs, fr, rt, oom) in vram.items():
        if fr < 1.0: flag(f"VRAM free {fr} GiB on TP{r} (alloc {al}, peak {pk}, reserved {rs})")
        if rt > 0 or oom > 0: flag(f"TP{r}: {int(rt)} allocator retries, {int(oom)} OOMs")
    sl = tail(f"{L}/solver.log", 4_000_000)
    died = sl.count("server died"); err = len(re.findall(r"stop=error", sl)); tb = sl.count("Traceback")
    if died > seen_died: flag(f"solver: server died x{died}: " + [l for l in sl.splitlines() if "server died" in l][-1][:200])
    if err > seen_err: flag(f"solver: {err} attempts finalized stop=error: " + [l for l in sl.splitlines() if "stop=error" in l][-1][:200])
    if tb > seen_tb: flag(f"solver.log Traceback x{tb}")
    seen_died, seen_err, seen_tb = max(seen_died, died), max(seen_err, err), max(seen_tb, tb)
    s = st.get("state")
    if s != "up":
        not_up_since = not_up_since or time.time()
        if time.time() - not_up_since > 300: flag(f"supervisor state {s} for {time.time() - not_up_since:.0f}s: {st}"); not_up_since = time.time()
    else:
        not_up_since = None
    nb = open(f"{R}/nb.timeline").read() if os.path.exists(f"{R}/nb.timeline") else ""
    if "NOTEBOOK_DONE" in nb and not done_flag:
        flag("notebook " + [l for l in nb.splitlines() if "NOTEBOOK_DONE" in l][-1][:300]); done_flag = True
    if time.time() - last_sum >= 600:
        last_sum = time.time()
        met = [l for l in sl.splitlines() if "] server:" in l]
        rq = sl.count("after a server-side error")
        n_att = sum(1 for _ in open(f"{L}/attempts.jsonl")) if os.path.exists(f"{L}/attempts.jsonl") else 0
        used = subprocess.run("free -g | awk 'NR==2{print $3}'", shell=True, capture_output=True, text=True).stdout.strip()
        v = " ".join(f"TP{r}:{a:.2f}/{p:.2f}/{rs:.2f}/{f:.2f}/{int(t)}/{int(o)}" for r, (a, p, rs, f, t, o) in sorted(vram.items()))
        with open(OUT + ".log", "a") as f:
            f.write(f"{time.strftime('%H:%M:%S', time.gmtime())} sup={s}/start{st.get('start')} died={died} requeues={rq} "
                    f"stop_error={err} attempt_records={n_att} host_used={used}G vram(alloc/peak/res/free/retry/oom) {v} | "
                    f"{(met[-1].split('] server: ')[1][:200]) if met else ''}\n")
    if done_flag:
        break
    time.sleep(60)
