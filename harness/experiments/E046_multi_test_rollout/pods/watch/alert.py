"""Every ~150 s: read each pod's run.log events; print ALERT lines (KV>=0.80 crossing for watched runs, thrash windows,
gate holds start, server start/died, done, solver gone, unreachable). State in alert_state.json."""
import json, os, re, statistics, subprocess, sys, time
D = os.path.dirname(os.path.abspath(__file__))
WATCH80 = set(sys.argv[1].split(",")) if len(sys.argv) > 1 else set()
pat = re.compile(r"\[(\S+) \+([\d.]+)h\] server: ([\d.]+) gen tok/s, prefix hits ([\d.]+), uncached ([\d.]+) tok/s, KV ([\d.]+), running ([\d.]+), waiting ([\d.]+), preemptions ([\d.]+).*?; (\d+) attempts done")
SF = f"{D}/alert_state.json"
st = json.load(open(SF)) if os.path.exists(SF) else {}
def out(s):
    print(f"ALERT {time.strftime('%H:%M:%S', time.gmtime())} {s}", flush=True)
while True:
    for line in open(f"{D}/pods.txt"):
        if not line.strip():
            continue
        pid, port, host, run = line.split()
        cmd = (f"grep -E '\\] server: |KV gate:|server up|server died|server start|server exited|server failed|done at|no restart|hard stop' /root/runs/{run}/run.log; "
               "echo SOLVER $(ps -eo pid,comm,args | awk '$2 ~ /^python/ && /solver_dsv41.py/ {print $1}')")
        try:
            r = subprocess.run(["timeout", "50", "ssh", "-n", "-o", "ConnectTimeout=15", "-p", port, f"root@{host}", cmd],
                               capture_output=True, text=True, timeout=60)
            txt = r.stdout
        except Exception as e:
            txt = ""
        s = st.setdefault(run, {"n": 0, "kv80": False, "gate": 0, "starts": 1, "died": 0, "done": False, "fail": 0, "nosol": 0})
        if "SOLVER" not in txt:
            s["fail"] += 1
            if s["fail"] == 3:
                out(f"{run}: pod {pid} unreachable for 3 checks")
            continue
        s["fail"] = 0
        L = txt.splitlines()
        W = [m.groups() for m in map(pat.search, L) if m]
        steady = [float(w[2]) for w in W if float(w[1]) >= 0.3 and float(w[5]) < 0.75]
        plateau = statistics.median(steady) if steady else None
        for w in W[s["n"]:]:
            t, h, tps, hits, unc, kv, rn, wt, pre, att = w
            h, tps, hits, unc, kv = float(h), float(tps), float(hits), float(unc), float(kv)
            desc = f"+{h:.2f}h ({t}) {tps} tok/s, hits {hits}, uncached {unc}, KV {kv}, running {rn}, preempt {pre}, {att} att; plateau {plateau}"
            if h >= 0.4 and ((plateau and tps < 0.75 * plateau and kv > 0.8) or (unc > 350 and hits < 0.9)):
                out(f"THRASH {run}: {desc}")
            if run in WATCH80 and kv >= 0.80 and not s["kv80"]:
                s["kv80"] = True
                out(f"KV80 {run}: {desc}")
            if run in WATCH80 and kv >= 0.88 and not s.get("kv88"):
                s["kv88"] = True
                out(f"KV88 {run}: {desc}")
            if float(pre) > 0:
                out(f"PREEMPT {run}: {desc}")
        s["n"] = len(W)
        g = sum("KV gate" in l for l in L)
        if g > 0 and s["gate"] == 0:
            out(f"GATE {run}: first hold: {[l for l in L if 'KV gate' in l][0]}")
        s["gate"] = g
        ns = sum("server start" in l for l in L); nd = sum(("server died" in l) or ("server exited" in l) or ("server failed" in l) for l in L)
        if ns > s["starts"] or nd > s["died"]:
            out(f"SERVER {run}: starts {ns}, died/exited {nd}: {[l for l in L if 'server' in l and 'server:' not in l][-1][:200]}")
        s["starts"], s["died"] = max(ns, 1), nd
        if any("done at" in l for l in L) and not s["done"]:
            s["done"] = True
            out(f"DONE {run}: {[l for l in L if 'done at' in l][-1]}")
        sol = L[-1].split()[1:] if L and L[-1].startswith("SOLVER") else []
        if not sol and not s["done"]:
            s["nosol"] += 1
            if s["nosol"] == 2:
                out(f"NOSOLVER {run}: no solver process (2 checks)")
        else:
            s["nosol"] = 0
    json.dump(st, open(SF, "w"))
    time.sleep(150)
