"""status.json for the dashboard: Kaggle clock, first attempts done / fully correct (every pair right, scored against
the x2t solutions), pairs answered, errored attempt records, server deaths / restarts, max allocator retries and OOMs
over ranks, running / waiting (newest scrape), and a one-line verdict (argv[1], else an automatic one).
Usage: status_json.py [verdict]"""
import json, subprocess, sys, time
T0 = 1793331040.0
s = json.loads(subprocess.run(["/usr/bin/python3", "/root/e066/b32mv3/state_now.py"], capture_output=True, text=True).stdout)
sol = json.load(open("/root/data/evalx2t_s2026_solutions.json"))
st = json.load(open("/kaggle/working/dsv41_logs/status.json"))
g = lambda x: json.loads(x) if isinstance(x, str) else x
done = corr = 0
for l in open("/kaggle/working/dsv41_logs/attempts.jsonl"):
    r = json.loads(l)
    if r.get("attempt") != 0 or r.get("pair", 0) != 0 or r.get("error"):
        continue
    t = sol[r["task_id"]]; pr = g(r.get("pred")); done += 1
    corr += (isinstance(pr, list) and len(pr) == len(t) and all(pr[i] == t[i] for i in range(len(t)))) if r.get("multi_test") else pr == t[0]
# cumulative over every server log of the run (server.log, server_restart1.log, ...): the max allocator retries and
# vLLM OOM counts per rank and server, OOM events = distinct (rank, second) of torch.OutOfMemoryError lines; then a
# floor file so no counter ever goes down
import glob, os, re
retries_all, oom_counter, oom_events = 0, 0, set()
for f in sorted(glob.glob("/kaggle/working/dsv41_logs/server*.log"), key=os.path.getmtime):
    per = {}
    out = subprocess.run(["grep", "-aE", "GPU memory:|OutOfMemoryError", f], capture_output=True, text=True).stdout
    for line in re.sub(r"\x1b\[[0-9;]*m", "", out).splitlines():
        m = re.search(r"Worker_TP(\d)\S* pid=\d+\).*GPU memory:.* (\d+) allocator retries, (\d+) OOMs", line)
        if m:
            a, b = per.get(m[1], (0, 0)); per[m[1]] = (max(a, int(m[2])), max(b, int(m[3])))
        e = re.search(r"Worker_TP(\d)\S* pid=\d+\) ERROR \d\d-\d\d (\d\d:\d\d:\d\d).*OutOfMemoryError", line)
        if e:
            oom_events.add((f, e[1], e[2]))
    if per:
        retries_all = max(retries_all, max(v[0] for v in per.values()))
        oom_counter += sum(v[1] for v in per.values())
rk = s["ranks"].values()
cp = [r["cap"] - r["peak"] for r in rk if "cap" in r and "peak" in r]
sc = (s.get("scrape") or [{}])[-1]
out = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(s["t"])), "kaggle_h": round((s["t"] - T0) / 3600, 2),
       "first_done": done, "first_correct": corr, "pairs_answered": st.get("pairs_with_answer"), "errored": s["att_err"],
       "deaths": s["died"], "restarts": max(0, s["starts"] - 1), "retries_max": retries_all,
       "ooms": max(oom_counter, len(oom_events)), "running": int(sc.get("num_requests_running", 0)),
       "waiting": int(sc.get("num_requests_waiting", 0))}
FL = "/root/e066/b32mv3/status_floor.json"
try:
    fl = json.load(open(FL))
except Exception:
    fl = {}
for k in ("first_done", "first_correct", "pairs_answered", "errored", "deaths", "restarts", "retries_max", "ooms"):
    out[k] = max(out[k] or 0, fl.get(k, 0))
json.dump({k: out[k] for k in ("first_done", "first_correct", "pairs_answered", "errored", "deaths", "restarts", "retries_max", "ooms")}, open(FL, "w"))
auto = (f"blew up: {out['ooms']} OOM(s), {out['deaths']} server death(s), {out['restarts']} restart(s); "
        + ("serving again" if s["ups"] > s["died"] else "restarting")) if out["deaths"] or out["ooms"] else \
       (f"stable; {out['retries_max']} allocator retries, cap-peak min {min(cp):.2f} GiB" if cp else "starting")
out["verdict"] = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] else auto
print(json.dumps(out))
