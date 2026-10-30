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
rk = s["ranks"].values()
cp = [r["cap"] - r["peak"] for r in rk if "cap" in r and "peak" in r]
sc = (s.get("scrape") or [{}])[-1]
out = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(s["t"])), "kaggle_h": round((s["t"] - T0) / 3600, 2),
       "first_done": done, "first_correct": corr, "pairs_answered": st.get("pairs_with_answer"), "errored": s["att_err"],
       "deaths": s["died"], "restarts": max(0, s["starts"] - 1), "retries_max": max([r.get("retries", 0) for r in rk] or [0]),
       "ooms": sum(r.get("ooms", 0) for r in rk), "running": int(sc.get("num_requests_running", 0)),
       "waiting": int(sc.get("num_requests_waiting", 0))}
auto = ("blew up: %d server death(s), %d restart(s)" % (out["deaths"], out["restarts"])) if out["deaths"] else \
       (f"stable; {out['retries_max']} allocator retries, cap-peak min {min(cp):.2f} GiB" if cp else "starting")
out["verdict"] = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] else auto
print(json.dumps(out))
