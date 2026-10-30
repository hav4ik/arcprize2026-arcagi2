"""What does a run that OOMs every U hours of server uptime score under B32's solver restart path?
Replays a full no-OOM replica's attempts (order, wall time, correctness) through: 80 workers, an OOM after U h of server
uptime, a hang + reload outage, at most 3 restarts, JOB_TRIES 2 (an HTTP-500 kill costs a try; a second kill loses the
attempt), stop-new / force deadlines. Scores the ARC Prize way (per task: fraction of pairs right by either attempt).
usage: oom_restart_sim.py ATTEMPTS.jsonl SOLUTIONS.json [--uptime 1.87] [--outage 0.30] [--slow 1.0] [--first-up 0.24]"""
import argparse, heapq, json

ap = argparse.ArgumentParser()
ap.add_argument("attempts"); ap.add_argument("solutions")
ap.add_argument("--uptime", type=float, default=1.87, help="server uptime (h) until each OOM; 0 = never")
ap.add_argument("--outage", type=float, default=0.30, help="hang + reload (h) after each OOM")
ap.add_argument("--slow", type=float, default=1.0, help="multiply attempt wall times (Kaggle slower than the pod)")
ap.add_argument("--first-up", type=float, default=0.24, help="first server start (h after solver start)")
ap.add_argument("--workers", type=int, default=80); ap.add_argument("--restarts", type=int, default=3)
ap.add_argument("--ooms", type=int, default=99, help="number of OOMs before the server stays healthy")
ap.add_argument("--stop-new", type=float, default=10.67); ap.add_argument("--force", type=float, default=11.0)
a = ap.parse_args()
sol = json.load(open(a.solutions))
att = [json.loads(l) for l in open(a.attempts) if l.strip()]
def depth(x):
    d = 0
    while isinstance(x, list) and x: x = x[0]; d += 1
    return d
def pair_ok(r):
    t = r["task_id"]; n = len(sol[t]); p = r["pred"]
    preds = p if (n > 1 and depth(p) == 3) else [p]
    return [i < len(preds) and preds[i] == sol[t][i] for i in range(n)]
jobs = []
for r in att:
    dur = float(r.get("wall_s") or r.get("seconds")) / 3600 * a.slow
    start = float(r["t_end_h"]) - float(r.get("wall_s") or r.get("seconds")) / 3600
    jobs.append({"key": (r["task_id"], int(r["attempt"])), "dur": dur, "start": start, "ok": pair_ok(r), "tries": 0})
jobs.sort(key=lambda j: j["start"])  # the replica's own launch order
queue, running, done, lost = list(jobs), {}, {}, []
t, up_at, restarts = 0.0, a.first_up, 0
oom_at = up_at + a.uptime if a.uptime > 0 else 1e9
events = []
while True:
    if t >= a.force: break
    if t >= up_at:  # server up: fill free workers
        while queue and len(running) < a.workers and t < a.stop_new:
            j = queue.pop(0); running[id(j)] = (j, t)
    ends = [(st + j["dur"], k) for k, (j, st) in running.items()]
    nxt = min([e for e, _ in ends] + [oom_at, a.force] + ([up_at] if t < up_at else []))
    t = nxt
    for e, k in ends:
        if e <= t + 1e-12:
            j, _ = running.pop(k); done[j["key"]] = j["ok"]
    if t >= oom_at - 1e-12 and t < a.force:
        events.append(f"OOM at {t:.2f} h, {len(running)} in flight")
        for k, (j, _) in list(running.items()):
            j["tries"] += 1
            if j["tries"] >= 2: lost.append(j["key"])
            else: queue.insert(0, j)
        running.clear()
        if restarts < a.restarts and a.stop_new - t > a.outage:
            restarts += 1; up_at = t + a.outage; oom_at = up_at + a.uptime if restarts < a.ooms else 1e9
            events.append(f"  restart {restarts}: up at {up_at:.2f} h")
        else:
            events.append("  no restart left"); break
    if not running and (not queue or t >= a.stop_new) and t >= up_at: break
score = 0.0
for task, pairs in sol.items():
    best = [False] * len(pairs)
    for att_i in (0, 1):
        ok = done.get((task, att_i))
        if ok: best = [b or o for b, o in zip(best, ok)]
    score += sum(best) / len(pairs)
first_done = sum(1 for k in done if k[1] == 0)
print("\n".join(events))
print(f"finished attempts {len(done)}/{len(jobs)} (first attempts {first_done}/240), lost after 2 kills {len(lost)}, "
      f"unstarted/unfinished {len(jobs) - len(done) - len(lost)}")
print(f"score (pass@2, task-weighted): {100 * score / len(sol):.1f}")
