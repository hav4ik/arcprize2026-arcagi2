"""Prepend one-line events for the live FINAL run to events.json (newest first); state in events_auto_state.json.
Prints the number of new events."""
import json, os, re, hashlib, datetime as dt
D = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN, BASE = "V12 + images FINAL", "Manh V12 replica"
DATA = "/home/vu/Workspace/ARC-AGI-2/experiments/E062_manh_vllm_fork/per_layer/phase_h/v12_img/data"
st_path = f"{D}/events_auto_state.json"
S = json.load(open(st_path)) if os.path.exists(st_path) else {"hours": [], "retries": 0, "deaths": 0, "restarts": 0, "ooms": 0, "gate": []}
status = json.load(open(f"{DATA}/status.json"))
h = status["kaggle_h"]; now = dt.datetime.utcnow().strftime("%H:%M")
ev = []
def add(kind, text, hh=None): ev.append({"utc": now, "run": RUN, "kaggle_h": h if hh is None else hh, "kind": kind, "event": text})
for k, kind, word in (("deaths", "bad", "death"), ("restarts", "warn", "restart"), ("ooms", "bad", "CUDA OOM")):
    if status.get(k, 0) > S[k]:
        add(kind, f"{status[k]} {word}(s) so far (was {S[k]}); verdict: {status.get('verdict','')[:160]}"); S[k] = status[k]
# allocator retries
mem_re = re.compile(r"Worker_TP(\d).*?GPU memory: .*?, (\d+) allocator retries, (\d+) OOMs")
mx = S["retries"]
try:
    for line in open(f"{DATA}/gpumem.log", errors="replace"):
        m = mem_re.search(line)
        if m: mx = max(mx, int(m[2]))
except OSError: pass
if mx > S["retries"]:
    add("warn", f"Allocator retries rose to {mx} (was {S['retries']}) on the worst rank."); S["retries"] = mx
# KV gate lines from the solver log
try:
    for line in open(f"{DATA}/dsv41_logs/solver.log", errors="replace"):
        if False:
            k = hashlib.md5(re.sub(r"^\[[^\]]*\]", "", line).strip()[:80].encode()).hexdigest()
            if k not in S["gate"] and len(S["gate"]) < 40:
                S["gate"].append(k); add("info", "KV gate: " + line.strip()[:200])
except OSError: pass
# hourly score vs baseline
acc = json.load(open(f"{D}/acc.json"))
def at(run, t):
    r = [x for x in acc if x["run"] == run and x["t_h"] <= t + 1e-9]
    return r[-1] if r else None
for hr in range(0):  # hourly rows off (30-min cadence, real events only)
    if h >= hr and hr not in S["hours"]:
        a, b = at(RUN, hr), at(BASE, hr)
        if a and b:
            add("info", f"Hour {hr}: score so far {a['score_so_far']} ({a['tasks_answered']} tasks, first-attempt acc {a['first_attempt_acc']}%) vs V12 {b['score_so_far']} ({b['tasks_answered']} tasks, {b['first_attempt_acc']}%) at the same time.", hr)
        S["hours"].append(hr)
if ev:
    old = json.load(open(f"{D}/events.json"))
    json.dump(list(reversed(ev)) + old, open(f"{D}/events.json", "w"), indent=1)
json.dump(S, open(st_path, "w"))
print(len(ev))
