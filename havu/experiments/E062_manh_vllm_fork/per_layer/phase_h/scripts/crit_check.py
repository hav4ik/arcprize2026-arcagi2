"""Critical-condition check for a replica run (pod side, read-only). Prints one line: "OK ..." or "CRIT <reasons>" or
"END ..." (notebook finished) or "STALE ..." (sampler heartbeat > 180 s). Conditions (coordinator, overnight duties):
any allocator retry or OOM; cap - reserved < 0.5 GiB on any rank; a server death, supervisor restart or state not up;
an errored attempt (error record or stop=error); Kaggle-eq RAM > 155 GiB.
usage: crit_check.py RUN"""
import json, os, sys, time
run = sys.argv[1]; R = f"/root/nbruns/{run}"; L = f"/root/kwork_{run}/dsv41_logs"
tl = open(f"{R}/nb.timeline").read() if os.path.exists(f"{R}/nb.timeline") else ""
if "NOTEBOOK_DONE" in tl:
    print("END " + [l for l in tl.splitlines() if "NOTEBOOK_DONE" in l][-1][:200]); sys.exit(0)
try:
    hb = time.time() - float(open(f"{R}/samp.heartbeat").read().split()[0])
except Exception:
    hb = 1e9
if hb > 180:
    print(f"STALE sampler heartbeat {hb:.0f} s"); sys.exit(0)
last = json.loads(open(f"{R}/samp.jsonl").read().strip().splitlines()[-1])
v = last.get("vllm_gpu_mem") or {}; c = last.get("alloc_cap") or {}; sup = last.get("supervisor") or {}
why = []
r = {k: x["retries"] for k, x in v.items() if x.get("retries")}; o = {k: x["ooms"] for k, x in v.items() if x.get("ooms")}
if r: why.append(f"allocator retries {r}")
if o: why.append(f"OOMs {o}")
cr = {k: x["cap_minus_reserved_now"] for k, x in c.items() if x.get("cap_minus_reserved_now") is not None}
CR_MIN = float(os.environ.get("CRIT_CAP_RES", "0.5")); RAM_MAX = float(os.environ.get("CRIT_RAM", "155"))
SLOPE_MIN = float(os.environ.get("CRIT_SLOPE", "-99")); SMI_MIN = float(os.environ.get("CRIT_SMI_FREE", "0"))
low = {k: x for k, x in cr.items() if x < CR_MIN}
if low: why.append(f"cap - reserved < {CR_MIN}: {low}")
smi = {g["gpu"]: round(g["free_mib"] / 1024, 2) for g in last.get("gpus") or [] if g.get("used_mib", 0) > 4000 and g["free_mib"] / 1024 < SMI_MIN}
if smi: why.append(f"nvidia-smi free < {SMI_MIN} GiB: {smi}")
T_NB = os.path.getmtime(f"{R}/notebook.sha256")
if SLOPE_MIN > -99 and (last["t"] - T_NB) / 3600 > 1.5:  # least-squares slope of each rank's cap - reserved over the last hour
    pts = {}
    for l in open(f"{R}/samp.jsonl"):
        try: q = json.loads(l)
        except ValueError: continue
        if q["t"] < last["t"] - 3600: continue
        for k, x in (q.get("alloc_cap") or {}).items():
            if x.get("cap_minus_reserved_now") is not None: pts.setdefault(k, []).append(((q["t"] - last["t"]) / 3600, x["cap_minus_reserved_now"]))
    for k, xs in pts.items():
        if len(xs) > 20:
            mx = sum(a for a, _ in xs) / len(xs); my = sum(b for _, b in xs) / len(xs)
            den = sum((a - mx) ** 2 for a, _ in xs) or 1
            sl = sum((a - mx) * (b - my) for a, b in xs) / den
            if sl < SLOPE_MIN: why.append(f"{k} cap - reserved slope {sl:.3f} GiB/h < {SLOPE_MIN}")
if int(sup.get("start") or 0) > 0 or (sup.get("state") not in (None, "up")): why.append(f"supervisor start {sup.get('start')} state {sup.get('state')}")
sl = open(f"{L}/solver.log", errors="replace").read() if os.path.exists(f"{L}/solver.log") else ""
if sl.count("server died"): why.append(f"server died x{sl.count('server died')}")
ne = 0
if os.path.exists(f"{L}/attempts.jsonl"):
    for l in open(f"{L}/attempts.jsonl"):
        if '"error": "' in l or '"stop_reason": "error"' in l: ne += 1
if ne: why.append(f"errored attempt records {ne}")
keq = last.get("kaggle_eq_gib") or 0
if keq > RAM_MAX: why.append(f"Kaggle-eq RAM {keq} GiB > {RAM_MAX}")
state = f"utc {last.get('utc')} cap-res {cr} keq {keq} retries 0 errored {ne}"
print(("CRIT " + "; ".join(why) + " | " + state) if why else ("OK " + state))
