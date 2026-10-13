"""E019 guard for the extra sample processes (pids in /tmp/e019_s1.pids), every 5 min from
evals/results/cache_gw_dev56.jsonl (windows with > 200k prompt tokens):
  * ALERT (logged, no kill) on KV peak > 0.92 in two consecutive windows, hit < 0.85 in two consecutive windows,
    or any retraction line;
  * SIGTERM the extra processes only when clearly broken: retractions in two consecutive windows.
Log: evals/results/log_s1_guard.txt (alerts are relayed to the coordinator by hand)."""
import json, os, signal, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
LOG = ROOT / "evals/results/cache_gw_dev56.jsonl"
start = time.time()
while True:
    time.sleep(300)
    recs = [json.loads(l) for l in LOG.read_text().splitlines() if l.strip()]
    per = {}
    for r in recs:
        if r["t"] >= start and "error" not in r and r["new_tokens"] + r["cached_tokens"] > 200000:
            per.setdefault(r["worker"], []).append(r)
    alerts, kill = [], None
    for w, rs in per.items():
        last2 = rs[-2:]
        if len(last2) == 2 and all(r.get("max_token_usage", r.get("token_usage", 0)) > 0.92 for r in last2):
            alerts.append(f"{w} KV>0.92 x2")
        if len(last2) == 2 and all(r["window_hit"] < 0.85 for r in last2):
            alerts.append(f"{w} hit<0.85 x2")
        if rs[-1]["retract_lines"] > 0:
            alerts.append(f"{w} retractions {rs[-1]['retract_lines']}")
        if len(last2) == 2 and all(r["retract_lines"] > 0 for r in last2):
            kill = f"{w} retractions in 2 windows"
    line = time.strftime("%H:%M:%S", time.gmtime()) + " " + json.dumps(
        {w: (rs[-1]["window_hit"], rs[-1].get("max_token_usage"), rs[-1].get("mean_running"), rs[-1].get("mean_gen_tps"))
         for w, rs in per.items()})
    if alerts:
        line += " ALERT " + "; ".join(alerts)
    if kill:
        for pid in Path("/tmp/e019_s1.pids").read_text().split():
            try:
                os.kill(int(pid), signal.SIGTERM)
            except ProcessLookupError:
                pass
        line += f" STOPPED extras: {kill}"
    with open(ROOT / "evals/results/log_s1_guard.txt", "a") as f:
        f.write(line + "\n")
    if kill:
        break
