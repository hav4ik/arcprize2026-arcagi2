"""score_job.py <run> <workload> [--live]: reads /root/kwork_<run>/dsv41_logs/attempts.jsonl and
/root/queue/wl/<workload>_solutions.json; writes /root/nbruns/<run>/score.json (per-task results + summary, dev/holdout halves
when /root/queue/wl/<workload>_{dev,holdout}_ids.txt exist). Also checks submission.json (the notebook's own output) when it exists."""
import json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import arcscore as A
run, wl = sys.argv[1], sys.argv[2]; W = f"/root/kwork_{run}"; R = f"/root/nbruns/{run}"
sol = json.load(open(f"/root/queue/wl/{wl}_solutions.json"))
recs = A.load_jsonl(f"{W}/dsv41_logs/attempts.jsonl")
res = A.task_results(recs, sol)
out = {"run": run, "workload": wl, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "n_records": len(recs),
       "errored_records": sum(1 for r in recs if r.get("error") or r.get("stop_reason") == "error"),
       "summary": A.summarize(res), "per_task": res}
for half in ("dev", "holdout"):
    p = f"/root/queue/wl/{wl}_{half}_ids.txt"
    if os.path.exists(p):
        out[half] = A.summarize(res, set(open(p).read().split()))
sp = f"{W}/submission.json"
if os.path.exists(sp):  # the notebook's own submission, Kaggle scoring
    try:
        sub = json.load(open(sp)); tot = 0
        for t, s in sol.items():
            rows = sub.get(t, []); ok = 0
            for q, g in enumerate(s):
                if q < len(rows) and g in (rows[q].get("attempt_1"), rows[q].get("attempt_2")): ok += 1
            tot += ok / len(s)
        out["submission_json_score"] = round(100 * tot / len(sol), 2)
    except Exception as e:
        out["submission_json_error"] = repr(e)
os.makedirs(R, exist_ok=True)
json.dump(out, open(f"{R}/score.json.tmp", "w")); os.replace(f"{R}/score.json.tmp", f"{R}/score.json")
print(json.dumps({k: v for k, v in out.items() if k != "per_task"}))
