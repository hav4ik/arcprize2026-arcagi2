#!/bin/bash
# hourly_wait.sh <until UTC "YYYY-mm-dd HH:MM"> <known flag count>: poll the pod every 60 s; return early on a new FLAG
# line; at the deadline (or on a flag) print the report, the watcher summaries since the last call, the flags and
# the follow-up of the kill-test's re-queued jobs.
T=$(date -u -d "$1" +%s); K=$2
while [ $(date +%s) -lt $T ]; do
  n=$(timeout 30 ssh -o BatchMode=yes -o ConnectTimeout=15 -p 18070 root@213.173.105.27 'cat /root/nbruns/r1_rerun/watch.flags 2>/dev/null | wc -l' 2>/dev/null)
  [ -n "$n" ] && [ "$n" -gt "$K" ] && { echo "NEW FLAG"; break; }
  sleep 60
done
timeout 200 ssh -o BatchMode=yes -p 18070 root@213.173.105.27 'bash /root/e062pl/p2_report.sh /root/nbruns/r1_rerun 1617253 2>&1 | grep -v "^== GPU memory lines" | grep -v "gpu_worker.py"; echo "== watch.log (last 7)"; tail -7 /root/nbruns/r1_rerun/watch.log | cut -c1-330; echo "== flags"; cat /root/nbruns/r1_rerun/watch.flags | cut -c1-250; python3 -I - <<PY
import json, collections
rs = [json.loads(l) for l in open("/kaggle/working/dsv41_logs/attempts.jsonl")]
key = lambda r: (r["task_id"], r.get("pair", 0), r.get("attempt"))
er = [r for r in rs if r.get("error")]; fin = {key(r) for r in rs if not r.get("error")}
print("error records", len(er), "| of their jobs finalized OK since:", sum(key(r) in fin for r in er), "| error strings:", dict(collections.Counter(str(r["error"])[:70] for r in er)))
PY'
