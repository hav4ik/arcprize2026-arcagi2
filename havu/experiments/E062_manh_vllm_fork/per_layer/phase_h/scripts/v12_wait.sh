#!/bin/bash
# v12_wait.sh <until UTC "YYYY-mm-dd HH:MM"> <known alert count> <short|full>: poll every 60 s: new alert lines -> return
# at once; sampler heartbeat older than 180 s -> restart the sampler, report it, return. At the deadline print the
# short (or full) report. Full reports also copy the data to phase_h/manh_v12/data/.
T=$(date -u -d "$1" +%s); K=$2; MODE=${3:-short}; D=/home/vu/Workspace/ARC-AGI-2/experiments/E062_manh_vllm_fork/per_layer/phase_h/manh_v12/data
SSH="timeout 60 ssh -o BatchMode=yes -o ConnectTimeout=15 -p 18070 root@213.173.105.27"
while [ $(date +%s) -lt $T ]; do
  out=$($SSH 'echo "N $(cat /root/nbruns/m12_rerun/samp.alerts 2>/dev/null | wc -l) HB $(( $(date +%s) - $(cut -d" " -f1 /root/nbruns/m12_rerun/samp.heartbeat 2>/dev/null || echo 0) ))"' 2>/dev/null)
  n=$(echo "$out" | awk '{print $2}'); hb=$(echo "$out" | awk '{print $4}')
  if [ -n "$hb" ] && [ "$hb" -gt 180 ]; then
    echo "SAMPLER HEARTBEAT STALE (${hb}s): restarting"
    $SSH 'pgrep -f "^python3 -I /root/e062pl/v12_sampler.py" || (setsid nohup python3 -I /root/e062pl/v12_sampler.py /root/kwork_m12_rerun /root/nbruns/m12_rerun /root/nbruns/m12_rerun/samp >> /root/nbruns/m12_rerun/samp.out 2>&1 < /dev/null &); sleep 5; pgrep -f "^python3 -I /root/e062pl/v12_sampler.py"; tail -3 /root/nbruns/m12_rerun/samp.out'
    MODE=short; break
  fi
  [ -n "$n" ] && [ "$n" -gt "$K" ] && { echo "NEW ALERT ($n lines)"; MODE=short; break; }
  sleep 60
done
$SSH 'echo "== alerts"; cat /root/nbruns/m12_rerun/samp.alerts 2>/dev/null | cut -c1-700; echo "== $(date -u +%T)"; tail -1 /root/nbruns/m12_rerun/samp.jsonl | python3 -c "
import json,sys; r=json.loads(sys.stdin.read())
c = r.get(\"alloc_cap\") or {}; v = r.get(\"vllm_gpu_mem\") or {}
print(\"cap-peak\", {k: x[\"cap_minus_peak\"] for k, x in c.items()}, \"retries\", {k: x[\"retries\"] for k, x in v.items()}, \"ooms\", sum(x[\"ooms\"] for x in v.values()))
print(\"kaggle-eq (cgroup)\", r.get(\"kaggle_eq_gib\"), \"| engine\", r.get(\"engine_last\"), \"| supervisor\", (r.get(\"supervisor\") or {}).get(\"start\"), (r.get(\"supervisor\") or {}).get(\"state\"))"; python3 -I /root/e062pl/first_correct.py /kaggle/working/dsv41_logs/attempts.jsonl /root/data/evalx2t_s2026_solutions.json; grep -a "] server:" /kaggle/working/dsv41_logs/solver.log | tail -1 | cut -c1-200'
if [ "$MODE" = full ]; then
  $SSH 'P=1608578; python3 -I /root/e062pl/v12_report.py /root/nbruns/m12_rerun/samp.jsonl /kaggle/working/dsv41_logs /root/data/evalx2t_s2026_solutions.json $P | grep -v "^| 0[12]:"; T0=$(python3 -c "import os; print(os.path.getmtime(\"/root/nbruns/m12_rerun/notebook.sha256\"))"); python3 -I /root/e062pl/v12_tput.py /kaggle/working/dsv41_logs $T0'
  mkdir -p $D && timeout 280 ssh -o BatchMode=yes -p 18070 root@213.173.105.27 'cd /root && tar -cf - nbruns/m12_rerun -C /root/kwork_m12_rerun dsv41_logs/attempts.jsonl dsv41_logs/solver.log dsv41_logs/status.json dsv41_logs/metrics.jsonl $(cd /root/kwork_m12_rerun && find dsv41_logs -name "*.log" -o -name "supervisor*.json" | grep -v traces) 2>/dev/null' | tar -C $D -xf - 2>/dev/null; echo "copied to $D: $(du -sh $D | cut -f1)"
fi
