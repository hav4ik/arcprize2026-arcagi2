#!/bin/bash
# replica_watch.sh (REPLICA_RUN, REPLICA_DATA) <until UTC "YYYY-mm-dd HH:MM"> <unused> [short|full]: replica_wait.sh, but the 60 s poll
# returns early only on crit_check.py's CRIT (retry/OOM, cap - reserved < 0.5, death/restart, errored attempt, RAM > 155),
# END (notebook done) or STALE (sampler restarted); EARLY/INFO alert lines no longer wake it. Original header: poll
# every 60 s; a new samp.alerts line -> return at once; sampler heartbeat older than 180 s -> restart it, report, return.
# Then: alerts, the latest sample (cap - reserved / cap - peak per rank, retries, OOMs, Kaggle-eq RAM, engine,
# supervisor), the dashboard sync, and the hour-by-hour comparison with V12 (hourly_vs_v12.py). full: also v12_report.py.
T=$(date -u -d "$1" +%s); K=$2; MODE=${3:-short}; RUN=${REPLICA_RUN:?}
P=/home/vu/Workspace/ARC-AGI-2/experiments/E062_manh_vllm_fork/per_layer/phase_h; D=${REPLICA_DATA:?}
SC=/tmp/claude-1000/-home-vu-Workspace-ARC-AGI-2/81e68821-e1f4-4b05-bad6-9a606f77ad54/scratchpad/sftv01
SSH="timeout 60 ssh -o BatchMode=yes -o ConnectTimeout=15 -p 18070 root@213.173.105.27"
while [ $(date +%s) -lt $T ]; do
  c=$($SSH "CRIT_CAP_RES=${CRIT_CAP_RES:-0.5} CRIT_RAM=${CRIT_RAM:-155} CRIT_SLOPE=${CRIT_SLOPE:--99} CRIT_SMI_FREE=${CRIT_SMI_FREE:-0} python3 -I /root/sftv01/scripts/crit_check.py $RUN" 2>/dev/null)
  case "$c" in
    CRIT*) echo "$c"; break;;
    END*) echo "$c"; break;;
    STALE*) echo "$c: restarting the sampler"
      $SSH "pgrep -f '^python3 -I /root/e062pl/v12_sampler.py /root/kwork_$RUN' || (setsid nohup python3 -I /root/e062pl/v12_sampler.py /root/kwork_$RUN /root/nbruns/$RUN /root/nbruns/$RUN/samp >> /root/nbruns/$RUN/samp.out 2>&1 < /dev/null &); sleep 5; pgrep -f '^python3 -I /root/e062pl/v12_sampler.py /root/kwork_$RUN'"; break;;
  esac
  sleep 60
done
echo "last check: ${c:-none}"
$SSH "echo '== alerts'; tail -3 /root/nbruns/$RUN/samp.alerts 2>/dev/null | cut -c1-300; echo \"== \$(date -u +%T)\"; tail -2 /root/nbruns/$RUN/nb.timeline; tail -1 /root/nbruns/$RUN/samp.jsonl | python3 -c '
import json,sys; r=json.loads(sys.stdin.read())
c = r.get(\"alloc_cap\") or {}; v = r.get(\"vllm_gpu_mem\") or {}
print(\"cap-reserved\", {k: x[\"cap_minus_reserved_now\"] for k, x in c.items()}, \"cap-peak\", {k: x[\"cap_minus_peak\"] for k, x in c.items()}, \"retries\", {k: x[\"retries\"] for k, x in v.items()}, \"ooms\", sum(x[\"ooms\"] for x in v.values()))
print(\"kaggle-eq (cgroup)\", r.get(\"kaggle_eq_gib\"), r.get(\"cgroup_gib\"), \"| rss\", r.get(\"rss_gib\"), \"| engine\", r.get(\"engine_last\"), \"| supervisor\", (r.get(\"supervisor\") or {}).get(\"start\"), (r.get(\"supervisor\") or {}).get(\"state\"))'; df -h / | tail -1; grep -a '] server:' /root/kwork_$RUN/dsv41_logs/solver.log | tail -1 | cut -c1-220"
if [ "$MODE" = full ]; then
  $SSH "python3 -I /root/e062pl/v12_report.py /root/nbruns/$RUN/samp.jsonl /root/kwork_$RUN/dsv41_logs /root/data/evalx2t_s2026_solutions.json \$(grep -aoE 'GPU KV cache size: [0-9,]+' /root/kwork_$RUN/dsv41_logs/server/server.log | head -1 | grep -oE '[0-9,]+' | tr -d ,) | grep -A12 '^slopes'"
fi
echo "== dashboard sync"; $P/scripts/replica_dash_sync.sh 2>&1 | tail -1
T0=$($SSH "python3 -c \"import os; print(os.path.getmtime('/root/nbruns/$RUN/notebook.sha256'))\"")
NOW_H=$(python3 -c "import time; print(round((time.time() - $T0) / 3600, 2))")
mkdir -p $D/dsv41_logs; for f in attempts.jsonl metrics.jsonl; do [ -e $D/dsv41_logs/$f ] || : > $D/dsv41_logs/$f; done
echo "== hourly vs V12 (Kaggle h now $NOW_H)"
python3 $P/scripts/hourly_vs_v12.py /home/vu/Workspace/ARC-AGI-2/experiments/E055_kaggle_replica/toolkit/workloads/evalx2t_s2026_solutions.json \
  $D/dsv41_logs/attempts.jsonl $D/tps.tsv $D/nbruns/$RUN/samp.jsonl $T0 \
  $SC/v12ref/replicas/manh_v12/attempts.jsonl $P/manh_v12/data/tps.tsv $SC/v12ref/replicas/manh_v12/samp.jsonl 1793325064.3 $NOW_H
echo "== serving metrics vs V12 (hourly_extra_vs_v12.py)"
python3 $P/scripts/hourly_extra_vs_v12.py /home/vu/Workspace/ARC-AGI-2/experiments/E055_kaggle_replica/toolkit/workloads/evalx2t_s2026_solutions.json \
  $D/dsv41_logs/metrics.jsonl $D/dsv41_logs/attempts.jsonl $D/nbruns/$RUN/samp.jsonl $T0 \
  $P/manh_v12/data/dsv41_logs/metrics.jsonl $SC/v12ref/replicas/manh_v12/attempts.jsonl $SC/v12ref/replicas/manh_v12/samp.jsonl 1793325064.3 $NOW_H
