#!/bin/bash
# E034 A: the E032 notebook agent (E031 mode) on the same server, 25 min at c24 on the probe-48 tasks; then B (v4b agent) on the box.
cd /home/vu/Workspace/ARC-AGI-2
J=/home/vu/.claude/jobs/821d4fe1/tmp/box; S="ssh -o BatchMode=yes -o ConnectTimeout=20 -p 40095 root@162.210.22.189"
X=/home/vu/.claude/jobs/821d4fe1/tmp/v5e032_extract
m() { $S "curl -s localhost:8001/metrics" 2>/dev/null | awk '/^vllm:prefix_cache_(queries|hits)_total/ {split($1,a,"{"); s[a[1]]+=$2} /^vllm:num_preemptions_total/ {p+=$2} END {printf "%.0f %.0f %.0f\n", s["vllm:prefix_cache_queries_total"], s["vllm:prefix_cache_hits_total"], p}'; }
echo "== A: E032 notebook agent (E031 mode) $(date -u +%H:%M)"
read q0 h0 p0 <<< "$(m)"; w0=$($S "grep -c 'Avg generation' /root/arc/runs/e034_cache/server.log" 2>/dev/null | tail -1)
python3 - <<'PY'
import json
s = json.load(open("experiments/E027_l4_throughput/e032_probe48_settings.json"))
s.update(ARC_LOCAL_FORCE_MIN=25, ARC_LOCAL_FORCE_CAP_MIN=28, ARC_LOCAL_HARD_MIN=3, ARC_LOCAL_WALL_MIN=32)
json.dump(s, open("/home/vu/.claude/jobs/821d4fe1/tmp/box/e034_e031mode_settings.json", "w"), indent=1)
PY
TASKS_OVERRIDE=$(cat $J/probe48_tasks.txt) NT=71 timeout 3000 $J/smoke_sft_solver.sh $X $J/e034_e031mode_settings.json 25 24 > $J/e034_A.log 2>&1
read q1 h1 p1 <<< "$(m)"
echo "E031-mode agent: prefix hit $(python3 -c "print(round(($h1-$h0)/max(1,$q1-$q0),3))") over $(( (q1-q0)/1000 ))k prompt tokens | preemptions $((p1-p0))"
$S "grep -E 'Avg generation' /root/arc/runs/e034_cache/server.log | tail -n +$((w0+12)) | sed -E 's/.*Avg prompt throughput: ([0-9.]+).*Avg generation throughput: ([0-9.]+).*Running: ([0-9]+).*/\1 \2 \3/' | awk '\$3>=20 {p+=\$1; g+=\$2; r+=\$3; n++} END {if(n) printf \"windows %d gen %.0f per-stream %.2f prompt %.0f\n\", n, g/n, g/r, p/n}'" 2>/dev/null
$S "mkdir -p /root/arc/runs/e034_cache/e031_agent_logs && cp -r /root/arc/work_sft/arc_logs /root/arc/runs/e034_cache/e031_agent_logs/" 2>/dev/null
timeout 3000 $S "/root/arc/ab_harness_box.sh" 2>&1 | grep -v "Welcome\|Have fun\|AI agents"
echo E034_AB_DONE
