#!/bin/bash
cd /home/vu/Workspace/ARC-AGI-2
J=/home/vu/.claude/jobs/821d4fe1/tmp/box; S="ssh -o BatchMode=yes -o ConnectTimeout=20 -p 40095 root@162.210.22.189"
X=/home/vu/.claude/jobs/821d4fe1/tmp/v5e032_extract
timeout 1500 $S "/root/arc/e032_serve_kaggleflags.sh /workspace/models/e032_sft_fp8block e032_probe48_kflags" 2>&1 | grep -v "Welcome\|Have fun\|AI agents"
TASKS_OVERRIDE=$(cat $J/probe48_tasks.txt) NT=71 timeout 12000 $J/smoke_sft_solver.sh $X experiments/E027_l4_throughput/e032_probe48_settings.json 120 24 > $J/e032_probe48_kflags.log 2>&1
tail -4 $J/e032_probe48_kflags.log | cut -c1-500
timeout 300 $S "cd /root/arc/work_sft && tar czf /root/e032_probe48_kflags.tgz summary_local.json submission.json solver.log arc_logs" 2>/dev/null
D=experiments/E027_l4_throughput/e032_probe48_kflags; mkdir -p $D && scp -q -P 40095 -o BatchMode=yes root@162.210.22.189:/root/e032_probe48_kflags.tgz $D/ && tar xzf $D/e032_probe48_kflags.tgz -C $D/
timeout 30 $S "grep -E 'Avg generation' /root/arc/runs/e032_probe48_kflags/server.log | sed -E 's/.*Avg prompt throughput: ([0-9.]+).*Avg generation throughput: ([0-9.]+).*Running: ([0-9]+).*Prefix cache hit rate: ([0-9.]+)%.*/\1 \2 \3 \4/' | awk '\$3>=20 {g+=\$2; r+=\$3; n++} {h=\$4} END {printf \"server: gen %.0f tok/s, per-stream %.2f (%d windows >=20 running), final prefix hit %.1f%%\n\", g/n, g/r, n, h}'" 2>&1 | grep -v "Welcome\|Have fun\|AI agents"
echo RERUN_DONE
