#!/bin/bash
# Dashboard data for the B32-M v3 replica (overwritten each time): data/gpumem.log = the original "GPU memory:",
# "Capped the PyTorch allocator" and "Avg generation throughput" lines of server.log, then of server_restart1.log, 2, ... in order; data/ram.tsv = the
# sampler's Kaggle-eq RAM, 1 point per 10 min (ram_tsv.py); data/status.json (status_json.py; verdict = argv 1 or automatic).
# Also the solver's attempts.jsonl and solver.log (copies). Read-only on the pod. Usage: export_dash.sh [verdict]
S=$(dirname "$0"); read IP PORT < $S/pod3_ssh.txt; O=/home/vu/Workspace/ARC-AGI-2/experiments/E062_manh_vllm_fork/per_layer/pod3_runs/b32m_v3/data; mkdir -p $O
timeout 120 ssh -o BatchMode=yes -p $PORT root@$IP 'cd /kaggle/working/dsv41_logs && for f in server.log $(ls server_restart*.log 2>/dev/null | sort -V); do grep -aE "GPU memory:|Capped the PyTorch allocator|Avg generation throughput" $f; done' > $O/gpumem.log.tmp && mv $O/gpumem.log.tmp $O/gpumem.log
timeout 120 ssh -o BatchMode=yes -p $PORT root@$IP '/usr/bin/python3 /root/e066/b32mv3/ram_tsv.py /root/e066/b32mv3/mon.jsonl 1793331040' > $O/ram.tsv.tmp && mv $O/ram.tsv.tmp $O/ram.tsv
timeout 120 ssh -o BatchMode=yes -p $PORT root@$IP "/usr/bin/python3 /root/e066/b32mv3/status_json.py $(printf '%q' "${1:-}")" > $O/status.json.tmp && [ -s $O/status.json.tmp ] && mv $O/status.json.tmp $O/status.json
timeout 120 scp -q -P $PORT root@$IP:/kaggle/working/dsv41_logs/attempts.jsonl $O/attempts.jsonl.tmp && mv $O/attempts.jsonl.tmp $O/attempts.jsonl
timeout 120 scp -q -P $PORT root@$IP:/kaggle/working/dsv41_logs/solver.log $O/solver.log.tmp && mv $O/solver.log.tmp $O/solver.log
echo "status.json: $(cat $O/status.json)"
echo "gpumem.log $(wc -l < $O/gpumem.log) lines; ram.tsv $(($(wc -l < $O/ram.tsv) - 1)) points; last: $(tail -1 $O/ram.tsv)"
