#!/bin/bash
# Run 3 hourly check: copy the solver logs (no traces), metrics scrape, sampler, mem.log and run.log from pod 3, then
# print supervisor events, server deaths, errors, attempts, GPU memory, host RAM and the per-window report.
# Read-only on the pod. Usage: pull_r3.sh [bucket_s=600]
S=$(dirname "$0"); read IP PORT < $S/pod3_ssh.txt; D=$S/r3copy; mkdir -p $D/out/dsv41_logs
timeout 120 ssh -o BatchMode=yes -p $PORT root@$IP 'cd /kaggle/working/dsv41_logs && tar czf - --exclude=traces --exclude="*/server/*.log" attempts.jsonl solver.log status.json metrics.jsonl e066_supervisor.json' > $D/logs.tgz && tar xzf $D/logs.tgz -C $D/out/dsv41_logs
for f in metrics_scrape.txt sampler.jsonl mem.log run.log; do timeout 120 scp -q -P $PORT root@$IP:/root/e066/r3/$f $D/; done
[ -f $D/evalx2t_s2026_solutions.json ] || cp $S/run2_final/evalx2t_s2026_solutions.json $D/
timeout 60 ssh -o BatchMode=yes -p $PORT root@$IP 'date -u +%T; for f in /kaggle/working/dsv41_logs/server/server.log /kaggle/working/dsv41_logs/r*/server/server.log; do [ -f $f ] && echo "$f: errors $(grep -acE "Traceback|EngineDead|out of memory|CUDA error" $f), 400s $(grep -ac " 400 " $f), 500s $(grep -ac " 500 " $f)"; done; nvidia-smi --query-gpu=index,memory.used,memory.free,power.draw,clocks.sm --format=csv,noheader; nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | tr "\n" " "; echo; free -g | sed -n 2p'
echo "== supervisor"; grep -a "E066 SUPERVISOR" $D/run.log | cut -c1-200; cat $D/out/dsv41_logs/e066_supervisor.json; echo
echo "== solver deaths/ups"; grep -aE "server died|server up after|\(wait [0-9]+\)" $D/out/dsv41_logs/solver.log | cut -c1-160
python3 - $D <<'PY'
import json, sys, collections
D = sys.argv[1]
R = [json.loads(l) for l in open(f"{D}/out/dsv41_logs/attempts.jsonl")]
ok = [r for r in R if not r.get("error")]
print(f"== attempts.jsonl: {len(R)} records, {len(ok)} finished without error; error records by kind:",
      dict(collections.Counter(r["error"][:60] for r in R if r.get("error"))))
G = 1048576
S = [json.loads(l) for l in open(f"{D}/sampler.jsonl") if l.strip()]
last = S[-1]
pk = max(S, key=lambda r: r["mem_kb"]["AnonPages"])
print(f"== host RAM: now AnonPages {last['mem_kb']['AnonPages'] / G:.1f} GiB, all-process RSS {last['all_rss_kb'] / G:.1f} GiB, "
      f"MemAvailable {last['mem_kb']['MemAvailable'] / G:.1f} GiB; run peak AnonPages {pk['mem_kb']['AnonPages'] / G:.1f} GiB at {pk['utc']}; "
      f"peak all RSS {max(r['all_rss_kb'] for r in S) / G:.1f} GiB; vLLM pids now {[p for p, _, _ in last['vllm']]}")
PY
st=$D/out/dsv41_logs/status.json; cat $st; echo
python3 $S/windows.py $D $D/metrics_scrape.txt $D/evalx2t_s2026_solutions.json ${1:-600} 2>&1 | tail -14
