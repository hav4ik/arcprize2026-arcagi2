#!/bin/bash
# E055 poll: both 12 h replicas; mirror attempts/metrics, print score so far (eval x2 solutions), KV/tok/s, restarts
D=$(cd "$(dirname "$0")" && pwd)/mirror
X=/home/vu/Workspace/ARC-AGI-2/experiments/E055_kaggle_replica
for b in "30092 root@213.173.105.28 v6rep_x2t $X/toolkit/workloads/evalx2t_s2026" "43647 root@157.157.221.29 candBf_x2t $X/toolkit/workloads/evalx2t_s2026"; do
  set -- $b; S="ssh -o ConnectTimeout=15 -o ServerAliveInterval=10 -o BatchMode=yes -p $1 $2"; M=$D/mirror_$3; mkdir -p $M
  echo "=== $3 $(date -u +%T)"
  timeout 120 $S "cd /root/runs/$3/out/dsv41_logs && tar cf - metrics.jsonl attempts.jsonl solver.log status.json 2>/dev/null" 2>/dev/null | tar xf - -C $M || echo SSH_FAIL
  grep -h "server:" $M/solver.log | tail -3 | cut -c1-170; timeout 60 $S "tail -1 /root/runs/$3/mem.log; tail -1 /root/runs/$3/memdetail.log 2>/dev/null | cut -c1-230" 2>/dev/null
  grep -hE "server died|restart|no restart|out of memory|admission closed|force deadline|local score|submission written \(final\)|done$" $M/solver.log | tail -4 | cut -c1-150
  python3 -c "import json; s=json.load(open('$M/status.json')); print({k: s.get(k) for k in ('elapsed_h','done','errors','restarts','running','queued','pairs_with_answer','pairs')})" 2>/dev/null
  python3 - "$M/metrics.jsonl" <<'PY'
import json, sys
w = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
if w:
    kv = max(x.get("kv", 0) for x in w); low = [x for x in w if x.get("elapsed_h", 0) > 0.4 and (x.get("prefix_hit") or 1) < 0.85 and x.get("uncached_tok_s", 0) > 500]
    print(f"windows {len(w)}, max KV {kv:.3f}, thrash-signature windows {len(low)}, mean gen tok/s {sum(x['gen_tok_s'] for x in w)/len(w):.0f}")
PY
  python3 $X/score.py $M/attempts.jsonl ${4}_challenges.json ${4}_challenges.json ${4}_solutions.json 2>/dev/null | head -3
done
