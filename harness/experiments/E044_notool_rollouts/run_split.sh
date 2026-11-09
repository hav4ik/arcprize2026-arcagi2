#!/usr/bin/env bash
# E044: run one batch split round-robin over providers, each in list order (partial runs stay prefixes).
# PROVIDERS="DeepInfra:128,InferenceNet:64,deepseek:48" (name:workers; deepseek = DeepSeek API, else OpenRouter provider).
# Usage: run_split.sh <name> <challenges> <solutions> <pairs_file>. Done pairs in any provider's file are skipped.
set -euo pipefail
ROOT=/home/vu/Workspace/ARC-AGI-2; HERE=$ROOT/experiments/E044_notool_rollouts
if [ -z "${NO_PID:-}" ]; then echo $$ > "$HERE/run.pid"; fi
PROVIDERS=${PROVIDERS:-DeepInfra:128,deepseek:48}
$ROOT/.venv/bin/python - "$1" "$4" "$PROVIDERS" <<'PY'
import glob, json, sys
name, pairs, provs = sys.argv[1], open(sys.argv[2]).read().strip().split(","), [p.split(":")[0] for p in sys.argv[3].split(",")]
done = set()
for f in glob.glob(f"/home/vu/Workspace/ARC-AGI-2/evals/results/e044_notool_*_{name}.jsonl"):
    done |= {f"{r['task_id']}:{r['pair']}" for r in map(json.loads, open(f)) if "error" not in r}
rest = [x for x in pairs if x not in done]
H = "/home/vu/Workspace/ARC-AGI-2/experiments/E044_notool_rollouts/"
for k, p in enumerate(provs):
    mine = [x for i, x in enumerate(rest) if i % len(provs) == k]
    open(f"{H}split_{name}_{p.lower()}.txt", "w").write(",".join(mine))
    print(f"{name}: {len(done)} done, {len(rest)} left -> {p} {len(mine)}")
PY
for pw in ${PROVIDERS//,/ }; do
  p=${pw%%:*}; w=${pw##*:}
  PROVIDER=$p WORKERS=$w "$HERE/launch.sh" "$1" "$2" "$3" "$HERE/split_$1_$(echo "$p" | tr 'A-Z' 'a-z').txt" &
done
wait
