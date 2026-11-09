#!/usr/bin/env bash
# E020 gateway launches. Usage: launch.sh <arm> <shard0|shard1|all> <attempts> <concurrency> [extra args...]
# Each arm x shard writes its own resumable file evals/results/qwen_tools_e020_<arm>_<shard>.jsonl (+ log_...txt);
# shard0/shard1 = alternate pairs of the 167 (experiments/E020_qwen_harness_80/shard*.txt), so a process can be added later.
set -euo pipefail
cd "$(dirname "$0")/../.."
arm=$1; shard=$2; attempts=$3; conc=$4; shift 4
COMMON=(--base-url http://127.0.0.1:30100/v1 --model nvidia/Qwen3.8-27B-NVFP4 --routing-key-header x-smg-routing-key
        --sandbox-url http://172.18.0.3:6000 --attempts "$attempts" --concurrency "$conc" --session-mem-mb 2048 --request-timeout 1200)
PROG=(--answer-mode program)
KAG=(--tier medium --hard 48000 --soft 24000 --context 131072 --record-alternatives)  # Kaggle submission defaults
case $arm in
  K0)     ARGS=("${KAG[@]}" --record-train-check) ;;
  K1)     ARGS=("${KAG[@]}" --record-train-check --turn-think-cap 8192) ;;
  K2)     ARGS=("${KAG[@]}" "${PROG[@]}" --turn-think-cap 8192 --prompt lean2progroles --task-summary --helpers --feedback rich) ;;
  K2nc)   ARGS=("${KAG[@]}" "${PROG[@]}" --prompt lean2progroles --task-summary --helpers --feedback rich) ;;
  K3)     ARGS=(--tier xhigh --hard 48000 --soft 24000 --context 131072 --record-alternatives "${PROG[@]}" --turn-think-cap 8192
                --prompt lean2progroles --task-summary --helpers --feedback rich) ;;
  x96p)   ARGS=("${PROG[@]}" --tier xhigh --hard 96000 --soft 48000 --prompt lean2progroles --task-summary --helpers --feedback rich) ;;
  x96pc8) ARGS=("${PROG[@]}" --tier xhigh --hard 96000 --soft 48000 --prompt lean2progroles --task-summary --helpers --feedback rich
                --turn-think-cap 8000 --turn-think-grace 2048) ;;
  m96pc8) ARGS=("${PROG[@]}" --tier medium --hard 96000 --soft 48000 --prompt lean2progroles --task-summary --helpers --feedback rich
                --turn-think-cap 8000 --turn-think-grace 2048) ;;
  x96pal) ARGS=("${PROG[@]}" --tier xhigh --hard 96000 --soft 48000 --prompt lean2progroles --task-summary --helpers --feedback rich
                --test-audit --lint) ;;
  x96pex) ARGS=("${PROG[@]}" --tier xhigh --hard 96000 --soft 48000 --prompt lean2progroles --task-summary --helpers --feedback rich
                --worked-example) ;;
  x96)    ARGS=(--tier xhigh --hard 96000 --soft 48000) ;;
  *) echo "unknown arm $arm"; exit 1 ;;
esac
# SLOTS=auto:N (or url=n,...) uses the shared worker-slot scheduler (evals/arc_harness/slots.py): direct per-worker
# URLs, per-box caps shared by all slot-aware processes (state: experiments/E019_qwen_program_answers/router/slots.json).
[ -n "${SLOTS:-}" ] && COMMON+=(--worker-slots "$SLOTS")
[ -n "${SLOT_STATE:-}" ] && COMMON+=(--slot-state "$SLOT_STATE")
SH=()
[ "$shard" != all ] && SH=(--pairs-file experiments/E020_qwen_harness_80/$shard.txt)
name=e020_${arm}_${shard}
out=evals/results/qwen_tools_${name}.jsonl
nohup .venv/bin/python evals/run_qwen_tools.py "${COMMON[@]}" "${ARGS[@]}" "${SH[@]}" --out "$out" --run-name "$name" "$@" \
  >> evals/results/log_qwen_tools_${name}.txt 2>&1 &
echo "$! $arm -> $out"
