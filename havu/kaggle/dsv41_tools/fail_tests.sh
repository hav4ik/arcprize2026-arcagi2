#!/bin/bash
# DSV41 Kaggle notebook (local_run.py) failure-mode tests on a Runpod 4xL4 with the real checkpoint. Each test uses a
# fresh working dir under /root/kaggle_sim/ft_<name>; results go to /root/kaggle_sim/fail_tests.log.
#   T1 never_up : a bogus vLLM flag -> server exits at startup, one restart, then the fallback stays; solver exits 0.
#   T2 kill     : 6 tasks; the server's process group is SIGKILLed once attempts run -> watchdog restart, errored
#                 attempts re-queued, force deadline cuts the rest, final submission valid.
#   T3 sigterm  : 6 tasks; SIGTERM to the solver once attempts run -> final write, server group stopped, exit.
# Usage: fail_tests.sh <model dir> [tests...]   (env: KIT, CHAL, SOL as in the defaults below)
M=$1; shift; TESTS=${*:-"never_up kill sigterm"}
S=/root/kaggle_sim; L=$S/fail_tests.log
log() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a $L; }
gpu_free() { [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)" -lt 1000 ]; }
check_sub() {  # $1 working dir: validity and how many rows differ from the fallback (test input)
  /usr/bin/python3 - "$1" <<'EOF'
import json, sys
w = sys.argv[1]
ch = json.load(open("/root/kaggle_sim/input/arc-prize-2026-arc-agi-2/arc-agi_test_challenges.json"))
sub = json.load(open(f"{w}/submission.json"))
ok = set(sub) == set(ch) and all(len(sub[t]) == len(v["test"]) for t, v in ch.items())
answered = sum(r["attempt_1"] != v["test"][i]["input"] for t, v in ch.items() for i, r in enumerate(sub[t]))
print(f"valid={ok} rows_with_model_answer={answered}")
EOF
}
run() {  # $1 name, rest: env assignments
  name=$1; shift; W=$S/ft_$name; rm -rf $W; mkdir -p $W/work $W/scratch
  env ARC_INPUT_ROOT=$S/input ARC_OUT=$W/work ARC_SCRATCH=$W/scratch ARC_MODE=submit ARC_SKIP_INSTALL=1 \
    ARC_VENV=/root/dsv41/venv ARC_MODEL_SRC=$M ARC_STAGE_MODE=none ARC_TOOL_PYTHON=/root/dsv41/toolvenv/bin/python \
    ARC_SOLUTIONS=$S/eval_solutions.json ARC_CACHE_SALT=ft_$name CUDA_HOME=/usr/local/cuda-13.0 \
    PATH=/usr/local/cuda-13.0/bin:/root/dsv41/venv/bin:$PATH "$@" \
    setsid /usr/bin/python3 $S/local_run.py > $W/run.log 2>&1 < /dev/null &
  echo $! > $W/nb.pid
}
wait_nb() { while kill -0 $(cat $S/ft_$1/nb.pid) 2>/dev/null; do sleep 10; done; }
wait_attempts() {  # until the solver log shows the server up and >= 60 s of attempts
  for i in $(seq 180); do grep -q "server up" $S/ft_$1/work/dsv41_logs/solver.log 2>/dev/null && break; sleep 10; done
  sleep 120
}
for t in $TESTS; do
  for i in $(seq 30); do gpu_free && break; sleep 10; done
  case $t in
  never_up)
    log "T1 never_up start"
    run never_up ARC_TASK_LIMIT=4 ARC_SERVER_EXTRA_ARGS='["--this-flag-does-not-exist"]' ARC_RESTART_MIN_LEFT_S=60
    wait_nb never_up
    log "T1 never_up: $(grep -E 'server|fallback' $S/ft_never_up/work/dsv41_logs/solver.log | cut -c1-160 | tr '\n' ' ' | cut -c1-900)"
    log "T1 never_up: $(check_sub $S/ft_never_up/work); GPU free: $(gpu_free && echo yes || echo NO)"
    ;;
  kill)
    log "T2 kill start"
    run kill ARC_TASK_LIMIT=6 ARC_WORKERS=12 ARC_STOP_NEW_S=1500 ARC_FORCE_AT_S=1800 ARC_HARD_STOP_S=2000 \
      ARC_RESTART_MIN_LEFT_S=600 ARC_MIN_ATTEMPT_TOKENS=2000
    wait_attempts kill
    sid=$(pgrep -f "vllm serve $M" | head -1)
    log "T2 kill: SIGKILL server process group $sid at $(tail -1 $S/ft_kill/work/dsv41_logs/solver.log | cut -c1-30)"
    kill -9 -- -$sid
    wait_nb kill
    log "T2 kill: $(grep -E 'died|restart|server up|retry|force|hard stop|local score' $S/ft_kill/work/dsv41_logs/solver.log | cut -c1-140 | tr '\n' ' ' | cut -c1-1500)"
    log "T2 kill: $(check_sub $S/ft_kill/work); attempts logged $(wc -l < $S/ft_kill/work/dsv41_logs/attempts.jsonl); GPU free: $(gpu_free && echo yes || echo NO)"
    ;;
  sigterm)
    log "T3 sigterm start"
    run sigterm ARC_TASK_LIMIT=6 ARC_WORKERS=12
    wait_attempts sigterm
    pid=$(pgrep -f "solver_dsv41.py" | head -1)
    log "T3 sigterm: SIGTERM solver $pid"
    kill -TERM $pid
    sleep 60
    log "T3 sigterm: $(grep -E 'signal|submission written|done' $S/ft_sigterm/work/dsv41_logs/solver.log | cut -c1-140 | tr '\n' ' ')"
    log "T3 sigterm: $(check_sub $S/ft_sigterm/work); vllm left: $(pgrep -fc "vllm serve $M"); GPU free after 60 s: $(gpu_free && echo yes || echo NO)"
    ;;
  esac
done
log "FAIL_TESTS_DONE"
