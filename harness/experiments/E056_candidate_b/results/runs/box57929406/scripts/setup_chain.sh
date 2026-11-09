#!/usr/bin/env bash
# Box 57929406 bring-up chain (runs on the box, detached with setsid; log /root/e056/runs/setup_chain.log):
#   1 wait for box_install.sh "INSTALL OK"  2 podprep.sh (kit, 12-file overlay, toolenv, data)
#   3 setup_candb_venv.sh (venv_candb, 16 shas, E051 CPU tests)  4 wait for the model download (DL_DONE, byte count)
#   5 verify_candb_start.sh (one candidate-B server start, checks, chat, stop by process group)
# Each step's own log stays in its usual place; this log has the step lines, durations and disk/RAM.
set -uo pipefail
L=/root/e056/runs/setup_chain.log
mkdir -p /root/e056/runs
log() { echo "$(date -u +%FT%TZ) $*" >> "$L"; }
state() { echo "disk $(df -h / | awk 'NR==2{print $3" used, "$4" free"}'); RAM $(free -g | awk 'NR==2{print $3" GiB used, "$7" GiB avail"}')"; }
fail() { log "CHAIN FAILED at $1"; log "CHAIN END"; exit 1; }
T0=$(date +%s)
log "CHAIN START; $(state)"

# 1. stack install
while ! grep -q "INSTALL OK" /root/dsv41/install.log; do
  if ! pgrep -x -f "/bin/bash /root/dsv41/box_install.sh" > /dev/null; then
    sleep 5; grep -q "INSTALL OK" /root/dsv41/install.log && break
    log "box_install.sh exited without INSTALL OK; tail:"; tail -15 /root/dsv41/install.log | cut -c1-300 >> "$L"; fail "install"
  fi
  sleep 20
done
log "step 1 install OK ($(grep -m1 -o '^\[[0-9:]*\] install start' /root/dsv41/install.log) -> $(grep -o '^\[[0-9:]*\] INSTALL OK' /root/dsv41/install.log)); $(state)"
tail -4 /root/dsv41/install.log | cut -c1-300 >> "$L"

# 2. podprep (kit to /root/kit, the kit's 12-file overlay over /root/dsv41/venv sha-checked, /root/toolenv, /root/data)
t=$(date +%s)
bash /root/podkit/podprep.sh > /root/e056/runs/podprep.out 2>&1 || { tail -15 /root/e056/runs/podprep.out >> "$L"; fail "podprep"; }
log "step 2 podprep OK in $(( $(date +%s) - t ))s: $(grep -E 'overlay ok|toolenv|PREP OK' /root/e056/runs/podprep.out | tr '\n' ' ')"
log "  /root/kit MANIFEST shas: solver $(sha256sum /root/kit/solver_dsv41.py | cut -c1-8), harness $(sha256sum /root/kit/repo/evals/run_deepseek_tools.py | cut -c1-8), overlay tgz $(sha256sum /root/kit/vllm_overlay_rw.tgz | cut -c1-8); $(state)"

# 3. candidate B venv
t=$(date +%s)
bash /root/e056/gpu/setup_candb_venv.sh > /root/e056/runs/setup_candb_venv.log 2>&1
rc=$?
tail -12 /root/e056/runs/setup_candb_venv.log | cut -c1-300 >> "$L"
[ "$rc" -eq 0 ] || fail "setup_candb_venv (rc=$rc)"
log "step 3 setup_candb_venv OK in $(( $(date +%s) - t ))s; venv $(du -sh /workspace/dsv41/venv | cut -f1), venv_candb $(du -sh /workspace/dsv41/venv_candb | cut -f1); $(state)"

# 4. model download
while ! grep -qE "DL_DONE|DL_FAILED" /root/dsv41/runs/dl_e128t.log; do
  if ! pgrep -x -f "/bin/bash /root/dsv41/dl_e128_full.sh" > /dev/null; then
    sleep 5; grep -qE "DL_DONE|DL_FAILED" /root/dsv41/runs/dl_e128t.log && break
    fail "download (process gone without DL_DONE)"
  fi
  sleep 30
done
grep -q DL_FAILED /root/dsv41/runs/dl_e128t.log && { grep -iE "error|DL_" /root/dsv41/runs/dl_e128t.log | tail -10 | cut -c1-300 >> "$L"; fail "download"; }
M=/root/dsv41/model-e128t
BYTES=$(find $M -path $M/.cache -prune -o -type f -printf '%s\n' | awk '{s+=$1} END {print s}')
NF=$(find $M -path $M/.cache -prune -o -type f -print | wc -l)
log "step 4 download: $(grep -o '^\[[0-9:]*\] start' /root/dsv41/runs/dl_e128t.log) -> $(grep -o '^\[[0-9:]*\] DL_DONE' /root/dsv41/runs/dl_e128t.log); $NF files, $BYTES bytes (expect 121 files, 131090911045)"
[ "$BYTES" = 131090911045 ] || fail "download byte count"
log "  model dir $(du -sh $M | cut -f1); $(state)"

# 5. one candidate-B server start
t=$(date +%s)
RUN=verify_box57929406 bash /root/e056/gpu/verify_candb_start.sh > /root/e056/runs/verify.driver.out 2>&1
rc=$?
cat /root/e056/runs/verify_box57929406.progress >> "$L"
grep -q "verify_box57929406: DONE" /root/e056/runs/verify_box57929406.progress || fail "verify (rc=$rc, no stop line)"
grep -q "verify_box57929406: ready after" /root/e056/runs/verify_box57929406.progress || fail "verify (server not ready)"
grep -q "chat rc=0 content=" /root/e056/runs/verify_box57929406.progress || fail "verify (chat completion)"
log "step 5 verify finished in $(( $(date +%s) - t ))s (rc=$rc); GPUs $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr '\n' ' ')MiB; $(state)"
log "CHAIN OK in $(( $(date +%s) - T0 ))s"
log "CHAIN END"
