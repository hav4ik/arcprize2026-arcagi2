#!/usr/bin/env bash
# E059 B16 on Vast 4x4090 box 57983124: bring-up chain (on the box, detached with setsid; log /root/e059/runs/setup_chain.log).
# E057b's chain (experiments/E057b_workers64_stress/scripts/setup_chain.sh) for candidate B (venv_candb, no overlay v2):
#   1 wait for box_install.sh "INSTALL OK"   2 podprep.sh (kit a9706290, 12-file overlay, toolenv, eval data)
#   3 E056 setup_candb_venv.sh (/root/dsv41/venv_candb, 16/16 sha lines, E051's 26 CPU tests)
#   4 /root/kit_b (= /root/kit with kaggle/dsv41_tools/pinned_candb's 5 files) and the evalx2t_s2026 workload
#   5 wait for the model download (DL_DONE, 121 files, 131,090,911,045 B)
#   6 memory check: one B server at 16 GiB plans 94 + 16 + 20 = 130 GiB; must fit half of MemTotal and the cgroup limit
#   7 E056 verify_candb_start.sh (one B server start at 16 GiB, util 0.92, 21490 blocks; checks; chat; stop by PGID)
#   8 gates on 7: offload line once, KV pool 1,591,736 tokens, override 21490, 0 tracebacks at ready and after the chat,
#     chat answer 42, GPUs idle after the stop -> "CHAIN OK"
# FROM=5 bash setup_chain.sh resumes at the download wait.
set -uo pipefail
L=/root/e059/runs/setup_chain.log
mkdir -p /root/e059/runs
log() { echo "$(date -u +%FT%TZ) $*" >> "$L"; }
state() { echo "disk $(df -h / | awk 'NR==2{print $3" used, "$4" free"}'); RAM $(free -g | awk 'NR==2{print $3" GiB used, "$7" GiB avail"}')"; }
fail() { log "CHAIN FAILED at $1"; log "CHAIN END"; exit 1; }
T0=$(date +%s)
FROM=${FROM:-1}
log "CHAIN START (from step $FROM); $(state)"

if [ "$FROM" -le 4 ]; then
# 1. stack install
while ! grep -q "INSTALL OK" /root/dsv41/install.log 2>/dev/null; do
  if ! pgrep -x -f "/bin/bash /root/dsv41/box_install.sh" > /dev/null; then
    sleep 5; grep -q "INSTALL OK" /root/dsv41/install.log && break
    log "box_install.sh exited without INSTALL OK; tail:"; tail -15 /root/dsv41/install.log | cut -c1-300 >> "$L"; fail "install"
  fi
  sleep 20
done
log "step 1 install OK ($(grep -m1 -o '^\[[0-9:]*\] install start' /root/dsv41/install.log) -> $(grep -o '^\[[0-9:]*\] INSTALL OK' /root/dsv41/install.log)); $(state)"
tail -4 /root/dsv41/install.log | cut -c1-300 >> "$L"

# 2. podprep
t=$(date +%s)
bash /root/podkit/podprep.sh > /root/e059/runs/podprep.out 2>&1 || { tail -15 /root/e059/runs/podprep.out >> "$L"; fail "podprep"; }
log "step 2 podprep OK in $(( $(date +%s) - t ))s: $(grep -E 'overlay ok|toolenv|PREP OK' /root/e059/runs/podprep.out | tr '\n' ' ')"
log "  /root/kit shas: solver $(sha256sum /root/kit/solver_dsv41.py | cut -c1-8), harness $(sha256sum /root/kit/repo/evals/run_deepseek_tools.py | cut -c1-8), overlay tgz $(sha256sum /root/kit/vllm_overlay_rw.tgz | cut -c1-8)"

# 3. candidate B venv (E056 script; its logs go to /root/e056/runs)
mkdir -p /root/e056/runs
t=$(date +%s)
bash /root/e056/gpu/setup_candb_venv.sh > /root/e056/runs/setup_candb_venv.log 2>&1
rc=$?
tail -12 /root/e056/runs/setup_candb_venv.log | cut -c1-300 >> "$L"
[ "$rc" -eq 0 ] || fail "setup_candb_venv (rc=$rc)"
grep -q "installed: all 16 sha lines match" /root/e056/runs/setup_candb_venv.log || fail "setup_candb_venv (16 sha lines)"
grep -q "^Ran 26 tests" /root/e056/runs/e051_cpu_tests.log && tail -1 /root/e056/runs/e051_cpu_tests.log | grep -q "^OK" \
  || fail "E051 CPU tests (expected 26 OK)"
log "step 3 setup_candb_venv OK in $(( $(date +%s) - t ))s; venv_candb $(du -sh /root/dsv41/venv_candb | cut -f1); $(state)"

# 4. kit_b + workload
rm -rf /root/kit_b && cp -a /root/kit /root/kit_b || fail "kit_b copy"
P=/root/e059/pinned_candb
cp "$P/solver_dsv41.py" /root/kit_b/solver_dsv41.py
for f in run_deepseek_tools.py train_check.py auto_repair.py auto_repair_names.json; do cp "$P/$f" /root/kit_b/repo/evals/$f; done
got="$(sha256sum /root/kit_b/solver_dsv41.py | cut -c1-8) $(sha256sum /root/kit_b/repo/evals/run_deepseek_tools.py | cut -c1-8) $(sha256sum /root/kit_b/repo/evals/train_check.py | cut -c1-8) $(sha256sum /root/kit_b/repo/evals/auto_repair.py | cut -c1-8) $(sha256sum /root/kit_b/repo/evals/auto_repair_names.json | cut -c1-8)"
log "step 4 kit_b: solver/harness/train_check/auto_repair/names $got (expect 77fe80a3 882c817f 8fe36175 2292369c 571f47eb)"
[ "$got" = "77fe80a3 882c817f 8fe36175 2292369c 571f47eb" ] || fail "kit_b shas"
cp /root/e059/workloads/evalx2t_s2026_challenges.json /root/e059/workloads/evalx2t_s2026_solutions.json /root/data/
w="$(sha256sum /root/data/evalx2t_s2026_challenges.json | cut -c1-12) $(sha256sum /root/data/evalx2t_s2026_solutions.json | cut -c1-12) $(sha256sum /root/e059/workloads/evalx2t_s2026_heavy90_ids.txt | cut -c1-12)"
log "  workload challenges/solutions/heavy90 $w (expect 83b249b1398d 8ed7202a78eb 3c6db22a194e)"
[ "$w" = "83b249b1398d 8ed7202a78eb 3c6db22a194e" ] || fail "workload shas"
fi

# 5. model download
while ! grep -qE "DL_DONE|DL_FAILED" /root/dsv41/runs/dl_e128t.log 2>/dev/null; do
  if ! pgrep -f "^/bin/bash /root/dsv41/dl_e128_(full|http).sh" > /dev/null; then
    sleep 60; grep -qE "DL_DONE|DL_FAILED" /root/dsv41/runs/dl_e128t.log && break
    pgrep -f "^/bin/bash /root/dsv41/dl_e128_(full|http).sh" > /dev/null && continue   # a watchdog restart
    fail "download (process gone without DL_DONE)"
  fi
  sleep 30
done
grep -q DL_FAILED /root/dsv41/runs/dl_e128t.log && ! grep -q DL_DONE /root/dsv41/runs/dl_e128t.log && { grep -iE "error|DL_" /root/dsv41/runs/dl_e128t.log | tail -10 | cut -c1-300 >> "$L"; fail "download"; }
M=/root/dsv41/model-e128t
BYTES=$(find $M -path $M/.cache -prune -o -type f -printf '%s\n' | awk '{s+=$1} END {print s}')
NF=$(find $M -path $M/.cache -prune -o -type f -print | wc -l)
log "step 5 download: $(grep -m1 -o '^\[[0-9:]*\] start' /root/dsv41/runs/dl_e128t.log) -> $(grep -o '^\[[0-9:]*\] DL_DONE' /root/dsv41/runs/dl_e128t.log | tail -1); $NF files, $BYTES bytes (expect 121 files, 131090911045)"
[ "$BYTES" = 131090911045 ] && [ "$NF" = 121 ] || fail "download byte/file count"
log "  model dir $(du -sh $M | cut -f1); $(state)"

# 6. memory check (half of host RAM and the cgroup limit; one B server at 16 GiB plans 130 GiB)
MT_KB=$(awk '/^MemTotal:/{print $2}' /proc/meminfo)
CG=$(cat /sys/fs/cgroup/memory.max 2>/dev/null || cat /sys/fs/cgroup/memory/memory.limit_in_bytes 2>/dev/null || echo max)
LIM=$(python3 -c "
mt = $MT_KB * 1024 / 2**30; cg = '$CG'.strip()
cgg = float(cg) / 2**30 if cg.isdigit() else float('inf')
print(f'{min(mt / 2, cgg):.1f} {mt:.1f} {cgg:.1f}')")
set -- $LIM
log "step 6 memory: MemTotal $2 GiB (half $(python3 -c "print(round($2/2,1))")), cgroup memory.max $CG ($3 GiB) -> limit $1 GiB; plan 130 GiB for one B server at 16 GiB"
python3 -c "import sys; sys.exit(0 if float('$1') >= 130 else 1)" || { log "  MEMCHECK BLOCKED"; fail "memory check"; }
log "  MEMCHECK OK; cpu.max $(cat /sys/fs/cgroup/cpu.max 2>/dev/null) nproc $(nproc)"

# 7. one candidate-B server start (E056 verify script, 16 GiB offload)
t=$(date +%s)
RUN=verify_b16_57983124
RUN=$RUN bash /root/e056/gpu/verify_candb_start.sh > /root/e059/runs/verify.driver.out 2>&1
rc=$?
PR=/root/e056/runs/$RUN.progress
cat "$PR" >> "$L"
grep -q "$RUN: DONE" "$PR" || fail "verify (rc=$rc, no stop line)"
grep -q "$RUN: ready after" "$PR" || fail "verify (server not ready)"
grep -q "chat rc=0 content='42'" "$PR" || fail "verify (chat completion)"
# 8. gates
grep -q ": 1 offload lines; GPU KV cache size: 1,591,736 tokens; Overriding num_gpu_blocks=[0-9]* with num_gpu_blocks_override=21490; skip-on lines [0-9]*; tracebacks 0$" "$PR" \
  || fail "verify gates (offload line / KV pool 1,591,736 / override 21490 / tracebacks at ready)"
grep -q "tracebacks after chat 0" "$PR" || fail "verify gates (tracebacks after chat)"
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)
[ "$used" -lt 500 ] || fail "GPUs not idle after the verify stop (${used} MiB)"
log "step 7-8 verify finished in $(( $(date +%s) - t ))s (rc=$rc); gates OK; GPUs $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr '\n' ' ')MiB; $(state)"
log "CHAIN OK in $(( $(date +%s) - T0 ))s"
log "CHAIN END"
