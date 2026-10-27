#!/usr/bin/env bash
# E057b box 57937248 bring-up chain (on the box, detached with setsid; log /root/e057b/runs/setup_chain.log).
# E056 box 57929406's chain (results/runs/box57929406/scripts/setup_chain.sh) plus B+:
#   1 wait for box_install.sh "INSTALL OK"   2 podprep.sh (kit, 12-file overlay, toolenv, eval data)
#   3 E056 setup_candb_venv.sh (venv_candb, 16 shas, E051 CPU tests)
#   4 setup_candbp_venv.sh (venv_candbp = venv_candb + E051 overlay v2, 31 CPU tests)
#   5 /root/kit_bp (= /root/kit with kaggle/dsv41_tools/pinned_candb's 5 files) and the evalx2t_s2026 workload
#   6 wait for the model download (DL_DONE, 121 files, 131,090,911,045 B)
#   7 memory check: 55% of MemTotal (coordinator's update, 2026-10-27; the original rule was half, which on this
#     251.5 GiB box is 125.8 GiB, below one server's ~130-140 GiB) and the cgroup limit; if the smaller is below the
#     planned 130 GiB (Engram 94 + 16 GiB pool + 20) the chain stops here ("MEMCHECK BLOCKED"); else 8.
#   8 verify_candbp_start.sh (one B+ server start, checks, chat, stop by process group)
# FROM=6 bash setup_chain.sh resumes at the download wait (steps 1-5 are skipped; used after a stalled download
# was restarted, which ends a running chain at step 6).
set -uo pipefail
L=/root/e057b/runs/setup_chain.log
mkdir -p /root/e057b/runs
log() { echo "$(date -u +%FT%TZ) $*" >> "$L"; }
state() { echo "disk $(df -h / | awk 'NR==2{print $3" used, "$4" free"}'); RAM $(free -g | awk 'NR==2{print $3" GiB used, "$7" GiB avail"}')"; }
fail() { log "CHAIN FAILED at $1"; log "CHAIN END"; exit 1; }
T0=$(date +%s)
FROM=${FROM:-1}
log "CHAIN START (from step $FROM); $(state)"

# 1. stack install
if [ "$FROM" -le 5 ]; then
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
bash /root/podkit/podprep.sh > /root/e057b/runs/podprep.out 2>&1 || { tail -15 /root/e057b/runs/podprep.out >> "$L"; fail "podprep"; }
log "step 2 podprep OK in $(( $(date +%s) - t ))s: $(grep -E 'overlay ok|toolenv|PREP OK' /root/e057b/runs/podprep.out | tr '\n' ' ')"
log "  /root/kit shas: solver $(sha256sum /root/kit/solver_dsv41.py | cut -c1-8), harness $(sha256sum /root/kit/repo/evals/run_deepseek_tools.py | cut -c1-8), overlay tgz $(sha256sum /root/kit/vllm_overlay_rw.tgz | cut -c1-8)"

# 3. candidate B venv (E056 script; its logs go to /root/e056/runs)
mkdir -p /root/e056/runs
t=$(date +%s)
bash /root/e056/gpu/setup_candb_venv.sh > /root/e056/runs/setup_candb_venv.log 2>&1
rc=$?
tail -12 /root/e056/runs/setup_candb_venv.log | cut -c1-300 >> "$L"
[ "$rc" -eq 0 ] || fail "setup_candb_venv (rc=$rc)"
log "step 3 setup_candb_venv OK in $(( $(date +%s) - t ))s; $(state)"

# 4. B+ venv
t=$(date +%s)
bash /root/e057b/scripts/setup_candbp_venv.sh > /root/e057b/runs/setup_candbp_venv.log 2>&1
rc=$?
tail -12 /root/e057b/runs/setup_candbp_venv.log | cut -c1-300 >> "$L"
[ "$rc" -eq 0 ] || fail "setup_candbp_venv (rc=$rc)"
log "step 4 setup_candbp_venv OK in $(( $(date +%s) - t ))s; venv_candbp $(du -sh /workspace/dsv41/venv_candbp | cut -f1); $(state)"

# 5. kit_bp + workload
rm -rf /root/kit_bp && cp -a /root/kit /root/kit_bp || fail "kit_bp copy"
P=/root/e057b/pinned_candb
cp "$P/solver_dsv41.py" /root/kit_bp/solver_dsv41.py
for f in run_deepseek_tools.py train_check.py auto_repair.py auto_repair_names.json; do cp "$P/$f" /root/kit_bp/repo/evals/$f; done
got="$(sha256sum /root/kit_bp/solver_dsv41.py | cut -c1-8) $(sha256sum /root/kit_bp/repo/evals/run_deepseek_tools.py | cut -c1-8) $(sha256sum /root/kit_bp/repo/evals/train_check.py | cut -c1-8) $(sha256sum /root/kit_bp/repo/evals/auto_repair.py | cut -c1-8) $(sha256sum /root/kit_bp/repo/evals/auto_repair_names.json | cut -c1-8)"
log "step 5 kit_bp: solver/harness/train_check/auto_repair/names $got (expect 77fe80a3 882c817f 8fe36175 2292369c 571f47eb)"
[ "$got" = "77fe80a3 882c817f 8fe36175 2292369c 571f47eb" ] || fail "kit_bp shas"
cp /root/e057b/workloads/evalx2t_s2026_challenges.json /root/e057b/workloads/evalx2t_s2026_solutions.json /root/data/
w="$(sha256sum /root/data/evalx2t_s2026_challenges.json | cut -c1-12) $(sha256sum /root/data/evalx2t_s2026_solutions.json | cut -c1-12) $(sha256sum /root/e057b/workloads/evalx2t_s2026_heavy90_ids.txt | cut -c1-12)"
log "  workload challenges/solutions/heavy90 $w (expect 83b249b1398d 8ed7202a78eb 3c6db22a194e)"
[ "$w" = "83b249b1398d 8ed7202a78eb 3c6db22a194e" ] || fail "workload shas"
fi

# 6. model download
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
log "step 6 download: $(grep -o '^\[[0-9:]*\] start' /root/dsv41/runs/dl_e128t.log) -> $(grep -o '^\[[0-9:]*\] DL_DONE' /root/dsv41/runs/dl_e128t.log); $NF files, $BYTES bytes (expect 121 files, 131090911045)"
[ "$BYTES" = 131090911045 ] && [ "$NF" = 121 ] || fail "download byte/file count"
log "  model dir $(du -sh $M | cut -f1); $(state)"

# 7. memory check (shared rented boxes: half of host RAM; the user allows ~55% for E057b; one B/B+ server at 16 GiB
#    plans 94 + 16 + 20 = 130 GiB)
MT_KB=$(awk '/^MemTotal:/{print $2}' /proc/meminfo)
CG=$(cat /sys/fs/cgroup/memory.max 2>/dev/null || cat /sys/fs/cgroup/memory/memory.limit_in_bytes 2>/dev/null || echo max)
LIM=$(python3 -c "
mt = $MT_KB * 1024 / 2**30; cg = '$CG'.strip()
cgg = float(cg) / 2**30 if cg.isdigit() else float('inf')
print(f'{min(mt * 0.55, cgg):.1f} {mt:.1f} {cgg:.1f} {mt / 2:.1f}')")
set -- $LIM
log "step 7 memory: MemTotal $2 GiB (half $4, 55% $(python3 -c "print(round($2*0.55,1))") GiB), cgroup memory.max $CG ($3 GiB) -> limit $1 GiB; plan 130 GiB for one B/B+ server at 16 GiB"
if python3 -c "import sys; sys.exit(0 if float('$1') >= 130 else 1)"; then
  log "  MEMCHECK OK (55% rule)"
else
  log "  MEMCHECK BLOCKED: limit $1 GiB < 130 GiB; no server started"
  log "SETUP OK in $(( $(date +%s) - T0 ))s (server start not run)"
  log "CHAIN END"
  exit 0
fi

# 8. one B+ server start
t=$(date +%s)
RUN=verify_candbp bash /root/e057b/scripts/verify_candbp_start.sh > /root/e057b/runs/verify.driver.out 2>&1
rc=$?
cat /root/e057b/runs/verify_candbp.progress >> "$L"
grep -q "verify_candbp: DONE" /root/e057b/runs/verify_candbp.progress || fail "verify (rc=$rc, no stop line)"
grep -q "verify_candbp: ready after" /root/e057b/runs/verify_candbp.progress || fail "verify (server not ready)"
grep -q "chat rc=0 content=" /root/e057b/runs/verify_candbp.progress || fail "verify (chat completion)"
log "step 8 verify finished in $(( $(date +%s) - t ))s (rc=$rc); GPUs $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr '\n' ' ')MiB; $(state)"
log "CHAIN OK in $(( $(date +%s) - T0 ))s"
log "CHAIN END"
