#!/bin/bash
# E062b L4 pod #2 (run from the workspace): wait for the B32+R1 replica (b32r1_x2t) to end and the pod to be free,
# then a same-pod throughput A/B of Manh's fork at x80: arm A = pod #1's replica config, arm B = A + Manh's flags,
# 55-min windows (stop-new at 55 min), copying each run off. Log: e062b/runs/chain_pod2.log
cd /home/vu/Workspace/ARC-AGI-2/experiments/E062_manh_vllm_fork/e062b
mkdir -p runs; L=runs/chain_pod2.log
SSH=(ssh -o ConnectTimeout=20 -o ServerAliveInterval=15 -o BatchMode=yes -p 11660 root@213.173.105.27)
log() { echo "[$(date -u +%T)] $*" >> $L; }
wait_done() {
  until "${SSH[@]}" "grep -q '^done at' /root/runs/$1/run.log" 2>/dev/null; do sleep 60; done
  log "$1 done: $("${SSH[@]}" "grep -E '^done at|local score' /root/runs/$1/run.log | tr '\n' ' '" 2>/dev/null | cut -c1-300)"
}
copy_off() {
  mkdir -p runs/$1/pod
  "${SSH[@]}" "tar czf - --exclude=scratch -C /root/runs $1" 2>/dev/null | tar xzf - -C runs/$1/pod
  log "$1 copied: $(du -sh runs/$1/pod | cut -f1)"
}
wait_free() {
  for _ in $(seq 90); do
    u=$("${SSH[@]}" "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1; pgrep -fc 'python[0-9.]* -u [^ ]*/solver_dsv41[.]py' || true" 2>/dev/null | tr '\n' ' ')
    set -- $u
    [ "${1:-99999}" -lt 500 ] && [ "${2:-1}" = 0 ] && { log "pod free (gpu max $1 MiB)"; return 0; }
    sleep 10
  done
  log "pod not free after 15 min: $u"; return 1
}
log "chain start"
until "${SSH[@]}" "grep -q MANH_INSTALL_OK /root/manh_setup/install.log" 2>/dev/null; do sleep 60; done
log "install: $("${SSH[@]}" "tail -3 /root/manh_setup/install.log | tr '\n' ' '" 2>/dev/null | cut -c1-300)"
wait_done b32r1_x2t
mkdir -p ../l4pod2/runs/b32r1_x2t
rsync -a --exclude scratch -e "ssh -o BatchMode=yes -p 11660" root@213.173.105.27:/root/runs/b32r1_x2t/ ../l4pod2/runs/b32r1_x2t/ && log "b32r1_x2t mirrored"
wait_free || exit 1
bash ab_run_pod2.sh base e062b_A_base 55 80 >> $L 2>&1; log "arm A launched"
wait_done e062b_A_base; copy_off e062b_A_base
wait_free || exit 1
bash ab_run_pod2.sh flags e062b_B_flags 55 80 >> $L 2>&1; log "arm B launched"
wait_done e062b_B_flags; copy_off e062b_B_flags
log "CHAIN_DONE"
