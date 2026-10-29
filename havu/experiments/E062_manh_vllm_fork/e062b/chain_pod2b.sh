#!/bin/bash
# E062b L4 pod #2 (run from the workspace), replacing chain_pod2.sh: wait for the B32+R1 replica (b32r1_x2t) to end,
# then the same-pod A/B at x80, 55-min windows: arm A = pod #1's replica config on b9ad670a67 + patches
# (e062b_A_base); arm B = the merge 95376a1 (Manh's main 5e1bccd07c + our branch) + his flags with MODE 572 and
# the decode indexer row shard (e062b_B_main5e1b), or, if that venv isn't installed by then, the old build with
# his MODE 60 flags (e062b_B_flags). Log: e062b/runs/chain_pod2.log
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
log "chain_pod2b start"
wait_done b32r1_x2t
mkdir -p ../l4pod2/runs/b32r1_x2t
rsync -a --exclude scratch -e "ssh -o BatchMode=yes -p 11660" root@213.173.105.27:/root/runs/b32r1_x2t/ ../l4pod2/runs/b32r1_x2t/ && log "b32r1_x2t mirrored"
wait_free || exit 1
bash ab_run_pod2.sh base e062b_A_base 55 80 >> $L 2>&1; log "arm A launched"
wait_done e062b_A_base; copy_off e062b_A_base
wait_free || exit 1
if "${SSH[@]}" "grep -q MANH2_INSTALL_OK /root/manh2_setup/install.log" 2>/dev/null; then
  bash ab_run_pod2.sh flags2 e062b_B_main5e1b 55 80 >> $L 2>&1; B=e062b_B_main5e1b; log "arm B launched: new build 95376a1, MODE 572 + row shard"
else
  bash ab_run_pod2.sh flags e062b_B_flags 55 80 >> $L 2>&1; B=e062b_B_flags; log "arm B launched: FALLBACK old build b9ad670a67, MODE 60 (new venv not installed)"
fi
wait_done $B; copy_off $B
log "CHAIN_DONE $B"
