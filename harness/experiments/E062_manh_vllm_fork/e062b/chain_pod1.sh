#!/bin/bash
# E062b L4 A/B on pod #1 (Runpod ddj7xv53e5rgre; pod #2 was reassigned to the B32-M replica). Run from the workspace.
# Waits for the manh_x2t_w80 replica to end normally (final submission + report), mirrors it, then a same-pod A/B at
# x80, 55-min windows: arm A = the replica's config on b9ad670a67 + patches (e062b_A_base, /root/manh/venv); arm B =
# the merge 95376a1 (Manh's main 5e1bccd07c + our branch) + his flags with MODE 572 + decode indexer row shard
# (e062b_B_main5e1b, /root/manh2/venv); fallback: old build with MODE 60 (e062b_B_flags). Log: e062b/runs/chain_pod1.log
cd /home/vu/Workspace/ARC-AGI-2/experiments/E062_manh_vllm_fork/e062b
mkdir -p runs; L=runs/chain_pod1.log
export E062_PORT=19843 E062_IP=213.173.105.17
SSH=(ssh -o ConnectTimeout=20 -o ServerAliveInterval=15 -o BatchMode=yes -p $E062_PORT root@$E062_IP)
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
log "chain_pod1 start"
wait_done manh_x2t_w80
sleep 30
rsync -a --exclude scratch -e "ssh -o BatchMode=yes -p $E062_PORT" root@$E062_IP:/root/runs/manh_x2t_w80/ ../l4/runs/manh_x2t_w80/ && log "manh_x2t_w80 mirrored to l4/runs/manh_x2t_w80"
(cd ../l4 && bash interim.sh final > /dev/null 2>&1) && log "final report: l4/reports/manh_x2t_w80_final.md, l4/reports/interim_final.txt"
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
