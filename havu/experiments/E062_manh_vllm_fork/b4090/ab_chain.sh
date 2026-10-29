#!/bin/bash
# E062 4090: run the rest of the throughput A/B unattended (from the workspace):
#   wait for ab_manh_w56 to finish -> copy it off -> B32 x56 (same deadlines; render parity vs the fork on B32's
#   server while it runs) -> copy -> fork x80 (L4 config, max-num-seqs 96, 45-min window) -> copy -> CHAIN_DONE.
# Log: b4090/runs/ab_chain.log
cd /home/vu/Workspace/ARC-AGI-2/experiments/E062_manh_vllm_fork/b4090
L=runs/ab_chain.log
SSH=(ssh -o ConnectTimeout=20 -o ServerAliveInterval=15 -o BatchMode=yes -p 41451 root@66.114.157.224)
log() { echo "[$(date -u +%T)] $*" >> $L; }
wait_done() {  # run name
  until "${SSH[@]}" "grep -q '^done at' /root/runs/$1/run.log" 2>/dev/null; do sleep 30; done
  log "$1 done: $("${SSH[@]}" "grep -E '^done at|local score|solver exit' /root/runs/$1/run.log | tr '\n' ' '" 2>/dev/null | cut -c1-300)"
}
copy_off() {  # run name
  mkdir -p runs/$1/pod
  "${SSH[@]}" "tar czf - --exclude=scratch -C /root/runs $1" 2>/dev/null | tar xzf - -C runs/$1/pod
  log "$1 copied: $(du -sh runs/$1/pod | cut -f1)"
}
wait_free() {
  for _ in $(seq 60); do
    u=$("${SSH[@]}" "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1; pgrep -fc 'python[0-9.]* -u [^ ]*/solver_dsv41[.]py' || true" 2>/dev/null | tr '\n' ' ')
    set -- $u
    [ "${1:-99999}" -lt 500 ] && [ "${2:-1}" = 0 ] && { log "box free (gpu max $1 MiB)"; return 0; }
    sleep 10
  done
  log "box not free after 10 min: $u"; return 1
}
log "chain start"
wait_done ab_manh_w56; copy_off ab_manh_w56
wait_free && bash ab_run.sh b32 ab_b32_w56 50 56 >> $L 2>&1
log "b32 launched"
until "${SSH[@]}" "grep -q 'server up' /root/runs/ab_b32_w56/out/dsv41_logs/solver.log" 2>/dev/null; do sleep 20; done
log "b32 server up: $("${SSH[@]}" "grep 'server up' /root/runs/ab_b32_w56/out/dsv41_logs/solver.log | head -1" 2>/dev/null | cut -c1-200)"
"${SSH[@]}" "cd /root/e062 && /root/dsv41/venv/bin/python -I parity2.py render --prompts prompts.json --url http://127.0.0.1:8100 --out render_b32.json && /root/dsv41/venv/bin/python -I parity2.py rendercompare render_b32.json render_manh_def.json" >> $L 2>&1
wait_done ab_b32_w56; copy_off ab_b32_w56
wait_free && E062_SEQS=96 bash ab_run.sh manh ab_manh_w80 45 80 >> $L 2>&1
log "fork x80 launched"
wait_done ab_manh_w80; copy_off ab_manh_w80
log "CHAIN_DONE"
