#!/bin/bash
# E043 side request (E042 / dsv4-infer): DSpark drafter acceptance on the real 1017 weights.
# Restarts server B (GPUs 4-7, port 18001) with the checkpoint's DSpark drafter (method dspark, N=5 = its block
# size; the drafter writes the block in one pass, so per-position counters give the acceptance for any N <= 5),
# then sends ~15 min of real eval traffic per configuration and records the /metrics spec-decode counter deltas.
# Configs: full / E72 w30 (tool-round keep set + Engram 64 GiB) on the tool harness, full / E72 x10 (no-tool keep
# set + Engram 48 GiB) on the no-tool harness. Runtime masks leave the drafter's own 128 experts unmasked.
# Usage (local): dspark_probe.sh [wait-for-local-pid]; results in results/dspark_probe.jsonl
set -u
E=/home/vu/Workspace/ARC-AGI-2/experiments/E043_dsv41_real_pruning
OUT=$E/results/dspark_probe.jsonl
S="ssh -o BatchMode=yes -p 50293 root@147.185.60.21"
WINDOW=${WINDOW:-900}
[ -n "${1:-}" ] && while kill -0 "$1" 2>/dev/null; do sleep 20; done
q() { timeout 300 $S "$1" 2>&1 | grep -v "^Welcome\|^Have fun\|^AI agents"; }
q 'cd /workspace/dsv4; bash code/scripts/stop.sh tp4B 4,5,6,7; echo "{\"engram_keep\": null, \"expert_keep\": null, \"observe\": null}" > /dev/shm/e043/ctrlB.json;
   GPUS=4,5,6,7 TP=4 PORT=18001 CTRL=/dev/shm/e043/ctrlB.json UTIL=0.88 SEQS=192 MBT=8192 bash code/scripts/serve.sh tp4B_dspark \
     --speculative-config "{\"method\":\"dspark\",\"num_speculative_tokens\":5}"'
until q 'grep -qE "Application startup complete|Traceback|Error" /workspace/dsv4/runs/tp4B_dspark/server.log && echo ok' | grep -q ok; do sleep 20; done
q 'grep -E "Application startup complete|Traceback|Error|DSpark|speculative|weights took|KV cache" /workspace/dsv4/runs/tp4B_dspark/server.log | head -20 | cut -c1-220'
metrics() { q 'curl -s localhost:18001/metrics' | grep -E '^vllm:(spec_decode|generation_tokens_total|num_requests_running)' ; }
setmask() {  # engram expert
  q "cd /workspace/dsv4; E=null; [ $1 != none ] && E='\"$1\"'; X=null; [ $2 != none ] && X='\"$2\"';
     echo \"{\\\"engram_keep\\\": \$E, \\\"expert_keep\\\": \$X, \\\"observe\\\": null}\" > /dev/shm/e043/ctrlB.json.tmp && mv /dev/shm/e043/ctrlB.json.tmp /dev/shm/e043/ctrlB.json;
     curl -s -m 120 localhost:18001/v1/completions -H 'Content-Type: application/json' -d '{\"model\":\"dsv41\",\"prompt\":[0,5,6,7],\"max_tokens\":1}' >/dev/null; sleep 3;
     grep 'E043\] ctrl' runs/tp4B_dspark/server.log | tail -1 | cut -c60-260"
}
idle() { for i in $(seq 1 60); do r=$(metrics | awk '/^vllm:num_requests_running/ {s+=$2} END {print s+0}'); [ "${r%.*}" = 0 ] && return; sleep 10; done; }
window() {  # tag harness(tool|notool) extra...
  local tag=$1 h=$2; shift 2
  idle; local m0; m0=$(metrics); local t0; t0=$(date +%s)
  if [ $h = tool ]; then
    (cd /home/vu/Workspace/ARC-AGI-2/evals && exec ../.venv/bin/python run_deepseek_tools.py --mode grid --effort low --prompt lean2sub --answer-tool \
      --final-mode session --lenient-parse --max-tokens 64000 --max-total-output 60000 --budget-notes 0.01 --budget-warn 0.75 \
      --budget-soft 20000 --soft-wording strong --effort-header minimal --turns-note 5 --train-check --attempts 1 --workers 24 \
      --sandbox-url http://172.18.0.4:6000 --base-url http://127.0.0.1:18001/v1 --model dsv41 --limit 24 \
      --extra-body '{"chat_template_kwargs":{"thinking":true,"reasoning_effort":50},"temperature":1.0,"top_p":1.0,"cache_salt":"dspark_'$tag'"}' \
      --challenges /home/vu/Workspace/ARC-AGI-2/data/arc-agi-2-github/data/evaluation --tag "e043_dspark_$tag" "$@" \
      > results/log_e043_dspark_$tag.txt 2>&1) &
  else
    (cd /home/vu/Workspace/ARC-AGI-2/evals && exec ../.venv/bin/python run_deepseek_arcprize.py --base-url http://127.0.0.1:18001/v1 --model dsv41 \
      --effort low --effort-header minimal --task-summary --attempts 1 --workers 24 --context 131072 --limit 24 \
      --extra-body '{"chat_template_kwargs":{"thinking":true,"reasoning_effort":50},"temperature":1.0,"top_p":1.0,"cache_salt":"dspark_'$tag'"}' \
      --out results/e043_dspark_$tag.jsonl > results/log_e043_dspark_$tag.txt 2>&1) &
  fi
  local pid=$!
  sleep $WINDOW
  local m1; m1=$(metrics); local t1; t1=$(date +%s)
  kill $pid 2>/dev/null; wait $pid 2>/dev/null
  python3 - "$tag" "$h" $((t1 - t0)) "$OUT" <<PY
import sys, json, re
tag, h, dt, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
def parse(txt):
    d = {}
    for l in txt.splitlines():
        m = re.match(r'^(vllm:[a-z_]+)(\{[^}]*\})? ([0-9.e+-]+)$', l)
        if not m: continue
        pos = re.search(r'position="(\d+)"', m.group(2) or "")
        k = m.group(1) + (f"[{pos.group(1)}]" if pos else "")
        d[k] = d.get(k, 0.0) + float(m.group(3))
    return d
a, b = parse("""$m0"""), parse("""$m1""")
dd = {k: b.get(k, 0) - a.get(k, 0) for k in b}
drafts = dd.get("vllm:spec_decode_num_drafts_total", 0)
per = [dd.get(f"vllm:spec_decode_num_accepted_tokens_per_pos_total[{i}]", 0) / max(1, drafts) for i in range(5)]
acc = dd.get("vllm:spec_decode_num_accepted_tokens_total", 0)
r = {"tag": tag, "harness": h, "seconds": dt, "drafts": drafts, "draft_tokens": dd.get("vllm:spec_decode_num_draft_tokens_total", 0),
     "accepted_tokens": acc, "per_pos_accept": [round(x, 4) for x in per],
     "mean_accept_len_N": {n: round(1 + sum(per[:n]), 3) for n in (1, 2, 3, 4, 5)},
     "gen_tokens": dd.get("vllm:generation_tokens_total", 0), "gen_tok_per_s": round(dd.get("vllm:generation_tokens_total", 0) / dt, 1)}
open(out, "a").write(json.dumps(r) + "\n"); print(json.dumps(r))
PY
}
setmask none none
window full_tool tool
setmask /workspace/dsv4/data/engram/masks/engram_top64gib.npz /workspace/dsv4/data/reap/protect/keep_contrib_dec_sp0_w30_e72.npy
window e72w30_tool tool --rescue-code
setmask /workspace/dsv4/data/engram/masks_nt2_calib/engram_top48gib.npz /workspace/dsv4/data/reap/keep_nt2/keep_contrib_dec_sp0_w10_e72.npy
window e72x10_notool notool
setmask none none
window full_notool notool
q 'grep -E "SpecDecoding metrics" /workspace/dsv4/runs/tp4B_dspark/server.log | tail -3 | cut -c1-300'
echo DSPARK-DONE
