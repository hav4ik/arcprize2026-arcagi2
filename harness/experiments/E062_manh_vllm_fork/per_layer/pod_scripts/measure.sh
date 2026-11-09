#!/bin/bash
# measure.sh <ckpt> <tag> [dcp=4] [cvd=0,1,2,3] [lmo=0]: start B32-M's server on a dummy checkpoint, wait for it, log
# memory lines, nvidia-smi peaks and cgroup-v1 unreclaimable host memory, run a fixed synthetic load
# (vllm bench serve: random 1024 in / 512 out, 160 prompts at concurrency 80, ignore-eos), then stop the server by its
# process group (setsid) and write runs/<tag>/summary.json. Ends with "MEASURE_DONE <tag>".
set -u
CK=$1; TAG=$2; DCP=${3:-4}; CVD=${4:-0,1,2,3}; LMO=${5:-0}
D=/root/e062pl; OUT=$D/runs/$TAG; V=/root/b32m/venv
rm -rf $OUT; mkdir -p $OUT
bash $D/serve_dummy.sh "$CK" "$TAG" "$DCP" "$CVD" "$LMO" > $OUT/serve_start.txt
PID=$(cat $OUT/server.pid)
( while kill -0 $PID 2>/dev/null; do
    nvidia-smi --query-gpu=index,memory.used,power.draw --format=csv,noheader,nounits | tr -d ' ' | paste -sd';' | sed "s/^/$(date -u +%T) /"
    awk '/^total_rss /{r=$2} /^total_shmem /{s=$2} END {printf "cg_unrec_gib=%.1f\n", (r+s)/2^30}' /sys/fs/cgroup/memory/memory.stat
    sleep 15; done ) > $OUT/gpu_host.log 2>&1 &
SAMP=$!
T0=$(date +%s); UP=0
while kill -0 $PID 2>/dev/null; do
  if curl -sf -m 5 http://127.0.0.1:8100/health > /dev/null; then UP=1; break; fi
  [ $(( $(date +%s) - T0 )) -gt 2400 ] && break
  sleep 10
done
echo "up=$UP after $(( $(date +%s) - T0 ))s" > $OUT/startup.txt
if [ $UP = 1 ] && [ -z "${NOBENCH:-}" ]; then
  $V/bin/vllm bench serve --backend vllm --base-url http://127.0.0.1:8100 --model dsv41 --tokenizer "$CK" \
      --tokenizer-mode deepseek_v41 --trust-remote-code --dataset-name random --random-input-len 1024 \
      --random-output-len 512 --num-prompts 160 --max-concurrency 80 --ignore-eos --save-result \
      --result-dir $OUT --result-filename bench.json > $OUT/bench.log 2>&1 || echo "BENCH_FAIL $?" >> $OUT/bench.log
fi
PGID=$(ps -o pgid= -p $PID | tr -d ' ')
[ -n "$PGID" ] && kill -TERM -- -$PGID 2>/dev/null
for i in $(seq 1 60); do kill -0 $PID 2>/dev/null || break; sleep 2; done
[ -n "$PGID" ] && kill -KILL -- -$PGID 2>/dev/null
kill $SAMP 2>/dev/null
$V/bin/python -I - "$OUT" <<'EOF'
import json, re, sys
out = sys.argv[1]
log = open(f"{out}/server.log", errors="replace").read()
pick = lambda pat: [m.group(0)[:200] for m in re.finditer(pat, log)]
peaks, unrec = {}, 0.0
for line in open(f"{out}/gpu_host.log"):
    if line.startswith("cg_unrec"):
        unrec = max(unrec, float(line.split("=")[1]))
    elif " " in line:
        for g in line.split(" ", 1)[1].strip().split(";"):
            i, mem, _ = g.split(",")
            peaks[i] = max(peaks.get(i, 0), int(float(mem)))
s = {"tag": out.rsplit("/", 1)[1], "startup": open(f"{out}/startup.txt").read().strip(),
     "model_loading": pick(r"Model loading took [0-9.]+ GiB[^\n]*"),
     "available_kv": pick(r"Available KV cache memory: [0-9.]+ GiB"),
     "kv_tokens": pick(r"GPU KV cache size: [0-9,]+ tokens"),
     "graph": pick(r"Graph capturing finished[^\n]*"),
     "encoder": pick(r"[^\n]*(?i:encoder cache|multimodal|mm_|vision|image)[^\n]*(?:budget|profil|cache size|tokens)[^\n]*")[:6],
     "errors": pick(r"[^\n]*(?:Error|error:|Traceback)[^\n]*")[:8],
     "gpu_peak_mib": peaks, "host_unreclaimable_peak_gib": unrec}
try:
    b = json.load(open(f"{out}/bench.json"))
    s["bench"] = {k: b.get(k) for k in ("duration", "completed", "output_throughput", "total_token_throughput",
                                        "mean_ttft_ms", "median_ttft_ms", "mean_tpot_ms", "median_tpot_ms",
                                        "mean_itl_ms", "total_input_tokens", "total_output_tokens")}
except Exception as exc:
    s["bench"] = f"none: {exc}"
json.dump(s, open(f"{out}/summary.json", "w"), indent=1)
print(json.dumps(s)[:3000])
EOF
echo "MEASURE_DONE $TAG"
