#!/bin/bash
# E042 P12b: NCCL copy-engine SHM all-reduce (NCCL_SHM_USE_CUDA_MEMCPY=1). P12's micro-bench: 4x faster at prefill
# sizes (2048 tokens 13.96 -> 3.53 ms) but 2x slower at 64 tokens (0.30 -> 0.64 ms). Waits for P12, then:
#   0. the Humming runs P12 lost to the fork's V4.1 whitelists (patched by patches/humming_v41.py at 00:16 UTC):
#      experts only (--moe-backend humming) and dense only (--linear-backend humming), P12's 4 points; and CED
#      prefill, which needs max-num-batched-tokens >= 128 x max-num-seqs (P12's MBT 2048 x 128 seqs failed): CED at
#      16 seqs x MBT 2048 (CED alone, vs P12 base), and 64 seqs x MBT 8192 with and without CED (deployable shape),
#   1. micro-bench with decode-batch sizes (96-192 tokens) for default, CE and CE + protocol variants,
#   2. end-to-end decode/prefill (P12's 4 points) for CE and for the CE variant with the fastest decode-size
#      all-reduces (if another variant beats plain CE there),
#   3. R4: tool replay E64 x 64 with the retention flag and the CE variant whose end-to-end points give the best
#      tool-mode estimate, 1 / (1/decode(1K x 64) + 0.93/prefill(8K x 16)) (box R3: 0.93 re-prefilled per generated
#      token); same seed as R3's r3_e64ret (209.0 tok/s), so it is an A/B against it.
# Log: /root/dsv41/runs/p12b.log
S=/root/dsv41/scripts; L=/root/dsv41/runs/p12b.log; PY=/root/dsv41/venv/bin/python; TS=/root/dsv41/metadata/trace_shapes
log() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a $L; }
busy() { curl -s -m 3 localhost:8100/v1/models > /dev/null || [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" != 0 ]; }
wait_up() {
  local R=/root/dsv41/runs/$1 t=0
  until curl -s -m 3 localhost:8100/v1/models > /dev/null; do
    sleep 10; t=$((t + 10))
    if grep -q Traceback $R/server.log || ! kill -0 $(cat $R/server.pid) 2>/dev/null || [ $t -gt 2700 ]; then
      log "$1 SERVE FAILED after ${t}s: $(grep -E 'Error' $R/server.log | grep -v Warning | tail -2 | cut -c1-300)"; return 1
    fi
  done
  log "$1 up after ${t}s: $(grep -o 'GPU KV cache size: [0-9,]* tokens' $R/server.log | tail -1)"
}
until grep -qE "P12_DONE|ABORT" /root/dsv41/runs/p12.log 2>/dev/null; do sleep 30; done
for i in $(seq 90); do busy || break; sleep 2; done
busy && { log "ABORT: GPUs busy after P12"; exit 1; }
log "p12b start"

dw() {  # dw <name> [points]
  local R=/root/dsv41/runs/$1/dw p ctx c; mkdir -p $R
  for p in ${2:-1024:64 1024:128 8192:16 32768:4}; do
    ctx=${p%%:*}; c=${p##*:}
    log "$1 ctx $ctx C $c: $($PY $S/decode_window.py --ctx $ctx --conc $c --window 60 --seed $c --sock-read 10800 \
      --json $R/ctx${ctx}_c${c}.json 2>&1 | tail -1 | sed -E 's/"first_token_s": \[[^]]*\], //' | cut -c1-400)"
  done
}

# 0. Humming experts only, dense only; CED
CED=(--hf-overrides '{"ced_prefill": true}')
for v in hmoe hlin ced16 mbt8k64 ced64; do
  N=p12b_$v; X=(); Q=128; M=2048; P=
  case $v in
    hmoe) X=(--moe-backend humming);;
    hlin) X=(--linear-backend humming);;
    ced16) X=("${CED[@]}"); Q=16; P="1024:16 8192:16 32768:4";;
    mbt8k64) Q=64; M=8192; P="1024:64 8192:16 32768:4";;
    ced64) X=("${CED[@]}"); Q=64; M=8192; P="1024:64 8192:16 32768:4";;
  esac
  busy && { log "ABORT before $N: busy"; exit 1; }
  DSV41_INDEXER_Q_PREDECODE=1 SEQS=$Q MBT=$M UTIL=0.88 $S/serve.sh $N "${X[@]}" >> $L
  (setsid nohup $S/gpumon.sh $N > /dev/null 2>&1 < /dev/null &)
  if wait_up $N; then
    log "$N weights/KV: $(grep -o -E 'Model loading took [0-9.]+ GiB|Available KV cache memory: [0-9.]+ GiB' /root/dsv41/runs/$N/server.log | sort -u | tr '\n' ' ')"
    dw $N "$P"
  fi
  $S/stop.sh $N | tee -a $L
done

# 1. micro-bench. Variants are "label:ENV=V,ENV=V" (no env for default).
O=/root/dsv41/runs/nccl; T=/root/dsv41/venv/bin/torchrun; B=$S/nccl_allreduce_bench.py
V="b_default: b_ce1:NCCL_SHM_USE_CUDA_MEMCPY=1 b_ce1_m2:NCCL_SHM_USE_CUDA_MEMCPY=1,NCCL_SHM_MEMCPY_MODE=2
   b_ce1_ll:NCCL_SHM_USE_CUDA_MEMCPY=1,NCCL_PROTO=LL b_ce1_ll128:NCCL_SHM_USE_CUDA_MEMCPY=1,NCCL_PROTO=LL128
   b_ce1_llsimple:NCCL_SHM_USE_CUDA_MEMCPY=1,NCCL_PROTO=LL,Simple"
for v in $V; do
  lab=${v%%:*}; ev=${v#*:}
  (cd $O; unset NCCL_PROTO NCCL_ALGO NCCL_P2P_LEVEL
   env $(echo "$ev" | sed 's/,NCCL_/ NCCL_/g') $T --nproc-per-node 4 $B --label $lab --out $lab.json 2>&1 | grep -E "^$lab" > $lab.txt)
  log "nccl $lab ($ev) ms at 64/128/256/2048 tokens: $($PY -c "import json,sys; r={x['tokens']: x['ms'] for x in json.load(open('$O/$lab.json'))['rows']}; print(*[r.get(t) for t in (64, 128, 256, 2048)])" 2>&1 | tail -1)"
done
# The CE variant with the fastest decode-size all-reduces (64 + 128 tokens), if it keeps prefill sizes within 1.5x of CE.
ALT=$($PY - <<EOF
import json
O = "$O"; vs = {v.split(":", 1)[0]: v.split(":", 1)[1] for v in """$V""".split()}
ms = {}
for lab in vs:
    try:
        ms[lab] = {x["tokens"]: x["ms"] for x in json.load(open(f"{O}/{lab}.json"))["rows"]}
    except Exception:
        pass
ce = ms.get("b_ce1")
best = min((l for l in ms if l.startswith("b_ce1_") and ce and ms[l][2048] <= 1.5 * ce[2048]),
           key=lambda l: ms[l][64] + ms[l][128], default=None)
if best and ms[best][64] + ms[best][128] < 0.8 * (ce[64] + ce[128]):
    print(best, vs[best])
EOF
)
log "alt CE variant: ${ALT:-none}"

# 2. end-to-end points
CFG="p12b_ce1:NCCL_SHM_USE_CUDA_MEMCPY=1"
[ -n "$ALT" ] && CFG="$CFG p12b_${ALT%% *}:${ALT#* }"
for c in $CFG; do
  N=${c%%:*}; ev=${c#*:}
  busy && { log "ABORT before $N: busy"; exit 1; }
  env $(echo "$ev" | sed 's/,NCCL_/ NCCL_/g') DSV41_INDEXER_Q_PREDECODE=1 SEQS=128 UTIL=0.88 $S/serve.sh $N >> $L
  (setsid nohup $S/gpumon.sh $N > /dev/null 2>&1 < /dev/null &)
  wait_up $N && dw $N
  $S/stop.sh $N | tee -a $L
done
EST=$($PY - <<EOF
import json
best = None
for c in """$CFG""".split():
    n, ev = c.split(":", 1)
    try:
        d = json.load(open(f"/root/dsv41/runs/{n}/dw/ctx1024_c64.json"))["decode_tok_s"]
        p = json.load(open(f"/root/dsv41/runs/{n}/dw/ctx8192_c16.json"))["prefill_tok_s"]
    except Exception:
        continue
    est = 1 / (1 / d + 0.93 / p)
    print(f"# {n}: decode {d} prefill {p} tool-mode estimate {est:.1f}")
    if best is None or est > best[0]:
        best = (est, n, ev)
try:
    d = json.load(open("/root/dsv41/runs/p12_base/dw/ctx1024_c64.json"))["decode_tok_s"]
    p = json.load(open("/root/dsv41/runs/p12_base/dw/ctx8192_c16.json"))["prefill_tok_s"]
    print(f"# p12_base: decode {d} prefill {p} tool-mode estimate {1 / (1 / d + 0.93 / p):.1f}")
except Exception:
    pass
if best:
    print(best[1], best[2])
EOF
)
log "estimates: $(echo "$EST" | grep '^#' | tr '\n' ' ')"
PICK=$(echo "$EST" | grep -v '^#' | tail -1)
[ -z "$PICK" ] && { log "ABORT: no CE end-to-end result"; log "P12B_DONE (server stopped; box idle)"; exit 1; }

# 3. R4 tool replay, A/B against r3_e64ret
N=r4_e64ce; ev=${PICK#* }
busy && { log "ABORT before $N: busy"; exit 1; }
log "$N uses ${PICK%% *} ($ev)"
env $(echo "$ev" | sed 's/,NCCL_/ NCCL_/g') DSV41_INDEXER_Q_PREDECODE=1 MODEL=/root/dsv41/model-e64 SEQS=128 UTIL=0.88 \
  MAXLEN=163840 $S/serve.sh $N --enable-prefix-caching --prefix-cache-retention-interval 128 >> $L
(setsid nohup $S/gpumon.sh $N > /dev/null 2>&1 < /dev/null &)
if wait_up $N; then
  (setsid nohup $S/kvmon.sh $N > /dev/null 2>&1 < /dev/null &)
  log "$N R2 tools C 64: $($PY $S/tool_replay.py --shapes $TS/tools_e030_hdrmin.json --conc 64 --warm-s 300 \
    --window-s 2400 --tool-delay 1 --seed 1 --json /root/dsv41/runs/$N/r2_tools_c64.json 2>&1 | tail -1 | cut -c1-900)"
fi
$S/stop.sh $N | tee -a $L
log "P12B_DONE (server stopped; box idle)"
