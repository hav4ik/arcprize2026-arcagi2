#!/bin/bash
# E112 2-bit tool model (user, 2026-10-24 18:00): same recipe as tool E128 2-bit, only the keep set differs.
# keep = tool stats, contrib, tool-block weight 10, 112 experts (E96 < E112 < E128 nested); Engram 64 GiB tool table;
# calibration = the E128 replay (tool calib tokens, 1.4M, seed 61, 10% rows) under the E112 mask;
# GPTQ w13 u2 g128 + w2 u2 g64, percdamp 1.0 (8 GPUs); then two TP4 servers (A 0-3, B 4-7) with runtime masks,
# dev NLL + first-turn probe on B, then the sub80s40 + header tool eval across both servers:
# run 1 = attempt 0 only (pass@1), run 2 = same tag with --attempts 2 (adds attempt 1 -> pass@2).
S="ssh -o BatchMode=yes -p 50293 root@147.185.60.21"
q() { timeout 7200 $S "$1" 2>&1 | grep -v "^Welcome\|^Have fun\|^AI agents"; }
up() { until timeout 20 $S "grep -qE 'Application startup complete|Traceback' /workspace/dsv4/runs/$1/server.log" 2>/dev/null; do sleep 20; done; q "grep -E 'Application startup complete|KV cache size|Error' /workspace/dsv4/runs/$1/server.log | grep -v UserWarning | grep -v 'multiproc_executor.py:944\]   ' | tail -3 | cut -c1-200"; }
D=/workspace/dsv4/data
G=$D/engram/masks/engram_top64gib.npz
K=$D/reap/keep_tool/keep_contrib_dec_sp0_w10_e112.npy
X=/dev/shm/e043/calx_tool_e112_by_layer
O=/dev/shm/e043/gptq_tool_e112_u2
TAG=e112t_w10_u2_gptq_eng64_sub80
# download.log is appended across downloads: only a DONE after this restart's marker counts, plus a shard check
until q "awk '/^=== restart 2026-10-24T18:00/{f=1} f && /^DONE/{d=1} END{exit !d}' /workspace/dsv4/logs/download.log && [ \$(ls /dev/shm/dsv4/DeepSeek-V4.1-Flash-1017/*.safetensors | wc -l) = 48 ] && [ -z \"\$(ls /dev/shm/dsv4/DeepSeek-V4.1-Flash-1017/.cache/huggingface/download/*.incomplete 2>/dev/null)\" ] && echo y" | grep -q y; do sleep 60; done
date -u; echo "checkpoint in /dev/shm; native server B for the calibration dump"
q "cd /workspace/dsv4; du -sh /dev/shm/dsv4; ls /dev/shm/dsv4/DeepSeek-V4.1-Flash-1017/*.safetensors | wc -l; mkdir -p /dev/shm/e043; echo '{\"engram_keep\": null, \"expert_keep\": null, \"observe\": null}' > /dev/shm/e043/ctrlB.json; GPUS=4,5,6,7 TP=4 EP=1 PORT=18001 CTRL=/dev/shm/e043/ctrlB.json UTIL=0.88 SEQS=64 MBT=2048 bash code/scripts/serve.sh tp4B_tool112"
up tp4B_tool112
date -u; echo "calibration dump under the E112 mask"
q "cd /workspace/dsv4; rm -rf /dev/shm/e043/obs/calx_tool112; tvenv/bin/python code/scripts/replay_reap.py --tok data/tok_tool_calib --tokens 1400000 --seed 61 --server http://127.0.0.1:18001 --ctrl /dev/shm/e043/ctrlB.json --obs-dir /dev/shm/e043/obs/calx_tool112 --dump-x --dump-x-frac 0.1 --engram-keep $G --expert-keep $K > logs/calx_tool_e112_replay.log 2>&1; tail -1 logs/calx_tool_e112_replay.log; venv/bin/python code/scripts/calx_transpose.py --obs /dev/shm/e043/obs/calx_tool112 --out $X > logs/calx_tool112_transpose.log 2>&1 && rm -rf /dev/shm/e043/obs/calx_tool112; tail -1 logs/calx_tool112_transpose.log; df -h /dev/shm | tail -1"
q "cd /workspace/dsv4; bash code/scripts/stop.sh tp4B_tool112 4,5,6,7; sleep 10; mkdir -p $O logs/gptq_tool_e112; rm -f logs/gptq_tool_e112/pids; for g in 0 1 2 3 4 5 6 7; do L=''; for l in \$(seq \$g 8 39); do L=\"\$L \$l\"; done; CUDA_VISIBLE_DEVICES=\$g PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True setsid nohup venv/bin/python code/scripts/gptq_experts.py --calx $X --layers \$L --bits13 2 --bits2 2 --group13 128 --group2 64 --batch 4 --keep $K --percdamp 1.0 --out $O > logs/gptq_tool_e112/gpu\$g.log 2>&1 < /dev/null & echo \$! >> logs/gptq_tool_e112/pids; done; echo gptq-launched"
while q "for p in \$(cat /workspace/dsv4/logs/gptq_tool_e112/pids); do kill -0 \$p 2>/dev/null && echo run; done" | grep -q run; do sleep 30; done
N=$(q "ls $O/*.json | wc -l" | tail -1)
[ "$N" = 40 ] || { echo "GPTQ incomplete ($N/40); stopping"; exit 1; }
q "cd /workspace/dsv4; cat logs/gptq_tool_e112/gpu*.log | tvenv/bin/python -c \"
import json,sys
r=[json.loads(l) for l in sys.stdin if l.startswith('{')]
m=lambda k: sum(x['mean_rel_out_err'][k] for x in r)/len(r)
print(len(r),'layers; mean rel out err w13 %.4f (rtn %.4f) w2 %.4f (rtn %.4f); worst w13 %.4f w2 %.4f; decode max %s' % (m('w13_gptq'),m('w13_rtn'),m('w2_gptq'),m('w2_rtn'),max(x['mean_rel_out_err']['w13_gptq'] for x in r),max(x['mean_rel_out_err']['w2_gptq'] for x in r),max(max(x['decode_check_max_abs'].values()) for x in r)))\"; rm -rf $X; du -sh $O; df -h /dev/shm | tail -1"
date -u; echo "GPTQ done; starting the two 2-bit servers"
SCH13='{"quant_method": "humming", "weight_dtype": "uint2", "weight_scale_dtype": "bfloat16", "group_size": 128, "has_zero_point": true}'
for s in A B; do
  [ $s = A ] && { GP=0,1,2,3; PO=18000; } || { GP=4,5,6,7; PO=18001; }
  q "cd /workspace/dsv4; echo '{\"engram_keep\": \"$G\", \"expert_keep\": \"$K\", \"observe\": null}' > /dev/shm/e043/ctrl$s.json; export DSV41_EXPERT_HUMMING='$SCH13'; export DSV41_EXPERT_HUMMING_W2='{\"group_size\": 64}'; export E043_GPTQ_DIR=$O; GPUS=$GP TP=4 EP=0 PORT=$PO CTRL=/dev/shm/e043/ctrl$s.json UTIL=0.86 SEQS=192 MBT=8192 bash code/scripts/serve.sh tp4${s}_$TAG"
  sleep 25
done
up tp4A_$TAG
up tp4B_$TAG
date -u; echo "dev screen on B (64 tool dev traces, first-turn probe)"
q "cd /workspace/dsv4; SERVER=http://127.0.0.1:18001 CTRL=/dev/shm/e043/ctrlB.json TOK=data/tok_tool_dev NSEQ=64 MAXLEN=60000 bash code/scripts/nll_candidates.sh data/nll_tool_dev.jsonl e112t_w10_u2_gptq_eng64:$G:$K | grep nll_gen | cut -c1-260; tvenv/bin/python code/scripts/gen_probe.py --server http://127.0.0.1:18001 --ctrl /dev/shm/e043/ctrlB.json --parquet data/tool_dev_probe.parquet --n 8 --samples 2 --max-tokens 6000 --workers 25 --out data/gen_probe_tool_dev.jsonl e112t_w10_u2_gptq_eng64:$G:$K | grep '^=='"
# make sure both servers run the E112 masks (the screen tools write the control file), then poke each server
for s in A B; do
  [ $s = A ] && PO=18000 || PO=18001
  q "cd /workspace/dsv4; echo '{\"engram_keep\": \"$G\", \"expert_keep\": \"$K\", \"observe\": null}' > /dev/shm/e043/ctrl$s.json.tmp && mv /dev/shm/e043/ctrl$s.json.tmp /dev/shm/e043/ctrl$s.json; curl -s -m 120 localhost:$PO/v1/completions -H 'Content-Type: application/json' -d '{\"model\":\"dsv41\",\"prompt\":[0,5,6,7],\"max_tokens\":1}' >/dev/null; sleep 5; grep 'E043\] ctrl' runs/tp4${s}_$TAG/server.log | tail -1 | cut -c60-300"
done
cd /home/vu/Workspace/ARC-AGI-2
BASES=http://127.0.0.1:18000/v1,http://127.0.0.1:18001/v1
date -u; echo "run 1: attempt 0 ($TAG)"
PRESET=sub80s40 SALT=tool_$TAG WORKERS=167 BASE=$BASES bash experiments/E043_dsv41_real_pruning/scripts/eval_arc.sh $TAG --rescue-code --attempts 1
date -u; echo RUN1-DONE
PRESET=sub80s40 SALT=tool_$TAG WORKERS=167 BASE=$BASES bash experiments/E043_dsv41_real_pruning/scripts/eval_arc.sh $TAG --rescue-code --attempts 2
date -u; echo CHAIN-E112-DONE
