#!/bin/bash
# E066: 2-bit GPTQ of a candidate's kept experts with the tool E128 recipe (E043 chain_t2: uint2, group 128 gate/up,
# group 64 down, zero points, percdamp 1.0), calibrated on training rollouts of both harnesses replayed under the
# candidate's own mask (MoE inputs of 10% of positions, 1.4M tokens: half image harness, half text twin).
# Usage: gptq_candidate.sh dump  <name> <keep.npy>   server up (native weights, observer patches): capture + transpose
#        gptq_candidate.sh quant <name> <keep.npy>   server stopped (needs the GPUs): GPTQ on 4 GPUs, layers g, g+4, ...
#   env: SERVER (http://127.0.0.1:18001), CTRL (/dev/shm/e043/ctrlB.json), TOKENS (1400000)
# Output: /workspace/dsv4/gptq/<name> (E043_GPTQ_DIR for serving, see serve_gptq.sh).
set -eu
D=/workspace/dsv4
S=$D/repo/experiments/E066_dsv41_image_pruning/scripts
S43=$D/repo/experiments/E043_dsv41_real_pruning/scripts
NAME=$2; K=$(realpath "$3")
G=$D/data/e066/engram/e043tool_top64gib.npz
OBS=$D/obs/calx_$NAME; X=$D/calx/${NAME}_by_layer; O=$D/gptq/$NAME
mkdir -p $D/logs/gptq_$NAME
case $1 in
  dump)  # half image-harness, half text-harness traffic (TOKENS each / 2), merged by calx_transpose
    rm -rf ${OBS}_img ${OBS}_txt $X
    HALF=$(( ${TOKENS:-1400000} / 2 ))
    $D/tvenv/bin/python $S/replay_mm.py --tok $D/data/e066/tok_img_calib_s/shard* \
      --parquet $D/data/e066/e065_image.parquet --tokens $HALF --seed 61 \
      --server ${SERVER:-http://127.0.0.1:18001} --ctrl ${CTRL:-/dev/shm/e043/ctrlB.json} --obs-dir ${OBS}_img \
      --dump-x --dump-x-frac 0.1 --engram-keep $G --expert-keep $K > $D/logs/gptq_$NAME/replay_img.log 2>&1
    tail -1 $D/logs/gptq_$NAME/replay_img.log
    $D/tvenv/bin/python $S43/replay_reap.py --tok $D/data/e066/tok_txt_calib_sub --tokens $HALF --seed 61 \
      --server ${SERVER:-http://127.0.0.1:18001} --ctrl ${CTRL:-/dev/shm/e043/ctrlB.json} --obs-dir ${OBS}_txt \
      --dump-x --dump-x-frac 0.1 --engram-keep $G --expert-keep $K > $D/logs/gptq_$NAME/replay_txt.log 2>&1
    tail -1 $D/logs/gptq_$NAME/replay_txt.log
    mkdir -p $D/calx
    $D/venv/bin/python $S43/calx_transpose.py --obs ${OBS}_img ${OBS}_txt --out $X > $D/logs/gptq_$NAME/transpose.log 2>&1
    rm -rf ${OBS}_img ${OBS}_txt
    tail -1 $D/logs/gptq_$NAME/transpose.log ;;
  quant)
    mkdir -p $O
    pids=""
    for g in 0 1 2 3; do
      LAYERS=$(seq $g 4 39 | tr '\n' ' ')
      CUDA_VISIBLE_DEVICES=$g PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $D/venv/bin/python $S43/gptq_experts.py \
        --model-dir $D/DeepSeek-V4.1-Flash-1017 --calx $X --layers $LAYERS --bits13 2 --bits2 2 --group13 128 \
        --group2 64 --batch 4 --keep $K \
        --percdamp 1.0 --out $O > $D/logs/gptq_$NAME/gpu$g.log 2>&1 &
      pids="$pids $!"
    done
    rc=0
    for p in $pids; do wait $p || rc=1; done
    [ $rc = 0 ] && echo "GPTQ_DONE $NAME $(du -sh $O | cut -f1)" || { echo "GPTQ_FAILED $NAME"; exit 1; } ;;
  *) echo "usage: gptq_candidate.sh dump|quant <name> <keep.npy>" >&2; exit 2 ;;
esac
