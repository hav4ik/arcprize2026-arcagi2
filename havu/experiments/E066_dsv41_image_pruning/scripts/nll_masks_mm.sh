#!/bin/bash
# E066: paired dev NLL of expert keep masks on held-out training rollouts of both harnesses (runtime masks on one
# server, E043 nll_candidates.sh with nll_eval_mm.py). Per mask, NSEQ dev rollouts of each set in SETS (img = E058
# image harness, txt = its text twin), one per held-out family, the same rollouts for every mask (fixed seed); every
# request has a unique cache_salt, so no KV cached under another mask is reused. Records carry `batch` (e065_* =
# image, e065t_* = text), so per-harness NLL can be split afterwards.
# Usage: nll_masks_mm.sh <out.jsonl> <tag>:<expert_keep.npy|none> [...]
#   env: SERVER (http://127.0.0.1:18001), CTRL (/dev/shm/e043/ctrlB.json), ENGRAM (tool 64 GiB mask; none = full),
#        SETS ("img txt"), NSEQ (32 per set), MAXLEN (60000), SEED (0), WORKERS (4), DATA (/workspace/dsv4/data/e066),
#        MODE (nll; gen = free-generation probe of the first turn, gen_probe_mm.py, NSEQ prompts per set, MAXTOK)
# The control file is restored to its previous content at the end (it used to be reset to "no expert mask", which
# silently unmasked a runtime-mask eval server: the E066 C1 evals of 12:17-16:35 UTC on 2026-10-29 ran unmasked).
# Layer sweep: NSEQ=8 MAXLEN=40000 nll_masks_mm.sh sweep.jsonl base:DIR/keep_base.npy base2:DIR/keep_base.npy \
#   l00:DIR/keep_l00.npy ... l39:DIR/keep_l39.npy, then class_stats.py sens --nll sweep.jsonl.
set -eu
OUT=$1; shift
D=/workspace/dsv4
DATA=${DATA:-$D/data/e066}
S=$D/repo/experiments/E066_dsv41_image_pruning/scripts
SERVER=${SERVER:-http://127.0.0.1:18001}
CTRL=${CTRL:-/dev/shm/e043/ctrlB.json}
ENG=${ENGRAM:-$DATA/engram/e043tool_top64gib.npz}
e=null; [ "$ENG" != none ] && e="\"$(realpath $ENG)\""
PREV=$(cat $CTRL 2>/dev/null || true)
for spec in "$@"; do
  IFS=: read -r tag exp <<< "$spec"
  x=null; [ "$exp" != none ] && x="\"$(realpath $exp)\""
  echo "{\"engram_keep\": $e, \"expert_keep\": $x, \"observe\": null}" > $CTRL.tmp && mv $CTRL.tmp $CTRL
  curl -s -m 300 $SERVER/v1/completions -H 'Content-Type: application/json' \
    -d '{"model":"dsv41","prompt":[0,5,6,7],"max_tokens":1}' > /dev/null  # the control file is re-read per forward
  for set in ${SETS:-img txt}; do
    case $set in
      img) TOK=$(ls -d $DATA/tok_img_dev_s/shard*); PQ=$DATA/e065_image.parquet ;;
      txt) TOK=$DATA/tok_txt_dev; PQ=$DATA/e065_text.parquet ;;
      *) echo "unknown set $set" >&2; exit 2 ;;
    esac
    if [ "${MODE:-nll}" = gen ]; then  # free-generation probe: first turn of NSEQ dev prompts per set
      $D/tvenv/bin/python $S/gen_probe_mm.py --tok $TOK --parquet $PQ --n-seq ${NSEQ:-25} --seed ${SEED:-0} \
        --max-tokens ${MAXTOK:-16000} --workers ${NSEQ:-25} --server $SERVER --tag "$tag" --out "$OUT"
    else
      $D/tvenv/bin/python $S/nll_eval_mm.py --tok $TOK --parquet $PQ --per-task --n-seq ${NSEQ:-32} --seed ${SEED:-0} \
        --max-len ${MAXLEN:-60000} --workers ${WORKERS:-4} --server $SERVER --tag "$tag" --out "$OUT"
    fi
  done
done
if [ -n "$PREV" ]; then echo "$PREV" > $CTRL.tmp; else echo "{\"engram_keep\": $e, \"expert_keep\": null, \"observe\": null}" > $CTRL.tmp; fi
mv $CTRL.tmp $CTRL
