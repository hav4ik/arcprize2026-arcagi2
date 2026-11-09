#!/bin/bash
# E066: upload a candidate export (export_candidate.sh, files <= 5 GB) to a private HF model repo, with its model card
# (cards/README_<name>.md: {SIZE} and the 2-bit dev NLL of <qtag> in results/q_screen.jsonl filled in, {PARITY} from
# <ptag> if given) and serving/ (the E043 patches the layout needs), then compare the repo with the folder by name and
# size (hf_verify_upload.py). Token: $HF_TOKEN (environment only).
# Usage: upload_candidate.sh <name> <repo> <qtag> [<ptag>]
set -uo pipefail
[ -n "${HF_TOKEN:-}" ] || { echo "HF_TOKEN not set"; exit 1; }
D=/workspace/dsv4
S=$D/repo/experiments/E066_dsv41_image_pruning/scripts
E43=$D/repo/experiments/E043_dsv41_real_pruning
NAME=$1; REPO=$2; QTAG=$3; PTAG=${4:-}
X=$D/export/${NAME}_gptq
cp $S/cards/README_${NAME}.md $X/README.md
mkdir -p $X/serving
cp $E43/patches/apply_compact_engram.py $E43/patches/apply_e043_humming.py $E43/patches/e043_humming.py $X/serving/
rm -rf $X/.cache
$D/venv/bin/python - $X/README.md $D/results/q_screen.jsonl "$QTAG" "$PTAG" $(du -sb $X | cut -f1) <<'PY'
import json, sys
card, res, qtag, ptag, size = sys.argv[1:]
full = {"e065_": 0.75221, "e065t_": 0.71902}
by = {}
for l in open(res):
    r = json.loads(l); by.setdefault(r["tag"], {})[(r["batch"], r["task_id"], r["attempt"])] = r
def nll(t, h):
    ks = [k for k in by[t] if k[0].startswith(h)]
    assert len(ks) == 32, (t, h, len(ks))
    return sum(by[t][k]["nll_gen"] for k in ks) / sum(by[t][k]["n_gen"] for k in ks)
s = open(card).read().replace("{SIZE}", f"{int(size):,}")
for key, h in (("{Q_IMG}", "e065_"), ("{Q_TXT}", "e065t_")):
    v = nll(qtag, h); s = s.replace(key, f"{v:.4f} (+{100 * (v / full[h] - 1):.1f}%)")
if ptag:
    s = s.replace("{PARITY}", f"{nll(ptag, 'e065_'):.4f} image / {nll(ptag, 'e065t_'):.4f} text")
assert "{" + "SIZE}" not in s and "{Q_" not in s and "{PARITY}" not in s
open(card, "w").write(s)
PY
[ $? = 0 ] || { echo "CARD_FAILED $NAME"; exit 1; }
date -u; echo "uploading $X -> $REPO"
for i in 1 2 3; do
  $D/tvenv/bin/python $E43/scripts/hf_upload.py --repo $REPO --type model --path $X \
    --message "E066 $NAME: 2-bit GPTQ, tool use with images, safetensors <= 5 GB" && break
  echo "upload attempt $i failed; retrying in 60 s"; sleep 60
done
date -u
if $D/tvenv/bin/python $E43/scripts/hf_verify_upload.py --repo $REPO --path $X; then echo "UPLOAD_DONE $NAME $(du -sb $X | cut -f1)"
else echo "UPLOAD_FAILED $NAME"; exit 1; fi
