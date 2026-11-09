#!/bin/bash
# stage_model.sh <kind> [--prefetch]: make a job's model present. Kinds: e128 (Kaggle chankhavu/dsv41-flash-1017-arc-tool-e128-2bit
# v1, 136 files vs bin/e128_manifest.json, sha256-verified for every file this script downloads), none.
# e128: missing/short files are downloaded from Kaggle per file (handshake: need_kaggle_token -> local token_watcher.sh -> bin/dl_with_token.sh, token on ssh stdin only).
# When the free disk is short the img-causal model (/root/models/img_causal + /root/img_causal_model, re-downloadable from HF) is
# deleted first (full mode only, and only with no notebook running). --prefetch downloads only what fits while keeping 14 GB free
# and deletes nothing, so it is safe next to a running notebook.
set -uo pipefail
KIND=$1; MODE=${2:-full}; B=/root/queue/bin; L=/root/queue/logs; mkdir -p $L
[ "$KIND" = none ] && exit 0
[ "$KIND" = e128 ] || { echo "unknown model kind $KIND"; exit 2; }
T=/root/queue/stage_tmp_$$; mkdir -p $T; trap "rm -rf $T" EXIT; D=/root/models/e128; H=chankhavu/dsv41-flash-1017-arc-tool-e128-2bit/pyTorch/default/1; mkdir -p $D
python3 - $D $B/e128_manifest.json $T/missing.tsv <<'PY'
import json, os, sys
d, m, out = sys.argv[1:4]; man = json.load(open(m))
miss = [(k, v[0]) for k, v in sorted(man.items()) if not os.path.exists(os.path.join(d, k)) or os.path.getsize(os.path.join(d, k)) != v[0]]
open(out, "w").write("".join(f"{k}\t{s}\n" for k, s in miss)); print(f"missing {len(miss)} files, {sum(s for _, s in miss) / 1e9:.1f} GB")
PY
[ -s $T/missing.tsv ] || { echo "e128 complete"; exit 0; }
free_gb() { df -BG --output=avail / | tail -1 | tr -dc 0-9; }
need_gb=$(awk -F'\t' '{s+=$2} END {printf "%d", s/1e9+1}' $T/missing.tsv)
if [ "$MODE" = --prefetch ]; then
  budget=$(( $(free_gb) - 14 )); [ $budget -gt 1 ] || { echo "no room to prefetch"; exit 0; }
  awk -F'\t' -v b=$budget 'BEGIN{b*=1e9} {if (s+$2<=b) {print; s+=$2}}' $T/missing.tsv > $T/pre.tsv
  cp $T/pre.tsv $T/missing.tsv; echo "prefetch $(wc -l < $T/missing.tsv) files within $budget GB"
elif [ $(free_gb) -lt $(( need_gb + 15 )) ]; then
  pgrep -f "e062pl/run_nb.py" > /dev/null && { echo "notebook running: not freeing disk"; exit 3; }
  echo "free $(free_gb) GB < need $need_gb + 15: deleting img_causal model (HF re-downloadable)"
  rm -rf /root/img_causal_model /root/models/img_causal
fi
# no token on the pod: ask the local watcher (sync/token_watcher.sh), which runs bin/dl_with_token.sh over ssh with the token on stdin
Q=/root/queue; rm -f $Q/token_ok $Q/token_fail; cp $T/missing.tsv $Q/need_kaggle_token.tmp; mv $Q/need_kaggle_token.tmp $Q/need_kaggle_token
echo "waiting for the local token watcher ($(wc -l < $Q/need_kaggle_token) files)"
w0=$(date +%s); until [ -e $Q/token_ok ] || [ -e $Q/token_fail ]; do [ $(( $(date +%s) - w0 )) -gt 10800 ] && { echo "watcher timeout"; rm -f $Q/need_kaggle_token; exit 4; }; sleep 10; done
[ -e $Q/token_ok ] || { echo "download failed"; exit 6; }
rc=0
[ "$MODE" = --prefetch ] && exit 0
python3 - $D $B/e128_manifest.json <<'PY' || exit 5
import hashlib, json, os, sys
from concurrent.futures import ThreadPoolExecutor
d, m = sys.argv[1:3]; man = json.load(open(m)); fs = list(man)
def h(k):
    p = os.path.join(d, k)
    if not os.path.exists(p) or os.path.getsize(p) != man[k][0]: return k, "missing/size"
    x = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""): x.update(b)
    return k, "ok" if x.hexdigest() == man[k][1] else "sha"
with ThreadPoolExecutor(8) as ex: res = list(ex.map(h, fs))
bad = [r for r in res if r[1] != "ok"]; print(f"VERIFY {len(res) - len(bad)}/{len(res)} ok; bad {bad}"); sys.exit(1 if bad else 0)
PY
v=$?
if [ $v -eq 0 ] && [ $rc -eq 0 ]; then echo "e128 staged"; exit 0; fi; exit 6
