#!/bin/bash
# The candidate's dataset inputs, version-pinned, sequential (one stream besides the model's two). Token on stdin.
#  - nguyennguyen599/vllm-wheels/versions/11 (pinned by kernel-metadata): the 294 top-level files of wheels/wheels/
#    (v11's vllm wheel g8cf0c0664b + 293 others, listed by pod 2 through the API) + .gitattributes, per file, size-checked.
#    The 4 nested source dirs (1,374 files; no cell installs from them) are left out, as on pods 2 and 3.
#  - chankhavu/dsv41-tools-kit/versions/5 and nguyennguyen599/arc2-2026-src/versions/$SRC_VER: whole-version downloads.
#  - nguyennguyen599/cu13-install/versions/2: 102 files per file; the 5 epoch-colon names (Kaggle answers 404) come from
#    the Ubuntu 24.04 archive via apt-get download <pkg>=<version>. Sizes checked against the listing.
# Logs: /root/p4/logs/dl_*.log; ends with INPUTS_DONE.
set -uo pipefail
read -r KAGGLE_API_TOKEN; export KAGGLE_API_TOKEN
S=/root/p4/scripts; L=/root/p4/logs; SRC_VER=${SRC_VER:-23}
W=/root/kin/vllm-wheels-v11
KAGGLEHUB_CACHE=$W/.khcache /root/kgl/bin/python -I $S/dl_kaggle_files.py dataset nguyennguyen599/vllm-wheels/versions/11 $S/v11_min.tsv $W > $L/dl_v11_min.log 2>&1
KAGGLEHUB_CACHE=$W/.khcache /root/kgl/bin/python -I $S/dl_kaggle_files.py dataset nguyennguyen599/vllm-wheels/versions/11 $S/v11_rest.tsv $W > $L/dl_v11_rest.log 2>&1
rm -rf $W/.khcache
echo "v11: $(tail -1 $L/dl_v11_min.log) / $(tail -1 $L/dl_v11_rest.log); $(ls $W/wheels/wheels/*.whl | wc -l) wheels; vllm wheel sha256 $(sha256sum $W/wheels/wheels/vllm-0.32.1rc1.dev0+g8cf0c0664b-*.whl | cut -c1-12)" | tee $L/dl_v11.summary
for h in chankhavu/dsv41-tools-kit/versions/5 nguyennguyen599/arc2-2026-src/versions/$SRC_VER; do
  for i in 1 2 3; do
    KAGGLEHUB_CACHE=/root/kin/kh /root/kgl/bin/python -I -c "import kagglehub, sys; print('GOT', kagglehub.dataset_download(sys.argv[1]))" $h >> $L/dl_whole.log 2>&1 && break
    sleep $((i * 20))
  done
done
C=/root/kin/cu13; mkdir -p $C
KAGGLEHUB_CACHE=$C/.khcache /root/kgl/bin/python -I $S/dl_files2.py nguyennguyen599/cu13-install/versions/2 $S/cu13_kaggle.tsv $C > $L/dl_cu13.log 2>&1
unset KAGGLE_API_TOKEN; rm -rf $C/.khcache
cd $C && apt-get update -qq > /dev/null 2>&1; while IFS=$'\t' read -r f size; do
  n=$(python3 -c 'import sys,urllib.parse; f=urllib.parse.unquote(sys.argv[1]); p,v,_=f.split("_",2); print(f"{p}={v}")' "$f")
  apt-get download -q "$n" > /dev/null 2>&1; [ -f "$f" ] || mv "$(python3 -c 'import sys,urllib.parse; print(urllib.parse.unquote(sys.argv[1]))' "$f")" "$f" 2>/dev/null
  echo "APT $n -> $f $( [ -f "$f" ] && stat -c %s "$f") (listed $size)"
done < $S/cu13_apt.tsv >> $L/dl_cu13.log 2>&1
python3 - $C $S/cu13_kaggle.tsv $S/cu13_apt.tsv <<'PY' | tee -a $L/dl_cu13.log
import os, sys
d = sys.argv[1]; rows = [l.rstrip("\n").split("\t") for f in sys.argv[2:] for l in open(f) if l.strip()]
bad = [p for p, s in rows if not os.path.exists(os.path.join(d, p)) or os.path.getsize(os.path.join(d, p)) != int(s)]
print(f"CU13 {len(rows) - len(bad)}/{len(rows)} files at the listed size; bad: {bad}")
PY
echo "INPUTS_DONE $(date -u +%T)" >> $L/dl_whole.log
