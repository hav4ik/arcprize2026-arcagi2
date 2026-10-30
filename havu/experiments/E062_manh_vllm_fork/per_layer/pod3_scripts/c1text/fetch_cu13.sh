#!/bin/bash
# cu13-install v2 at /root/kin/cu13 (Manh's cell 1 dpkg-installs it; commit runs raise without it): 102 files per file
# from Kaggle (version-pinned, token on stdin), and the 5 whose names hold an epoch colon (the Kaggle API answers 404)
# from the Ubuntu 24.04 archive with apt-get download <pkg>=<exact version>. Sizes checked against the listing.
set -uo pipefail
read -r KAGGLE_API_TOKEN; export KAGGLE_API_TOKEN
D=/root/kin/cu13; mkdir -p $D; C=/root/e066/c1text
/root/kgl/bin/python -I $C/dl_files2.py nguyennguyen599/cu13-install/versions/2 $C/cu13_kaggle.tsv $D > $C/dl_cu13.log 2>&1
unset KAGGLE_API_TOKEN; rm -rf $D/.khcache; tail -1 $C/dl_cu13.log
cd $D && apt-get update -qq > /dev/null 2>&1; while IFS=$'\t' read -r f size; do
  n=$(python3 -c 'import sys,urllib.parse; f=urllib.parse.unquote(sys.argv[1]); p,v,_=f.split("_",2); print(f"{p}={v}")' "$f")
  apt-get download -q "$n" > /dev/null 2>&1; [ -f "$f" ] || mv "$(python3 -c 'import sys,urllib.parse; print(urllib.parse.unquote(sys.argv[1]))' "$f")" "$f" 2>/dev/null
  echo "APT $n -> $f $( [ -f "$f" ] && stat -c %s "$f") (listed $size)"
done < $C/cu13_apt.tsv
python3 - $D $C/cu13_kaggle.tsv $C/cu13_apt.tsv <<'PY'
import os, sys
d = sys.argv[1]; rows = [l.rstrip("\n").split("\t") for f in sys.argv[2:] for l in open(f) if l.strip()]
bad = [p for p, s in rows if not os.path.exists(os.path.join(d, p)) or os.path.getsize(os.path.join(d, p)) != int(s)]
print(f"CU13 {len(rows) - len(bad)}/{len(rows)} files at the listed size; bad: {bad}")
PY
