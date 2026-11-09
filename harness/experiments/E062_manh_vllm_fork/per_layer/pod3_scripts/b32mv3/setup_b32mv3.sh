#!/bin/bash
# B32-M v3 replica layout (rerun mode): archive the previous /kaggle/working; fresh /kaggle/working, /tmp/dsv41,
# /tmp/arc_vision, /dev/shm/wheels; inputs as at the submission (2026-10-28 23:37 UTC): vllm-wheels v9 (= v8),
# arc2-2026-src v22, cu13-install v2, kit v5, the E128 Kaggle model copy (verified against cell 1b's manifest);
# competition dir = the local data/ files with the test file replaced by the 240-task evalx2t_s2026 workload; the
# kernel venv /root/kenv reset from /root/kenv_fresh (Manh's cells install the v9 stack; cell 2r patches it).
# Usage: setup_b32mv3.sh <archive tag of the previous working dir>
set -euo pipefail
TAG=$1; D=/root/e066/b32mv3; NB=$D/nb_build.ipynb; KM=$D/kernel-metadata.json; exec > >(tee -a $D/setup.log) 2>&1
echo "setup_b32mv3 $(date -u +%T)"
grep -q "VERIFY OK: 136/136" $D/verify_e128.log || { echo "E128 not verified"; exit 1; }
[ -e /root/e066/${TAG}_working ] && { echo "archive tag $TAG exists"; exit 1; }
mv /kaggle/working /root/e066/${TAG}_working; rm -rf /tmp/dsv41 /tmp/arc_vision /dev/shm/wheels; mkdir -p /kaggle/working
I=/kaggle/input/datasets/nguyennguyen599
rm -f $I/vllm-wheels $I/cu13-install $I/arc2-2026-src
ln -s /root/b32m/dl9 $I/vllm-wheels; ln -s /root/kin/cu13 $I/cu13-install; ln -s /root/khsrc/datasets/nguyennguyen599/arc2-2026-src/versions/22 $I/arc2-2026-src
rm -rf /kaggle/input/models; MS=$(python3 -c 'import json,sys; m=json.load(open(sys.argv[1]))["model_sources"]; assert len(m)==1, m; o,n,f,var,v=m[0].split("/"); print(f"{o}/{n}/{f.lower()}/{var}/{v}")' $KM)
mkdir -p /kaggle/input/models/$(dirname $MS); ln -s /root/models/e128 /kaggle/input/models/$MS
C=/kaggle/input/competitions/arc-prize-2026-arc-agi-2; rm -rf $C; mkdir -p $C; cp /root/e066/c1text/comp/*.json $C/; cp /root/data/evalx2t_s2026_challenges.json $C/arc-agi_test_challenges.json
echo "inputs: $(ls -la $I/ /kaggle/input/models/$(dirname $MS)/ | grep -- '->' | awk '{print $9, $10, $11}' | tr '\n' ';')"
echo "wheels $(ls $I/vllm-wheels/wheels/wheels/*.whl | wc -l), $(ls $I/vllm-wheels/wheels/wheels/ | grep vllm-0); cu13 $(ls $I/cu13-install | wc -l) files; kit $(sha256sum /kaggle/input/datasets/chankhavu/dsv41-tools-kit/dsv41_kit.tgz | cut -c1-16); test file $(sha256sum $C/arc-agi_test_challenges.json | cut -c1-16)"
rm -rf /root/kenv && cp -a /root/kenv_fresh /root/kenv; echo "kenv reset: $(/usr/bin/uv pip freeze --python /root/kenv/bin/python | wc -l) packages"
cp $NB $D/nb_rerun.ipynb; sha256sum $D/nb_rerun.ipynb | cut -c1-16
echo "SETUP_DONE $(date -u +%T)"
