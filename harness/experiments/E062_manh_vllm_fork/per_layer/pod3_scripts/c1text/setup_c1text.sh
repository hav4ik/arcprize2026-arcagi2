#!/bin/bash
# c1text run layout: archive the previous /kaggle/working, /tmp/dsv41, /tmp/arc_vision under /root/e066/<tag>_*;
# fresh /kaggle/working; vllm-wheels -> the v11 tree; cu13-install -> /root/kin/cu13; competition dir = the local data/ files (commit) or with the
# test file replaced by the evalx2t_s2026 workload (rerun); C1 under the build's model_sources path; cell 1b manifest
# check; the kernel venv /root/kenv reset from /root/kenv_fresh (Manh's cells install v11 into it; no 2r).
# The previous run's /tmp/dsv41, /tmp/arc_vision, /dev/shm/wheels are removed (regenerable).
# Usage: setup_c1text.sh <nb.ipynb> <kernel-metadata.json> <commit|rerun> <archive tag of the previous run>
set -euo pipefail
NB=$1; KM=$2; MODE=$3; TAG=$4; D=/root/e066/c1text; exec > >(tee -a $D/setup.log) 2>&1
echo "setup_c1text $MODE $(date -u +%T)"
[ -e /root/e066/${TAG}_working ] && { echo "archive tag $TAG exists"; exit 1; }
mv /kaggle/working /root/e066/${TAG}_working; rm -rf /tmp/dsv41 /tmp/arc_vision /dev/shm/wheels  # regenerable (logs are in the working dir)
mkdir -p /kaggle/working
rm -f /kaggle/input/datasets/nguyennguyen599/vllm-wheels; ln -s /root/b32m/dl11 /kaggle/input/datasets/nguyennguyen599/vllm-wheels
rm -f /kaggle/input/datasets/nguyennguyen599/cu13-install; ln -s /root/kin/cu13 /kaggle/input/datasets/nguyennguyen599/cu13-install
C=/kaggle/input/competitions/arc-prize-2026-arc-agi-2; rm -rf $C; mkdir -p $C; cp $D/comp/*.json $C/
[ "$MODE" = rerun ] && cp /root/data/evalx2t_s2026_challenges.json $C/arc-agi_test_challenges.json
echo "test file $(sha256sum $C/arc-agi_test_challenges.json | cut -c1-16) ($MODE); wheels $(ls /kaggle/input/datasets/nguyennguyen599/vllm-wheels/wheels/wheels/*.whl | wc -l), vllm: $(ls /kaggle/input/datasets/nguyennguyen599/vllm-wheels/wheels/wheels/ | grep vllm-0)"
MS=$(python3 -c 'import json,sys; m=json.load(open(sys.argv[1]))["model_sources"]; assert len(m)==1, m; o,n,f,var,v=m[0].split("/"); print(f"{o}/{n}/{f.lower()}/{var}/{v}")' $KM)
[ -e /kaggle/input/models/$MS ] || { mkdir -p /kaggle/input/models/$(dirname $MS); ln -s /root/models/c1 /kaggle/input/models/$MS; }
echo "models: $(ls /kaggle/input/models/) -> $(readlink -f /kaggle/input/models/$MS); src -> $(readlink -f /kaggle/input/datasets/nguyennguyen599/arc2-2026-src); kit $(sha256sum /kaggle/input/datasets/chankhavu/dsv41-tools-kit/dsv41_kit.tgz | cut -c1-16)"
python3 - $NB /root/e066/c1_gptq_sha256_manifest.json <<'PY'
import json, re, sys
nb = json.load(open(sys.argv[1])); m = json.load(open(sys.argv[2]))
s = "".join("".join(c["source"]) for c in nb["cells"] if 'INTEGRITY = json.loads(r"""' in "".join(c["source"]))
model = json.loads(re.search(r'INTEGRITY = json.loads\(r"""(.*?)"""\)', s, re.S)[1])["model"]
print("cell 1b manifest == C1's:", model == {e["path"]: [e["size"], e["sha256"]] for e in m["entries"]}, len(model))
PY
rm -rf /root/kenv && cp -a /root/kenv_fresh /root/kenv  # fresh kernel venv (Manh's cells install the stack into it)
echo "kenv reset: $(/usr/bin/uv pip freeze --python /root/kenv/bin/python | wc -l) packages"
cp $NB $D/nb_${MODE}.ipynb; sha256sum $D/nb_${MODE}.ipynb | cut -c1-12
echo "SETUP_DONE $(date -u +%T)"
