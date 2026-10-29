#!/bin/bash
# Run 3 layout: fresh /kaggle/working and /tmp/dsv41 (run 2's are archived, not deleted), the build's model source mapped
# to the local C1 copy, Manh's src at the given local version dir, vLLM files restored to the wheel's (cell 2r patches).
# Usage: setup_run3.sh <notebook.ipynb> <kernel-metadata.json> <arc2-2026-src version dir> [kit tgz to map]
set -euo pipefail
NB=$1; KM=$2; SRC=$3; KIT=${4:-}; D=/root/e066/r3; mkdir -p $D; exec > >(tee -a $D/setup.log) 2>&1
echo "setup_run3 $(date -u +%T)"
[ -d /root/e066/run2_working ] || { mv /kaggle/working /root/e066/run2_working; mv /tmp/dsv41 /root/e066/run2_tmp_dsv41; mv /tmp/arc_vision /root/e066/run2_arc_vision 2>/dev/null || true; }
mkdir -p /kaggle/working
[ -n "$KIT" ] && cp "$KIT" /kaggle/input/datasets/chankhavu/dsv41-tools-kit/dsv41_kit.tgz
echo "kit $(sha256sum /kaggle/input/datasets/chankhavu/dsv41-tools-kit/dsv41_kit.tgz | cut -c1-16)"
MS=$(python3 -c 'import json,sys; m=json.load(open(sys.argv[1]))["model_sources"]; assert len(m)==1, m; o,n,f,v=m[0].split("/"); print(f"{o}/{n}/{f.lower()}/{v}")' $KM)
rm -f /kaggle/input/datasets/nguyennguyen599/arc2-2026-src; ln -s $SRC /kaggle/input/datasets/nguyennguyen599/arc2-2026-src
mkdir -p /kaggle/input/models/$(dirname $MS); [ -e /kaggle/input/models/$MS ] || ln -s /root/models/c1 /kaggle/input/models/$MS
echo "model source /kaggle/input/models/$MS -> $(readlink -f /kaggle/input/models/$MS); src -> $(readlink -f /kaggle/input/datasets/nguyennguyen599/arc2-2026-src)"
python3 - $NB /root/e066/c1_gptq_sha256_manifest.json <<'PY'
import json, re, sys
nb = json.load(open(sys.argv[1])); m = json.load(open(sys.argv[2]))
s = "".join("".join(c["source"]) for c in nb["cells"] if 'INTEGRITY = json.loads(r"""' in "".join(c["source"]))
model = json.loads(re.search(r'INTEGRITY = json.loads\(r"""(.*?)"""\)', s, re.S)[1])["model"]
print("cell 1b manifest == C1's:", model == {e["path"]: [e["size"], e["sha256"]] for e in m["entries"]}, len(model))
PY
cp $NB $D/nb_run3.ipynb; sha256sum $D/nb_run3.ipynb | cut -c1-16
bash /root/e066/nbrun/nb_restore_vllm.sh
echo "SETUP3_DONE $(date -u +%T)"
