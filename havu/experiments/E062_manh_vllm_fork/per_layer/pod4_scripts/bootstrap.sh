#!/bin/bash
# Pod 4 bootstrap: the download venv /root/kgl (system Python 3.12 + kagglehub + huggingface_hub), the directory layout,
# the sudo check, and the Kaggle-like kernel venv (/root/kenv built from kenv_fresh_freeze.txt, pristine copy
# /root/kenv_fresh). Log: /root/p4/logs/bootstrap.log
set -euo pipefail
S=/root/p4/scripts; L=/root/p4/logs; mkdir -p $L /root/models/e128 /root/kin /root/data /root/runs
exec > >(tee -a $L/bootstrap.log) 2>&1
echo "bootstrap $(date -u +%T)"
python3 -m venv /root/kgl
/root/kgl/bin/pip install -q --upgrade pip > /dev/null
/root/kgl/bin/pip install -q kagglehub "huggingface_hub[hf_xet]" > $L/pip_kgl.log 2>&1
echo "kgl: $(/root/kgl/bin/python -c 'import kagglehub, huggingface_hub; print(kagglehub.__version__, huggingface_hub.__version__)')"
# Kaggle-like kernel venv: Python 3.13 (Kaggle's image Python), the 48 packages of pod 2's fresh-kernel freeze
uv python install 3.13 > $L/uv_python.log 2>&1
PYB=$(uv python find 3.13)
rm -rf /root/kenv /root/kenv_fresh
uv venv -p "$PYB" /root/kenv > $L/kenv.log 2>&1
uv pip install --python /root/kenv/bin/python -r $S/kenv_fresh_freeze.txt >> $L/kenv.log 2>&1
cp -a /root/kenv /root/kenv_fresh
echo "kenv: $(/root/kenv/bin/python -V), $(uv pip freeze --python /root/kenv/bin/python | wc -l) packages, freeze == listing: $(diff <(uv pip freeze --python /root/kenv/bin/python) $S/kenv_fresh_freeze.txt > /dev/null && echo yes || echo NO); $(du -sh /root/kenv_fresh | cut -f1)"
echo "BOOTSTRAP_DONE $(date -u +%T)"
