#!/bin/bash
# A Kaggle-like fresh kernel venv (pod 2's kenv_fresh_freeze.txt: 48 packages; ipykernel, numpy, scipy, pandas, ...):
# built at /root/kenv (so its shebangs are right), copied pristine to /root/kenv_fresh; nb_run.sh resets /root/kenv from
# it before every run, so Manh's `uv pip install` cells install the full stack into it as on Kaggle. Registers the
# papermill kernelspec py313b32m -> /root/kenv/bin/python.
set -euo pipefail
PYB=$(/root/kgl/bin/uv python find 3.13); rm -rf /root/kenv /root/kenv_fresh
/usr/bin/uv venv -p "$PYB" /root/kenv > /root/e066/c1text/kenv.log 2>&1
/usr/bin/uv pip install --python /root/kenv/bin/python -r /root/e066/c1text/kenv_fresh_freeze.txt >> /root/e066/c1text/kenv.log 2>&1
/root/kenv/bin/python -m ipykernel install --prefix /root/kgl --name py313b32m --display-name "py313 kenv (Kaggle-like)" > /dev/null
cp -a /root/kenv /root/kenv_fresh
echo "kenv: $(/root/kenv/bin/python -V), $(/usr/bin/uv pip freeze --python /root/kenv/bin/python | wc -l) packages, freeze == listing: $(diff <(/usr/bin/uv pip freeze --python /root/kenv/bin/python) /root/e066/c1text/kenv_fresh_freeze.txt > /dev/null && echo yes || echo NO); $(du -sh /root/kenv_fresh | cut -f1)"
cat /root/kgl/share/jupyter/kernels/py313b32m/kernel.json | tr -d '\n '; echo
