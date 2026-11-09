#!/bin/bash
# Pod prep after box_install.sh + the E128 download: kit (+ its vLLM overlay over the venv, sha-checked), tool venv
# (Kaggle's tool packages), eval data. Log: /root/podprep.log
set -euo pipefail
exec > >(tee -a /root/podprep.log) 2>&1
echo "[$(date -u +%T)] prep start"
rm -rf /root/kit && mkdir -p /root/kit && tar xzf /root/podkit/dsv41_kit.tgz -C /root/kit
SPK=/root/dsv41/venv/lib/python3.12/site-packages
tar xzf /root/kit/vllm_overlay_rw.tgz -C $SPK
while read -r h rel; do echo "$h  $SPK/$rel" | sha256sum -c --quiet -; done < /root/kit/vllm_overlay_rw.sha256
echo "overlay ok"
export PATH=$HOME/.local/bin:$PATH
[ -x /root/toolenv/bin/python ] || uv venv -q --python 3.12 /root/toolenv
uv pip install -q --python /root/toolenv/bin/python numpy==2.2.6 scipy==1.15.3 scikit-image==0.25.2 networkx==3.4.2 \
  pandas==2.3.3 sympy==1.14.0 matplotlib pillow
/root/toolenv/bin/python -c "import numpy, scipy.ndimage, skimage; print('toolenv', numpy.__version__, scipy.__version__)"
mkdir -p /root/data && cp /root/podkit/arc-agi_evaluation_*.json /root/data/
cp /root/podkit/local_run.py /root/podkit/podrun.sh /root/podkit/memlog.sh /root/ && chmod +x /root/podrun.sh /root/memlog.sh
echo "[$(date -u +%T)] PREP OK"
