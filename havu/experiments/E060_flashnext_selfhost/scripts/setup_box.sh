#!/bin/bash
# E060: build the serving and harness environments on a box after push_to_box.sh. Idempotent.
#   bash /root/e060/repo/scripts/setup_box.sh            (then, with HF_TOKEN on stdin: see dl_model.sh)
#
# svenv (vLLM 0.32.0 for serving). Two cases:
#  a) The Vast image vastai/vllm_v0.32.0-cuda-13.0 (box 57983124) has vLLM 0.32.0, torch 2.13.0+cu130, FlashInfer
#     0.7.0.post1 (with sm89 JIT cache) and transformers 5.19.0 in /usr/local/lib/python3.12/dist-packages. svenv
#     then gets its own copy of the `vllm` package (so the patches never touch the image) and a .pth that adds the
#     image's dist-packages for everything else.
#  b) Any other image: a fresh venv with `vllm==0.32.0` from PyPI (not tested in E060; check torch/CUDA match).
# Either way the E060 expert-count hook is applied to svenv's vllm copy (patches/apply_expert_counts.py).
#
# hvenv (harness + tool-code Python, like the Kaggle toolenv): openai, httpx, and the toolenv packages.
set -euo pipefail
E=/root/e060
SYS=/usr/local/lib/python3.12/dist-packages
mkdir -p $E/runs/results
command -v uv >/dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH=$HOME/.local/bin:$PATH; }

if [ ! -x $E/svenv/bin/python ]; then
  if [ -d $SYS/vllm ] && grep -q '^Version: 0.32.0$' $SYS/vllm-0.32.0.dist-info/METADATA 2>/dev/null; then
    uv venv -q --python /usr/bin/python3.12 $E/svenv
    SP=$E/svenv/lib/python3.12/site-packages
    cp -a $SYS/vllm $SYS/vllm-0.32.0.dist-info $SP/
    echo "import site; site.addsitedir(\"$SYS\")" > $SP/zz_e060_system_dist_packages.pth
    echo "svenv: copy of the image's vLLM 0.32.0"
  else
    uv venv -q --python 3.12 $E/svenv
    uv pip install -q --python $E/svenv/bin/python vllm==0.32.0
    echo "svenv: vLLM 0.32.0 from PyPI (untested path)"
  fi
fi
$E/svenv/bin/python $E/repo/patches/apply_expert_counts.py "$($E/svenv/bin/python -c 'import vllm, os; print(os.path.dirname(vllm.__file__))' 2>/dev/null | tail -1)"
(cd $E && $E/svenv/bin/python -c "import vllm, torch, vllm.e060_expert_counts; print('svenv ok', vllm.__version__, torch.__version__, torch.cuda.get_device_capability(0))")

if [ ! -x $E/hvenv/bin/python ]; then
  uv venv -q --python 3.12 $E/hvenv
fi
uv pip install -q --python $E/hvenv/bin/python openai==3.40.0 httpx numpy==2.2.6 scipy==1.15.3 scikit-image==0.25.2 \
  networkx==3.4.2 pandas==2.3.3 sympy==1.14.0 matplotlib pillow tokenizers
$E/hvenv/bin/python -c "import openai, numpy, scipy, skimage; print('hvenv ok', openai.__version__, numpy.__version__)"
(cd $E && $E/hvenv/bin/python repo/scripts/run_local_tools.py --model flashnext --base-url http://127.0.0.1:18321/v1 \
  --sandbox-url local --challenges repo/data/arc-agi-2-github/data/evaluation --attempts 1 --dry-run \
  --out runs/dry.jsonl | head -1)
echo "SETUP OK; next: model download (dl_model.sh), then serve_flashnext.sh"
