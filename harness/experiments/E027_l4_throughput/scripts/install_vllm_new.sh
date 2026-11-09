#!/bin/bash
# Installs the newest vLLM release into /root/vllmnew (torch cu130), after clearing old torch.compile caches.
rm -rf /root/.cache/vllm/torch_compile_cache /root/.cache/vllm/torch_aot_compile /root/.cache/uv
df -h / | tail -1
uv venv --python 3.12 /root/vllmnew > /dev/null 2>&1
. /root/vllmnew/bin/activate
uv pip install -U vllm --extra-index-url https://download.pytorch.org/whl/cu130 --index-strategy unsafe-best-match > /root/arc/logs/install_vllmnew.log 2>&1
grep -E "^ \+ (vllm|torch|transformers|flashinfer)" /root/arc/logs/install_vllmnew.log
python -c "import vllm, torch; print('vllm', vllm.__version__, 'torch', torch.__version__, torch.version.cuda)" 2>&1 | tail -1
V=$(python -c "import vllm, os; print(os.path.dirname(vllm.__file__))" 2>/dev/null)
echo "turboquant + spec-decode mentions:"
grep -rn -iE "speculat|num_spec|spec_decode|query_len|max_query" $V/v1/attention/backends/turboquant_attn.py | head -12 | cut -c1-200
rm -rf /root/.cache/uv
df -h / | tail -1
echo INSTALL_DONE
