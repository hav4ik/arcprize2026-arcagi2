"""Replay plans for the E014 notebooks. Usage: P=$(python3 experiments/E014_sglang_kaggle/plans.py); then
.venv/bin/python kaggle/build_notebook.py --variant replay --env "$(echo $P | python3 -c 'import json,sys;print(json.dumps(json.load(sys.stdin)["sgl"]))')"  (or ["vllm"])
"""
import json
lazy = {"ARC_SGL_MAMBA_STRATEGY": "extra_buffer_lazy", "ARC_SGL_SKIP_LOCK": "1", "ARC_SGL_MAMBA_SLOTS": "108"}
sgl = [{"name": "sgl_radix", "max_seqs": 36, "env": lazy,
        "fallback_envs": [{"ARC_SGL_SKIP_LOCK": "1", "ARC_SGL_MAMBA_SLOTS": "144"}, {}],
        "levels": [[16, 300], [24, 300], [35, 300], [48, 300]]},
       {"name": "sgl_radix_hicache", "max_seqs": 36, "base": "sgl_radix", "extra_env": {"ARC_SGL_HICACHE": "2"},
        "levels": [[24, 300], [35, 300], [48, 300]]}]
vllm = [{"name": "vllm_prefix", "max_seqs": 64, "env": {"ARC_SERVER_EXTRA": json.dumps(["--enable-prompt-tokens-details"])},
         "levels": [[16, 300], [24, 300], [35, 300], [48, 300]]}]
print(json.dumps({"sgl": {"ARC_SERVER": "sglang", "ARC_BENCH_STOP_MIN": "58", "ARC_REPLAY_PLAN": json.dumps(sgl)},
                  "vllm": {"ARC_SERVER": "vllm", "ARC_BENCH_STOP_MIN": "38", "ARC_REPLAY_PLAN": json.dumps(vllm)}}))
