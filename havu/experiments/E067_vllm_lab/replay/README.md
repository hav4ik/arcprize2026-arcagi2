# Replay bench inputs: b32m_x2t token cache

The prepared token cache of `evals/serve_replay_bench.py` for the b32m_x2t trace set is 4.4 MB, so it lives on
Hugging Face. It is in the private dataset `livobereshka/arc-agi2-vllm-serving-lab` (team only) at
`replay/b32m_x2t.tokens.json.gz`.

| file | where | what |
|---|---|---|
| `b32m_x2t.tokens.json.gz` | HF only | token plans of 480 attempts / 8,703 turns (sha256 `3ea09fa3c281862b3dcff48b7153ad7c8b3213071797ef443a3ca9678dcb2511`) |
| `tools.json` | here and HF | the tool set the replay sends, auto-detected as `--answer-tool --multi-test` (pass it with `--tools`) |
| `prepare.json` | here and HF | the prep run's config, workload stats and fidelity (`prompt_len_exact_frac` 1.0) |

## Download

```bash
export HF_TOKEN="$(sed -n 's/^huggingface_api_key:[[:space:]]*//p' SECRETS.yaml | tr -d "\"' ")"
.venv/bin/python - <<'EOF'
from huggingface_hub import hf_hub_download
p = hf_hub_download("livobereshka/arc-agi2-vllm-serving-lab", "replay/b32m_x2t.tokens.json.gz",
                    repo_type="dataset", local_dir="prep")
print(p)  # prep/replay/b32m_x2t.tokens.json.gz
EOF
sha256sum prep/replay/b32m_x2t.tokens.json.gz   # 3ea09fa3...2511
```

Then pass `--prep-cache prep/replay/b32m_x2t.tokens.json.gz --tools experiments/E067_vllm_lab/replay/tools.json`,
together with the trace tarball
`experiments/E064_manh_notebook_b32_harness/l4e2e/runs/b32m_x2t/out/dsv41_traces.tar.gz` (the cache is keyed by trace
digest). The usage and the pod recipe are in `notes/2026-10-30_serve_replay_bench.md`.

## How it was made

On the dev box (no GPU, 4.4 min):
```bash
.venv/bin/python evals/serve_replay_bench.py --prepare-only \
    --traces experiments/E064_manh_notebook_b32_harness/l4e2e/runs/b32m_x2t/out/dsv41_traces.tar.gz \
    --local-render data/dsv4_prune/model_meta --prep-cache prep/b32m_x2t.tokens.json.gz --out prep/b32m_x2t
```
- Renderer: the DeepSeek-V4.1 `encoding.py` + `tokenizer.json` (sha c90dfa01, the Kaggle Model copy's file).
- The cache is valid for any server with that tokenizer and chat template. A bench run checks one plan against the
  server's `/tokenize` first (`fidelity.server_render_check`).

The prompts contain ARC-AGI-2 evaluation tasks: workload `evalx2t_s2026` is the 120 public eval tasks x2, with seeded
transforms. Do not train on them.
