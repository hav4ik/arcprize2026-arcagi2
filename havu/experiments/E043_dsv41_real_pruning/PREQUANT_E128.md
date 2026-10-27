# Reproducing the pre-quantization tool E128 (DeepSeek-V4.1-Flash-1017, 128 of 384 experts)

Our Kaggle model is **tool E128 2-bit**: DeepSeek-V4.1-Flash-1017 with 128 of 384 routed experts per layer kept, and
those experts GPTQ-quantized to 2 bits. This guide rebuilds the same pruned model *before* the 2-bit step. Those
weights were never saved; the box that could have held them is gone. They can be rebuilt exactly on CPU from public
weights plus a 40 KB manifest. No GPU is needed.

## What "pre-quantization" means here

- DeepSeek released Flash-1017 with routed experts in **MXFP4**, and no higher-precision copy exists. Attention, the
  shared expert and dense layers are FP8/BF16.
- Our GPTQ took each kept MXFP4 expert, decoded it exactly to BF16, and quantized it to uint2 (`scripts/gptq_experts.py`).
- So the pre-quant E128 is:
  - the released model's own MXFP4 experts at the kept ids, renumbered 0..127 in ascending original id;
  - the router (`ffn.gate.weight`, `bias`, `bias_vl`) sliced to the same rows;
  - everything else unchanged.
- To get BF16, decode on load (see [Format](#format-of-the-output)). The decode is exact.
- Everything except the routed experts is already in the 2-bit HF repo, byte-equal to the base. This was checked
  tensor by tensor (`README.md`, "Re-verified from HF alone").

## Links

All `chankhavu` repos are **private**. Ask Vu for read access (a collaborator invite or a read token) before starting.

| what | where | used here |
|---|---|---|
| Base model, MIT, **revision `1bae0e1088993b9b82de08f59de5be5a34e42170`** | https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017 | source of every weight (510 GB) |
| Tool E128 2-bit (HF, 131.1 GB) | https://huggingface.co/chankhavu/DeepSeek-V4.1-Flash-1017-arc-tool-E128-2bit | `e043_manifest.json` (kept ids), `config.json`, `model-engram-*` (64 GiB Engram table), `gptq/` (only for the optional check) |
| Tool E128 2-bit (Kaggle Model, same weights; Engram split into ≤5 GB row chunks) | https://www.kaggle.com/models/chankhavu/dsv41-flash-1017-arc-tool-e128-2bit (`pyTorch/default/1`) | not needed: use the HF repo, whose Engram tensors have the standard layout |
| Tool E112 2-bit | https://huggingface.co/chankhavu/DeepSeek-V4.1-Flash-1017-arc-tool-E112-2bit | same recipe with its own manifest |
| No-tool E128 2-bit / E96 2/3-bit / E72 MXFP4 | `chankhavu/DeepSeek-V4.1-Flash-1017-arc-notool-{E128-2bit,E96-2bit3bit,E72}` | same recipe with its own manifest |
| Expert-importance dataset (scores, keep sets, Engram masks) | https://huggingface.co/datasets/chankhavu/dsv41-flash1017-expert-importance | how the keep set was chosen (card: `importance_dataset_README.md`) |
| Pruning artifacts (observer records, REAP stats, token streams, NLL results, code snapshot) | https://huggingface.co/datasets/chankhavu/dsv41-flash1017-e043-pruning-artifacts | recomputing or changing keep sets |
| GPTQ artifacts (E192 layers 0-19, box snapshot) | https://huggingface.co/datasets/chankhavu/dsv41-flash1017-e043-gptq-artifacts | not needed |
| Training tool rollouts that the keep set and GPTQ calibration came from | https://huggingface.co/datasets/livobereshka/arc-agi2-deepseek-flash1017-rollouts-e031-e036 | not needed |
| Code (this folder) | GitHub `hav4ik/arcprize2026-arcagi2`, `havu/experiments/E043_dsv41_real_pruning/` | `scripts/export_pruned.py`, `scripts/verify_prequant.py` |
| vLLM fork used for serving (wheel) | https://github.com/yhfgyyf/vllm-deepseek-v4-sm89/releases/tag/v0.28.1rc1-vision9-sm89-sm120-cu130 | only for serving, plus our `patches/` |

The keep set behind tool E128 is `keep_contrib_dec_sp0_w10_e128.npy`. It ranks experts by contribution on generated
tokens of 10M tokens of training tool rollouts, with tool-call blocks weighted ×10. Its ids are in the 2-bit repo's
`e043_manifest.json` (`keep_original_ids`), and its sha256 `f4c946c3…a56d` is in `config.json`
(`e043_pruning.expert_keep_sha256`). Step 2 checks both.

## Recipe

This is the code path that built the 2-bit model (`box_code/code/scripts/export_tool_e128.sh`) minus
`--no-routed-experts`, so the MXFP4 experts are written instead of skipped. It was tested end to end on a 2-layer
slice of the real base (see [Tests](#tests)).

**Needs:**
- Python ≥ 3.10, `torch >= 2.7` (CPU is fine; it needs `float8_e8m0fnu`), `safetensors >= 0.6`, `numpy`, `huggingface_hub`, `requests`.
- About 760 GB of disk: base 510 GB + Engram download 72 GB + output 178 GB. If the Engram download and `--out` sit
  on the same filesystem, the Engram files are hardlinked into the output and the total is about 690 GB.
- About 16 GB of RAM.

```bash
pip install "torch>=2.7" --index-url https://download.pytorch.org/whl/cpu
pip install "safetensors>=0.6" numpy huggingface_hub requests
export HF_TOKEN=<token with read access to chankhavu/*>
E043=<checkout>/havu/experiments/E043_dsv41_real_pruning

# 1. downloads: the base at the pinned revision, and from our 2-bit repo only the Engram files and the small json files
hf download deepseek-ai/DeepSeek-V4.1-Flash-1017 --revision 1bae0e1088993b9b82de08f59de5be5a34e42170 --local-dir base
hf download chankhavu/DeepSeek-V4.1-Flash-1017-arc-tool-E128-2bit \
    --include "model-engram-*" "*.json" --local-dir e128_2bit          # ~72 GB; gptq/ is not needed

# 2. keep mask [40, 384] from the manifest, checked against the config's sha256
python $E043/scripts/verify_prequant.py keep --manifest e128_2bit/e043_manifest.json \
    --config e128_2bit/config.json --out keep_tool_e128.npy               # must print "matches ... True"

# 3. export: kept MXFP4 experts renumbered 0..127, router sliced, DSpark/vision dropped, Engram from the 2-bit repo
python $E043/scripts/export_pruned.py --src base --expert-keep keep_tool_e128.npy \
    --engram-from e128_2bit --text-only --out e128_prequant
cp -r base/encoding base/inference e128_prequant/ 2>/dev/null            # export_pruned.py copies these if present

# 4. verify: router rows and all 5,120 experts byte-equal to the base; optional cosine vs the 2-bit GPTQ experts
python $E043/scripts/verify_prequant.py check --export e128_prequant --base base --all \
    --gptq hf:chankhavu/DeepSeek-V4.1-Flash-1017-arc-tool-E128-2bit@main  # range reads; prints PASS
```

Expected output: about 178 GB, made of backbone 9.7 GB + routed experts 96.3 GB + compact Engram 71.8 GB. The export
writes `model-0000N.safetensors` shards of at most 4.9 GB, plus the `model-engram-*` files, `config.json`,
`model.safetensors.index.json` and its own `e043_manifest.json`. Step 3 reads about 106 GB and takes minutes on fast
disks.

`verify_prequant.py` also takes `hf:<repo>@<rev>` for `--base`, so step 4 can run against the HF base without a local
copy. Step 5 of the check (`--gptq`) expects cosine ≈ 0.89–0.93 between each GPTQ expert and its MXFP4 original (the
2-bit error), and |cos| < 0.01 with the neighbouring expert, so it also confirms the id mapping.

### Variant: full Engram instead of the 64 GiB table

The deployed model uses a compact Engram table: 35% / 32% of rows on layers 1 / 14, chosen by lookup frequency on
tool traffic. Dropped rows read as zero vectors through `layers.{1,14}.engram.embed_tokens.row_map`, and
`text_config.engram_compact_kept` records the kept rows per hash column. That needs our `apply_compact_engram.py`
patch, or equivalent handling in your own loader.

For the stock Engram layout, drop `--engram-from e128_2bit` in step 3. The original 203 GB tables are then copied
as-is, giving about 310 GB of output with no `row_map`.

- Caveat: `export_pruned.py` loads each 98 GB table into RAM and `safetensors.save_file` copies it once more, so this
  needs about 200 GB of RAM.
- On the 2-bit model, dev NLL is 0.79650 with the full table vs 0.79679 with 64 GiB, so the compact table costs
  nothing measurable.

## Format of the output

- Routed experts, per layer `L` (0-39) and compact id `j` (0-127):
  - `layers.L.ffn.experts.j.{w1,w3}.weight`: I8 [2304, 2560];
  - `layers.L.ffn.experts.j.w2.weight`: I8 [5120, 1152];
  - `.scale` tensors: F8_E8M0, [2304, 160] / [5120, 72].
- The original id is `e043_manifest.json["keep_original_ids"][str(L)][j]`.
- `config.json` is the base config plus:
  - `text_config.n_routed_experts = 128`, top-6 routing unchanged;
  - `text_config.engram_compact_kept`;
  - `e043_pruning` (keep sha256, `text_only: true`).
  - It has no `e043_lowbit` block; that block marks the 2-bit model.
  - `quantization_config` is the base's (`fp8` with `expert_dtype: fp4`).
- **MXFP4 → BF16, exact.** This is the decode the GPTQ consumed. It was checked bit-exact against the vLLM fork's
  DeepGEMM decoder. Each byte holds two E2M1 codes, low nibble first; each scale byte `s` means 2^(s-127) for 32
  consecutive inputs:

```python
E2M1 = np.array([0, .5, 1, 1.5, 2, 3, 4, 6], np.float32)
def mxfp4_to_f32(q, s):            # q: uint8 [out, in/2], s: uint8 [out, in/32]  ->  float32 [out, in] (exact in bf16)
    c = np.stack([q & 15, q >> 4], -1).reshape(q.shape[0], -1)
    v = np.where((c & 8) != 0, -E2M1[c & 7], E2M1[c & 7])
    return v * np.repeat(np.exp2(s.astype(np.float32) - 127), 32, -1)
```

  In BF16, the experts are about 362 GB.

## Serving (optional)

- Use the vLLM fork above with `patches/apply_e043_patches.py`, `patches/apply_compact_engram.py` (for the compact
  Engram) and `patches/apply_e043_humming.py`, then run `--language-model-only`.
- The humming patch is needed only because it adds the router shape (128, 6). Without `DSV41_EXPERT_HUMMING` set,
  the experts stay on the fork's native MXFP4 path.
- Untested on this exact checkpoint. MXFP4 E64/E72 exports built the same way served at parity: dev NLL within
  0.0003 of runtime masks.
- Reference numbers on 64 family-disjoint dev tool traces (teacher-forced NLL on generated tokens; `README.md`, tool
  round). Use them to check a served copy of this checkpoint:

  | model | dev NLL | tool-block NLL | valid first-turn calls (of 25) |
  |---|---:|---:|---:|
  | full model | 0.7130 | 0.118 | - |
  | **E128 MXFP4 = this checkpoint** (runtime masks on the full checkpoint) | +7.9% (≈ 0.769) | 0.135 | 22 |
  | E128 2-bit export | 0.79640 (+11.7%) | 0.149 | 22 |
  | E128 2-bit, runtime masks | 0.79652 | | |

  So pruning accounts for about two thirds of the 2-bit model's NLL gap and the 2-bit step for the remaining third.
  The no-tool E128 split the same way: MXFP4 +8.3%, 2-bit +12.1%.

## Tests

`scripts/verify_prequant.py` was tested on 2026-10-26 on a 2-layer slice of the real base: layers 0 and 39, kept
ids at ranks 0 / 1 / 127 plus one dropped expert, the routers, and MTP / vision tensors. The slice was fetched by
range reads and run through the unmodified `export_pruned.py` with a stub `--engram-from`.

- The dropped expert and the MTP / vision tensors are absent; experts are renumbered 0 / 1 / 127.
- `n_routed_experts` is 128, the keep sha256 matches, and there is no `e043_lowbit` block.
- Router rows and experts are byte-equal to the HF base.
- Cosine vs our GPTQ experts is 0.889–0.927 for the right expert and |cos| ≤ 0.002 for the neighbouring one.
- An export with experts 0 and 1 swapped fails the byte check, as it should.
- The `--engram-from` copy fallback across filesystems (hardlink → copy) is tested, and the output is byte-identical
  across runs.
