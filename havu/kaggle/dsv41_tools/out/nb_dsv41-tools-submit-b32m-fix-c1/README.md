# Next candidate: B32-M fix on the C1 model (text harness). NOT pushed to Kaggle.

Notebook sha256 `73bd117b6a1d`, slug `dsv41-tools-submit-b32m-fix-c1`. It is the build of tonight's
`dsv41-tools-submit-b32m-fix` V1 with two differences:
- **Model:** C1 (new E128 2-bit with the vision tower, unused here: `MANH_VISION=0`, `--language-model-only`),
  `arazzd/deepseek-v41-flash-1017-arc-tool-img-e128-2bit/PyTorch/default/1`. It is shared with chankhavu and verified
  (137 files, 132 GB). Cell 1b checks `c1_gptq_sha256_manifest.json`.
- **Solver:** `a9983f4c`, V1's `1ea2b25d` plus `APITimeoutError` / "Request timed out" as a connection error. Pod 3 lost
  an attempt to it.

**Why:** dsv4-pruning's final E066 results (120 tasks x 2, ARC scoring). On the text harness, C1 ties B32's E128,
slightly ahead:

| | C1 | B32's E128 | Diff |
|---|---|---|---|
| pass@1 | 75.1 | 73.7 | +1.5 [-3.1, +6.0] |
| pass@2 | 85.8 | 84.6 | +1.2 [-3.8, +6.2] |
| Tokens per attempt | 38.9K | 40.8K | ~5% fewer |
| Tool-error rate | 14.4% | 15.1% | the same |

Images still hurt on our own stack (-5.4), and E72-152 is retired. See `experiments/E066_dsv41_image_pruning/README.md`,
"Final ARC-AGI-2 results".

**Not tested yet:**
- C1 on the text path under Manh's server. Pod 3 ran C1 with images: 1.45M KV at util 0.90.
- Text without the vision encoder loaded should give a little more KV than that.
- Before a submission: a pod commit smoke plus a few hours of replica, per the testing protocol.

```
python3 kaggle/dsv41_tools/build_img_manh.py --text --manh-nb <Manh v6 pull> --slug dsv41-tools-submit-b32m-fix-c1 \
  --model-source arazzd/deepseek-v41-flash-1017-arc-tool-img-e128-2bit/PyTorch/default/1 \
  --model-manifest experiments/E066_dsv41_image_pruning/results/c1_gptq_sha256_manifest.json \
  --max-num-seqs 64 --gpu-util 0.90 --kv-offload-gib 10 --exec-timeout-s 900 --pin-dataset nguyennguyen599/vllm-wheels/11
```
