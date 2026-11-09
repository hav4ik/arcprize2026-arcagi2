# vLLM ARC patches

`vllm_c6_arc_stability.patch` targets the exact vLLM source snapshot used by
the Kaggle runtime:

```text
c6fe94b4d5b418fa213af0e5884eddd304333dcd
```

It combines five narrowly scoped fixes:

1. upstream [PR 50820](https://github.com/vllm-project/vllm/pull/50820),
   which includes `offset_start` in the LoRA expand-pointer cache key;
2. upstream [PR 51597](https://github.com/vllm-project/vllm/pull/51597),
   which preserves the `lm_head` LoRA delta while vLLM
   computes prompt log-probabilities, plus the equivalent Model Runner V2
   path used by this snapshot;
3. bounds checks around sampled/custom token IDs before Triton or Torch indexes
   the logits tensor, returning `-inf` and an invalid-rank sentinel instead of
   poisoning CUDA with an illegal address;
4. the no-speculative-decoding portion of upstream
   [PR 51865](https://github.com/vllm-project/vllm/pull/51865), which prevents
   a one-token prefill chunk from being dispatched through a FULL decode CUDA
   graph merely because its shape matches a decode batch; and
5. a sampled-token containment guard in Model Runner V2's post-update kernel.
   Invalid token IDs, impossible sampled-token counts, and overflowing request
   histories no longer become offsets into the output-frequency or token-history
   buffers. The ARC service can observe the malformed response and restart the
   EngineCore without first poisoning CUDA with an illegal address.

Apply it from a clean vLLM checkout at that commit, then rebuild the wheel:

```bash
git apply --check /path/to/vllm_c6_arc_stability.patch
git apply /path/to/vllm_c6_arc_stability.patch
```

For the existing Kaggle wheel, use the package-only runtime variant before
`starter.py` (and before importing `vllm` in that kernel):

```bash
python - <<'PY'
from importlib.metadata import version

expected = "0.23.1rc1.dev1424+gc6fe94b4d"
actual = version("vllm")
assert actual == expected, f"runtime patch expects {expected}, found {actual}"
PY
echo '94178443f3724953f48461f2b98be0ffe2f70fe34d412e89ce1d20c1ca1822f8  /kaggle/working/vllm_c6_arc_stability_runtime.patch' \
  | sha256sum --check -
patch --dry-run --binary --forward -p1 \
  -d /usr/local/lib/python3.12/dist-packages \
  < /kaggle/working/vllm_c6_arc_stability_runtime.patch
patch --binary --forward -p1 \
  -d /usr/local/lib/python3.12/dist-packages \
  < /kaggle/working/vllm_c6_arc_stability_runtime.patch
```

`vllm_c6_arc_stability_runtime.patch` omits repository tests and changes only
the seven installed `vllm/` modules. It was dry-run applied, actually applied,
and byte-compiled against both a clean archive of commit `c6fe94b4d5` and the
exact pre-edit module backups from the Kaggle wheel; its SHA-256 is
`94178443f3724953f48461f2b98be0ffe2f70fe34d412e89ce1d20c1ca1822f8`.
Do not apply both variants to the same source tree.

The patch intentionally does not backport PR 51865's speculative-decoder
changes. The ARC runner uses no speculative model. Upgrade to a newer vLLM
revision instead if speculative decoding is required.

Do not infer compatibility from the package label
`0.23.1rc1.dev1424+gc6fe94b4d`: the commit is chronologically newer than the
v0.25.1 tag, but the two revisions are divergent and neither contains every
commit from the other.

## Async sampler-output lifetime fix

`vllm_c6_async_sampler_snapshot_runtime.patch` fixes a separate async-output
lifetime bug in the same c6 snapshot. A compiled sampler can return token IDs
and logprob tensors backed by reusable Inductor/CUDA-graph storage. The model
runner performs prompt-logprob bookkeeping before it constructs
`AsyncGPUModelRunnerOutput` and starts the device-to-host copy. Under a mixed
multi-LoRA decode/prompt-score workload, that bookkeeping can reuse the sampled
token buffer while Python still holds a reference to it.

The patch clones the small sampler payload immediately after `_sample()`, on
the producing CUDA stream and before bookkeeping. It is unconditional when
vLLM async scheduling is active, adds no host synchronization, and leaves the
synchronous scheduler unchanged. Apply it before importing `vllm`; it works
against either the clean c6 package or the package after
`vllm_c6_arc_stability_runtime.patch`:

```bash
echo '6a050c49c5ea7567faae75bbd96ab6d78d9b096948ea02b83b84c36fb38428a5  /kaggle/working/vllm_c6_async_sampler_snapshot_runtime.patch' \
  | sha256sum --check -
patch --dry-run --binary --forward -p1 \
  -d /usr/local/lib/python3.12/dist-packages \
  < /kaggle/working/vllm_c6_async_sampler_snapshot_runtime.patch
patch --binary --forward -p1 \
  -d /usr/local/lib/python3.12/dist-packages \
  < /kaggle/working/vllm_c6_async_sampler_snapshot_runtime.patch
python -m py_compile \
  /usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu_model_runner.py
```

Do not apply the diagnostic-only `vllm_c6_async_output_probe.patch`,
`vllm_c6_async_output_guard.patch`, or
`vllm_c6_sampler_snapshot_fix.patch` in a production notebook. Those files
were used to localize the corruption and deliberately retain opt-in forensic
controls. The production patch above contains only the early snapshot.
