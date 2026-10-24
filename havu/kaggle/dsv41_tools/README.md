# DeepSeek-V4.1 tool solver for Kaggle 4×L4 (E042 RW / E043 tool E128)

Build with `.venv/bin/python kaggle/dsv41_tools/build.py` (add `--no-model` to leave model_sources empty). It writes:
- `out/kit_ds/`: the dataset `chankhavu/dsv41-tools-kit`.
  - `dsv41_kit.tgz`: an exact snapshot of the harness files the solver imports (`evals/run_deepseek_tools.py` with E043's
    `--token-exact`, its imports, the ARC Prize prompt), `solver_dsv41.py`, `stage_model.py`, and the vLLM overlay.
    The overlay holds the 12 files of the pod's patched fork that differ from the wheel, with sha256.
  - `MANIFEST.json`: the sha256 of every kit file.
- `out/nb_submit/` (`chankhavu/dsv41-tools-submit`): the submission notebook. The commit run only writes the fallback
  submission (seconds); the competition rerun runs everything.
- `out/nb_smoke/` (`chankhavu/dsv41-tools-smoke`): the same cells. Its commit runs the whole pipeline on 6 tasks, with
  deadlines at 60 / 75 / 85 min.
- `out/local_run.py`: the cells as one script, for the Runpod pods (`ARC_*` environment variables set the paths).

Inputs:
- the competition;
- the kit dataset;
- the wheel set: output of `chankhavu/dsv41-sm89-fork-py312-cu130`, 198 pinned packages incl. the vLLM and FlashInfer
  forks and humming-kernels 0.1.12;
- the model `chankhavu/dsv41-flash-1017-arc-tool-e128-2bit/pyTorch/default/1`: the E128 export resharded to ≤ 5 GB
  files, Engram as row chunks.

Cells:
0. T0 and paths; fallback submission; kit unpack + sha256 check.
1. Model staging (`stage_model.py copy`, detached): copy to local disk while vLLM installs. Its pre-flight checks every
   index file and gptq layer against its safetensors header size; a failure keeps the fallback.
2. Offline install, overlay (sha256 checked), CUDA 13 shim, import check.
3. `solver_dsv41.py` (see its docstring):
   - the vLLM server in its own process group, with watchdog and one restart;
   - the tool harness: sub80s40 + minimal header, `--token-exact --rescue-code --train-check`, 48 workers,
     LocalExecutor tool sessions on the image's Python;
   - deadlines: stop-new 10h50, wrap-up note 15 min before force, force 11h10, hard stop 11h30;
   - admission control; the gzip loop guard (thinking only, 10 windows); second-attempt priority;
   - an atomic, validated submission after every attempt; SIGTERM handling.
4. Final validity check (else fallback) and the trace archive.

Tested on Runpod 4×L4 (pods A and B) with the real E128 and the eval set as the test file. `fail_tests.sh` covers
three cases:
- the server never comes up;
- the server is killed mid-run;
- SIGTERM to the solver.

Two bugs found and fixed:
- a task limit wrote a partial submission;
- after a server restart, attempts relaunched before the death was noticed and used up their retry.
