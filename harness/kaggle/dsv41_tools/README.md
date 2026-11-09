# DeepSeek-V4.1 tool solver for Kaggle 4×L4 (E042 RW / E043 tool E128)

**Current candidate: B32** (submitted 2026-10-27): start at `CANDIDATE_B32.md`. Replicating it end to end: `REPLICATE.md`.
Every submission follows `../SUBMISSION_TESTING_PROTOCOL.md`.

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
   - the vLLM server in its own process group, with watchdog and up to 3 restarts (one before 2026-10-26);
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

Unit tests (no GPU; the V5 Kaggle kit `kaggle_kit_v5/` with the live solver and harness over it, as cell 0b installs
them): `.venv/bin/python -m unittest discover -s kaggle/dsv41_tools/tests` (solver, notebook cells 00 / 0b, E042
tooling) and `evals/tests/test_dsv41_harness.py` (the harness's Kaggle paths). Open and fixed gaps of the next
candidate: `next_candidate_fixes.md`.

Input lookup (rule from Manh, 2026-10-30): notebook, cell and solver code never scans `/kaggle/input` (or
`ARC_INPUT_ROOT`) recursively. With his datasets attached, one `**` glob, `rglob` or `os.walk` there walks thousands of
files, about 5 min per call. Cell 0 defines `find_input(name, *exact)` and `find_inputs` (`input_paths.HELPER`). They
look at the exact candidates first, then the attached sources' mount dirs (`INPUT_ROOTS`), then fixed depths only:
`datasets/<owner>/<slug>[/<dir>]`, `competitions/<slug>`, `notebooks/<owner>/<slug>[/<dir>]`,
`models/<owner>/<model>/<framework>/<variation>/<version>`, and up to two levels for older mounts and the pods. Every
builder ends with `input_paths.finish(nb, meta)`, which sets `INPUT_ROOTS` from kernel-metadata.json, rewrites the old
lookups in the Version 5 and Manh notebooks it starts from, and fails the build if any code cell still scans the input
root. The solvers use `ARC_MODEL_DIR` (every solver cell sets it) and otherwise `find_model_dir()` at fixed depths.
`tests/test_input_scans.py` checks the live code under `kaggle/` and every builder's notebook.
