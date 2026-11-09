#!/usr/bin/env python3
"""E066 image candidate (2026-10-29): B32-M (below) with the E058 image harness and an E066 checkpoint that keeps the
vision tower (uniform E128 = C1, or per-layer expert counts such as E72-152 avg 120). Differences from B32-M:
- pinned_img/: B32's pinned solver 77fe80a3 + the opt-in ARC_GRID_FORMAT=text+image switch (drops --token-exact
  and the server's --language-model-only, adds --grid-format text+image --image-cell 0); the live harness frozen
  on 2026-10-29 (E058 text+image, chat-path --context-limit = chat_context_cap); the live arc_harness/grid_image.py
  (token-aligned images); B32's train_check and auto-repair;
- cell 2r applies two sha256-guarded patches to his wheel: model.py (Engram row chunks + per-layer expert counts)
  and dsv4_topk.py (router whitelist 6..512) -- branch dynamic-count-moe b3a3f642a9;
- the server also gets --limit-mm-per-prompt (one image per grid) and the model source is the E066 Kaggle Model.

B32-M: candidate B32 with Manh's vLLM environment (E062/E064, 2026-10-28).

B32's notebook as build_frozen_mt.py builds it (B32's pinned solver, harness, train_check and auto-repair,
byte-identical; cell 0b checks; B32's emergency writer and v3 fallback; integrity and sandbox checks; smoke report),
with only the environment layer swapped:
- Kaggle's latest docker image (`docker_image_pinning_type: latest`; his code needs the newest environment, and
  smoke V1 on his pinned sha saw no GPU) and his datasets (CUDA 13 debs, his wheels incl. vllm 0.32.1rc1+g<sha>, src);
- a GPU gate right after the clock cell (cell 0a): 4 GPUs via nvidia-smi within 90 s (with an LD_LIBRARY_PATH repair
  when NVML is present but not found) and Python 3.13, else commit runs raise at once and reruns keep the fallback;
- his install cells 1-5 (nguyennguyen599/arc2-2026-harness-deepseek) instead of B32's fork wheels / overlay / venv,
  verbatim except: `rm -r /kaggle/working/*` dropped (it would delete our fallback submission and smoke logs) and his
  "CUDA installation failed" raise kept for commit runs only (a rerun keeps the fallback instead of erroring);
- the solver launches his vLLM itself (a `vllm` wrapper around the kernel's Python, as his dsv41_vllm.py does) with
  the E062 L4 replica's tested server config: DCP4, interleave 64, retention 512, embed_tokens UVA, OffloadingConnector
  32 GiB (offload_prompt_only false), FULL CUDA graphs up to 96, max-num-seqs 96, Humming pad 64,64, driver reserve
  512 MiB, VLLM_PLUGINS empty; 80 workers, KV gate 0.85 + queue hold;
- Manh's extra flags (host all-reduce, indexer logits, humming tuning, his GPU order, ...) in one cell, OFF unless
  built with --manh-flags (E062b is still testing them);
- plan B (cell 2r): his row-chunk fix a6c3ee8c2b (= our E062 patch 01) patched into his installed
  vllm/models/deepseek_v41/nvidia/model.py, sha256-guarded (base 5e1bccd07c -> patch + verify; fixed -> skip; other ->
  commit runs raise, reruns leave it), because his Kaggle wheel g5e1bccd07c cannot load the Kaggle copy of E128;
- smoke (commit runs, --smoke-minutes 37): kill switch 36:30 after the first cell (stop-new 32, force 34, hard 35),
  the 2 smallest tasks (00576224, and 25ff71a9 with 2 test inputs), metrics every 30 s; extras: an environment record (versions, his wheel, the
  row-chunk path, sha256 of the files his weight loading runs), a 16-session tool-sandbox check on Python 3.13, a
  background API contract check of the solver's server, and a report with the timeline (install, server start/up),
  gen tok/s / running / KV windows and, if the server never came up, how far it got; the verdict cell raises on any
  failure. Competition reruns ignore all smoke settings (this build is also the submission build).

Usage: build_b32m.py --manh-nb DIR [--smoke-minutes 37] [--smoke-tasks IDS] [--util 0.945] [--manh-flags] [--slug ...] [--out DIR]
       [--manh-datasets owner/a,owner/b,...] [--docker-image SHA]
Nothing is pushed: the coordinator reviews and pushes.
"""
import argparse
import base64
import re
import hashlib
import json
import sys
from pathlib import Path

import build_frozen_mt as B
import input_paths as IP

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
P = HERE / "pinned_img"
MANIFEST = None  # E066: --model-manifest (the export's path/size/sha256 list)
MANH_DATASETS = ["nguyennguyen599/arc2-2026-src", "nguyennguyen599/cu13-install", "nguyennguyen599/unsloth-main",
                 "nguyennguyen599/vllm-wheels"]
CELLS = {n: HERE / f"cells/b32m_{n}.py" for n in ("gpugate", "env", "manh_flags", "contract", "report", "verdict")}
CELLS["patches"] = HERE / "cells/img_patches.py"
PATCH_DIFFS = {"MODEL5E1B": HERE / "b32m_patches/dynamic_count_rowchunk_model.diff",  # g5e1bccd07c: a6c3ee8c2b + per-layer
               "MODELF506": HERE / "b32m_patches/dynamic_count_model_f506.diff",  # gf506afa210+: per-layer only
               "TOPK": HERE / "b32m_patches/dynamic_count_dsv4_topk.diff"}  # router whitelist 6..512
# one image per grid: a task's training pairs (2 images each) and test inputs; ARC-AGI-2 tasks have up to ~10 pairs
MM_LIMIT = 64
# the <= 50-min smoke: B32's report cell waits up to 90 s (not 4 min) for the model hash, and reports an unfinished one
INTEG_WAIT = ("        for _ in range(48):  # wait up to 4 min for the model hash to finish",
              "        for _ in range(18):  # B32-M: wait up to 90 s for the model hash (<= 50-min smoke)")
INTEG_DONE = ('        if _ig.get("error") or not _ig.get("done") or _mbad or (_ig.get("wheels") or {}).get("bad"):',
              '        if not _ig.get("done") and not (_ig.get("error") or _mbad):  # B32-M: an unfinished hash is reported\n'
              '            print("INPUT INTEGRITY INCOMPLETE (short smoke): no mismatch among the files hashed so far",'
              ' flush=True)\n'
              '        elif _ig.get("error") or _mbad or (_ig.get("wheels") or {}).get("bad"):')
SMOKE_METRICS = ('    os.environ.setdefault("ARC_COMMIT_MODE", "smoke")\n',
                 '    os.environ.setdefault("ARC_COMMIT_MODE", "smoke")\n'
                 '    os.environ.setdefault("ARC_METRICS_EVERY_S", "30")  # B32-M: server metrics every 30 s\n')
# load test (--load-test, a separate build, never the submission): 80 typical-size eval tasks (sizes 4,044-11,375, the
# middle 80 of 120), so the 80 workers are busy; the report prints server throughput, the verdict fails only on a
# server error / crash or no submission
LOAD_EVAL_TASKS = ("135a2760,13e47133,142ca369,1818057f,1ae2feb7,21897d95,247ef758,269e22fb,271d71e2,2ba387bc,2c181942,"
                   "2d0172a1,332f06d7,35ab12c3,36a08778,38007db0,3a25b0d8,409aa875,446ef5d2,45a5af55,4a21e3da,4c416de3,"
                   "4c7dc4dd,4e34c42c,53fb4810,5545f144,581f7754,58490d8a,58f5dbd5,5dbc8537,64efde09,65b59efc,67e490f4,"
                   "6e453dd6,6e4f6532,6ffbe589,71e489b6,7491f3cf,7666fa5d,7b0280bc,7b3084d4,7b80bb43,7ed72f31,800d221b,"
                   "80a900e0,8698868d,88bcf3b4,88e364bc,89565ca0,898e7135,8b7bacbf,8b9c3697,8e5c0c38,8f215267,8f3a5a89,"
                   "9385bd28,9bbf930d,a251c730,a395ee82,a47bf94d,a6f40cea,abc82100,b10624e5,b5ca7ac4,b6f77b65,b9e38dc0,"
                   "c7f57c3e,cb2d8a2c,cbebaa4b,d59b0160,da515329,db0c5428,db695cfb,de809cff,dfadab01,e376de54,edb79dae,"
                   "f560132c,faa9f03d,fc7cae8d")
SEQS = 96
OFFLOAD = {"kv_connector": "OffloadingConnector", "kv_role": "kv_both",
           "kv_connector_extra_config": {"cpu_bytes_to_use": 34359738368, "offload_prompt_only": False,
                                         "eviction_policy": "lru"}}
# the L4 replica's server flags (E062 l4/launch_manh.sh) beyond what B32's settings cell already passes (Humming
# dense, FULL graphs, --retention); the later --compilation-config wins (graphs up to max-num-seqs, as on the pod)
EXTRA = ["--decode-context-parallel-size", "4", "--cp-kv-cache-interleave-size", "64",
         "--cpu-offload-gb", "1", "--cpu-offload-params", "embed_tokens",
         "--compilation-config", json.dumps({"cudagraph_mode": "FULL", "max_cudagraph_capture_size": SEQS}),
         "--kv-transfer-config", json.dumps(OFFLOAD, separators=(",", ":")),
         "--limit-mm-per-prompt", json.dumps({"image": MM_LIMIT})]
SERVER_ENV = {"VLLM_GPU_DRIVER_RESERVE_MB": "512", "VLLM_GPU_MEMORY_LOG_INTERVAL": "300",
              "VLLM_HUMMING_MOE_PAD_MULTIPLE": "64,64", "VLLM_PLUGINS": ""}


def b32_notebook(a, tmp):
    """B32's build (REPLICATE.md section 3) with B32-M's workers and server flags, no B32 overlay."""
    argv = ["build_frozen_mt.py", "--slug", a.slug, "--out", str(tmp), "--workers", str(a.workers), "--budget", "0.5",
            "--gate", "0.85", "--retention", "512", "--util", str(a.util), "--auto-repair", "--admit-max-waiting", "1",
            "--solver", str(P / "solver_dsv41.py"), "--harness", str(P / "run_deepseek_tools.py"),
            "--train-check", str(P / "train_check.py"), "--server-extra", json.dumps(EXTRA),
            "--extra-pin", f"repo/evals/arc_harness/grid_image.py={P / 'arc_harness/grid_image.py'}",
            "--expect-env", "ARC_GRID_FORMAT=text+image",
            "--need", "solver_dsv41.py=GRID_FORMAT = E(\"ARC_GRID_FORMAT\"",
            "--need", "repo/evals/run_deepseek_tools.py=def chat_context_cap",
            "--need", "repo/evals/run_deepseek_tools.py=def text_image_content",
            "--need", "repo/evals/arc_harness/grid_image.py=def aligned_cell"]
    if getattr(a, "external", False):  # S25: the solver attaches to Manh's server cell
        argv += ["--expect-env", f"ARC_EXTERNAL_SERVER=http://127.0.0.1:{a.server_port}/v1",
                 "--need", "solver_dsv41.py=EXTERNAL = E(\"ARC_EXTERNAL_SERVER\""]
    if a.smoke_minutes:
        argv += ["--smoke-minutes", str(a.smoke_minutes)]
        if a.model_manifest:
            argv += ["--smoke-integrity", str(a.model_manifest)]
    old = sys.argv
    try:
        sys.argv = argv
        B.main()
    finally:
        sys.argv = old
    nb = json.load(open(tmp / f"{a.slug}.ipynb"))
    return nb, json.load(open(tmp / "kernel-metadata.json"))


def src(c):
    return "".join(c["source"])


def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": text.splitlines(keepends=True)}


def replace_once(text, old, new):
    assert text.count(old) == 1, f"expected once: {old[:70]!r} ({text.count(old)})"
    return text.replace(old, new)


def manh_install_cells(manh_nb, version_note):
    """His cells 1-5 (CUDA 13 install, his src copy, wheel copy to /dev/shm, uv installs, uninstalls)."""
    cells = manh_nb["cells"]
    pick = [next(c for c in cells if "cu13-install" in src(c) and "dpkg" in src(c)),
            next(c for c in cells if "arc2-2026-src" in src(c) and src(c).lstrip().startswith("!")),
            next(c for c in cells if "/dev/shm/wheels" in src(c) and "copy_item" in src(c) and "UV_FIND_LINKS" in src(c)),
            next(c for c in cells if "uv pip install --no-build-isolation vllm" in src(c)),
            next(c for c in cells if "uv pip uninstall tensorflow jax" in src(c))]
    out = []
    for i, c in enumerate(pick, 1):
        text = src(c)
        note = f"# Manh's cell {i} ({version_note}), verbatim"
        if i == 2:
            text = replace_once(text, "!rm -r /kaggle/working/*\n", "")
            note += " except `!rm -r /kaggle/working/*` (it would delete our fallback submission and smoke logs)"
        if i == 4:
            text = replace_once(text, '    raise RuntimeError("CUDA installation failed")',
                                '    if not RERUN:  # B32-M: a rerun keeps the fallback submission instead of erroring\n'
                                '        raise RuntimeError("CUDA installation failed")\n'
                                '    print("CUDA installation failed (rerun: continuing; the fallback stands)", flush=True)')
            note += " except: his CUDA-install raise only in commit runs"
        timing = (f'_B32M_T = globals().get("_B32M_T", {{}}); _B32M_T["manh_cell_{i}"] = round(time.time() - T0)'
                  f'; print("[B32-M] Manh\'s cell {i} starts at +", _B32M_T["manh_cell_{i}"], "s", flush=True)\n')
        out.append(code(note + "\n" + timing + text))
    return out


PRESERVER = """# Cell 2s (E066 --external): serving is Manh's server cell below (his launcher dsv41_vllm.py, his flags); the
# solver (cell 3) attaches to it with ARC_EXTERNAL_SERVER (S25) and never starts, stops or restarts it. Here: wait for
# the model staging pre-flight (cell 1) and hand his cell the checkpoint; a failed pre-flight or the fallback mode
# gives his launcher a missing path, so no server comes up and the solver keeps the fallback submission.
E066_MODEL = MODEL_SRC or "/nonexistent-model"
if MODE == "fallback":
    E066_MODEL = "/nonexistent-model-fallback-mode"
elif STAGE_DONE:
    _t = time.time()
    while not os.path.exists(STAGE_DONE) and time.time() - _t < 3600:
        time.sleep(5)
    _mk = open(STAGE_DONE).read().strip() if os.path.exists(STAGE_DONE) else "fail staging timeout"
    print("E066 staging marker:", _mk[:300], flush=True)
    if _mk.startswith("ok "):
        E066_MODEL = _mk[3:].strip()
    elif _mk.startswith("fail preflight"):
        E066_MODEL = "/nonexistent-model-preflight-failed"
os.environ["ARC_MODEL_DIR"] = E066_MODEL
print("E066 serving checkpoint:", E066_MODEL, flush=True)
"""


def external_server(cells, manh_nb, a):
    """--external: drop B32-M's flags cell (his server cell sets his env), insert cell 2s and his server cell (the one
    that runs dsv41_vllm.py --qwen38-port, \"server only\"), edited only where the user agreed: our checkpoint,
    --context, --concurrency, the lossless all-reduce mode, and the chat-path tool parser our harness needs."""
    cells = [c for c in cells if not src(c).startswith("# Cell 2m (B32-M)")]
    srv = next(src(c) for c in manh_nb["cells"] if "dsv41_vllm.py" in src(c) and "--qwen38-port" in src(c)
               and not src(c).lstrip().startswith("#"))
    edits = []
    for pat, rep in ((r"--dsv41-vllm-model \S+", "--dsv41-vllm-model {E066_MODEL}"),
                     (r"--context \d+", f"--context {a.server_context}"),
                     (r"--concurrency \d+", f"--concurrency {a.server_concurrency}"),
                     (r"--qwen38-port \d+", f"--qwen38-port {a.server_port}")):
        srv, n = re.subn(pat, rep, srv)
        assert n == 1, f"his server cell: {pat!r} matched {n} times"
        edits.append(rep)
    # his launcher sets --limit-mm-per-prompt from --vision-max-images (vision-repl-audit recipe: 16); the image harness
    # sends one image per grid, up to 14 on the eval set and 22 on training tasks, so a hidden task could exceed 16
    if "--vision-max-images" in srv:
        srv, n = re.subn(r"--vision-max-images \d+", f"--vision-max-images {a.vision_max_images}", srv)
        assert n == 1
    else:
        srv, n = re.subn(r"(--vision-max-pixels \d+ \\\n)", rf"\1  --vision-max-images {a.vision_max_images} \\\n", srv)
        assert n == 1, "his server cell: --vision-max-pixels line not found once"
    edits.append(f"--vision-max-images {a.vision_max_images}")
    if a.allreduce_mode:
        srv, n = re.subn(r'"VLLM_HOST_ALLREDUCE_MODE": "\d+"', f'"VLLM_HOST_ALLREDUCE_MODE": "{a.allreduce_mode}"', srv)
        assert n == 1, "his server cell: VLLM_HOST_ALLREDUCE_MODE not found once"
        edits.append(f"VLLM_HOST_ALLREDUCE_MODE={a.allreduce_mode}")
    lines = srv.split("\n")
    last = max(i for i, l in enumerate(lines) if l.strip().startswith("--vllm-arg="))
    assert not lines[last].rstrip().endswith("\\"), "expected the last --vllm-arg line to end his command"
    lines[last] = lines[last].rstrip() + " \\"
    lines[last + 1:last + 1] = ["  --vllm-arg=--tool-call-parser=deepseek_v41 \\", "  --vllm-arg=--enable-auto-tool-choice"]
    edits.append("--tool-call-parser deepseek_v41 --enable-auto-tool-choice")
    srv = "\n".join(lines)
    head = ("# Manh's server cell (his notebook as pulled), verbatim except (E066, agreed with the user): "
            + "; ".join(edits) + "\n")
    tail = ('\nos.environ["ARC_EXTERNAL_SERVER_LOG"] = f"{server_root}/server/server.log"\n'
            'print("E066 external server log:", os.environ["ARC_EXTERNAL_SERVER_LOG"], flush=True)\n')
    k = next(k for k, c in enumerate(cells) if src(c).startswith("# Cell 2 (B32-M): Manh's environment"))
    cells[k + 1:k + 1] = [code(PRESERVER), code(head + srv + tail)]
    s0 = src(cells[0])
    s0 = replace_once(s0, '               ("ARC_GRID_FORMAT", "text+image"),\n',
                      '               ("ARC_GRID_FORMAT", "text+image"),\n'
                      f'               ("ARC_EXTERNAL_SERVER", "http://127.0.0.1:{a.server_port}/v1"),\n'
                      f'               ("ARC_PORT", "{a.server_port}"),\n')
    s0 = replace_once(s0, '("ARC_CONTEXT_LIMIT", "163840")', f'("ARC_CONTEXT_LIMIT", "{a.server_context}")')
    cells[0] = code(s0)
    return cells


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manh-nb", type=Path, required=True, help="`kaggle kernels pull -m` folder of his notebook")
    ap.add_argument("--slug", default="dsv41-tools-submit-img")
    ap.add_argument("--model-source", required=True, help="the E066 Kaggle Model handle, owner/model/framework/instance/version")
    ap.add_argument("--model-manifest", type=Path, help="the export's manifest (path, size, sha256): smoke integrity check")
    ap.add_argument("--external", action="store_true",
                    help="serving = Manh's own server cell (his launcher and flags, the source of truth for serving); the "
                         "solver attaches to it with ARC_EXTERNAL_SERVER (S25) and never starts it")
    ap.add_argument("--server-port", type=int, default=18084, help="--external: his --qwen38-port")
    ap.add_argument("--server-context", type=int, default=163840, help="--external: his --context (= ARC_CONTEXT_LIMIT)")
    ap.add_argument("--server-concurrency", type=int, default=120, help="--external: his --concurrency (max-num-seqs)")
    ap.add_argument("--vision-max-images", type=int, default=32,
                    help="--external: his --vision-max-images (images per prompt); his launcher accepts 2-32 "
                         "(qwen38_vision.validate_options); ARC's worst case is ~24 (10 train pairs x 2 + tests)")
    ap.add_argument("--allreduce-mode", default="556",
                    help="--external: VLLM_HOST_ALLREDUCE_MODE in his cell (60 = lossy q8; 556 lossless); '' keeps his")
    ap.add_argument("--workers", type=int, default=80)
    ap.add_argument("--util", type=float, default=0.945, help="Kaggle's equivalent of the pods' 0.938 (B32: 0.945)")
    ap.add_argument("--smoke-minutes", type=int, default=37, help="kill switch at N-0.5 min, stop-new N-5, force N-3")
    ap.add_argument("--smoke-tasks", default="00576224,25ff71a9", help="commit-run tasks (the 2 smallest; 25ff71a9 has 2 test inputs)")
    ap.add_argument("--manh-flags", action="store_true", help="turn Manh's extra flags ON (after E062b)")
    ap.add_argument("--load-test", action="store_true", help="commit run = an 80-task eval load test (not a candidate)")
    ap.add_argument("--manh-datasets", default=",".join(MANH_DATASETS))
    ap.add_argument("--docker-image", help="pin this image (\"original\"); default: Kaggle's latest image, which his code needs (smoke V1 on his pinned image saw no GPU)")
    ap.add_argument("--versions", type=Path, default=HERE / "b32m_manh_versions.json",
                    help="his dataset versions / wheel names recorded at build time (printed in the notebook)")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    out = a.out or HERE / f"out/nb_{a.slug}"
    tmp = out.with_name(out.name + "_b32base")
    nb, meta = b32_notebook(a, tmp)
    manh_meta = json.load(open(a.manh_nb / "kernel-metadata.json"))
    manh_nb = json.load(open(a.manh_nb / manh_meta["code_file"]))
    versions = json.load(open(a.versions)) if a.versions.exists() else {}
    version_note = (f"{manh_meta['id']} as pulled; vllm wheel {versions.get('vllm_wheel', '?')}")

    cells = nb["cells"]
    # settings: max-num-seqs 96 and the server env, as on the L4 replica
    s0 = src(cells[0])
    s0 = replace_once(s0, '("ARC_MAX_NUM_SEQS", "64")', f'("ARC_MAX_NUM_SEQS", "{SEQS}")')
    s0 = replace_once(s0, '               ("ARC_SERVER_EXTRA_ARGS",',
                      '               ("ARC_GRID_FORMAT", "text+image"),\n'
                      f'               ("ARC_SERVER_ENV_JSON", {json.dumps(json.dumps(SERVER_ENV))}),\n'
                      '               ("ARC_SERVER_EXTRA_ARGS",')
    s0 = ("# E066 image candidate (2026-10-29, build_img.py): B32-M + the E058 image harness (ARC_GRID_FORMAT) and the\n"
          f"# E066 checkpoint {a.model_source} (vision tower kept).\n"
          "# B32-M (2026-10-28): B32 with Manh's vLLM environment (build_b32m.py): his image, CUDA 13 and wheels; the\n"
          f"# E062 L4 replica's server config (DCP4, offload 32 GiB, graphs/seqs {SEQS}); {a.workers} workers, KV gate 0.85.\n"
          + s0)
    cells[0] = code(s0)
    i = next(i for i, c in enumerate(cells) if src(c).startswith("# Cell 2: offline install"))
    flags = CELLS["manh_flags"].read_text()
    flags = replace_once(flags, "MANH_FLAGS = False", f"MANH_FLAGS = {bool(a.manh_flags)}")
    env_cell = replace_once(CELLS["env"].read_text(), '"""__VERSIONS__"""', '"""' + json.dumps(versions) + '"""')
    rowchunk = CELLS["patches"].read_text()
    for key, path in PATCH_DIFFS.items():
        diff = path.read_bytes()
        rowchunk = replace_once(rowchunk, f"__{key}_DIFF_B64__", base64.b64encode(diff).decode())
        rowchunk = replace_once(rowchunk, f"__{key}_DIFF_SHA__", hashlib.sha256(diff).hexdigest())
    new = manh_install_cells(manh_nb, version_note) + [code(rowchunk), code(env_cell), code(flags)]
    cells[i:i + 1] = new
    g = next(g for g, c in enumerate(cells) if src(c).startswith("# Cell 0: clock"))
    cells.insert(g + 1, code(CELLS["gpugate"].read_text()))  # before anything slow (Manh's install, staging)
    if a.smoke_minutes:
        j = next(j for j, c in enumerate(cells) if src(c).startswith("# Cell 3: the solver"))
        # B32's tool-sandbox check on the new image's Python (B32 ran it via --smoke-stress), then the background
        # contract check of the solver's server
        cells[j:j] = [code(B.SANDBOX_CELL.read_text()), code(CELLS["contract"].read_text())]
        k = next(k for k, c in enumerate(cells) if src(c).startswith("# Cell 5 (smoke report"))
        rep5 = replace_once(replace_once(src(cells[k]), *INTEG_WAIT), *INTEG_DONE)
        # our report prints first (B32's report may raise on integrity), the verdict raises last
        cells[k:k + 1] = [code(CELLS["report"].read_text()), code(rep5), code(CELLS["verdict"].read_text())]
        s0 = replace_once(src(cells[0]), *SMOKE_METRICS)
        if a.load_test:  # the eval challenges file of the competition data, 80 typical tasks
            tasks = (f'os.environ["B32M_LOAD_TEST"] = "1"\n'
                     f'    import glob as _lg\n'
                     # cell 00 runs before cell 0's find_input: fixed depths, never a recursive scan
                     f'    _lf = sorted({{_p for _d in {IP.PRE_CELL0_DIRS}\n'
                     f'                  for _p in _lg.glob(f"/kaggle/input/{{_d}}arc-agi_evaluation_challenges.json")}})\n'
                     f'    if _lf:\n'
                     f'        os.environ.setdefault("ARC_CHALLENGES", _lf[0])\n'
                     f'        os.environ.setdefault("ARC_TASK_IDS", "{LOAD_EVAL_TASKS}")\n'
                     f'    print("B32-M LOAD TEST: eval file", _lf[:1], flush=True)')
        else:
            tasks = f'os.environ.setdefault("ARC_TASK_IDS", "{a.smoke_tasks}")'
        cells[0] = code(replace_once(s0, 'os.environ.setdefault("ARC_TASK_IDS", "25ff71a9,27a28665,00576224")', tasks))
    if a.external:
        cells = external_server(cells, manh_nb, a)
    nb["cells"] = cells
    out.mkdir(parents=True, exist_ok=True)
    meta["dataset_sources"] = [d for d in meta["dataset_sources"] if d == "chankhavu/dsv41-tools-kit"] + [
        d for d in a.manh_datasets.split(",") if d]
    meta["kernel_sources"] = []  # B32's fork wheel kernel is not used
    meta["model_sources"] = [a.model_source]  # E066: the image checkpoint (replaces E128)
    meta.pop("docker_image", None)
    if a.docker_image:
        meta.update(docker_image=a.docker_image, docker_image_pinning_type="original")
    else:  # Manh: his code only works on the newest Kaggle environment
        meta["docker_image_pinning_type"] = "latest"
    meta.update(id=f"chankhavu/{a.slug}", title=a.slug, code_file=f"{a.slug}.ipynb", is_private=True,
                enable_gpu=True, enable_internet=False, machine_shape="NvidiaL4")
    assert "arc-prize-2026-arc-agi-2" in meta["competition_sources"], "4xL4 needs the competition attached"
    IP.finish(nb, meta)  # cell 0's INPUT_ROOTS from these sources; no cell scans the input root
    json.dump(nb, open(out / f"{a.slug}.ipynb", "w"), indent=1, ensure_ascii=False)
    json.dump(meta, open(out / "kernel-metadata.json", "w"), indent=1)
    print(out, len(nb["cells"]), "cells;", {k: meta[k] for k in ("id", "dataset_sources", "kernel_sources",
                                                                 "model_sources", "docker_image_pinning_type")},
          "manh flags", "ON" if a.manh_flags else "OFF")


if __name__ == "__main__":
    main()
