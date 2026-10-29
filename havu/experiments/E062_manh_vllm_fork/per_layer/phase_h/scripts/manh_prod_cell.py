# Cell 2m (B32-M + Manh's server): MANH'S PRODUCTION vLLM SERVER, started and owned by this cell, not by the solver.
# dsv41-vllm-prod.md's "Server only" launch: his env (MANH_SERVER_ENV, his env cell), /kaggle/working/dsv41_vllm.py
# (Manh's cell 2 copies it from arc2-2026-src) and his production flags, plus what B32's harness needs on chat turns
# (tools with tool_choice "auto": --enable-auto-tool-choice --tool-call-parser deepseek_v41) and B32's
# --prefix-cache-retention-interval 128. Waits for the model staging marker (cell 1: the FUSE mount read once), launches,
# waits until /v1/models serves dsv41, then defines MANH_BASE_URL, MANH_PORT (= ARC_PORT), MANH_MODEL, MANH_SERVER_PID,
# MANH_SERVER_LOG. Cell 3 hands MANH_BASE_URL to the solver as ARC_EXTERNAL_SERVER (S23, cell 0b's pinned solver): it
# never starts, stops or restarts a server, and a death of this one ends new attempts (0 restarts).
# MANH_VISION=0: --language-model-only (text prompts only, more KV) instead of his vision-repl-audit recipe.
# No server: commit runs RAISE; the competition rerun keeps the fallback submission (MODE = "fallback").
import signal, urllib.request

# Tuned humming MoE tiles for the TP4 routed-expert shards, L4 only (num_sms 58); dsv41-vllm-prod.md, Requirements.
_moe_cfg = lambda block, warp, stages, ctas, stream_k: {
    "block_shape": block, "warp_shape": warp, "num_stages": stages, "num_ctas_per_sm": ctas,
    "use_stream_k": stream_k, "num_sms": 58, "mma_type": "mma", "output_chunk_rows": 0,
    "use_f16_accum": False, "use_pdl": False, "use_tma_a": False, "use_tma_as": False,
    "use_tma_as2": False, "use_tma_c": False}
DSV41_MOE_TUNING = {  # "NxK": [[lo, hi, config], ...], used when lo < tokens x top_k 6 <= hi
    "1152x5120": [
        [1926, 3680, _moe_cfg([32, 128, 128], [32, 64, 64], 5, 2, False)],
        [3888, 5504, _moe_cfg([48, 128, 128], [48, 64, 64], 3, 1, False)],
        [5504, 8192, _moe_cfg([64, 128, 128], [64, 32, 128], 3, 1, True)],
        [8192, 12288, _moe_cfg([48, 128, 128], [48, 64, 64], 3, 1, True)],
    ],
    "5120x576": [
        [1840, 4248, _moe_cfg([32, 256, 64], [32, 64, 64], 5, 2, False)],
        [4248, 5592, _moe_cfg([48, 256, 64], [48, 64, 32], 5, 1, False)],
        [5592, 8192, _moe_cfg([64, 256, 64], [64, 32, 64], 5, 1, False)],
    ],
}
MANH_SERVER_ENV = {  # dsv41-vllm-prod.md's env cell (os.environ.update there)
    "VLLM_HOST_ALLREDUCE": "1",
    "VLLM_HOST_ALLREDUCE_MODE": "556",            # lossless 44 + keep-warm (512), no Q8 (16): g5e1bccd07c+ wheels
    "VLLM_HOST_ALLREDUCE_PREFETCH_BLOCKS": "16",
    "VLLM_HOST_ALLREDUCE_PREFETCH_MAX_MB": "36",
    "VLLM_HOST_ALLREDUCE_LARGE_SHAPE": "32x32x256",
    "VLLM_HOST_ALLREDUCE_SHAPE": "12x12x256",
    "VLLM_HOST_ALLREDUCE_UNIT_KB": "16",
    "VLLM_TP_HOST_GATHER": "1",                   # host TP all-gather
    "VLLM_DCP_ASYNC_QUERY_GATHER": "1",           # DCP only: fp8 query all-gather on a side stream (default)
    "VLLM_HUMMING_MOE_PAD_MULTIPLE": "64,64",     # routed experts unpadded: 576-wide TP4 shard, not 640
    "VLLM_HUMMING_MOE_TUNING_CONFIG": json.dumps(DSV41_MOE_TUNING),  # 321+ token steps only
    "VLLM_TRITON_SPARSE_INDEXER_LOGITS": "1",     # indexer consumers score only layer 20's candidates
    "VLLM_DSA_DECODE_INDEXER_ROW_SHARD": "1",     # each TP rank scores 1/4 of the decode indexer rows
    "VLLM_DSV41_AR_OVERLAP_CHUNKS": "2",          # prefill all-reduces under row chunks (gf506afa210+ wheel)
    "VLLM_GPU_DRIVER_RESERVE_MB": "512",          # keep 512 MiB out of the torch cache (dsv41-vllm-prod.md, Requirements)
    "VLLM_GPU_MEMORY_LOG_INTERVAL": "60",         # per-rank "GPU memory: ..." line every 60 s
}
MANH_PORT = int(os.environ.get("ARC_PORT", "18084"))
MANH_BASE_URL = f"http://127.0.0.1:{MANH_PORT}/v1"
# --context (max-model-len) and --concurrency (max-num-seqs; CUDA graphs up to it): B32-M's 163840 and 80 workers.
# dsv41-vllm-prod.md's KV and memory numbers were measured at 131072 / 64: check the KV line of the first start.
MANH_CONTEXT, MANH_SEQS = 163840, 80
MANH_VISION = os.environ.get("MANH_VISION", "1") == "1"
MANH_VISION_MAX_IMAGES = int(os.environ.get("MANH_VISION_MAX_IMAGES", "16"))  # per prompt; his recipe's 16
MANH_MODEL, MANH_SERVER_PID, MANH_SERVER_LOG = MODEL_SRC, None, None


def _manh_alive(pid):
    try:
        with open(f"/proc/{pid}/stat") as f:
            return f.read().rsplit(")", 1)[1].split()[0] not in ("Z", "X")
    except OSError:
        return False


def _manh_tail(n=2500):
    try:
        return " | ".join(open(MANH_SERVER_LOG, errors="replace").read().splitlines()[-15:])[-n:]
    except Exception as exc:
        return f"<no server.log: {type(exc).__name__}>"


if MODE != "fallback" and MODEL_SRC:
    try:
        assert int(os.environ.get("ARC_CONTEXT_LIMIT") or MANH_CONTEXT) <= MANH_CONTEXT, "ARC_CONTEXT_LIMIT > --context"
        assert os.path.isfile(f"{WORK}/dsv41_vllm.py"), f"no {WORK}/dsv41_vllm.py (Manh's cell 2 copies arc2-2026-src)"
        if STAGE_DONE:  # the solver waited for it before loading; so does this cell
            _t = time.time()
            while not os.path.exists(STAGE_DONE) and time.time() - _t < float(os.environ.get("ARC_STAGE_WAIT_S", 2400)):
                time.sleep(5)
            _staged = open(STAGE_DONE).read().strip() if os.path.exists(STAGE_DONE) else ""
            print(f"model staging: {_staged or 'not finished'} after {time.time() - _t:.0f}s wait", flush=True)
            if _staged.startswith("ok "):
                MANH_MODEL = _staged.split(" ", 1)[1]
        _env = dict(BASE_ENV)  # cell 2's: CUDA 13 first on PATH, HOME and caches under SCRATCH
        _env.update(MANH_SERVER_ENV)
        _cudnn = "/usr/local/lib/python3.13/dist-packages/nvidia/cudnn"  # his cudnn cell
        if not os.path.isdir(_cudnn):
            import importlib.util
            _cudnn = list(importlib.util.find_spec("nvidia.cudnn").submodule_search_locations)[0]
        _env.update(CUDNN_HOME=_cudnn, CUDNN_PATH=_cudnn, LD_LIBRARY_PATH=f"{_cudnn}/lib:" + _env.get("LD_LIBRARY_PATH", ""))
        _root = f"{WORK}/dsv41_server_{time.time_ns()}"
        _mode = (["--qwen38-recipe", "vision-repl-audit", "--vision-max-pixels", "2097152",
                  "--vision-max-images", str(MANH_VISION_MAX_IMAGES)] if MANH_VISION else
                 ["--qwen38-recipe", "manual", "--solve-mode", "tir", "--tir-interface", "repl", "--input-mode", "text"])
        _cmd = [PY, "-u", f"{WORK}/dsv41_vllm.py", "--qwen38-variant", "dsv4-vllm", *_mode,
                "--dsv41-vllm-model", MANH_MODEL, "--dsv41-vllm-triton-cache", f"{WORK}/dsv41_triton",
                "--output-root", _root, "--tensor-parallel-size", "4", "--gpu-ids", "0,2,1,3",
                "--qwen38-port", str(MANH_PORT), "--temp-root", "/tmp/arc_vision", "--context", str(MANH_CONTEXT),
                "--concurrency", str(MANH_SEQS), "--gpu-memory-utilization", "0.95", "--kv-offload-gib-per-rank", "12",
                "--max-num-batched-tokens", "2048",
                "--vllm-arg=--offload-backend=uva", "--vllm-arg=--cpu-offload-gb=0.35",
                "--vllm-arg=--cpu-offload-params=embed_tokens", "--vllm-arg=--engram-config.use_thp=true",
                "--vllm-arg=--watermark=0.1", "--vllm-arg=--attention-config.sparse_mla_prefill_workspace_factor=2",
                "--vllm-arg=--enable-auto-tool-choice", "--vllm-arg=--tool-call-parser=deepseek_v41",
                "--vllm-arg=--prefix-cache-retention-interval=128"]
        _r = subprocess.run(_cmd, cwd=WORK, env=_env, stdin=subprocess.DEVNULL, capture_output=True, text=True,
                            errors="replace", timeout=900)
        print("\n".join(l for l in (_r.stdout + _r.stderr).splitlines() if not l.startswith("Command:"))[-4000:],
              flush=True)  # not its "Command:" line (the full argv)
        assert _r.returncode == 0, f"dsv41_vllm.py exited with {_r.returncode}"
        MANH_SERVER_PID = int(json.load(open(f"{_root}/server/launch.json"))["pid"])
        MANH_SERVER_LOG = f"{_root}/server/server.log"
        _t, _last, _models = time.time(), time.time(), []
        _deadline = min(_t + float(os.environ.get("ARC_SERVER_UP_TIMEOUT_S", 7200)),
                        T0 + float(os.environ.get("ARC_STOP_NEW_S", 39000)))
        while True:
            try:
                with urllib.request.urlopen(f"{MANH_BASE_URL}/models", timeout=5) as _u:
                    _models = json.loads(_u.read())["data"]
                if any(m.get("id") == "dsv41" for m in _models):
                    break
            except Exception:
                pass
            if not _manh_alive(MANH_SERVER_PID):
                raise RuntimeError(f"server exited during startup; server.log: {_manh_tail()}")
            if time.time() > _deadline:
                raise RuntimeError(f"server not up after {time.time() - _t:.0f}s; server.log: {_manh_tail()}")
            if time.time() - _last > 120:
                _last = time.time()
                print(f"[+{time.time() - T0:.0f}s] server loading ({time.time() - _t:.0f}s): {_manh_tail(300)}", flush=True)
            time.sleep(5)
        print(f"MANH SERVER UP after {time.time() - _t:.0f}s (+{time.time() - T0:.0f}s): {MANH_BASE_URL} model dsv41, "
              f"max_model_len {[m.get('max_model_len') for m in _models]}, pid {MANH_SERVER_PID}, log {MANH_SERVER_LOG}",
              flush=True)
    except Exception as exc:
        print("=" * 100, f"MANH SERVER FAILED: {type(exc).__name__}: {exc}", "=" * 100, sep="\n", flush=True)
        if MANH_SERVER_PID:
            try:
                os.killpg(MANH_SERVER_PID, signal.SIGKILL)
            except Exception:
                pass
        MANH_SERVER_PID = None
        if not RERUN:
            raise
        MODE = "fallback"  # rerun: the fallback submission (cell 0) stands