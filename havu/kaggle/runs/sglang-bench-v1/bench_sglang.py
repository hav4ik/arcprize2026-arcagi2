"""Offline SGLang throughput test on Kaggle 4xL4 (E013). Runs next to solver.py (imports its helpers).

Steps, time-boxed by ARC_BENCH_STOP_MIN (default 47 min from notebook start; the push also sets a session timeout):
1. Install SGLang 0.5.20 from the attached wheel source (chankhavu/sglang-0-5-20-py312-cu130) with --no-index,
   pinned to its requirements.lock. Print torch/CUDA/driver info.
2. Config A, radix cache ON: start server (TP=4, FP8 weights, fp8_e4m3 KV), record startup time and the memory pool
   sizes from the log; decode benchmark with 16 sequences at ~40k context (same prompts as the vLLM bench), then 32
   sequences reusing those 16 prompts (radix hits, so little extra prefill).
3. Config B, radix cache OFF: restart, record pool sizes; short decode benchmark (16 sequences, ~10k context).
Results: /kaggle/working/sglang_bench.json, server logs in /kaggle/working/arc_logs/.
"""
import asyncio
import glob
import json
import os
import re
import subprocess
import sys
import time

sys.path.insert(0, "/kaggle/working")
import solver as S  # noqa: E402

STOP = S.T0 + float(os.environ.get("ARC_BENCH_STOP_MIN", 47)) * 60
OUT = os.path.join(S.WORK, "sglang_bench.json")
RESULTS = {"configs": {}}


def save():
    json.dump(RESULTS, open(OUT, "w"), indent=1)


def left():
    return STOP - time.time()


def wheel_dir_sglang():
    p = S.find_one("/kaggle/input/**/sglang-0.5.20-*.whl")
    if not p:
        raise FileNotFoundError("sglang wheel source not attached")
    return os.path.dirname(p)


def install():
    wd = wheel_dir_sglang()
    t = time.time()
    code, out = S.sh(f"{sys.executable} -m pip install --no-index --find-links {wd} -r {wd}/requirements.lock",
                     timeout=1800)
    RESULTS["install_s"] = round(time.time() - t)
    S.log(f"pip install (offline, lock) rc={code} in {RESULTS['install_s']}s")
    if code != 0:
        print(out[-4000:])
        raise RuntimeError("pip install failed")
    code, out = S.sh(f"{sys.executable} -c 'import torch, sglang, flashinfer; print(torch.__version__, torch.version.cuda,"
                     f" torch.cuda.is_available(), torch.cuda.device_count(), sglang.__version__, flashinfer.__version__)'",
                     timeout=600)
    RESULTS["versions"] = out.strip().splitlines()[-1] if out.strip() else out
    _, smi = S.sh("nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader", timeout=60)
    RESULTS["gpus"] = smi.strip()
    S.log("versions:", RESULTS["versions"], "| gpus:", smi.strip().replace("\n", "; "))
    save()


def launch(mdir, name, radix):
    cmd = [sys.executable, "-m", "sglang.launch_server", "--model-path", mdir, "--served-model-name", S.SERVED,
           "--host", "127.0.0.1", "--port", str(S.PORT), "--tp-size", "4", "--kv-cache-dtype", "fp8_e4m3",
           "--context-length", "131072", "--mem-fraction-static", os.environ.get("ARC_SGL_MEM", "0.85"),
           "--max-running-requests", "64", "--trust-remote-code",
           "--attention-backend", os.environ.get("ARC_SGL_ATTN", "triton"), "--log-level", "info"]
    if not radix:
        cmd.append("--disable-radix-cache")
    env = os.environ.copy()
    env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false",
               PYTORCH_ALLOC_CONF="expandable_segments:True")
    path = os.path.join(S.LOG_DIR, f"sglang_{name}.log")
    S.log(f"[{name}] server cmd:", " ".join(cmd))
    fh = open(path, "w")
    proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                            start_new_session=True, env=env)
    return proc, path


def pools(path):
    txt = open(path, errors="ignore").read()
    out = {}
    for key in ["max_total_num_tokens", "max_running_requests", "max_mamba_cache_size", "context_len",
                "chunked_prefill_size", "max_prefill_tokens"]:
        m = re.findall(key + r"\W{1,3}(\d+)", txt)
        if m:
            out[key] = int(m[-1])
    for pat in [r"KV Cache is allocated\.[^\n]*", r"Mamba Cache is allocated\.[^\n]*", r"Memory pool end\.[^\n]*",
                r"Load weight end\.[^\n]*", r"Capture cuda graph end\.[^\n]*", r"attention_backend[^\n]{0,80}"]:
        m = re.findall(pat, txt)
        if m:
            out[pat[:20]] = m[-1][:300]
    return out


def stop_server(proc):
    try:
        os.killpg(proc.pid, 9)
    except Exception:
        pass
    S.sh("pkill -9 -f sglang.launch_server; pkill -9 -f 'sglang::'; sleep 8", timeout=60)


async def bench_cfg(name, radix, steps, mdir, tasks, prefixes):
    if left() < 12 * 60:
        S.log(f"[{name}] skipped: {left() / 60:.1f} min left")
        return
    res = RESULTS["configs"].setdefault(name, {"radix": radix})
    t = time.time()
    proc, path = launch(mdir, name, radix)
    ok = S.wait_ready(proc, path, timeout=min(1500, max(60, left() - 120)))
    res["startup_s"] = round(time.time() - t)
    res["ready"] = ok
    res["pools"] = pools(path)
    S.log(f"[{name}] ready={ok} in {res['startup_s']}s pools={json.dumps(res['pools'])}")
    save()
    if not ok:
        res["log_tail"] = S.server_log_tail(path, 80)
        save()
        stop_server(proc)
        return
    client = S.Client()
    for label, kw in steps:
        if left() < 4 * 60:
            S.log(f"[{name}] stopping before {label}: time box")
            break
        try:
            r = await asyncio.wait_for(S.bench_decode(client, prefixes, tasks=tasks, **kw), timeout=max(60, left() - 90))
            r["prompt_tokens_last"] = client.last_prompt_tokens
            res[label] = r
        except Exception as exc:
            res[label] = {"error": f"{type(exc).__name__}: {exc}"[:300]}
            S.log(f"[{name}] {label} failed: {exc}")
        save()
    res["log_metrics"] = [ln for ln in open(path, errors="ignore").read().splitlines()
                          if "Decode batch" in ln or "gen throughput" in ln][-12:]
    save()
    stop_server(proc)


async def main_async():
    tasks = json.load(open(S.comp_file("arc-agi_evaluation_challenges.json")))
    prefixes = json.load(open(S.find_one("/kaggle/input/**/bench_prefixes.json")))
    mdir = S.model_dir()
    await bench_cfg("radix_on", True, [
        ("decode_16x40k", dict(n_seqs=16, gen_tokens=1000)),
        ("decode_32x40k_shared", dict(n_seqs=32, gen_tokens=1000, n_unique=16)),
    ], mdir, tasks, prefixes)
    await bench_cfg("radix_off", False, [
        ("decode_16x10k", dict(n_seqs=16, gen_tokens=1000, max_chars=20000)),
    ], mdir, tasks, prefixes)


def driver_env():
    """Put the host driver's libcuda first (Kaggle images also carry an older compat libcuda)."""
    real = "/usr/local/nvidia/lib64"
    if os.path.exists(real + "/libcuda.so"):
        os.environ["LD_LIBRARY_PATH"] = real + ":" + os.environ.get("LD_LIBRARY_PATH", "")
        os.environ["LD_PRELOAD"] = real + "/libcuda.so"


def main():
    os.makedirs(S.LOG_DIR, exist_ok=True)
    driver_env()
    S.log("sglang bench start")
    try:
        install()
        asyncio.run(main_async())
    except BaseException as exc:
        import traceback
        traceback.print_exc()
        RESULTS["fatal"] = f"{type(exc).__name__}: {exc}"[:500]
    save()
    S.log("sglang bench done:", json.dumps(RESULTS)[:3000])


if __name__ == "__main__":
    main()
    os._exit(0)
