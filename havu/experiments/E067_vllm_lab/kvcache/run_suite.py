#!/usr/bin/env python3
"""Start a vLLM server on GPU 1 under the lab GPU lock, run load-generator configs against it,
then stop it (own process group only) so the lock is released between server configs.

Usage: run_suite.py SUITE [--only NAME ...]   (suites are defined in SUITES below)
"""
import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
PY = "/home/vu/miniconda3/envs/vllm/bin/python"
VLLM = "/home/vu/miniconda3/envs/vllm/bin/vllm"
LOCK = "/tmp/claude-1000/-home-vu-Workspace-ARC-AGI-2/81e68821-e1f4-4b05-bad6-9a606f77ad54/scratchpad/vllmlab/gpu1.lock"
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
PORT = 8067
RUNS = os.path.join(HERE, "runs")

# KV pool: Qwen2.5-0.5B = 24 layers x 2 kv heads x 64 dim x 2 (K,V) x 2 B = 12,288 B/token.
KV_BYTES = 2_400_000_000  # ~195k tokens

BASE = ["--max-model-len", "32768", "--gpu-memory-utilization", "0.5",
        "--kv-cache-memory-bytes", str(KV_BYTES), "--enable-prompt-tokens-details",
        "--served-model-name", "m", "--port", str(PORT), "--seed", "0",
        "--generation-config", "vllm",
        # observers only: instrumented scheduler (subclass of the default AsyncScheduler) and
        # sampled block-residency histograms
        "--scheduler-cls", "kvlab_sched.LabScheduler",
        "--kv-cache-metrics", "--kv-cache-metrics-sample", "0.05"]


def R(name, extra=(), env=None, **kw):
    """one run = one fresh server (fresh cache, lock released between runs)"""
    return (name, list(extra), env or {}, kw)


OFF = lambda gib: ["--kv-offloading-size", str(gib)]  # noqa: E731  native -> OffloadingConnector

SUITES = {
    "smoke": [R("smoke_c8", concurrency=8, duration=60, stagger=5)],
    # Exp 1: no offload, concurrency sweep past KV capacity.
    "exp1": [R("e1_c8", concurrency=8), R("e1_c16", concurrency=16),
             R("e1_c32", concurrency=32), R("e1_c64", concurrency=64)],
    # Exp 1b: scheduler knobs (V1 has no watermark): admit-on-first-chunk, max-num-seqs cap.
    "exp1b": [R("e1b_c32_noreserve", ["--no-scheduler-reserve-full-isl"], concurrency=32),
              R("e1b_c32_maxseqs12", ["--max-num-seqs", "12"], concurrency=32)],
    # Exp 2: CPU offload (OffloadingConnector, LRU) at two host sizes.
    "exp2": [R("e2_off4_c32", OFF(4), concurrency=32), R("e2_off4_c64", OFF(4), concurrency=64),
             R("e2_off12_c32", OFF(12), concurrency=32), R("e2_off12_c64", OFF(12), concurrency=64),
             R("e2_off4_c16", OFF(4), concurrency=16)],
    # Exp 3: client-side admission gate on KV usage (new attempts only), no offload.
    "exp3": [R("e3_c64_gate70", concurrency=64, gate_kv=0.70, gate_min=4),
             R("e3_c64_gate85", concurrency=64, gate_kv=0.85, gate_min=4),
             R("e3_c64_gate70_off4", OFF(4), concurrency=64, gate_kv=0.70, gate_min=4)],
}


def wait_health(proc, timeout=600):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            return False
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(2)
    return False


def stop(proc):
    try:
        os.killpg(proc.pid, signal.SIGTERM)  # our own session/process group only
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=60)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()


def run_one(lname, extra, env_extra, kw):
    os.makedirs(RUNS, exist_ok=True)
    out = os.path.join(RUNS, lname)
    server_args = BASE + extra
    log = open(out + ".server.log", "w")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="1", HF_HUB_OFFLINE="1", PYTHONPATH=HERE,
               KVLAB_SCHED_LOG=out + ".sched.jsonl", **env_extra)
    if os.path.exists(out + ".sched.jsonl"):
        os.remove(out + ".sched.jsonl")
    cmd = ["flock", LOCK, VLLM, "serve", MODEL] + server_args
    print("[suite]", time.strftime("%H:%M:%S"), "start", lname, " ".join(extra), kw, flush=True)
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
    try:
        if not wait_health(proc):
            print("[suite] server failed; see", log.name, flush=True)
            return
        txt = open(log.name).read()
        m = re.search(r"GPU KV cache size: ([\d,]+) tokens", txt)
        kv_tokens = int(m.group(1).replace(",", "")) if m else None
        a = [PY, os.path.join(HERE, "loadgen.py"), "--out", out, "--server-pid", str(proc.pid),
             "--url", f"http://127.0.0.1:{PORT}"]
        for k, v in kw.items():
            a += ["--" + k.replace("_", "-"), str(v)]
        t0 = time.time()
        subprocess.run(a, check=False)
        with open(out + ".server.json", "w") as f:
            json.dump({"server_args": server_args, "env": env_extra, "kv_tokens": kv_tokens,
                       "wall": time.time() - t0}, f, indent=1)
    finally:
        stop(proc)
        log.close()
        print("[suite]", time.strftime("%H:%M:%S"), "stopped", lname, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("suites", nargs="+")
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    for suite in a.suites:
        for lname, extra, env, kw in SUITES[suite]:
            if a.only and lname not in a.only:
                continue
            run_one(lname, extra, env, kw)
            time.sleep(3)  # let a waiting lab agent take the lock


if __name__ == "__main__":
    sys.exit(main())
