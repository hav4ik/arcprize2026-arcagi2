#!/usr/bin/env python3
"""First real-server smoke test of evals/serve_replay_bench.py --mode synthetic (coordinator request).
Starts our lab server (same flags as run_suite BASE, longer max-model-len for the V4.1 trace lengths),
runs the bench unmodified, stops the server. Output: runs/bench_smoke/."""
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_suite as rs  # noqa: E402

REPO = "/home/vu/Workspace/ARC-AGI-2"
OUT = os.path.join(rs.RUNS, "bench_smoke")
TRACES = f"{REPO}/experiments/E064_manh_notebook_b32_harness/l4e2e/runs/b32m_x2t/out/dsv41_traces.tar.gz"


def main():
    os.makedirs(OUT, exist_ok=True)
    args = list(rs.BASE)
    args[args.index("--max-model-len") + 1] = "131072"
    i = args.index("--scheduler-cls")
    del args[i:i + 2]  # plain upstream scheduler for the bench smoke
    logp = os.path.join(OUT, "server.log")
    log = open(logp, "w")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="1", HF_HUB_OFFLINE="1", VLLM_ALLOW_LONG_MAX_MODEL_LEN="1")
    cmd = ["flock", rs.LOCK, rs.VLLM, "serve", rs.MODEL] + args
    print("[bench_smoke] start", " ".join(cmd), flush=True)
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
    try:
        if not rs.wait_health(proc, logp):
            print("[bench_smoke] server failed", flush=True)
            return 1
        bench = [rs.PY, f"{REPO}/evals/serve_replay_bench.py", "--base-url", f"http://127.0.0.1:{rs.PORT}",
                 "--traces", TRACES, "--mode", "synthetic", "--synthetic-vocab", "100:150000",
                 "--concurrency", "8", "--duration", "240", "--warmup-s", "60", "--think-time", "2",
                 "--seed", "0", "--server-log", logp, "--out", os.path.join(OUT, "out")]
        print("[bench_smoke] run", " ".join(bench), flush=True)
        t0 = time.time()
        r = subprocess.run(bench, cwd=REPO, stdout=open(os.path.join(OUT, "bench.log"), "w"),
                           stderr=subprocess.STDOUT)
        print(f"[bench_smoke] bench rc={r.returncode} wall={time.time() - t0:.0f}s", flush=True)
    finally:
        rs.stop(proc)
        log.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
