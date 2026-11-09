#!/usr/bin/env python3
"""E059 OOM kill-order test (scaled-down Kaggle memory situation, run inside a memory-limited cgroup).

Kaggle kills by cgroup: 173 GiB for everything. A B32 run holds ~123 GiB of shared memory (Engram + pinned KV pool) and
~13-20 GiB of anonymous memory in the model server. If tool sandboxes then allocate more than the rest, the kernel's
memcg OOM killer must pick a sandbox (the executor sets oom_score_adj 1000), never the server; the executor restarts the
killed session and the attempt goes on.

This script reproduces that at small scale with the real executor: a stand-in "server" (oom_score_adj 0) holds
SERVER_SHM_GIB of touched shared memory (/dev/shm) plus SERVER_ANON_GIB of anonymous memory; then N sandbox sessions
each allocate SANDBOX_GIB at once (under their own 4 GiB caps), together exceeding the cgroup limit. Pass: the server
stays alive, at least one sandbox is OOM-killed and its session restarts and works.

Run:  systemd-run --user --scope -p MemoryMax=6G -p MemorySwapMax=0 .venv/bin/python oom_order_test.py
"""
import mmap
import os
import subprocess
import sys
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "evals"))
from arc_harness.executor import LocalExecutor  # noqa: E402

SERVER_SHM_GIB = float(os.environ.get("SERVER_SHM_GIB", "3.5"))
SERVER_ANON_GIB = float(os.environ.get("SERVER_ANON_GIB", "0.5"))
SANDBOXES = int(os.environ.get("SANDBOXES", "4"))
SANDBOX_GIB = float(os.environ.get("SANDBOX_GIB", "1.2"))

SERVER = r'''
import mmap, os, sys, time
shm_gib, anon_gib = float(sys.argv[1]), float(sys.argv[2])
fd = os.open("/dev/shm/e059_oom_server", os.O_CREAT | os.O_RDWR, 0o600)
os.ftruncate(fd, int(shm_gib * 2**30))
shm = mmap.mmap(fd, int(shm_gib * 2**30))
for i in range(0, len(shm), 4096):
    shm[i] = 1
anon = bytearray(int(anon_gib * 2**30))
for i in range(0, len(anon), 4096):
    anon[i] = 1
print("server ready", flush=True)
while True:
    time.sleep(1)
'''


def main():
    with open("/proc/self/cgroup") as f:
        print("cgroup:", f.read().strip())
    server = subprocess.Popen([sys.executable, "-c", SERVER, str(SERVER_SHM_GIB), str(SERVER_ANON_GIB)],
                              stdout=subprocess.PIPE, text=True)
    assert server.stdout.readline().strip() == "server ready"
    print(f"server pid {server.pid} holds {SERVER_SHM_GIB} GiB shared + {SERVER_ANON_GIB} GiB anon, oom_score_adj "
          f"{open(f'/proc/{server.pid}/oom_score_adj').read().strip()}")
    train = [{"input": [[0]], "output": [[1]]}]
    exs = [LocalExecutor(train, [[0]], 60, 16000, memory_mb=4096, python=sys.executable, style="ipython",
                         cpu_seconds=600) for _ in range(SANDBOXES)]
    for ex in exs:
        assert "42" in ex.execute("print(6 * 7)")["stdout"]
    print(f"{SANDBOXES} sandboxes up, oom_score_adj",
          [open(f"/proc/{ex.proc.pid}/oom_score_adj").read().strip() for ex in exs])
    results = [None] * SANDBOXES
    cell = (f"x = bytearray({int(SANDBOX_GIB * 2**30)})\nfor i in range(0, len(x), 4096):\n    x[i] = 1\n"
            "import time\ntime.sleep(5)\nprint('held')")

    def hog(i):
        results[i] = exs[i].execute(cell)

    threads = [threading.Thread(target=hog, args=(i,)) for i in range(SANDBOXES)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(120)
    died = [i for i, r in enumerate(results) if r and r.get("state_reset")]
    held = [i for i, r in enumerate(results) if r and "held" in (r.get("stdout") or "")]
    alive = server.poll() is None
    after = [("42" in (ex.execute("print(6 * 7)").get("stdout") or "")) for ex in exs]
    print(f"sandboxes that held their {SANDBOX_GIB} GiB: {held}; killed (session restarted): {died}")
    print(f"server alive: {alive}; every session works afterwards: {all(after)}")
    for ex in exs:
        ex.close()
    server.kill()
    os.unlink("/dev/shm/e059_oom_server")
    ok = alive and died and all(after)
    print("OOM ORDER TEST", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
