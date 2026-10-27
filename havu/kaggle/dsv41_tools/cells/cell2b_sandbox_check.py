# Cell 2b (smoke, commit runs only): the tool sandbox at full load, before the solver starts. Starts as many sandbox
# sessions as the solver runs workers (each its own process with a 4 GiB address-space cap, 64 MB file cap, nice 10 and
# oom_score_adj 1000, like solver.session_factory), runs a typical ARC cell in every one, then checks that an oversized
# allocation fails inside its session, that a runaway cell is interrupted and the session restarts, and that closing
# every session returns host memory. Commit run: a failure RAISES ("SANDBOX CHECK FAILED"); the competition rerun skips
# this cell. Uses cell 0's KIT, MODE and RERUN, and cell 2's TOOL_PY.
import os
import sys
import threading
import time


def sandbox_meminfo_gib():
    out = {}
    with open("/proc/meminfo") as f:
        for line in f:
            key, value = line.split(":", 1)
            if key in ("MemAvailable", "AnonPages", "Shmem"):
                out[key] = int(value.split()[0]) / 2**20
    return out


def sandbox_pss_gib(pids):
    total = 0
    for pid in pids:
        try:
            with open(f"/proc/{pid}/smaps_rollup") as f:
                total += sum(int(l.split()[1]) for l in f if l.startswith("Pss:"))
        except OSError:
            pass
    return total / 2**20


def sandbox_check(kit, python, sessions, mem_mb=4096, timeout=30, alloc_gib=6):
    """Returns (report lines, failures)."""
    sys.path.insert(0, f"{kit}/repo/evals")
    from arc_harness.executor import LocalExecutor
    train = [{"input": [[0, 1, 0], [1, 1, 1]], "output": [[1, 0, 1], [0, 0, 0]]}]
    test = [[1, 0, 0], [0, 1, 1]]
    cell = ("import numpy as np\nfrom scipy import ndimage\ng = np.array(test_input)\n"
            "lab, n = ndimage.label(g)\nprint(n, (1 - g).tolist())")
    lines, failures, results, lock = [], [], [], threading.Lock()
    before = sandbox_meminfo_gib()
    exs = [LocalExecutor(train, test, timeout, 16000, memory_mb=mem_mb, python=python, style="ipython",
                         cpu_seconds=7200) for _ in range(sessions)]

    def run(ex):
        t0 = time.time()
        try:
            r = ex.execute(cell)
            ok = r.get("process_status") in (None, "ok", "completed") and "[[0, 1, 1]" in (r.get("stdout") or "")
            with lock:
                results.append((ok, time.time() - t0, (r.get("stderr") or "")[-200:]))
        except Exception as exc:
            with lock:
                results.append((False, time.time() - t0, f"{type(exc).__name__}: {exc}"))

    threads = [threading.Thread(target=run, args=(ex,)) for ex in exs]
    t0 = time.time()
    for th in threads:
        th.start()
    for th in threads:
        th.join(300)
    pids = [ex.proc.pid for ex in exs if getattr(ex, "proc", None)]
    loaded = sandbox_meminfo_gib()
    good = sum(1 for ok, _, _ in results if ok)
    slow = max((dt for _, dt, _ in results), default=0)
    lines.append(f"{good}/{sessions} sessions ran a numpy/scipy cell in {time.time() - t0:.0f} s (slowest {slow:.0f} s); "
                 f"sandbox PSS {sandbox_pss_gib(pids):.2f} GiB; host anon {before['AnonPages']:.1f} -> "
                 f"{loaded['AnonPages']:.1f} GiB, MemAvailable {before['MemAvailable']:.1f} -> {loaded['MemAvailable']:.1f} GiB")
    if good < sessions:
        failures.append(f"{sessions - good} sessions failed: {[e for ok, _, e in results if not ok][:2]}")

    # an allocation over the session's cap must fail inside the session, which stays usable
    ex = exs[0]
    r = ex.execute(f"x = bytearray({alloc_gib} * 2**30)\nprint('allocated')")
    text = (r.get("stdout") or "") + (r.get("stderr") or "")
    contained = "allocated" not in text and ("MemoryError" in text or r.get("state_reset"))
    after = ex.execute("print(6 * 7)")
    lines.append(f"{alloc_gib} GiB allocation in one session: {'blocked' if contained else 'NOT blocked'} "
                 f"({text.strip().splitlines()[-1][:90] if text.strip() else r.get('process_status')}); session after: "
                 f"{(after.get('stdout') or '').strip() or after.get('process_status')}")
    if not contained or "42" not in (after.get("stdout") or ""):
        failures.append("memory cap did not contain the allocation or the session did not recover")

    # a runaway cell is interrupted at the timeout and the session keeps working
    ex = exs[1]
    t1 = time.time()
    r = ex.execute("while True:\n    pass")
    after = ex.execute("print(6 * 7)")
    lines.append(f"runaway cell: {r.get('process_status')} after {time.time() - t1:.0f} s; session after: "
                 f"{(after.get('stdout') or '').strip() or after.get('process_status')}")
    if "42" not in (after.get("stdout") or ""):
        failures.append("session did not recover after a timeout")

    for ex in exs:
        ex.close()
    time.sleep(3)
    alive = [p for p in pids if os.path.exists(f"/proc/{p}")]
    end = sandbox_meminfo_gib()
    lines.append(f"closed: {len(alive)} sandbox processes left; host anon {end['AnonPages']:.1f} GiB, "
                 f"MemAvailable {end['MemAvailable']:.1f} GiB")
    if alive:
        failures.append(f"{len(alive)} sandbox processes outlived close()")
    if end["AnonPages"] > before["AnonPages"] + 1.0:
        failures.append(f"host anon memory not returned ({before['AnonPages']:.1f} -> {end['AnonPages']:.1f} GiB)")
    return lines, failures


if not RERUN and MODE != "fallback":
    try:
        _sb_lines, _sb_fail = sandbox_check(KIT, TOOL_PY, int(os.environ.get("ARC_SMOKE_SANDBOXES", "56")),
                                            int(os.environ.get("ARC_EXEC_MEM_MB", "4096")),
                                            int(os.environ.get("ARC_SMOKE_SANDBOX_TIMEOUT", "30")))
    except Exception as exc:
        _sb_lines, _sb_fail = [], [f"{type(exc).__name__}: {exc}"]
    for _l in _sb_lines:
        print("sandbox check:", _l, flush=True)
    if _sb_fail:
        print("=" * 100, f"SANDBOX CHECK FAILED: {_sb_fail}", "=" * 100, sep="\n", flush=True)
        raise RuntimeError(f"SANDBOX CHECK FAILED: {_sb_fail}")
    print("SANDBOX CHECK OK", flush=True)
