"""E081 Kaggle 4xL4 machine survey: one probe per commit notebook. Writes /kaggle/working/probe.json.

1. identity: hostname, CPU, RAM, GPU uuid/serial/bus/power limit, nvidia-smi topo (tells repeated machines apart);
2. Manh Nguyen's gpucheck.py (verbatim, from /tmp/gpucheck.py): pinned H2D/D2H per GPU + NCCL latency per GPU pair;
3. read speed of each mounted input under /kaggle/input: cold parallel (8 threads) and cold single-stream reads of
   files not yet touched, time-boxed, then a warm re-read of the same bytes (page cache);
4. write/read speed of /kaggle/working, /tmp and /dev/shm (2 GiB with fsync).
Everything is time-boxed so a probe costs a few GPU-minutes; nothing is installed and no network is used.
"""
import json
import os
import re
import socket
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor

OUT = "/kaggle/working/probe.json"
CHUNK = 8 << 20
R = {"t_start": time.time()}


def sh(cmd, timeout=120):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout).stdout
    except Exception as exc:
        return f"ERR {type(exc).__name__}"


def identity():
    q = ("index,name,uuid,serial,pci.bus_id,driver_version,power.limit,power.max_limit,clocks.max.sm,"
         "pcie.link.gen.max,pcie.link.width.max,memory.total")
    rows = [dict(zip(q.split(","), (x.strip() for x in line.split(","))))
            for line in sh(f"nvidia-smi --query-gpu={q} --format=csv,noheader").strip().splitlines()]
    cpu = re.search(r"Model name:\s*(.+)", sh("lscpu"))
    R["identity"] = {"hostname": socket.gethostname(), "cpu": cpu.group(1).strip() if cpu else None,
                     "nproc": os.cpu_count(), "mem_total_gb": round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9, 1),
                     "kernel": sh("uname -r").strip(), "gpus": rows, "topo": sh("nvidia-smi topo -m"),
                     "boot_id": open("/proc/sys/kernel/random/boot_id").read().strip()}
    print(json.dumps({k: v for k, v in R["identity"].items() if k != "topo"}, indent=1))
    print(R["identity"]["topo"])


def gpucheck():
    t = time.time()
    out = sh("python3 -u /tmp/gpucheck.py 2>&1 | grep -v FutureWarning | grep -v 'return func'", timeout=900)
    print(out)
    copy = {int(m[0]): {"h2d": float(m[1]), "d2h": float(m[2])}
            for m in re.findall(r"gpu(\d+)\s+H2D\s+([\d.]+) GB/s\s+D2H\s+([\d.]+) GB/s", out)}
    pairs = []
    for line in out.splitlines():
        m = re.match(r"(alone|together)\s+(\d)-(\d)\s+(.*)", line)
        if m:
            vals = dict((k, float(v)) for k, v in re.findall(r"(\w+K)\s+([\d.]+)", m.group(4)))
            pairs.append({"mode": m.group(1), "pair": f"{m.group(2)}-{m.group(3)}", **vals})
    R["gpucheck"] = {"copy": copy, "pairs": pairs, "seconds": round(time.time() - t, 1), "raw": out[-4000:]}


def files_under(root):
    out = []
    for d, _, names in os.walk(root, followlinks=True):
        for n in names:
            p = os.path.join(d, n)
            try:
                out.append((p, os.path.getsize(p)))
            except OSError:
                pass
    return out


def read_range(path, nbytes, stop_at):
    got = 0
    with open(path, "rb", buffering=0) as f:
        while got < nbytes and time.time() < stop_at:
            b = f.read(min(CHUNK, nbytes - got))
            if not b:
                break
            got += len(b)
    return got


# Exactly what Manh's V12 (LB 76.25) mounts; walking all of /kaggle/input can take ~5 min, so only these roots.
ROOTS = ["/kaggle/input/models/chankhavu/dsv41-flash-1017-arc-tool-e128-2bit/pytorch/default/1",
         "/kaggle/input/datasets/nguyennguyen599/vllm-wheels",
         "/kaggle/input/datasets/nguyennguyen599/cu13-install"]


def mounts():
    res = {}
    for root in ROOTS:
        if not os.path.isdir(root):
            res[root] = {"missing": True}
            print(root, "MISSING", flush=True)
            continue
        t_list = time.time()
        fs = sorted(files_under(root), key=lambda x: -x[1])
        total = sum(s for _, s in fs)
        big = [x for x in fs if x[1] >= 256 << 20]
        r = {"files": len(fs), "total_gb": round(total / 1e9, 2), "big_files": len(big),
             "list_s": round(time.time() - t_list, 1)}
        if big:
            # cold single-stream: the first 2 GiB of the largest file, max 30 s
            p0, s0 = big[0]
            t = time.time(); n = read_range(p0, min(s0, 2 << 30), t + 30); dt = time.time() - t
            r["cold_single_MBps"] = round(n / dt / 1e6, 1); r["cold_single_gb"] = round(n / 1e9, 2)
            # cold parallel: 8 threads over the next files (untouched), up to 1 GiB each, max 45 s total
            todo = big[1:17] or big[:1]
            stop = time.time() + 45
            t = time.time()
            with ThreadPoolExecutor(8) as ex:
                n = sum(ex.map(lambda x: read_range(x[0], min(x[1], 1 << 30), stop), todo))
            dt = time.time() - t
            r["cold_par8_MBps"] = round(n / dt / 1e6, 1); r["cold_par8_gb"] = round(n / 1e9, 2)
            # warm re-read of the first file's bytes (page cache)
            t = time.time(); n = read_range(p0, min(s0, 2 << 30), t + 20); dt = time.time() - t
            r["warm_single_MBps"] = round(n / dt / 1e6, 1)
        else:  # small-file dataset: read everything, max 30 s
            stop = time.time() + 30; t = time.time()
            with ThreadPoolExecutor(8) as ex:
                n = sum(ex.map(lambda x: read_range(x[0], x[1], stop), fs))
            dt = time.time() - t
            r["cold_par8_MBps"] = round(n / max(dt, 1e-6) / 1e6, 1); r["cold_par8_gb"] = round(n / 1e9, 2)
        res[root] = r
        print(root, r, flush=True)
    R["mounts"] = res


def disks():
    res = {}
    for d in ("/kaggle/working", "/tmp", "/dev/shm"):
        p = os.path.join(d, "e081_probe.bin")
        try:
            t = time.time()
            out = sh(f"dd if=/dev/zero of={p} bs=8M count=256 conv=fsync 2>&1 | tail -1", timeout=300)
            w = 2 * 1024 / (time.time() - t)
            sh("sync; echo 3 > /proc/sys/vm/drop_caches 2>/dev/null")
            t = time.time(); n = read_range(p, 2 << 30, time.time() + 60); rr = n / (time.time() - t) / 1e6
            res[d] = {"write_MBps": round(w * 1.048576, 1), "read_MBps": round(rr, 1), "df": sh(f"df -h {d} | tail -1").strip()}
        except Exception as exc:
            res[d] = {"error": f"{type(exc).__name__}: {exc}"}
        finally:
            try:
                os.remove(p)
            except OSError:
                pass
        print(d, res[d], flush=True)
    R["disks"] = res


for step in (identity, gpucheck, mounts, disks):
    t = time.time()
    try:
        step()
    except Exception as exc:
        R[f"{step.__name__}_error"] = f"{type(exc).__name__}: {exc}"
        print("STEP FAILED", step.__name__, exc, flush=True)
    R[f"{step.__name__}_s"] = round(time.time() - t, 1)
    json.dump(R, open(OUT, "w"), indent=1)
R["t_total_s"] = round(time.time() - R["t_start"], 1)
json.dump(R, open(OUT, "w"), indent=1)
print("PROBE_DONE", R["t_total_s"], "s")
