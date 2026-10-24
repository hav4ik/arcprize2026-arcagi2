#!/usr/bin/env python3
"""Read-only runtime inventory for a Kaggle or rented GPU notebook.

Uses the standard library. PyTorch is inspected only if already installed; no
packages are installed, no model is loaded, and no environment variables or
credentials are printed. Missing GPUs/tools are represented in the JSON report.
"""

import argparse
import csv
import datetime
import io
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys

sys.dont_write_bytecode = True


def read_text(path, limit=16384):
    try:
        with open(path, encoding="utf-8") as handle:
            return handle.read(limit).strip()
    except (OSError, UnicodeError):
        return None


def run(command, timeout):
    try:
        r = subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=False)
        return {"status": "ok" if r.returncode == 0 else "error", "returncode": r.returncode,
                "stdout": r.stdout[:20000].strip(), "stderr": r.stderr[:2000].strip()}
    except FileNotFoundError:
        return {"status": "unavailable", "reason": "command not installed"}
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "timeout_seconds": timeout}
    except OSError as error:
        return {"status": "error", "reason": str(error)}


def memory_info():
    values = {}
    for line in (read_text("/proc/meminfo") or "").splitlines():
        fields = line.split()
        if len(fields) >= 2 and fields[1].isdigit():
            values[fields[0].rstrip(":")] = int(fields[1]) * (1024 if fields[-1] == "kB" else 1)
    cgroups = {}
    paths = ["/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory.current", "/sys/fs/cgroup/memory.high",
             "/sys/fs/cgroup/cpu.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes",
             "/sys/fs/cgroup/memory/memory.usage_in_bytes"]
    for line in (read_text("/proc/self/cgroup") or "").splitlines():
        fields = line.split(":", 2)
        if len(fields) == 3 and fields[0] == "0":
            relative = fields[2].lstrip("/")
            if ".." not in Path(relative).parts:
                paths.extend(str(Path("/sys/fs/cgroup") / relative / name)
                             for name in ["memory.max", "memory.current", "memory.high", "cpu.max"])
    for path in dict.fromkeys(paths):
        value = read_text(path)
        if value is not None:
            cgroups[path] = int(value) if value.isdigit() else value
    finite_limits = [v for k, v in cgroups.items() if k.endswith(("memory.max", "memory.limit_in_bytes"))
                     and isinstance(v, int) and 0 < v < 2**60]
    if values.get("MemTotal"):
        finite_limits.append(values["MemTotal"])
    return {"host_total_bytes": values.get("MemTotal"), "host_available_bytes": values.get("MemAvailable"),
            "effective_limit_bytes_estimate": min(finite_limits) if finite_limits else None,
            "cgroup_values": cgroups,
            "note": "Host MemAvailable can exceed the container allowance; inspect cgroup current usage too."}


def gpu_info(timeout):
    columns = ["index", "name", "memory.total", "memory.free", "compute_cap", "driver_version", "pci.bus_id"]
    command = ["nvidia-smi", "--query-gpu=" + ",".join(columns), "--format=csv,noheader,nounits"]
    result = run(command, timeout)
    if result["status"] == "error":
        # Older nvidia-smi versions can lack the compute_cap query field.
        columns.remove("compute_cap")
        command[1] = "--query-gpu=" + ",".join(columns)
        fallback = run(command, timeout)
        if fallback["status"] == "ok":
            result = fallback
            result["note"] = "compute_cap query unavailable; use optional PyTorch capability report"
    if result["status"] == "ok":
        result["devices"] = [dict(zip(columns, [cell.strip() for cell in row]))
                             for row in csv.reader(io.StringIO(result.pop("stdout"))) if row]
        result["memory_unit"] = "MiB (nvidia-smi)"
    return {"inventory": result, "topology": run(["nvidia-smi", "topo", "-m"], timeout)}


def torch_info(timeout):
    code = '''import importlib.util, json, sys
sys.dont_write_bytecode = True
if importlib.util.find_spec("torch") is None:
    print(json.dumps({"status": "unavailable", "reason": "torch not installed"}))
else:
    import torch
    result = {"status": "ok", "version": torch.__version__, "built_cuda_version": torch.version.cuda,
              "cuda_available": torch.cuda.is_available(), "devices": []}
    if result["cuda_available"]:
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            result["devices"].append({"index": i, "name": p.name,
                "compute_capability": [p.major, p.minor], "total_memory_bytes": p.total_memory,
                "multiprocessor_count": p.multi_processor_count})
    print(json.dumps(result))
'''
    result = run([sys.executable, "-B", "-c", code], timeout)
    if result["status"] == "ok":
        try:
            parsed = json.loads(result["stdout"])
            if result["stderr"]:
                parsed["diagnostic_stderr"] = result["stderr"]
            return parsed
        except json.JSONDecodeError:
            result["status"] = "error"
            result["reason"] = "torch subprocess did not produce JSON"
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", nargs="+", default=[str(Path.cwd()), "/tmp", "/dev/shm"])
    parser.add_argument("--timeout-seconds", type=int, default=15)
    parser.add_argument("--skip-torch", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.timeout_seconds <= 60:
        parser.error("timeout-seconds must be between 1 and 60")
    disks = []
    for item in args.paths:
        path = str(Path(item).absolute())
        try:
            d = shutil.disk_usage(path)
            disks.append({"path": path, "total_bytes": d.total, "used_bytes": d.used, "free_bytes": d.free})
        except OSError as error:
            disks.append({"path": path, "error": str(error)})
    commands = ["python", "python3", "nvidia-smi", "nvcc", "gcc", "g++", "cmake", "ninja", "git", "pip", "vllm", "uv"]
    try:
        affinity = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        affinity = None
    result = {
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "platform": {"system": platform.system(), "release": platform.release(), "machine": platform.machine(),
                     "python_version": platform.python_version(), "python_executable": sys.executable},
        "cpu": {"logical_count": os.cpu_count(), "affinity_count": affinity},
        "memory": memory_info(), "disks": disks,
        "commands": {name: shutil.which(name) for name in commands},
        "gpu": gpu_info(args.timeout_seconds),
        "torch": {"status": "skipped"} if args.skip_torch else torch_info(args.timeout_seconds),
        "notes": ["No model weights were loaded and no benchmarks were run.",
                  "This report reads hardware/runtime metadata and does not dump environment variables.",
                  "Disk entries may describe the same backing filesystem; do not sum their free space."],
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
