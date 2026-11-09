"""Test helpers: synthetic /proc trees, cgroup roots, a fake nvidia-smi, sampler records and decoy processes.
No GPU, no vLLM, no network."""
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import time
import unittest

KIT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIX = os.path.join(KIT, "tests", "fixtures")
if KIT not in sys.path:
    sys.path.insert(0, KIT)

GIB = 1024 ** 3


class TmpCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="reli_test_")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def path(self, *parts):
        p = os.path.join(self.tmp, *parts)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        return p


def fixture(name):
    return os.path.join(FIX, name)


def read(path):
    with open(path, errors="replace") as f:
        return f.read()


def make_proc(root, pid, argv, ppid=1, rss_gib=0.1, anon_gib=0.08, shmem_gib=0.01, uid=None, state="S", pad=0):
    """One /proc/<pid> with cmdline (NUL-separated, optionally NUL-padded like setproctitle), stat and status."""
    d = os.path.join(root, str(pid))
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "cmdline"), "wb") as f:
        f.write(b"\0".join(a.encode() for a in argv) + b"\0" + b"\0" * pad)
    comm = os.path.basename(argv[0])[:15]
    with open(os.path.join(d, "stat"), "w") as f:
        f.write("%d (%s) %s %d %d %d 0 -1 0 0 0 0 0 0 0 0 20 0 1 0 0 0 0\n" % (pid, comm, state, ppid, pid, pid))
    kb = lambda g: int(g * 1024 * 1024)
    uid = os.getuid() if uid is None else uid
    with open(os.path.join(d, "status"), "w") as f:
        f.write("Name:\t%s\nState:\t%s\nPPid:\t%d\nUid:\t%d\t%d\t%d\t%d\nVmRSS:\t%d kB\nRssAnon:\t%d kB\n"
                "RssFile:\t0 kB\nRssShmem:\t%d kB\n" % (comm, state, ppid, uid, uid, uid, uid, kb(rss_gib),
                                                        kb(anon_gib), kb(shmem_gib)))


def make_server_tree(root, api=19687, engine=20496, workers=(20827, 20829, 20830, 20828), dcp=False, parent=500):
    """API server -> EngineCore -> 4 TP workers (titles NUL-padded), as vLLM V1 runs them."""
    make_proc(root, api, ["/root/kenv/bin/python", "-m", "vllm.entrypoints.cli.main", "serve", "/kaggle/input/m"],
              ppid=parent, rss_gib=1.4, anon_gib=0.9)
    make_proc(root, engine, ["VLLM::EngineCore"], ppid=api, rss_gib=4.0, anon_gib=3.8, pad=40)
    for r, pid in enumerate(workers):
        title = "VLLM::Worker_TP%d_DCP%d" % (r, r) if dcp else "VLLM::Worker_TP%d" % r
        make_proc(root, pid, [title], ppid=engine, rss_gib=30.0 + r, anon_gib=28.0 + r, shmem_gib=2.0, pad=60)


def write_meminfo(root, src=None):
    os.makedirs(root, exist_ok=True)
    shutil.copy(src or fixture("meminfo_hostwide"), os.path.join(root, "meminfo"))


def write_self_cgroup(proc_root, text):
    os.makedirs(os.path.join(proc_root, "self"), exist_ok=True)
    with open(os.path.join(proc_root, "self", "cgroup"), "w") as f:
        f.write(text)


def cgroup_v1(root, limit="9223372036854771712"):
    os.makedirs(os.path.join(root, "memory"), exist_ok=True)
    shutil.copy(fixture("cgroup_v1_memory.stat"), os.path.join(root, "memory", "memory.stat"))
    with open(os.path.join(root, "memory", "memory.limit_in_bytes"), "w") as f:
        f.write(limit + "\n")


def cgroup_v2(root, sub="", memory_max="max"):
    d = os.path.join(root, sub) if sub else root
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(root, "cgroup.controllers"), "w") as f:
        f.write("cpuset cpu io memory pids\n")
    shutil.copy(fixture("cgroup_v2_memory.stat"), os.path.join(d, "memory.stat"))
    with open(os.path.join(d, "memory.max"), "w") as f:
        f.write(memory_max + "\n")


def fake_nvidia_smi(dirpath, gpus_csv=None, apps_csv=None):
    """An executable that answers the two queries the kit makes."""
    os.makedirs(dirpath, exist_ok=True)
    g = os.path.join(dirpath, "gpus.csv")
    a = os.path.join(dirpath, "apps.csv")
    shutil.copy(gpus_csv or fixture("nvidia_smi_gpus.csv"), g)
    if apps_csv is None:
        shutil.copy(fixture("nvidia_smi_apps.csv"), a)
    else:
        with open(a, "w") as f:
            f.write(apps_csv)
    p = os.path.join(dirpath, "nvidia-smi")
    with open(p, "w") as f:
        f.write('#!/bin/sh\ncase "$1" in\n  --query-gpu=*) cat "%s" ;;\n  --query-compute-apps=*) cat "%s" ;;\n'
                "esac\n" % (g, a))
    os.chmod(p, os.stat(p).st_mode | stat.S_IEXEC)
    return p


def record(t, keq=140.0, cmp=1.5, cmr=1.0, peak=19.5, reserved=20.0, retries=0, ooms=0, engines=1, key="log#0",
           died=0, up=1, requeue=0, retry=0, stop_error=0, errors=0, records=0, t0=None, sup=None, nb_end=None,
           gen=600.0, startup_t=None, src="cgroup_v1", fast=False, pids=(19687, 20496, 20827)):
    """A synthetic sampler.py record."""
    ranks = {"TP%d" % i: {"alloc": 18.9, "peak": peak, "reserved": reserved, "free": 1.2, "retries": retries,
                          "ooms": ooms, "cap": peak + cmp, "cap_minus_peak": cmp, "cap_minus_reserved": cmr,
                          "cap_minus_reserved0": 0.7} for i in range(4)}
    return {"kind": "sample", "t": t, "utc": time.strftime("%H:%M:%S", time.gmtime(t)), "kaggle_eq_gib": keq,
            "kaggle_eq_source": src, "kaggle_eq_meminfo_gib": keq + 11.3,
            "cgroup": {"version": 1, "anon_gib": keq - 14, "shmem_gib": 7.0},
            "groups": {"tp_workers": {"n": 4, "rss_gib": 127.8}, "engine_core": {"n": engines, "rss_gib": 4.0}},
            "sandboxes": {"n": 30, "total_gib": 3.0, "max_gib": 0.3}, "dev_shm_used_gib": 8.0, "engines": engines,
            "vllm_procs": [[p, "VLLM::EngineCore" if i == 1 else "x", 1.0, 0.0] for i, p in enumerate(pids)],
            "gpus": [{"index": i, "used_mib": 21600, "free_mib": 1400} for i in range(4)], "gpu_apps": [],
            "vllm": {"server_key": key, "ranks": ranks, "counts": {"oom": 0, "dead": 0, "traceback": 0},
                     "worst": {"cap_minus_peak": cmp, "cap_minus_reserved": cmr, "peak": peak, "reserved": reserved,
                               "retries": retries, "ooms": ooms}, "kv_tokens": 1453240, "startup_t": startup_t},
            "solver": {"counts": {"server_died": died, "server_up": up, "requeue": requeue, "retry": retry,
                                  "stop_error": stop_error, "timeout": 0, "traceback": 0},
                       "t0": t0, "metrics": {"gen_tok_s": gen, "kv": 0.9}},
            "attempts": {"records": records, "error_records": errors}, "supervisor": sup or {"state": "up", "start": 0},
            "notebook_end": nb_end, "fast": fast}


def write_records(path, recs):
    with open(path, "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")


def spawn_decoy(title):
    """A real process whose cmdline is exactly `title` (one token): /bin/sh with argv[0] = title, reading stdin."""
    return subprocess.Popen([title], executable="/bin/sh", stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL)


def proc_state(pid):
    with open("/proc/%d/stat" % pid) as f:
        return f.read().rpartition(")")[2].split()[0]
