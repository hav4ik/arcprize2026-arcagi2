"""Shared, read-only parsers and trend maths for the Kaggle reliability kit (stdlib only, Python >= 3.8).

Sources: host memory (/proc/meminfo is HOST-wide on pods, so the container's cgroup is the Kaggle-relevant number),
processes (/proc/<pid>), nvidia-smi, vLLM server logs (per-rank "GPU memory:" and "Capped the PyTorch allocator"
lines), the solver's log / attempts / supervisor status files, and the notebook runner's timeline.

All clocks are UTC. vLLM and the solver stamp their lines with UTC wall time on Kaggle and on our pods."""
from __future__ import annotations

import calendar
import glob
import json
import os
import re
import subprocess
import time

GIB = 1024 ** 3
SCHEMA = 1

# Kaggle's 4xL4 competition sessions: cgroup v2 memory.max = 173.0 GiB, no swap. The +7 GiB covers what Kaggle's
# container holds besides our processes (measured on Kaggle commit smokes against the same notebook on pods).
KAGGLE_RAM_LIMIT_GIB = 173.0
KAGGLE_OVERHEAD_GIB = 7.0
# cap - peak below this was where the B32-M memory audit put GPU OOMs (allocation spikes of ~0.25 GiB under load).
CAP_PEAK_LIMIT_GIB = 0.26

ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")


def utc(t):
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t))


def hms(t):
    return time.strftime("%H:%M:%S", time.gmtime(t))


# ----------------------------------------------------------------------------------------------- host memory
MEMINFO_KEYS = ("MemTotal", "MemAvailable", "AnonPages", "Shmem", "Cached", "Mlocked", "PageTables", "AnonHugePages")


def read_meminfo(proc_root="/proc"):
    """/proc/meminfo fields in bytes. HOST-wide on pods (other tenants' anon memory included): never compare it with
    Kaggle's limit directly."""
    out = {}
    try:
        with open(os.path.join(proc_root, "meminfo")) as f:
            for line in f:
                key, _, rest = line.partition(":")
                parts = rest.split()
                if key in MEMINFO_KEYS and parts:
                    out[key] = int(parts[0]) * 1024
    except (OSError, ValueError):
        pass
    return out


def _read_kv(path):
    d = {}
    try:
        with open(path) as f:
            for line in f:
                parts = line.split()
                if len(parts) == 2:
                    try:
                        d[parts[0]] = int(parts[1])
                    except ValueError:
                        pass
    except OSError:
        return None
    return d


def _read_limit(path):
    try:
        with open(path) as f:
            s = f.read().strip()
    except OSError:
        return None
    if s in ("", "max"):
        return None
    try:
        v = int(s)
    except ValueError:
        return None
    return None if v >= 1 << 60 else v  # v1 "unlimited" is 9223372036854771712


def _self_cgroup(proc_root):
    """{'v1': path of the memory controller, 'v2': unified path} from /proc/self/cgroup."""
    rel = {}
    try:
        with open(os.path.join(proc_root, "self", "cgroup")) as f:
            for line in f:
                parts = line.rstrip("\n").split(":", 2)
                if len(parts) != 3:
                    continue
                hid, ctrls, path = parts
                if hid == "0" and ctrls == "":
                    rel["v2"] = path
                elif "memory" in ctrls.split(","):
                    rel["v1"] = path
    except OSError:
        pass
    return rel


def read_cgroup_memory(cgroup_root="/sys/fs/cgroup", proc_root="/proc"):
    """The container's own memory accounting, as Kaggle's cgroup limit sees it.

    v1 (Runpod / Vast pods): memory/memory.stat total_rss + total_shmem (hierarchical totals; rss / shmem if the
    totals are missing). v2 (Kaggle, most new hosts): memory.stat anon + shmem (v2's anon excludes shmem).
    The memory controller is looked up on v1 first, since hybrid hosts keep it there.
    Returns {"version": 1 | 2, "anon", "shmem", "limit" (bytes or None), "path"} or {"version": None}."""
    rel = _self_cgroup(proc_root)
    v1 = []
    if rel.get("v1") not in (None, "", "/"):
        v1.append(os.path.join(cgroup_root, "memory", rel["v1"].lstrip("/")))
    v1.append(os.path.join(cgroup_root, "memory"))
    for d in v1:
        st = _read_kv(os.path.join(d, "memory.stat"))
        if not st:
            continue
        anon = st.get("total_rss", st.get("rss"))
        shmem = st.get("total_shmem", st.get("shmem"))
        if anon is None or shmem is None:
            continue
        return {"version": 1, "anon": anon, "shmem": shmem, "path": d,
                "limit": _read_limit(os.path.join(d, "memory.limit_in_bytes"))}
    if os.path.exists(os.path.join(cgroup_root, "cgroup.controllers")):
        v2 = []
        if rel.get("v2") not in (None, "", "/"):
            v2.append(os.path.join(cgroup_root, rel["v2"].lstrip("/")))
        v2.append(cgroup_root)
        for d in v2:
            st = _read_kv(os.path.join(d, "memory.stat"))
            if st and "anon" in st and "shmem" in st:
                return {"version": 2, "anon": st["anon"], "shmem": st["shmem"], "path": d,
                        "limit": _read_limit(os.path.join(d, "memory.max"))}
    return {"version": None}


def kaggle_equivalent(cg, meminfo, overhead_gib=KAGGLE_OVERHEAD_GIB):
    """(GiB, source): the container's anon + shmem + Kaggle's overhead, to compare with Kaggle's 173 GiB.
    Falls back to the host-wide /proc/meminfo form only when no cgroup accounting is readable (flagged by source)."""
    if cg and cg.get("version"):
        return (cg["anon"] + cg["shmem"]) / GIB + overhead_gib, "cgroup_v%d" % cg["version"]
    if meminfo and "AnonPages" in meminfo:
        return (meminfo["AnonPages"] + meminfo.get("Shmem", 0)) / GIB + overhead_gib, "meminfo_hostwide"
    return None, None


def dev_shm_used(path="/dev/shm"):
    try:
        sv = os.statvfs(path)
    except OSError:
        return None
    return (sv.f_blocks - sv.f_bfree) * sv.f_frsize


# ------------------------------------------------------------------------------------------------- processes
def read_argv(pid, proc_root="/proc"):
    """argv tokens of a process. Titles set with setproctitle ("VLLM::Worker_TP3") are padded with NULs, so empty
    tokens are dropped."""
    with open(os.path.join(proc_root, str(pid), "cmdline"), "rb") as f:
        raw = f.read()
    return [t.decode(errors="replace") for t in raw.split(b"\0") if t]


def exact_title(argv):
    """The process title when the cmdline is exactly one token (setproctitle style), else None."""
    return argv[0] if len(argv) == 1 else None


def scan_procs(proc_root="/proc"):
    """{pid: {"ppid", "comm", "argv", "cmd", "rss", "anon", "shmem" (bytes), "uid", "state"}} for every readable
    process. Kernel threads (empty cmdline) are skipped."""
    out = {}
    try:
        names = os.listdir(proc_root)
    except OSError:
        return out
    for name in names:
        if not name.isdigit():
            continue
        pid = int(name)
        base = os.path.join(proc_root, name)
        try:
            argv = read_argv(pid, proc_root)
            if not argv:
                continue
            with open(os.path.join(base, "stat")) as f:
                stat = f.read()
            head, _, tail = stat.rpartition(")")
            fields = tail.split()
            state, ppid = fields[0], int(fields[1])
            comm = head.partition("(")[2]
            st = {}
            with open(os.path.join(base, "status")) as f:
                for line in f:
                    k, _, v = line.partition(":")
                    if k in ("VmRSS", "RssAnon", "RssShmem"):
                        st[k] = int(v.split()[0]) * 1024
                    elif k == "Uid":
                        st["uid"] = int(v.split()[0])
        except (OSError, ValueError, IndexError):
            continue
        out[pid] = {"ppid": ppid, "comm": comm, "argv": argv, "cmd": " ".join(argv), "state": state,
                    "rss": st.get("VmRSS", 0), "anon": st.get("RssAnon", 0), "shmem": st.get("RssShmem", 0),
                    "uid": st.get("uid")}
    return out


def children_map(procs):
    kids = {}
    for pid, p in procs.items():
        kids.setdefault(p["ppid"], []).append(pid)
    return kids


def descendants(procs, root, kids=None):
    """All descendants of root (root excluded)."""
    kids = kids if kids is not None else children_map(procs)
    out, todo = [], list(kids.get(root, []))
    while todo:
        p = todo.pop()
        out.append(p)
        todo.extend(kids.get(p, []))
    return out


def is_vllm_worker(cmd):
    return cmd.startswith("VLLM::Worker")


def is_engine_core(cmd):
    return cmd.startswith("VLLM::EngineCore")


def is_api_server(argv):
    cmd = " ".join(argv)
    if cmd.startswith("VLLM::APIServer"):
        return True
    if "vllm.entrypoints" in cmd and ("serve" in argv or "api_server" in cmd):
        return True
    return any(os.path.basename(a) == "vllm" for a in argv[:2]) and "serve" in argv


GROUPS = ("tp_workers", "engine_core", "api_server", "vllm_other", "supervisor", "solver", "sandboxes", "kernel",
          "kit", "other")
KIT_SCRIPTS = ("sampler.py", "watcher.py", "gpu_hold.py", "fault_inject.py", "run_notebook.py")


def classify(procs, solver_pat="solver_dsv41.py", supervisor_pat="dsv41_vllm.py"):
    """{pid: group}. vLLM processes first (B32's solver launches its own server, so the server is a descendant of
    the solver); tool sandboxes are the solver's descendants outside the vLLM tree."""
    kids = children_map(procs)
    vllm_roots = [p for p, v in procs.items()
                  if is_api_server(v["argv"]) or is_engine_core(v["cmd"]) or is_vllm_worker(v["cmd"])]
    vllm_tree = set(vllm_roots)
    for r in vllm_roots:
        vllm_tree.update(descendants(procs, r, kids))
    solvers = [p for p, v in procs.items() if solver_pat and solver_pat in v["cmd"] and p not in vllm_tree]
    sand = set()
    for s in solvers:
        sand.update(x for x in descendants(procs, s, kids) if x not in vllm_tree)
    out = {}
    for pid, v in procs.items():
        cmd = v["cmd"]
        if is_vllm_worker(cmd):
            g = "tp_workers"
        elif is_engine_core(cmd):
            g = "engine_core"
        elif is_api_server(v["argv"]):
            g = "api_server"
        elif pid in vllm_tree:
            g = "vllm_other"
        elif supervisor_pat and supervisor_pat in cmd:
            g = "supervisor"
        elif pid in solvers:
            g = "solver"
        elif pid in sand or "import ast, contextlib" in cmd:
            g = "sandboxes"
        elif "ipykernel_launcher" in cmd:
            g = "kernel"
        elif any(("reliability/" + s) in cmd or cmd.endswith(s) for s in KIT_SCRIPTS):
            g = "kit"
        else:
            g = "other"
        out[pid] = g
    return out


def group_totals(procs, groups):
    tot = {g: {"n": 0, "rss_gib": 0.0, "anon_gib": 0.0, "shmem_gib": 0.0} for g in GROUPS}
    for pid, g in groups.items():
        p = procs[pid]
        t = tot[g]
        t["n"] += 1
        t["rss_gib"] += p["rss"] / GIB
        t["anon_gib"] += p["anon"] / GIB
        t["shmem_gib"] += p["shmem"] / GIB
    for t in tot.values():
        for k in ("rss_gib", "anon_gib", "shmem_gib"):
            t[k] = round(t[k], 3)
    return tot


# ------------------------------------------------------------------------------------------------------- GPU
GPU_FIELDS = ("index", "uuid", "memory.used", "memory.free", "memory.total", "power.draw", "clocks.sm",
              "temperature.gpu")


def _num(s):
    s = s.strip()
    try:
        return float(s)
    except ValueError:
        return None  # "[N/A]", "[No data]", "[Not Supported]"


def parse_gpu_csv(text):
    out = []
    for line in text.strip().splitlines():
        f = [x.strip() for x in line.split(",")]
        if len(f) < len(GPU_FIELDS) or not f[0].isdigit():
            continue
        out.append({"index": int(f[0]), "uuid": f[1], "used_mib": _num(f[2]), "free_mib": _num(f[3]),
                    "total_mib": _num(f[4]), "power_w": _num(f[5]), "sm_mhz": _num(f[6]), "temp_c": _num(f[7])})
    return out


def parse_apps_csv(text, gpus=()):
    """[[pid, gpu index or None, used MiB or None]]. A process that just exited shows "[No data]" for a while."""
    idx = {g["uuid"]: g["index"] for g in gpus}
    out = []
    for line in text.strip().splitlines():
        f = [x.strip() for x in line.split(",")]
        if len(f) < 3 or not f[0].isdigit():
            continue
        out.append([int(f[0]), idx.get(f[1]), _num(f[2])])
    return out


def query_gpus(nvidia_smi="nvidia-smi", timeout=30):
    """(gpus, apps, error). Never raises: a missing or hung nvidia-smi is recorded, not fatal."""
    try:
        q = subprocess.run([nvidia_smi, "--query-gpu=" + ",".join(GPU_FIELDS), "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=timeout)
        gpus = parse_gpu_csv(q.stdout)
        a = subprocess.run([nvidia_smi, "--query-compute-apps=pid,gpu_uuid,used_memory",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=timeout)
        return gpus, parse_apps_csv(a.stdout, gpus), (None if q.returncode == 0 else q.stderr.strip()[:200])
    except (OSError, subprocess.SubprocessError) as exc:
        return [], [], "%s: %s" % (type(exc).__name__, exc)


# ------------------------------------------------------------------------------------------- incremental logs
class LogFollower:
    """Yields only the lines appended since the last call, per file (ANSI codes stripped). A file that shrank was
    rotated or truncated: it is re-read from the start and the caller is told (reset=True)."""

    CHUNK = 16 << 20

    def __init__(self):
        self.pos = {}
        self.partial = {}

    def read(self, path):
        """(lines, reset)."""
        try:
            size = os.path.getsize(path)
        except OSError:
            return [], False
        pos = self.pos.get(path, 0)
        reset = size < pos
        if reset:
            pos = 0
            self.partial[path] = b""
        lines = []
        try:
            with open(path, "rb") as f:
                f.seek(pos)
                while True:
                    data = f.read(self.CHUNK)
                    if not data:
                        break
                    pos += len(data)
                    parts = (self.partial.get(path, b"") + data).split(b"\n")
                    self.partial[path] = parts.pop()
                    lines.extend(ANSI.sub("", p.decode(errors="replace")).rstrip("\r") for p in parts)
        except OSError:
            return lines, reset
        self.pos[path] = pos
        return lines, reset


def expand_globs(patterns):
    seen, out = set(), []
    for pat in patterns:
        for p in sorted(glob.glob(pat, recursive=True)):
            if p not in seen and os.path.isfile(p):
                seen.add(p)
                out.append(p)
    return out


# --------------------------------------------------------------------------------------------- vLLM server log
WHO = re.compile(r"\(([A-Za-z][\w]*?) pid=(\d+)\)")
LOGTIME = re.compile(r"\b(?:DEBUG|INFO|WARNING|ERROR|CRITICAL) (\d\d)-(\d\d) (\d\d):(\d\d):(\d\d)\b")
GPU_MEM = re.compile(r"GPU memory: ([\d.]+) GiB allocated \(peak ([\d.]+) GiB\), ([\d.]+) GiB reserved"
                     r"(?: \(([\d.]+) GiB in split segments\))?, ([\d.]+) GiB free, (\d+) allocator retries, (\d+) OOMs")
CAP = re.compile(r"Capped the PyTorch allocator at ([\d.]+) GiB \(([\d.]+) GiB reserved\)")
KV_SIZE = re.compile(r"GPU KV cache size: ([\d,]+) tokens")
OOM_LINE = re.compile(r"CUDA out of memory|OutOfMemoryError|CUDA error: out of memory")
DEAD_LINE = re.compile(r"EngineDeadError|Worker proc \S+ died|Engine core proc \S+ died|EngineCore encountered an "
                       r"issue|WorkerProc hit an exception|RPC call to execute_model timed out")
TRACEBACK = "Traceback (most recent call last)"


def rank_key(name):
    m = re.search(r"TP(\d+)", name)
    return "TP%s" % m.group(1) if m else name


def log_epoch(line, ref_epoch):
    """Epoch of a vLLM "INFO MM-DD HH:MM:SS" stamp (no year: the year that puts it at or before ref + 1 day)."""
    m = LOGTIME.search(line)
    if not m:
        return None
    mo, d, H, M, S = (int(x) for x in m.groups())
    year = time.gmtime(ref_epoch).tm_year
    try:
        t = calendar.timegm((year, mo, d, H, M, S))
        if t > ref_epoch + 86400:
            t = calendar.timegm((year - 1, mo, d, H, M, S))
    except (ValueError, OverflowError):
        return None
    return t


class VllmLog:
    """State of one server log: per-rank allocator cap (startup) and newest "GPU memory:" line, KV pool, startup,
    and error-line counters. A cap line from a new pid for a rank means a new server instance in the same file."""

    MAX_EVENTS = 60

    def __init__(self, path):
        self.path = path
        self.instance = 0
        self.reset()

    def reset(self):
        self.caps, self.mem = {}, {}
        self.kv_tokens = None
        self.startup_t = None
        self.counts = {"oom": 0, "dead": 0, "traceback": 0}
        self.events = []
        self.last_t = None

    def feed(self, line, ref_epoch):
        t = log_epoch(line, ref_epoch)
        if t is not None:
            self.last_t = t
        who = WHO.search(line)
        name, pid = (who.group(1), int(who.group(2))) if who else ("?", None)
        m = GPU_MEM.search(line)
        if m:
            a, pk, rs, sp, fr, rt, oo = m.groups()
            self.mem[rank_key(name)] = {"alloc": float(a), "peak": float(pk), "reserved": float(rs),
                                        "split": float(sp) if sp else None, "free": float(fr), "retries": int(rt),
                                        "ooms": int(oo), "pid": pid, "t": t}
            return
        c = CAP.search(line)
        if c:
            r = rank_key(name)
            old = self.caps.get(r)
            if old and old["pid"] != pid:
                self.instance += 1
                self.caps, self.mem = {}, {}
                self.startup_t = None
            self.caps[r] = {"cap": float(c.group(1)), "reserved": float(c.group(2)), "pid": pid, "t": t}
            return
        k = KV_SIZE.search(line)
        if k:
            self.kv_tokens = int(k.group(1).replace(",", ""))
        if "Application startup complete" in line:  # uvicorn's line has no stamp: use the log's last one
            self.startup_t = t if t is not None else (self.last_t or ref_epoch)
        for kind, hit in (("oom", OOM_LINE.search(line)), ("dead", DEAD_LINE.search(line)),
                          ("traceback", TRACEBACK in line)):
            if hit:
                self.counts[kind] += 1
                if sum(1 for e in self.events if e[1] == kind) < self.MAX_EVENTS:
                    self.events.append([t, kind, line.strip()[:240]])

    def ranks(self):
        """Per rank: the newest GPU memory line plus cap - peak and cap - reserved (only when the memory line comes
        from the same process as the cap line)."""
        out = {}
        for r in sorted(set(self.caps) | set(self.mem)):
            c, m = self.caps.get(r), self.mem.get(r)
            d = dict(m) if m else {}
            if c:
                d["cap"] = c["cap"]
                d["cap_reserved0"] = c["reserved"]
                d["cap_minus_reserved0"] = round(c["cap"] - c["reserved"], 3)
                if m and (m["pid"] == c["pid"] or None in (m["pid"], c["pid"])):
                    d["cap_minus_peak"] = round(c["cap"] - m["peak"], 3)
                    d["cap_minus_reserved"] = round(c["cap"] - m["reserved"], 3)
            out[r] = d
        return out

    @property
    def key(self):
        return "%s#%d" % (self.path, self.instance)


class ServerLogs:
    """All server logs matching the globs (a restart writes a new one, e.g. dsv41_logs/r1/server/server.log).
    The current server is the newest-modified log that has per-rank lines (else the newest log)."""

    def __init__(self, patterns):
        self.patterns = list(patterns)
        self.follow = LogFollower()
        self.logs = {}

    def update(self, now=None):
        now = now or time.time()
        for p in expand_globs(self.patterns):
            lg = self.logs.get(p)
            if lg is None:
                lg = self.logs[p] = VllmLog(p)
            lines, reset = self.follow.read(p)
            if reset:
                lg.reset()
            for line in lines:
                lg.feed(line, now)
        return self.summary()

    def current(self):
        def mtime(p):
            try:
                return os.path.getmtime(p)
            except OSError:
                return 0
        ranked = sorted(self.logs.values(), key=lambda lg: (bool(lg.caps or lg.mem), mtime(lg.path)))
        return ranked[-1] if ranked else None

    def summary(self):
        cur = self.current()
        counts = {"oom": 0, "dead": 0, "traceback": 0}
        for lg in self.logs.values():
            for k in counts:
                counts[k] += lg.counts[k]
        if cur is None:
            return {"logs": 0, "counts": counts}
        ranks = cur.ranks()
        return {"logs": len(self.logs), "current_log": cur.path, "server_key": cur.key, "ranks": ranks,
                "kv_tokens": cur.kv_tokens, "startup_t": cur.startup_t, "counts": counts,
                "worst": worst_ranks(ranks)}


def worst_ranks(ranks):
    """Worst-rank summary: min cap - peak / cap - reserved, max peak / reserved, max retries, sum of OOMs."""
    def vals(k):
        return [d[k] for d in ranks.values() if d.get(k) is not None]
    w = {}
    for k, f in (("cap_minus_peak", min), ("cap_minus_reserved", min), ("peak", max), ("reserved", max),
                 ("alloc", max), ("free", min), ("retries", max)):
        v = vals(k)
        w[k] = f(v) if v else None
    o = vals("ooms")
    w["ooms"] = sum(o) if o else None
    return w


# ------------------------------------------------------------------------------------------------- solver log
STAMP = re.compile(r"^\[(\d\d):(\d\d):(\d\d) \+([\d.]+)h\]")
METRICS = re.compile(r"\] server: ([-\d.]+) gen tok/s.*?KV ([\d.]+), running ([\d.]+), waiting ([\d.]+), "
                     r"preemptions ([\d.]+)")
UP_AFTER = re.compile(r"server up after (\d+)s")
FINAL_ERROR = re.compile(r"\] (?:\[\d+\] )?\S+#\d+ pred=\S+ stop=error")
RETRY = re.compile(r"^\[[^\]]*\] retry \S+: ")


def stamp_epoch(H, M, S, ref_epoch, slack=300):
    """Epoch of a UTC time-of-day stamp: the day that puts it at or before ref (+ slack)."""
    t = ref_epoch - ref_epoch % 86400 + H * 3600 + M * 60 + S
    if t > ref_epoch + slack:
        t -= 86400
    return t


def stamp_near(H, M, S, ref_epoch):
    """Epoch of a UTC time-of-day stamp: the day that puts it closest to ref (within +-12 h)."""
    t = ref_epoch - ref_epoch % 86400 + H * 3600 + M * 60 + S
    if t > ref_epoch + 43200:
        t -= 86400
    elif t < ref_epoch - 43200:
        t += 86400
    return t


class SolverLog:
    """Counters and stamped events from the solver's log ("[HH:MM:SS +X.XXh] ..."). t0 = the Kaggle clock's zero
    (the solver's elapsed hours are counted from the notebook start)."""

    KINDS = ("server_died", "server_up", "requeue", "retry", "stop_error", "kv_gate", "traceback", "timeout")
    MAX_EVENTS = 400

    def __init__(self, path):
        self.path = path
        self.follow = LogFollower()
        self.reset()

    def reset(self):
        self.counts = dict.fromkeys(self.KINDS, 0)
        self.events = []
        self.t0 = None
        self.last_h = None
        self.last_metrics = None
        self.last_kind = None

    def feed(self, line, ref_epoch):
        m = STAMP.match(line)
        h = float(m.group(4)) if m else None
        if m:
            self.last_h = h
            if self.t0 is None:
                self.t0 = stamp_epoch(int(m.group(1)), int(m.group(2)), int(m.group(3)), ref_epoch) - h * 3600
        kind = None
        if "server died" in line:
            kind = "server_died"
        elif UP_AFTER.search(line):
            kind = "server_up"
        elif "re-queue " in line:
            kind = "requeue"
        elif RETRY.search(line):
            kind = "retry"
        elif FINAL_ERROR.search(line):
            kind = "stop_error"
        elif "KV gate:" in line:
            kind = "kv_gate"
        if "Traceback" in line:
            self.counts["traceback"] += 1
        if "APITimeoutError" in line or "Request timed out" in line:
            self.counts["timeout"] += 1
        mm = METRICS.search(line)
        if mm:
            g, kv, run, wait, pre = (float(x) for x in mm.groups())
            self.last_metrics = {"gen_tok_s": g, "kv": kv, "running": run, "waiting": wait, "preemptions": pre,
                                 "h": h}
        if kind:
            self.counts[kind] += 1
            if kind in ("server_died", "server_up"):
                self.last_kind = kind
            if kind != "kv_gate" and len(self.events) < self.MAX_EVENTS:
                self.events.append([m.group(0)[1:9] if m else None, h, kind, line.strip()[:240]])

    def update(self, now=None):
        now = now or time.time()
        lines, reset = self.follow.read(self.path)
        if reset:
            self.reset()
        for line in lines:
            self.feed(line, now)
        return {"counts": dict(self.counts), "t0": self.t0, "last_h": self.last_h, "metrics": self.last_metrics,
                "down": self.last_kind == "server_died"}


class AttemptsFile:
    """Record and error-record counts of attempts.jsonl. Per-try error records are normal (re-queued jobs); the
    finalized errors are counted from the solver log ("stop=error")."""

    def __init__(self, path):
        self.path = path
        self.follow = LogFollower()
        self.records = self.errors = 0

    def update(self):
        lines, reset = self.follow.read(self.path)
        if reset:
            self.records = self.errors = 0
        for line in lines:
            if not line.strip():
                continue
            self.records += 1
            try:
                if json.loads(line).get("error"):
                    self.errors += 1
            except ValueError:
                pass
        return {"records": self.records, "error_records": self.errors}


def read_status(patterns):
    """The newest supervisor status JSON matching the globs, or {}."""
    files = expand_globs(patterns)
    files.sort(key=lambda p: os.path.getmtime(p) if os.path.exists(p) else 0)
    for p in reversed(files):
        try:
            with open(p) as f:
                d = json.load(f)
        except (OSError, ValueError):
            continue
        if isinstance(d, dict):
            keep = {k: d.get(k) for k in ("state", "start", "pid", "error", "up_s", "updated", "restarts")
                    if k in d}
            keep["file"] = p
            return keep
    return {}


def notebook_done(paths):
    """The runner's end line (run_notebook.py "NOTEBOOK_DONE", papermill wrappers' "nb end rc=") or None."""
    for p in paths:
        try:
            with open(p, "rb") as f:
                f.seek(max(0, os.path.getsize(p) - 65536))
                txt = f.read().decode(errors="replace")
        except OSError:
            continue
        hits = [ln for ln in txt.splitlines() if "NOTEBOOK_DONE" in ln or "nb end rc=" in ln]
        if hits:
            return hits[-1][:300]
    return None


# --------------------------------------------------------------------------------------------- record readers
def load_records(path, kind="sample"):
    out = []
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except ValueError:
                    continue  # a line cut by a copy taken mid-write
                if r.get("kind", "sample") == kind:
                    out.append(r)
    except OSError:
        pass
    return out


def kaggle_t0(records):
    """The Kaggle clock's zero: the solver's t0 from the newest record that has one, else None."""
    for r in reversed(records):
        t0 = (r.get("solver") or {}).get("t0")
        if t0:
            return t0
    return None


def clock(records):
    """(function epoch -> hours, label). Kaggle hours when the solver log gave t0, else hours since the first
    sample."""
    t0 = kaggle_t0(records)
    if t0:
        return (lambda t: (t - t0) / 3600.0), "Kaggle h"
    first = records[0]["t"] if records else 0
    return (lambda t: (t - first) / 3600.0), "h since first sample"


def segments(records):
    """Split records by vLLM server instance (server_key); records without a server are their own segment."""
    segs, cur, key = [], [], object()
    for r in records:
        k = (r.get("vllm") or {}).get("server_key")
        if k != key and cur:
            segs.append(cur)
            cur = []
        key = k
        cur.append(r)
    if cur:
        segs.append(cur)
    return segs


# ------------------------------------------------------------------------------------------------ trend maths
def linfit(points):
    """Least-squares slope (per x unit) of [(x, y)]; None with < 3 points or no x spread."""
    pts = [(x, y) for x, y in points if x is not None and y is not None]
    n = len(pts)
    if n < 3:
        return None
    mx = sum(x for x, _ in pts) / n
    my = sum(y for _, y in pts) / n
    sxx = sum((x - mx) ** 2 for x, _ in pts)
    if sxx <= 0:
        return None
    return sum((x - mx) * (y - my) for x, y in pts) / sxx


def eta_hours(current, slope, limit, direction):
    """Hours until current reaches limit at this slope: 0.0 if already past it, None if not moving toward it.
    direction "up": the limit is above (RAM); "down": the limit is below (cap - peak)."""
    if current is None or limit is None:
        return None
    if direction == "up":
        if current >= limit:
            return 0.0
        return (limit - current) / slope if slope and slope > 1e-9 else None
    if current <= limit:
        return 0.0
    return (current - limit) / -slope if slope and slope < -1e-9 else None


STATUS_ORDER = {"OK": 0, "N/A": 0, "WATCH": 1, "RISK": 2, "FAIL": 3}


def trend_verdict(name, current, slope, limit, direction, now_h, end_h, blew_at_h=None, unit="GiB",
                  stable_horizon_h=48.0):
    """(status, text): "stable" (OK), "growing toward X, ETA ~Y h" (RISK if before the run's end, WATCH after),
    or "blew up at Z" (FAIL)."""
    if blew_at_h is not None:
        return "FAIL", "%s blew up at %.2f h" % (name, blew_at_h)
    if current is None:
        return "N/A", "%s: no data" % name
    if slope is None:
        return "N/A", "%s: %.2f %s now, too little post-warm-up data for a slope" % (name, current, unit)
    eta = eta_hours(current, slope, limit, direction)
    if eta == 0.0:
        return "FAIL", "%s blew up at %.2f h (%.2f %s, limit %s)" % (name, now_h, current, unit, limit)
    toward = "growing" if direction == "up" else "shrinking"
    if eta is None or eta > stable_horizon_h:
        return "OK", "%s stable (%.2f %s, %+.3f %s/h)" % (name, current, unit, slope, unit)
    remaining = (end_h - now_h) if end_h is not None else None
    if remaining is not None and eta <= remaining:
        return "RISK", ("%s %s toward %s %s, ETA ~%.1f h (at %.1f h, before the %.1f h end; %+.3f %s/h)"
                        % (name, toward, limit, unit, eta, now_h + eta, end_h, slope, unit))
    return "WATCH", ("%s %s toward %s %s, ETA ~%.1f h (after the run's end; %+.3f %s/h)"
                     % (name, toward, limit, unit, eta, slope, unit))


def post_warmup(segment, warmup_s, window_s=None):
    """The part of a server segment after its warm-up, limited to the last window_s seconds."""
    if not segment:
        return []
    start = segment[0]["t"]
    st = ((segment[-1].get("vllm") or {}).get("startup_t"))
    if st and st > start:
        start = st
    out = [r for r in segment if r["t"] >= start + warmup_s]
    if window_s and out:
        out = [r for r in out if r["t"] >= out[-1]["t"] - window_s]
    return out


def get_path(rec, path):
    """rec["a"]["b"] for path "a.b"; None when missing."""
    cur = rec
    for part in path.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
    return cur
