#!/usr/bin/env python3
"""E046: build a frozen-input submission notebook: submit Version 5's exact notebook and inputs (model v1,
chankhavu/dsv41-tools-kit, the wheels kernel, competition data) + a settings cell + cell 0b, which installs a pinned
solver and harness over the kit (sha256, feature and setting checks; a failure raises in the commit run, falls back to v3 in the rerun). No new
dataset version is mounted (Kaggle mount errors cluster on fresh cache entries, discussion 752147).

Usage: build_frozen_mt.py --slug dsv41-tools-submit-mtf --workers 36 --budget 0.5 [--gate 0.6]
       [--solver PATH] [--harness PATH] [--out DIR]
The commit run ("Save & Run All") is a smoke test: server start + 3 tasks (2 multi-test); the competition rerun runs
every task with the settings below.
"""
import argparse
import base64
import gzip
import hashlib
import io
import json
import sys
import tarfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:  # tests load this file by path
    sys.path.insert(0, str(HERE))
import input_paths as IP  # noqa: E402

ROOT = HERE.parents[1]
V5 = HERE / "out/nb_submit_v4"  # submit Version 5's notebook (v3's cells + the v4 settings cell)
KIT_TGZ = HERE / "kaggle_kit_v5/dsv41_kit.tgz"  # the kit dataset as Kaggle mounts it (Version 5's inputs)


def check_imports(files, kit_tgz=KIT_TGZ):
    """Build-time guard (E055, 2026-10-27: the live harness imported grid_image.deepseek_cell, which the Kaggle kit's
    grid_image.py lacks, so every attempt would have failed on Kaggle; cell 0b checks features, not imports). Every
    `from <local module> import <names>` in a pinned repo/evals file must resolve against what the notebook runs: the
    pinned copy of that module if one is shipped, else the kit's. Returns the unresolved names."""
    import ast
    with tarfile.open(kit_tgz) as tf:
        view = {m.name.removeprefix("./"): tf.extractfile(m).read().decode() for m in tf.getmembers()
                if m.isfile() and m.name.endswith(".py")}
    pinned = {arc: Path(src).read_text() for arc, src in files.items() if arc.endswith(".py")}
    view.update(pinned)

    def top_names(text):
        names = set()
        for n in ast.parse(text).body:
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names.add(n.name)
            elif isinstance(n, (ast.Assign, ast.AnnAssign)):
                for t in (n.targets if isinstance(n, ast.Assign) else [n.target]):
                    names |= {x.id for x in ast.walk(t) if isinstance(x, ast.Name)}
            elif isinstance(n, (ast.Import, ast.ImportFrom)):
                names |= {(a.asname or a.name).split(".")[0] for a in n.names}
        return names
    bad = []
    for arc, text in pinned.items():
        if not arc.startswith("repo/evals/"):
            continue
        for node in ast.walk(ast.parse(text)):
            if isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                mod = "repo/evals/" + node.module.replace(".", "/") + ".py"
                if mod in view:  # a local module (not the standard library or site-packages)
                    have = top_names(view[mod])
                    bad += [f"{arc}: from {node.module} import {a.name}" for a in node.names
                            if a.name != "*" and a.name not in have]
    return bad



def opt_env(a):
    """Settings cell / expected env for the opt-in levers of the next candidate (empty for V6's settings)."""
    env, g = [], lambda k: getattr(a, k, 0)
    if g("effort"):
        env.append(("ARC_REASONING_EFFORT", str(g("effort"))))
    if g("auto_repair"):
        env.append(("ARC_AUTO_REPAIR", "1"))
    if g("best_guess_at"):
        env.append(("ARC_BEST_GUESS_AT", str(g("best_guess_at"))))
    if g("best_guess_turns"):
        env.append(("ARC_BEST_GUESS_TURNS", str(g("best_guess_turns"))))
    if g("post_pass_note"):
        env.append(("ARC_POST_PASS_NOTE", "1"))
    if g("admit_max_waiting"):
        env.append(("ARC_ADMIT_MAX_WAITING", str(g("admit_max_waiting"))))
    if g("stall_restart"):
        env.append(("ARC_STALL_RESTART_S", str(g("stall_restart"))))
    if g("conn_hold"):
        env.append(("ARC_CONN_HOLD", "1"))
    if g("restart_ramp"):
        env.append(("ARC_RESTART_RAMP", str(g("restart_ramp"))))
    return env


def opt_need(a):
    """Cell 0b's feature check for the opt-in levers: {pinned file: strings it must contain}."""
    need, g = {}, lambda k: getattr(a, k, 0)
    harness, solver = "repo/evals/run_deepseek_tools.py", "solver_dsv41.py"
    if g("auto_repair"):
        need.setdefault(harness, []).append("--auto-repair")
    if g("best_guess_at") or g("best_guess_turns"):
        need.setdefault(harness, []).append("--best-guess-at")
    if g("post_pass_note"):
        need.setdefault(harness, []).append("--post-pass-note")
        need.setdefault(solver, []).append("ARC_POST_PASS_NOTE")
    if g("effort"):
        need.setdefault(solver, []).append("ARC_REASONING_EFFORT")
    if g("admit_max_waiting"):
        need.setdefault(solver, []).append("ARC_ADMIT_MAX_WAITING")
    for key, var in (("stall_restart", "ARC_STALL_RESTART_S"), ("conn_hold", "ARC_CONN_HOLD"),
                     ("restart_ramp", "ARC_RESTART_RAMP")):
        if g(key):
            need.setdefault(solver, []).append(var)
    return need


def retention_args(a):
    """Server flags appended to Version 5's: the retention interval and any --server-extra flags (e.g. E051's
    --kv-transfer-config, which needs the --overlay that ships the patched offloading connector)."""
    r = getattr(a, "retention", 0)
    return (["--prefix-cache-retention-interval", str(r)] if r else []) + json.loads(getattr(a, "server_extra", "") or "[]")


# Version 5's server flags: what the rerun fallback (the kit's v3 code and the kit's own vLLM overlay) runs with
V5_SERVER_EXTRA = json.dumps(["--linear-backend", "humming", "--compilation-config",
                              json.dumps({"cudagraph_mode": "FULL", "max_cudagraph_capture_size": 64})])


def smoke_block(a):
    """Commit runs only (`--smoke-minutes N`): deadlines that end the smoke within N minutes of the first cell (stop-new
    N-5, force N-3, hard stop N-2, kill switch N-0.5 min), so model load, generation, the force cut and the final write
    all run, and a background sampler of host memory (cgroup), GPU power / clocks / memory and free disk every 30 s
    (ARC_SMOKE_ENV_LOG, read by the smoke report cell at the end)."""
    n = getattr(a, "smoke_minutes", 0)
    if not n:
        return ""
    dl = {"ARC_STOP_NEW_S": (n - 5) * 60, "ARC_FORCE_AT_S": (n - 3) * 60, "ARC_HARD_STOP_S": (n - 2) * 60,
          "ARC_KILL_AT_S": int((n - 0.5) * 60)}
    sets = "".join(f'    os.environ.setdefault("{k}", "{v}")\n' for k, v in dl.items())
    if getattr(a, "smoke_stress", False):
        # 24 tasks (31 pairs, 6 multi-input) and a GPU KV pool cut to ~220k tokens, so the gate, preemption, the KV
        # offload store AND load paths, the force cut and the final write all run in a 30-min smoke
        sets += (f'    os.environ["ARC_TASK_IDS"] = "{",".join(SMOKE_STRESS_TASKS)}"\n'
                 f'    os.environ.setdefault("ARC_SMOKE_SANDBOXES", "{a.workers}")\n'
                 '    _sx = json.loads(os.environ["ARC_SERVER_EXTRA_ARGS"])\n'
                 '    if "--num-gpu-blocks-override" not in _sx:\n'
                 f'        os.environ["ARC_SERVER_EXTRA_ARGS"] = json.dumps(_sx + ["--num-gpu-blocks-override", "{SMOKE_KV_BLOCKS}"])\n')
    return f'''    # smoke within {n} min of the first cell: model load, generation, force cut and final write all run
{sets}    import shutil, subprocess, threading, time as _st
    os.environ.setdefault("ARC_SMOKE_ENV_LOG", "/kaggle/working/smoke_env.jsonl")

    def _smoke_env_sampler(path=os.environ["ARC_SMOKE_ENV_LOG"]):
        def _rd(p):
            try:
                return open(p).read().strip()
            except Exception:
                return None
        while True:
            try:
                rec = {{"t": round(_st.time(), 1)}}
                for k in ("memory.current", "memory.peak", "memory.max", "memory.swap.current"):
                    rec[k] = _rd(f"/sys/fs/cgroup/{{k}}")
                rec["stat"] = {{l.split()[0]: int(l.split()[1]) for l in (_rd("/sys/fs/cgroup/memory.stat") or "").splitlines()
                               if l.split()[:1] and l.split()[0] in ("anon", "file", "shmem", "unevictable", "file_mapped")}}
                rec["meminfo_kb"] = {{l.split(":")[0]: int(l.split()[1]) for l in (_rd("/proc/meminfo") or "").splitlines()
                                     if l.split(":")[0] in ("MemTotal", "MemAvailable", "Shmem", "Mlocked", "Cached")}}
                try:
                    rec["gpu"] = subprocess.run(["nvidia-smi", "--query-gpu=index,power.draw,power.limit,clocks.sm,"
                                                 "memory.used,memory.total,temperature.gpu", "--format=csv,noheader,nounits"],
                                                capture_output=True, text=True, timeout=20).stdout.strip().splitlines()
                except Exception:
                    rec["gpu"] = []
                _sb_n, _sb_kb = 0, 0
                for _p in os.listdir("/proc"):
                    if _p.isdigit() and _rd(f"/proc/{{_p}}/oom_score_adj") == "1000":  # tool sandbox sessions
                        _sb_n += 1
                        _sb_kb += int(next((l.split()[1] for l in (_rd(f"/proc/{{_p}}/status") or "").splitlines()
                                            if l.startswith("VmRSS:")), 0))
                rec["sandbox"] = {{"n": _sb_n, "rss_gib": round(_sb_kb / 2**20, 2)}}
                rec["disk_free_gib"] = {{d: round(shutil.disk_usage(d).free / 2**30, 1)
                                        for d in ("/kaggle/working", "/kaggle/temp", "/kaggle/tmp", "/tmp", "/dev/shm")
                                        if os.path.exists(d)}}
                with open(path, "a") as f:
                    f.write(json.dumps(rec) + "\\n")
            except Exception:
                pass
            _st.sleep(30)
    threading.Thread(target=_smoke_env_sampler, daemon=True).start()
'''


SMOKE_REPORT_CELL = '''# Cell 5 (smoke report, commit runs only): peaks of host memory (cgroup, the 173 GiB limit), GPU power / clocks /
# memory / temperature and the lowest free disk, from the sampler started in the settings cell (every 30 s)
import json, os
if not os.environ.get("KAGGLE_IS_COMPETITION_RERUN") and os.environ.get("ARC_SMOKE_ENV_LOG"):
    try:
        _recs = [json.loads(l) for l in open(os.environ["ARC_SMOKE_ENV_LOG"]) if l.strip()]
        _gib = lambda v: round(int(v) / 2**30, 1) if v not in (None, "", "max") and str(v).isdigit() else v
        _mx = lambda xs: max([x for x in xs if isinstance(x, (int, float))], default=None)
        print(f"smoke report: {len(_recs)} samples over {(_recs[-1]['t'] - _recs[0]['t']) / 60:.1f} min", flush=True)
        print("  cgroup memory.max", _gib(_recs[-1].get("memory.max")), "GiB; peak memory.current",
              _mx([_gib(r.get("memory.current")) for r in _recs]), "GiB; memory.peak", _gib(_recs[-1].get("memory.peak")),
              "GiB; swap", _gib(_recs[-1].get("memory.swap.current")), flush=True)
        for k in ("anon", "shmem", "file", "unevictable"):
            print(f"  peak cgroup {k}", _mx([_gib(r.get("stat", {}).get(k)) for r in _recs]), "GiB", flush=True)
        _mi = [r.get("meminfo_kb", {}) for r in _recs]
        print("  host MemTotal", round(_mi[-1].get("MemTotal", 0) / 2**20, 1), "GiB; min MemAvailable",
              round(min(m.get("MemAvailable", 0) for m in _mi) / 2**20, 1), "GiB; peak Shmem",
              round(max(m.get("Shmem", 0) for m in _mi) / 2**20, 1), "GiB; peak Mlocked",
              round(max(m.get("Mlocked", 0) for m in _mi) / 2**20, 1), "GiB", flush=True)
        _g = {}
        for r in _recs:
            for line in r.get("gpu", []):
                f = [x.strip() for x in line.split(",")]
                if len(f) == 7:
                    d = _g.setdefault(f[0], {"draw": [], "limit": f[2], "sm": [], "mem": [], "total": f[5], "temp": []})
                    for k, i in (("draw", 1), ("sm", 3), ("mem", 4), ("temp", 6)):
                        try:
                            d[k].append(float(f[i]))
                        except ValueError:
                            pass
        for i, d in sorted(_g.items()):
            print(f"  GPU {i}: power limit {d['limit']} W, peak draw {_mx(d['draw'])} W, peak SM {_mx(d['sm'])} MHz, "
                  f"peak memory {_mx(d['mem'])} / {d['total']} MiB, peak {_mx(d['temp'])} C", flush=True)
        _df = {}
        for r in _recs:
            for k, v in r.get("disk_free_gib", {}).items():
                _df[k] = min(_df.get(k, v), v)
        print("  lowest free disk (GiB):", _df, flush=True)
        _sb = [r.get("sandbox", {}) for r in _recs]
        print("  tool sandboxes: peak", max((s.get("n", 0) for s in _sb), default=0), "processes,",
              max((s.get("rss_gib", 0) for s in _sb), default=0), "GiB RSS", flush=True)
    except Exception as exc:
        print("smoke report failed:", type(exc).__name__, exc, flush=True)
    try:  # which code paths the smoke exercised (solver log + server metrics)
        _lg = f"{os.environ.get('ARC_OUT', '/kaggle/working')}/dsv41_logs"
        _m = [json.loads(l) for l in open(f"{_lg}/metrics.jsonl") if l.strip()]
        _s = open(f"{_lg}/solver.log").read()
        _host = sum((x.get("host_tok_s") or 0) * (x.get("window_s") or 0) for x in _m)
        print("smoke coverage:", {"KV max": max((x.get("kv") or 0) for x in _m), "server running max":
              max((x.get("srv_running") or 0) for x in _m), "preemptions": max((x.get("preemptions") or 0) for x in _m),
              "tokens restored from host (KV offload loads)": int(_host), "KV gate holds logged": _s.count("KV gate:"),
              "server restarts": _s.count("server start (restart"), "attempts done": _s.count(" stop="),
              "force cut": "force deadline:" in _s, "final write": "submission written (final)" in _s}, flush=True)
    except Exception as exc:
        print("smoke coverage failed:", type(exc).__name__, exc, flush=True)
    _integrity_bad = None
    _hp = f"{os.environ.get('ARC_OUT', '/kaggle/working')}/smoke_hang.json"
    if os.path.exists(_hp):  # cell 2c: a deliberate engine hang; the solver's watchdog must restart the server
        _hg = json.load(open(_hp))
        try:
            _hs = open(f"{os.environ.get('ARC_OUT', '/kaggle/working')}/dsv41_logs/solver.log").read()
        except OSError:
            _hs = ""
        _after = _hs.split("server stalled", 1)[1] if "server stalled" in _hs else ""
        print("hang test:", _hg, "| watchdog fired:", "server stalled" in _hs, "| restarts:", _hs.count("server start (restart"),
              "| re-queued after the stall:", _after.count("re-queue "), "| attempts finished after the stall:",
              _after.count(" stop="), "| conn retries spent on a dead server:", _after.count("(2):") + _after.count("(3):"),
              flush=True)
        if _hg.get("error") or _hg.get("resumed_without_restart") or "server stalled" not in _hs:
            _integrity_bad = "HANG TEST FAILED: the watchdog did not restart the frozen server (see the line above)"
        else:
            print("HANG TEST OK: frozen engine detected and the server restarted", flush=True)
    _ip = f"{os.environ.get('ARC_OUT', '/kaggle/working')}/smoke_integrity.json"
    if os.path.exists(_ip):  # cell 1b's background check of every input (model hashed after the server came up)
        import time as _it
        for _ in range(48):  # wait up to 4 min for the model hash to finish
            _ig = json.load(open(_ip))
            if _ig.get("done") or _ig.get("error"):
                break
            _it.sleep(5)
        _md = _ig.get("model", {})
        print("input integrity: competition files", _ig.get("competition"), "| kit:", _ig.get("kit"), "| wheels:",
              _ig.get("wheels"), "| model:", {k: _md.get(k) for k in ("files", "checked", "missing", "size_mismatch",
              "sha_mismatch", "started_min", "minutes")}, "| error:", _ig.get("error"), flush=True)
        _mbad = any(_md.get(k) for k in ("missing", "size_mismatch", "sha_mismatch"))
        if _ig.get("error") or not _ig.get("done") or _mbad or (_ig.get("wheels") or {}).get("bad"):
            _integrity_bad = "INPUT INTEGRITY FAILED or incomplete: see the line above"
        else:
            print("INPUT INTEGRITY OK: every model file matches its sha256", flush=True)
    if _integrity_bad:
        print("=" * 100, _integrity_bad, "=" * 100, sep=chr(10), flush=True)
        raise RuntimeError(_integrity_bad)
'''


def settings_cell(a):
    env = [("ARC_WORKERS", str(a.workers)), ("ARC_MULTI_TEST", "1"), ("ARC_MULTI_BUDGET", str(a.budget)),
           ("ARC_CONTEXT_LIMIT", "163840"), ("ARC_MAX_NUM_SEQS", "64"), ("ARC_GPU_UTIL", str(getattr(a, "util", 0.945)))]
    if a.gate:
        env += [("ARC_KV_ADMIT_MAX", str(a.gate)), ("ARC_KV_ADMIT_MIN_WORKERS", "16")]
    env += opt_env(a)
    pairs = ",\n               ".join(f'("{k}", "{v}")' for k, v in env)
    # E051: vLLM rejects a KV connector when PYTORCH_CUDA_ALLOC_CONF has expandable_segments (set by some images)
    alloc_fix = ("print('PYTORCH_CUDA_ALLOC_CONF from the image:', repr(os.environ.get('PYTORCH_CUDA_ALLOC_CONF')), flush=True)\n"
                 "_ac = [x for x in os.environ.get('PYTORCH_CUDA_ALLOC_CONF', '').split(',') if x and 'expandable_segments' not in x]\n"
                 "os.environ['PYTORCH_CUDA_ALLOC_CONF'] = ','.join(_ac)\n"
                 if "--kv-transfer-config" in getattr(a, "server_extra", "") else "")
    return f'''# Cell 00 (E046 frozen-input candidate): submit Version 5's settings (Humming dense, CUDA graphs <= 64, max-num-seqs
# 64, util 0.945; Kaggle KV 1.55M) + one rollout per task for tasks with several test inputs (ARC_MULTI_TEST) with a
# (1 + {a.budget} (n - 1)) x output budget / turn cap (ARC_MULTI_BUDGET); {a.workers} workers{f", KV gate {a.gate}" if a.gate else ""}.
# E128 on the 45 multi-pair eval tasks (E046 run 3): pass@2 83.1% vs 77.5% per pair at -29% tokens per task.
# ARC_CONTEXT_LIMIT = the server's max-model-len: turns never request past it (a 3-input task overflowed in run 3).
import json, os
for _k, _v in ({pairs},
               ("ARC_SERVER_EXTRA_ARGS", json.dumps(["--linear-backend", "humming", "--compilation-config",
                                                     json.dumps({{"cudagraph_mode": "FULL",
                                                                 "max_cudagraph_capture_size": 64}})]
                                                    + {json.dumps(retention_args(a))}))):
    os.environ.setdefault(_k, _v)
{alloc_fix}# The commit run ("Save & Run All") is a smoke test: server start + 3 tasks (2 multi-test). The competition rerun
# ignores ARC_COMMIT_MODE (cell 0 always runs "submit") and never sees the task list.
if not os.environ.get("KAGGLE_IS_COMPETITION_RERUN"):
    os.environ.setdefault("ARC_COMMIT_MODE", "smoke")
    os.environ.setdefault("ARC_TASK_IDS", "25ff71a9,27a28665,00576224")
{smoke_block(a)}print("E046 settings:", {{k: os.environ.get(k) for k in ("ARC_WORKERS", "ARC_MULTI_TEST", "ARC_MULTI_BUDGET",
                                                     "ARC_CONTEXT_LIMIT", "ARC_KV_ADMIT_MAX", "ARC_MAX_NUM_SEQS", "ARC_GPU_UTIL",
                                                     "ARC_REASONING_EFFORT", "ARC_AUTO_REPAIR", "ARC_BEST_GUESS_AT",
                                                     "ARC_BEST_GUESS_TURNS", "ARC_POST_PASS_NOTE", "ARC_ADMIT_MAX_WAITING",
                                                     "ARC_STOP_NEW_S", "ARC_FORCE_AT_S", "ARC_HARD_STOP_S", "ARC_KILL_AT_S",
                                                     "ARC_SERVER_EXTRA_ARGS", "PYTORCH_CUDA_ALLOC_CONF")}}, flush=True)
'''


# Cell 0b's fixed feature checks (every candidate) and the settings its rerun fallback drops
SOLVER_NEED = ["MULTI_TEST", "MULTI_BUDGET", "CONTEXT_LIMIT", "KV_ADMIT_MAX", "split_multi", "restart_allowed",
               "shutdown_socket"]
HARNESS_NEED = ["--multi-test", "--multi-budget", "--context-limit", "class ContextFull", "import builtins as _b",
                "def answer_items", "def context_budget"]
FALLBACK_DROP = ["ARC_MULTI_TEST", "ARC_MULTI_BUDGET", "ARC_KV_ADMIT_MAX", "ARC_CONTEXT_LIMIT", "ARC_AUTO_REPAIR",
                 "ARC_BEST_GUESS_AT", "ARC_BEST_GUESS_TURNS", "ARC_POST_PASS_NOTE", "ARC_REASONING_EFFORT",
                 "ARC_ADMIT_MAX_WAITING"]
PINS_ARCHIVE = "dsv41_pins.tgz"
SANDBOX_CELL = HERE / "cells/cell2b_sandbox_check.py"
HANG_CELL = HERE / "cells/cell2c_hang_test.py"
SMOKE_KV_BLOCKS = 4500  # ~333k KV tokens (74 per block on this stack): 2x max-model-len 163,840, far below 1.55M
INTEGRITY_CELL = HERE / "cells/cell1b_integrity.py"


def integrity_cell(manifest):
    """Cell 1b: cells/cell1b_integrity.py with the model manifest (path -> [size, sha256]) filled in."""
    m = json.load(open(manifest))
    data = {"model": {e["path"]: [e["size"], e["sha256"]] for e in m["entries"]}}
    text = INTEGRITY_CELL.read_text()
    assert text.count('"""__INTEGRITY__"""') == 1
    return text.replace('"""__INTEGRITY__"""', '"""' + json.dumps(data, sort_keys=True) + '"""')
SMOKE_STRESS_TASKS = ["25ff71a9", "27a28665", "00576224", "337b420f", "25c199f5", "12997ef3", "2ccd9fef", "0ca9ddb6",
                      "412b6263", "351d6448", "178fcbfb", "137f0df0", "2204b7a8", "22eb0ac0", "292dd178", "05f2a901",
                      "2c737e39", "230f2e48", "20fb2937", "3bdb4ada", "2753e76c", "20818e16", "41e4d17e", "13f06aa5"]
PINS_CELL = HERE / "cells/cell0b_pins.py"
PINS_MTIME = 1793059200  # 2026-10-27 00:00 UTC: fixed member mtime, so a rebuild gives the same archive


def build_pins(files, out, slug, user="chankhavu"):
    """The pins dataset folder for `kaggle datasets create -p out`: dsv41_pins.tgz (the pinned files at their kit
    paths; deterministic: sorted members, fixed metadata, gzip mtime 0), SHA256SUMS and dataset-metadata.json.
    Returns (archive sha256, {member: sha256})."""
    out.mkdir(parents=True, exist_ok=True)
    sha = {arc: hashlib.sha256(Path(src).read_bytes()).hexdigest() for arc, src in sorted(files.items())}
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w", format=tarfile.USTAR_FORMAT) as tf:
        for arc, src in sorted(files.items()):
            data = Path(src).read_bytes()
            info = tarfile.TarInfo(arc)
            info.size, info.mtime, info.mode, info.uid, info.gid, info.uname, info.gname = (
                len(data), PINS_MTIME, 0o644, 0, 0, "", "")
            tf.addfile(info, io.BytesIO(data))
    gz = io.BytesIO()
    with gzip.GzipFile(fileobj=gz, mode="wb", mtime=0, filename="") as g:
        g.write(raw.getvalue())
    (out / PINS_ARCHIVE).write_bytes(gz.getvalue())
    archive_sha = hashlib.sha256(gz.getvalue()).hexdigest()
    (out / "SHA256SUMS").write_text(f"{archive_sha}  {PINS_ARCHIVE}\n"
                                    + "".join(f"{h}  {arc}\n" for arc, h in sha.items()))
    json.dump({"title": slug, "id": f"{user}/{slug}", "licenses": [{"name": "CC0-1.0"}]},
              open(out / "dataset-metadata.json", "w"), indent=1)
    return archive_sha, sha


def pins_cell(slug, archive_sha, sha, expect, new=(), need=None):
    """Cell 0b for a pins dataset: cells/cell0b_pins.py with its PINS data filled in (no code is generated)."""
    pins = {"dataset": slug, "archive": PINS_ARCHIVE, "archive_sha": archive_sha, "sha": sha, "new": sorted(new),
            "expect": expect, "solver_need": SOLVER_NEED, "harness_need": HARNESS_NEED, "opt_need": need or {},
            "fallback_drop": FALLBACK_DROP, "fallback_server_extra": V5_SERVER_EXTRA}
    text = PINS_CELL.read_text()
    assert text.count('"""__PINS__"""') == 1
    return text.replace('"""__PINS__"""', '"""' + json.dumps(pins, sort_keys=True) + '"""')


def code_cell(files, expect, new=(), need=None):
    need = need or {}
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for arc, src in files.items():
            tf.add(src, arcname=arc)
    b64 = base64.b64encode(buf.getvalue()).decode()
    sha = {arc: hashlib.sha256(Path(src).read_bytes()).hexdigest() for arc, src in files.items()}
    return f'''# Cell 0b (E046, frozen inputs): the pinned solver + harness ride in the notebook, so the inputs stay exactly
# Version 5's (no new dataset version to mount). Overwrites the kit's two files after cell 0 unpacked and checked it,
# then checks hashes, required features and settings.
# Commit run ("Save & Run All"): any failure RAISES, so the commit errors and the log shows "E046 CHECK FAILED".
# Competition rerun: the same failure restores the kit's original v3 files and runs per pair at 36 workers (a scored
# submission beats an error); the log line is printed either way. The rerun path never raises (S10): a file whose
# backup is missing or cannot be copied back keeps what is in the kit (the pinned file, or the kit's own if the
# overlay never ran), and the notebook still runs.
import base64, hashlib, io, shutil, tarfile
_MT = "{b64}"
_MT_SHA = {json.dumps(sha)}
_MT_EXPECT = {json.dumps(expect)}
_MT_NEW = {json.dumps(sorted(new))}
_MT_NEED = {json.dumps(need)}
if MODE != "fallback":
    _bak, _why = {{}}, ""
    try:
        assert KIT, "kit not unpacked"
        for _f in _MT_SHA:
            if _f in _MT_NEW:
                continue  # new to the kit (no v3 original); the v3 code never imports it
            shutil.copy2(f"{{KIT}}/{{_f}}", f"{{KIT}}/{{_f}}.v3orig")
            _bak[_f] = f"{{KIT}}/{{_f}}.v3orig"  # S10: only once the backup exists
        with tarfile.open(fileobj=io.BytesIO(base64.b64decode(_MT))) as _tf:
            _tf.extractall(KIT)
        for _f, _h in _MT_SHA.items():
            assert hashlib.sha256(open(f"{{KIT}}/{{_f}}", "rb").read()).hexdigest() == _h, f"overlay mismatch: {{_f}}"
        _sv = open(f"{{KIT}}/solver_dsv41.py").read()
        _hv = open(f"{{KIT}}/repo/evals/run_deepseek_tools.py").read()
        for _need in ("MULTI_TEST", "MULTI_BUDGET", "CONTEXT_LIMIT", "KV_ADMIT_MAX", "split_multi", "restart_allowed",
                      "shutdown_socket"):
            assert _need in _sv, f"solver lacks {{_need}}"
        for _need in ("--multi-test", "--multi-budget", "--context-limit", "class ContextFull", "import builtins as _b",
                      "def answer_items", "def context_budget"):
            assert _need in _hv, f"harness lacks {{_need}}"
        for _f, _needs in _MT_NEED.items():  # features the chosen opt-in settings rely on
            _txt = open(f"{{KIT}}/{{_f}}").read()
            for _need in _needs:
                assert _need in _txt, f"{{_f}} lacks {{_need}}"
        for _k, _v in _MT_EXPECT.items():
            assert os.environ.get(_k) == _v, f"{{_k}}={{os.environ.get(_k)!r}}, expected {{_v!r}}"
    except Exception as exc:
        _why = f"{{type(exc).__name__}}: {{exc}}"
    if not _why:
        print("E046 CHECK OK: multi-test code installed and checked:", {{f: h[:8] for f, h in _MT_SHA.items()}},
              _MT_EXPECT, flush=True)
    else:
        print("=" * 100, f"E046 CHECK FAILED: {{_why}}", "=" * 100, sep=chr(10), flush=True)
        if not RERUN:
            raise RuntimeError(f"E046 CHECK FAILED (commit run): {{_why}}")
        _kept = []
        try:
            for _f in _MT_SHA:  # rerun: back to the kit's v3 files, per pair, 36 workers
                if _f in _MT_NEW:
                    continue
                try:
                    shutil.copy2(_bak[_f], f"{{KIT}}/{{_f}}.restore")
                    os.replace(f"{{KIT}}/{{_f}}.restore", f"{{KIT}}/{{_f}}")
                except Exception as exc:
                    _kept.append(f"{{_f}} ({{type(exc).__name__}})")
            # (a kept pinned solver would pass the opt-in harness flags to the v3 harness, whose parser exits)
            for _k in ("ARC_MULTI_TEST", "ARC_MULTI_BUDGET", "ARC_KV_ADMIT_MAX", "ARC_CONTEXT_LIMIT", "ARC_AUTO_REPAIR",
                       "ARC_BEST_GUESS_AT", "ARC_BEST_GUESS_TURNS", "ARC_POST_PASS_NOTE", "ARC_REASONING_EFFORT",
                       "ARC_ADMIT_MAX_WAITING"):
                os.environ.pop(_k, None)
            os.environ["ARC_WORKERS"] = "36"
            os.environ["ARC_SERVER_EXTRA_ARGS"] = {json.dumps(V5_SERVER_EXTRA)}  # no flag that needs the new overlay
        except Exception as exc:
            _kept.append(f"{{type(exc).__name__}}: {{exc}}")
        print("E046 FALLBACK: the kit's v3 code, per pair, 36 workers"
              + (f"; not restored, kept as is: {{', '.join(_kept)}}" if _kept else ""), flush=True)
''', sha


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slug", required=True)
    ap.add_argument("--workers", type=int, required=True)
    ap.add_argument("--budget", type=float, default=0.5)
    ap.add_argument("--gate", type=float, default=0.0)
    ap.add_argument("--solver", type=Path, default=HERE / "solver_dsv41.py")
    ap.add_argument("--harness", type=Path, default=ROOT / "evals/run_deepseek_tools.py")
    ap.add_argument("--extra-pin", action="append", default=[], metavar="ARC=PATH",
                    help="E066: also pin PATH at kit path ARC (e.g. repo/evals/arc_harness/grid_image.py=...); "
                         "it replaces the kit's copy (a new file is created)")
    ap.add_argument("--expect-env", action="append", default=[], metavar="K=V",
                    help="E066: cell 0b also checks os.environ[K] == V (the settings cell must set it)")
    ap.add_argument("--need", action="append", default=[], metavar="ARC=TEXT",
                    help="E066: cell 0b also checks that the installed file ARC contains TEXT")
    ap.add_argument("--train-check", type=Path, help="pin repo/evals/train_check.py (default: the kit's)")
    ap.add_argument("--retention", type=int, default=0, help="--prefix-cache-retention-interval (0: the base 128)")
    ap.add_argument("--util", type=float, default=0.945, help="ARC_GPU_UTIL (V5/V6: 0.945)")
    ap.add_argument("--effort", type=int, default=0, help="ARC_REASONING_EFFORT (0: the solver's 50)")
    ap.add_argument("--auto-repair", action="store_true", help="ARC_AUTO_REPAIR=1 + embed evals/auto_repair*.")
    ap.add_argument("--best-guess-at", type=float, default=0.0)
    ap.add_argument("--best-guess-turns", type=int, default=0)
    ap.add_argument("--post-pass-note", action="store_true", help="ARC_POST_PASS_NOTE=1 (I8)")
    ap.add_argument("--overlay", type=Path, help="dir with a replacement vllm_overlay_rw.tgz + .sha256 (cell 2 installs it)")
    ap.add_argument("--server-extra", default="", help="JSON list of extra vLLM server flags (appended)")
    ap.add_argument("--admit-max-waiting", type=int, default=0, help="ARC_ADMIT_MAX_WAITING (S17: gate holds while "
                    "the server has >= N waiting requests)")
    ap.add_argument("--smoke-integrity", type=Path, help="model manifest (entries: path, size, sha256): commit runs hash "
                    "every model file and check every other input in the background (cell 1b)")
    ap.add_argument("--stall-restart", type=int, default=0, help="ARC_STALL_RESTART_S (S20 progress watchdog)")
    ap.add_argument("--conn-hold", action="store_true", help="ARC_CONN_HOLD=1 (S21 re-launch hold)")
    ap.add_argument("--restart-ramp", default="", help="ARC_RESTART_RAMP start,per_min (S22)")
    ap.add_argument("--smoke-hang", action="store_true", help="commit runs: freeze the engine after server up "
                    "(cells/cell2c_hang_test.py) to exercise the watchdog, restart and ramp")
    ap.add_argument("--smoke-stress", action="store_true", help="with --smoke-minutes: 24 tasks, a ~220k-token GPU KV "
                    "pool (gate, preemption, offload loads), and a tool-sandbox check cell (commit runs only)")
    ap.add_argument("--smoke-minutes", type=int, default=0, help="commit-run smoke ends within N min (deadlines + "
                    "environment sampler + report cell); 0: Version 5's smoke deadlines")
    ap.add_argument("--pins-dataset", default="", help="ship the pinned files as this private dataset (folder written to "
                    "out/pins_<slug>/ for `kaggle datasets create`) instead of base64 inside cell 0b")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    nb = json.load(open(V5 / "dsv41-tools-submit.ipynb"))
    meta = json.load(open(V5 / "kernel-metadata.json"))
    assert "".join(nb["cells"][1]["source"]).startswith("# Cell 0: clock, paths"), "unexpected cell order"
    expect = {"ARC_WORKERS": str(a.workers), "ARC_MULTI_TEST": "1", "ARC_MULTI_BUDGET": str(a.budget),
              "ARC_CONTEXT_LIMIT": "163840"}
    if a.gate:
        expect["ARC_KV_ADMIT_MAX"] = str(a.gate)
    expect.update(opt_env(a))
    expect.update(dict(kv.split("=", 1) for kv in a.expect_env))
    if a.util != 0.945:
        expect["ARC_GPU_UTIL"] = str(a.util)
    files = {"solver_dsv41.py": a.solver, "repo/evals/run_deepseek_tools.py": a.harness}
    new, need = [], opt_need(a)
    for arc_text in a.need:
        arc, text = arc_text.split("=", 1)
        need.setdefault(arc, []).append(text)
    for pin in a.extra_pin:
        arc, path = pin.split("=", 1)
        files[arc] = Path(path)
        with tarfile.open(KIT_TGZ) as tf:
            if arc not in {m.name.removeprefix("./") for m in tf.getmembers()}:
                new.append(arc)
    if a.train_check:  # pin train_check.py too (the kit's is 44c4d210; pods run 8fe36175); the fallback restores it
        files["repo/evals/train_check.py"] = a.train_check
    if a.auto_repair:  # taken from the pinned harness's folder when it has them, else the live evals/
        for name in ("auto_repair.py", "auto_repair_names.json"):
            src = a.harness.parent / name
            files[f"repo/evals/{name}"] = src if src.exists() else ROOT / "evals" / name
            new.append(f"repo/evals/{name}")
    if a.overlay:  # replaces the kit's overlay pair; cell 2 sha-checks every file it installs; the fallback restores the kit's
        for name in ("vllm_overlay_rw.tgz", "vllm_overlay_rw.sha256"):
            files[name] = a.overlay / name
    json.loads(a.server_extra or "[]")  # fail at build time on a malformed list
    bad = check_imports(files)
    assert not bad, f"pinned files import names the notebook would not have: {bad}"
    if a.pins_dataset:
        archive_sha, sha = build_pins(files, HERE / f"out/pins_{a.pins_dataset}", a.pins_dataset)
        cell0b = pins_cell(a.pins_dataset, archive_sha, sha, expect, new, need)
    else:
        cell0b, sha = code_cell(files, expect, new, need)
    nb["cells"][0]["source"] = settings_cell(a).splitlines(keepends=True)
    nb["cells"].insert(2, {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
                           "source": cell0b.splitlines(keepends=True)})
    if a.smoke_minutes:
        assert a.smoke_minutes >= 20, "the model alone takes ~15 min to load on Kaggle"
        nb["cells"].append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
                            "source": SMOKE_REPORT_CELL.splitlines(keepends=True)})
    if a.smoke_integrity:
        assert a.smoke_minutes, "--smoke-integrity needs --smoke-minutes"
        i = next(i for i, c in enumerate(nb["cells"]) if "".join(c["source"]).startswith("# Cell 1: stage the checkpoint"))
        nb["cells"].insert(i + 1, {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
                                   "source": integrity_cell(a.smoke_integrity).splitlines(keepends=True)})
    if a.smoke_hang:
        assert a.smoke_minutes >= 32 and a.stall_restart, "--smoke-hang needs --smoke-minutes >= 32 (a restart must fit before stop-new) and --stall-restart"
        i = next(i for i, c in enumerate(nb["cells"]) if "".join(c["source"]).startswith("# Cell 3: the solver"))
        nb["cells"].insert(i, {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
                               "source": HANG_CELL.read_text().splitlines(keepends=True)})
    if a.smoke_stress:
        assert a.smoke_minutes, "--smoke-stress needs --smoke-minutes"
        i = next(i for i, c in enumerate(nb["cells"]) if "".join(c["source"]).startswith("# Cell 3: the solver"))
        nb["cells"].insert(i, {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
                               "source": SANDBOX_CELL.read_text().splitlines(keepends=True)})
    out = a.out or HERE / f"out/nb_{a.slug}"
    out.mkdir(parents=True, exist_ok=True)
    meta.update(id=f"chankhavu/{a.slug}", title=a.slug, code_file=f"{a.slug}.ipynb")
    if a.pins_dataset:
        meta["dataset_sources"] = list(meta["dataset_sources"]) + [f"chankhavu/{a.pins_dataset}"]
    meta.pop("id_no", None)
    # Version 5's cells 0 and 2 found inputs with recursive scans: bounded lookups (input_paths.py), checked
    IP.finish(nb, meta)
    json.dump(nb, open(out / f"{a.slug}.ipynb", "w"), indent=1, ensure_ascii=False)
    json.dump(meta, open(out / "kernel-metadata.json", "w"), indent=1)
    print(out, {k: meta[k] for k in ("dataset_sources", "model_sources", "kernel_sources", "is_private")},
          {f: h[:8] for f, h in sha.items()})


if __name__ == "__main__":
    main()
