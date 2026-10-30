#!/usr/bin/env python3
"""Run an exact Kaggle notebook end to end on a pod, as Kaggle's "Run All" does: nbclient, one kernel, the cells in
order, stop at the first error. Never run this on Kaggle (it refuses when KAGGLE_KERNEL_RUN_TYPE is set).

/kaggle layout (--kaggle-root, default /kaggle):
  --map SRC=DEST   symlink <root>/input/DEST -> SRC (e.g. /root/models/e128=models/chankhavu/<slug>/pytorch/default/1,
                   /root/comp=competitions/arc-prize-2026-arc-agi-2). Symlinks only: an existing real file or
                   directory at DEST is never replaced. --map-file F.json takes {"DEST": "SRC"}.
  --prune-input    remove symlinks under <root>/input that are not in the map, so a glob finds one model dir as on
                   Kaggle (real files are never touched).
  --working DIR    a fresh working dir (must be absent or empty); <root>/working becomes a symlink to it. Without it
                   <root>/working itself is used and must be absent or empty; --archive-working DEST moves an old one
                   aside first. Nothing is deleted.
Mode: --mode commit = KAGGLE_IS_COMPETITION_RERUN is absent from the kernel's environment (removed if inherited);
      --mode rerun  = KAGGLE_IS_COMPETITION_RERUN=1 (the hidden-set rerun; map your stand-in test file yourself).
Environment: --clean-env starts from nothing, like `env -i`: HOME, LANG=C.UTF-8, TERM=dumb, PATH = <venv>/bin,
      /usr/local/cuda/bin and the system dirs, VIRTUAL_ENV=<venv>; --env K=V adds variables. Without --clean-env the
      runner's environment is inherited. Run the script with the kernel venv's Python: the "python3" kernel is the
      interpreter that runs this script (e.g. /root/kenv/bin/python -I run_notebook.py ...).
Outputs (prefix --out):
  .timeline       cell START / END lines (UTC, seconds since start, status) and NOTEBOOK_DONE ok|error
  .log            every cell output as it arrives (like papermill --log-output), so the 12 h solver cell's lines
                  (supervisor restarts, ...) are on disk during the run, not only at the end
  .ipynb          the executed notebook, rewritten after every cell
  .manifest.json  notebook sha256, mode, map, working dir, environment variable names, hooks
Hooks: --on-cell-start N:CMD runs `bash -c CMD` detached (own session) when cell N starts (index in the notebook,
      as --dry-run lists them), e.g. the GPU hold before the server cell. Output: <out>.hook<N>.log.
--dry-run prints the plan (links, working dir, environment, cells) and changes nothing.

Usage: run_notebook.py NB.ipynb --out /root/runs/r1/nb --mode rerun --clean-env --venv /root/kenv \\
         --map /root/models/e128=models/chankhavu/m/pytorch/default/1 --prune-input --working /root/kwork_r1"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

RERUN_VAR = "KAGGLE_IS_COMPETITION_RERUN"
SYSTEM_PATH = ["/usr/local/cuda/bin", "/usr/local/sbin", "/usr/local/bin", "/usr/sbin", "/usr/bin", "/sbin", "/bin"]


class Refused(Exception):
    pass


def utc(t=None):
    return time.strftime("%H:%M:%S", time.gmtime(t if t is not None else time.time()))


def parse_map(items, map_file=None):
    """[(dest relative to input/, absolute src)]."""
    pairs = []
    if map_file:
        with open(map_file) as f:
            for dest, src in json.load(f).items():
                pairs.append((dest, src))
    for it in items or []:
        if "=" not in it:
            raise Refused("--map wants SRC=DEST, got %r" % it)
        src, dest = it.split("=", 1)
        pairs.append((dest, src))
    out, seen = [], set()
    for dest, src in pairs:
        dest = dest.strip().strip("/")
        if not dest or os.path.isabs(dest) or ".." in dest.split("/"):
            raise Refused("map destination must be relative to input/ without '..': %r" % dest)
        if dest in seen:
            raise Refused("map destination given twice: %s" % dest)
        seen.add(dest)
        src = os.path.abspath(src)
        if not os.path.exists(src):
            raise Refused("map source does not exist: %s" % src)
        out.append((dest, src))
    for d1 in seen:  # one destination inside another would link through a symlink
        for d2 in seen:
            if d1 != d2 and d2.startswith(d1 + "/"):
                raise Refused("map destinations nest: %s and %s" % (d1, d2))
    return out


def plan_inputs(kaggle_root, mapping, prune):
    """[(action, path, src, old target)] with action in link / relink / keep / prune."""
    inp = os.path.join(kaggle_root, "input")
    acts, wanted = [], set()
    for dest, src in mapping:
        p = os.path.join(inp, dest)
        wanted.add(p)
        if os.path.islink(p):
            cur = os.readlink(p)
            same = os.path.abspath(os.path.join(os.path.dirname(p), cur)) == src
            acts.append(("keep" if same else "relink", p, src, cur))
        elif os.path.lexists(p):
            raise Refused("%s exists and is not a symlink; refusing to replace real data" % p)
        else:
            acts.append(("link", p, src, None))
    if prune and os.path.isdir(inp):
        for dirpath, dirnames, filenames in os.walk(inp):
            for name in dirnames + filenames:
                p = os.path.join(dirpath, name)
                if os.path.islink(p) and p not in wanted:
                    acts.append(("prune", p, None, os.readlink(p)))
    return acts


def apply_inputs(acts):
    for act, p, src, _old in acts:
        if act in ("link", "relink"):
            os.makedirs(os.path.dirname(p), exist_ok=True)
            if act == "relink":
                os.unlink(p)
            os.symlink(src, p)
        elif act == "prune":
            os.unlink(p)


def plan_working(kaggle_root, working, archive):
    """(kernel cwd, [(action, path, detail)])."""
    kw = os.path.join(kaggle_root, "working")
    acts = []

    def nonempty(p):
        return os.path.isdir(p) and not os.path.islink(p) and bool(os.listdir(p))
    if working:
        working = os.path.abspath(working)
        if os.path.exists(working) and (not os.path.isdir(working) or os.listdir(working)):
            raise Refused("--working %s exists and is not an empty directory" % working)
        acts.append(("mkdir", working, ""))
        if os.path.islink(kw):
            acts.append(("relink", kw, working))
        elif nonempty(kw):
            if not archive:
                raise Refused("%s is a non-empty directory; pass --archive-working DEST" % kw)
            acts.append(("archive", kw, os.path.abspath(archive)))
            acts.append(("link", kw, working))
        elif os.path.isdir(kw):
            acts.append(("rmdir_empty", kw, ""))
            acts.append(("link", kw, working))
        else:
            acts.append(("link", kw, working))
        return working, acts
    if os.path.islink(kw):
        target = os.path.realpath(kw)
        if os.path.isdir(target) and os.listdir(target):
            raise Refused("%s -> %s is not empty; use a fresh --working DIR" % (kw, target))
        return kw, acts
    if nonempty(kw):
        if not archive:
            raise Refused("%s is not empty; pass --archive-working DEST or --working DIR" % kw)
        acts.append(("archive", kw, os.path.abspath(archive)))
    acts.append(("mkdir", kw, ""))
    return kw, acts


def apply_working(acts):
    for act, p, detail in acts:
        if act == "mkdir":
            os.makedirs(p, exist_ok=True)
        elif act == "archive":
            if os.path.exists(detail):
                raise Refused("archive destination %s exists" % detail)
            os.makedirs(os.path.dirname(detail), exist_ok=True)
            os.rename(p, detail)
        elif act == "rmdir_empty":
            os.rmdir(p)
        elif act in ("link", "relink"):
            os.makedirs(os.path.dirname(p), exist_ok=True)
            if act == "relink":
                os.unlink(p)
            os.symlink(detail, p)


def build_env(base, mode, clean=False, venv=None, extra=(), home=None):
    if clean:
        path = ([os.path.join(venv, "bin")] if venv else []) + SYSTEM_PATH
        env = {"HOME": home or base.get("HOME", "/root"), "LANG": "C.UTF-8", "TERM": "dumb", "PATH": ":".join(path)}
        if venv:
            env["VIRTUAL_ENV"] = venv
    else:
        env = dict(base)
        if venv:
            env["VIRTUAL_ENV"] = venv
            env["PATH"] = os.path.join(venv, "bin") + ":" + env.get("PATH", ":".join(SYSTEM_PATH))
    for kv in extra or ():
        if "=" not in kv:
            raise Refused("--env wants K=V, got %r" % kv)
        k, v = kv.split("=", 1)
        if k == RERUN_VAR:
            raise Refused("set %s with --mode, not --env" % RERUN_VAR)
        env[k] = v
    env.pop(RERUN_VAR, None)
    if mode == "rerun":
        env[RERUN_VAR] = "1"
    return env


def parse_hooks(items):
    hooks = {}
    for it in items or []:
        n, sep, cmd = it.partition(":")
        if not sep or not n.strip().isdigit() or not cmd.strip():
            raise Refused("--on-cell-start wants N:CMD, got %r" % it)
        hooks.setdefault(int(n), []).append(cmd)
    return hooks


def read_cells(path):
    with open(path) as f:
        nb = json.load(f)
    out = []
    for i, c in enumerate(nb.get("cells", [])):
        src = c.get("source", "")
        src = "".join(src) if isinstance(src, list) else src
        out.append((i, c.get("cell_type"), (src.splitlines() or [""])[0][:100]))
    return out


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def build_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("notebook")
    ap.add_argument("--out", required=True, help="output prefix")
    ap.add_argument("--mode", choices=("commit", "rerun"), required=True)
    ap.add_argument("--kaggle-root", default="/kaggle")
    ap.add_argument("--map", action="append", default=[], metavar="SRC=DEST")
    ap.add_argument("--map-file")
    ap.add_argument("--prune-input", action="store_true")
    ap.add_argument("--working")
    ap.add_argument("--archive-working")
    ap.add_argument("--clean-env", action="store_true")
    ap.add_argument("--venv", help="kernel venv (PATH and VIRTUAL_ENV), e.g. /root/kenv")
    ap.add_argument("--env", action="append", default=[], metavar="K=V")
    ap.add_argument("--kernel", default="python3")
    ap.add_argument("--cell-timeout", type=int, default=50000)
    ap.add_argument("--startup-timeout", type=int, default=300)
    ap.add_argument("--on-cell-start", action="append", default=[], metavar="N:CMD")
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args(argv)


def make_client(nb, a, cwd, files, hooks, env):
    from nbclient import NotebookClient
    import nbformat

    t0 = time.time()
    tl = open(files["timeline"], "a", buffering=1)
    lg = open(files["log"], "a", buffering=1)

    def save():
        try:
            nbformat.write(nb, files["ipynb"])
        except Exception as exc:  # never let a save break the run
            tl.write("%s save failed: %s\n" % (utc(), exc))

    class Client(NotebookClient):
        def output(self, outs, msg, display_id, cell_index):
            c = msg.get("content", {})
            mt = msg.get("msg_type")
            if mt == "stream":
                lg.write(c.get("text", ""))
            elif mt in ("execute_result", "display_data"):
                lg.write(c.get("data", {}).get("text/plain", "") + "\n")
            elif mt == "error":
                lg.write("ERROR %s: %s\n%s\n" % (c.get("ename"), c.get("evalue"), "\n".join(c.get("traceback", []))))
            return super().output(outs, msg, display_id, cell_index)

    def on_start(cell, cell_index, **kw):
        first = (cell.source.splitlines() or [""])[0][:110]
        if cell.cell_type == "code" and cell.source.strip():  # nbclient also calls this for markdown cells
            tl.write("%s +%7.0fs START cell %d: %s\n" % (utc(), time.time() - t0, cell_index, first))
            lg.write("\n===== cell %d %s %s\n" % (cell_index, utc(), first))
        for j, cmd in enumerate(hooks.get(cell_index, [])):
            hl = open("%s.hook%d%s.log" % (a.out, cell_index, "" if j == 0 else "_%d" % j), "a")
            p = subprocess.Popen(["bash", "-c", cmd], stdout=hl, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                 env=env, start_new_session=True)
            tl.write("%s +%7.0fs HOOK cell %d pid %d: %s\n" % (utc(), time.time() - t0, cell_index, p.pid, cmd[:200]))

    def on_done(cell, cell_index, execute_reply=None, **kw):
        st = (execute_reply or {}).get("content", {}).get("status")
        tl.write("%s +%7.0fs END   cell %d: status %s\n" % (utc(), time.time() - t0, cell_index, st))
        save()

    client = Client(nb, timeout=a.cell_timeout, startup_timeout=a.startup_timeout, kernel_name=a.kernel,
                    allow_errors=False, record_timing=True, on_cell_start=on_start, on_cell_executed=on_done,
                    resources={"metadata": {"path": cwd}})
    return client, tl, save, t0


def main(argv=None):
    a = build_args(argv)
    try:
        if os.environ.get("KAGGLE_KERNEL_RUN_TYPE"):
            raise Refused("this looks like a Kaggle kernel (KAGGLE_KERNEL_RUN_TYPE is set): run only on pods")
        nb_path = os.path.abspath(a.notebook)
        cells = read_cells(nb_path)
        mapping = parse_map(a.map, a.map_file)
        in_acts = plan_inputs(a.kaggle_root, mapping, a.prune_input)
        cwd, w_acts = plan_working(a.kaggle_root, a.working, a.archive_working)
        env = build_env(dict(os.environ), a.mode, a.clean_env, a.venv, a.env)
        hooks = parse_hooks(a.on_cell_start)
        bad = [n for n in hooks if n >= len(cells)]
        if bad:
            raise Refused("hook cell index %s out of range (%d cells)" % (bad, len(cells)))
    except (Refused, OSError, ValueError) as exc:
        print("refused: %s" % exc, file=sys.stderr)
        return 2
    files = {k: a.out + ext for k, ext in (("timeline", ".timeline"), ("log", ".log"), ("ipynb", ".ipynb"),
                                           ("manifest", ".manifest.json"))}
    manifest = {"notebook": nb_path, "sha256": sha256(nb_path), "mode": a.mode, RERUN_VAR: env.get(RERUN_VAR),
                "kaggle_root": a.kaggle_root, "cwd": cwd, "map": [[d, s] for d, s in mapping],
                "input_actions": in_acts, "working_actions": w_acts, "env_names": sorted(env),
                "PATH": env.get("PATH"), "VIRTUAL_ENV": env.get("VIRTUAL_ENV"), "python": sys.executable,
                "kernel": a.kernel, "hooks": {str(k): v for k, v in hooks.items()}, "started_utc": utc()}
    if a.dry_run:
        print(json.dumps({k: v for k, v in manifest.items() if k != "started_utc"}, indent=1))
        print("cells:")
        for i, typ, first in cells:
            print("  %3d %-8s %s%s" % (i, typ, first, "   <- hook" if i in hooks else ""))
        return 0
    if os.path.exists(files["timeline"]):
        print("refused: %s exists (pick a new --out)" % files["timeline"], file=sys.stderr)
        return 2
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    try:
        apply_inputs(in_acts)
        apply_working(w_acts)
    except (Refused, OSError) as exc:
        print("refused: %s" % exc, file=sys.stderr)
        return 2
    with open(files["manifest"], "w") as f:
        json.dump(manifest, f, indent=1)
    try:
        import nbclient  # noqa: F401  (imported before the environment is replaced)
        import nbformat
    except ImportError as exc:
        print("refused: %s (run with the kernel venv's python, which needs nbclient + ipykernel)" % exc,
              file=sys.stderr)
        return 2
    nb = nbformat.read(nb_path, as_version=4)
    os.environ.clear()  # the kernel inherits the runner's environment
    os.environ.update(env)
    client, tl, save, t0 = make_client(nb, a, cwd, files, hooks, env)
    tl.write("%s NOTEBOOK_START %s sha256 %s mode %s %s=%s cwd %s\n"
             % (utc(), nb_path, manifest["sha256"][:16], a.mode, RERUN_VAR, env.get(RERUN_VAR), cwd))
    code = 0
    try:
        client.execute()
        tl.write("NOTEBOOK_DONE ok +%.0fs %s\n" % (time.time() - t0, utc()))
    except Exception as exc:
        code = 1
        tl.write("NOTEBOOK_DONE error +%.0fs %s: %s: %s\n" % (time.time() - t0, utc(), type(exc).__name__,
                                                            str(exc)[-1500:].replace("\n", " | ")))
    finally:
        save()
    return code


if __name__ == "__main__":
    sys.exit(main())
