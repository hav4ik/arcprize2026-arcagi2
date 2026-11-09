"""Vu's B32 tool harness on our supervised dsv41 server (--qwen38-variant dsv4-vllm --dsv41-harness havu).

His code lives in havu/ at his repo's paths (havu/README.md): kaggle/dsv41_tools/solver_dsv41.py (B32-M + S25-S28 +
E066) and the evals/ modules it imports (run_deepseek_tools, train_check, auto_repair, arc_harness, ...); edit it there.
The pipeline starts the dsv4-vllm server from a restart supervisor thread (qwen38_server.run_supervisor) and runs his
solver from that source as an external server (S25) that follows the supervisor's status file through restarts (S26) and
re-queues the attempts a dying engine answers (E066), with his B32-M settings. His submission is validated and copied to
--final-submission whenever it changes. --input-mode text-image runs his E066 image harness (ARC_GRID_FORMAT=text+image:
the JSON prompt plus one token-aligned image per grid, on the chat path) on a server that keeps its vision encoder.
"""

import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import threading
import time

import dsv41_vllm
import qwen38_server


HAVU = Path(__file__).with_name("havu")
SOLVER = Path("kaggle/dsv41_tools/solver_dsv41.py")
CODE = (str(SOLVER), "evals/*.py", "evals/*.json", "evals/arc_harness/*.py", "notes/prompts/*.txt")
# B32-M (his dsv41-tools-submit-b32m.ipynb); ARC_WORKERS and ARC_CONTEXT_LIMIT follow --concurrency and --context.
B32M = {"ARC_MULTI_TEST": "1", "ARC_MULTI_BUDGET": "0.5", "ARC_KV_ADMIT_MAX": "0.85", "ARC_KV_ADMIT_MIN_WORKERS": "16",
        "ARC_AUTO_REPAIR": "1", "ARC_ADMIT_MAX_WAITING": "1", "ARC_METRICS_EVERY_S": "30"}
PRIVATE = ("TOKEN", "KEY", "SECRET", "TILES", "PASSWORD")
ATTEMPTS = ("attempt_1", "attempt_2")


def add_arguments(parser):
    g = parser.add_argument_group("Vu's B32 tool harness on our dsv41 server (--qwen38-variant dsv4-vllm)")
    g.add_argument("--dsv41-harness", choices=("repl", "havu"), default="repl",
                   help="repl: our REPL-TIR solver; havu: Vu's B32 solver (--havu-dir) on our supervised server.")
    g.add_argument("--havu-dir", type=Path, default=HAVU,
                   help="His harness source: kaggle_submit/havu (default) or a checkout's havu/ folder with S26, S27 "
                        "and E066.")
    g.add_argument("--havu-env", action="append", default=[], metavar="KEY=VALUE",
                   help="Solver environment override (e.g. ARC_REASONING_EFFORT=50), applied last; repeatable.")
    g.add_argument("--havu-restarts", type=int, default=3, help="Server restarts after an exit or a stall.")
    g.add_argument("--havu-stall-s", type=float, default=1500.0,
                   help="Restart a server that is up but makes no token for this long while requests run (0: never).")


def parse_env(items):
    env = {}
    for item in items:
        key, sep, value = item.partition("=")
        if not sep or not re.fullmatch(r"[A-Z_][A-Z0-9_]*", key):
            raise ValueError(f"--havu-env takes KEY=VALUE, not {item!r}")
        env[key] = value
    return env


DEAD_ENGINE = "The server had an error while processing your request"  # vLLM's EngineDeadError.CLIENT_MESSAGE


def check_source(havu, images=False):
    """His solver and evals/ under havu; the solver must follow a supervised server (S26) from this layout (S27) and
    re-queue the attempts a dying engine answers with vLLM's 500 (E066); with images, it needs his image switch."""
    solver = Path(havu) / SOLVER
    if not solver.is_file() or not (Path(havu) / "evals" / "run_deepseek_tools.py").is_file():
        raise ValueError(f"--havu-dir {havu}: needs {SOLVER} and evals/run_deepseek_tools.py (his harness source)")
    text = solver.read_text()
    for marker, note in (("ARC_EXTERNAL_SERVER_STATUS", "S26: follow a supervised server"),
                         ("EVALS =", "S27: run from the havu/ layout"),
                         (DEAD_ENGINE, "E066: re-queue the attempts a dying vLLM engine answers")):
        if marker not in text:
            raise ValueError(f"{solver} has no {marker!r} ({note})")
    if images and ("ARC_GRID_FORMAT" not in text
                   or "text+image" not in (Path(havu) / "evals" / "run_deepseek_tools.py").read_text()):
        raise ValueError(f"--havu-dir {havu}: --input-mode text-image needs his E066 image switch "
                         "(ARC_GRID_FORMAT) and E058 harness (--grid-format text+image)")


def validate_options(args):
    if getattr(args, "dsv41_harness", "repl") != "havu":
        return
    if args.qwen38_variant != "dsv4-vllm":
        raise ValueError("--dsv41-harness havu runs on --qwen38-variant dsv4-vllm")
    if getattr(args, "qwen38_recipe", "manual") != "manual":
        raise ValueError("--qwen38-recipe sets our REPL solver; --dsv41-harness havu runs Vu's: drop the recipe")
    if args.solve_mode != "direct" or args.task_token_budget is not None:
        raise ValueError("--dsv41-harness havu runs his tool turns and budgets: drop --solve-mode and "
                         "--task-token-budget (his ARC_* settings go in --havu-env)")
    for name in ("primary_submission", "routing_manifest", "resume_from", "base_url"):
        if getattr(args, name, None):
            raise ValueError(f"--{name.replace('_', '-')} is not supported with --dsv41-harness havu")
    if args.havu_restarts < 0 or args.havu_stall_s < 0:
        raise ValueError("--havu-restarts and --havu-stall-s must be non-negative")
    args.havu_dir = args.havu_dir.resolve()
    images = args.input_mode == "text-image"
    check_source(args.havu_dir, images)
    args.havu_overrides = parse_env(args.havu_env)
    grid = grid_format(args)
    if args.havu_overrides.get("ARC_GRID_FORMAT", grid) != grid:
        raise ValueError(f"--input-mode {args.input_mode} runs ARC_GRID_FORMAT={grid}: drop that --havu-env")
    if images and args.vision_max_images < 32:
        raise ValueError("--dsv41-harness havu --input-mode text-image sends one image per grid (up to ~24 a task): "
                         "use --vision-max-images 32")
    if images:
        dsv41_vllm.read_metadata(args.dsv41_vllm_model)  # a checkpoint with the vision tower


def grid_format(args):
    return "text+image" if args.input_mode == "text-image" else "json"


def code_record(havu):
    """sha256 of each file of his harness (CODE) and of the whole set: which code a run used."""
    files = {p.relative_to(havu).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
             for pattern in CODE for p in sorted(Path(havu).glob(pattern)) if p.is_file()}
    digest = hashlib.sha256("".join(f"{path} {sha}\n" for path, sha in sorted(files.items())).encode()).hexdigest()
    return digest, files


def deadlines(total):
    """His solver's stop-new, force and hard-stop times (s from ARC_T0) for a run of total seconds, and when we stop
    it (after his hard stop): the B32 offsets (39000/40200/41400) for total 42300, scaled down for short runs."""
    hard = total - min(900.0, 0.05 * total)
    return dict(ARC_STOP_NEW_S=round(hard - min(2400.0, 0.2 * total)),
                ARC_FORCE_AT_S=round(hard - min(1200.0, 0.1 * total)),
                ARC_HARD_STOP_S=round(hard)), total - min(60.0, 0.02 * total)


def solver_env(args, t0, total, keys, task_count):
    """The solver's settings (ours, then --havu-env); the full environment is os.environ plus these."""
    root = args.output_root.resolve()
    env = {"ARC_T0": repr(t0), "ARC_KIT": str((args.havu_dir / SOLVER).parent), "ARC_OUT": str(root / "havu"),
           "ARC_CHALLENGES": str(Path(args.challenges).resolve()), "ARC_MODEL_DIR": str(args.dsv41_vllm_model),
           "ARC_STAGE_DONE": "", "ARC_TOOL_PYTHON": sys.executable, "ARC_PORT": str(args.qwen38_port),
           "ARC_EXTERNAL_SERVER": f"http://127.0.0.1:{args.qwen38_port}/v1",
           "ARC_EXTERNAL_SERVER_LOG": str(root / "server" / "server.log"),
           "ARC_EXTERNAL_SERVER_STATUS": str(root / "supervisor.json"),
           "ARC_MAX_SERVER_RESTARTS": str(args.havu_restarts),
           "ARC_SERVER_UP_TIMEOUT_S": str(round(args.health_timeout_seconds + 300)),
           "ARC_WORKERS": str(args.concurrency), "ARC_CONTEXT_LIMIT": str(args.context), **B32M,
           "ARC_GRID_FORMAT": grid_format(args),
           **{k: str(v) for k, v in deadlines(total)[0].items()},
           "ARC_TASK_IDS": ",".join(keys) if len(keys) < task_count else "",
           "CUDA_VISIBLE_DEVICES": ""}  # neither the solver nor its sandbox needs a GPU
    if args.solutions:
        env["ARC_SOLUTIONS"] = str(Path(args.solutions).resolve())
    env.update(args.havu_overrides)
    return env


def check_python(env, havu):
    """Before the server: his solver compiles and its evals/ modules import (an edit's error shows here, not after the
    server load); his sandbox (ARC_TOOL_PYTHON -I) imports numpy and scipy.ndimage."""
    full, solver = dict(os.environ, **env), str(havu / SOLVER)
    imports = (f"import sys; sys.path[:0] = [{str(havu / 'evals')!r}]; compile(open({solver!r}).read(), {solver!r}, "
               "'exec'); import openai, run_deepseek_tools, train_check, auto_repair, arc_harness.executor")
    for command in ([sys.executable, "-c", imports],
                    [env["ARC_TOOL_PYTHON"], "-I", "-c", "import numpy, scipy.ndimage"]):
        result = subprocess.run(command, env=full, cwd=env["ARC_OUT"], stdin=subprocess.DEVNULL, capture_output=True,
                                text=True, timeout=300)
        if result.returncode:
            raise RuntimeError(f"{command[0]} cannot run {command[-1]!r}: {result.stderr.strip()[-500:]}")


def fallback(queries, validate):
    """His fallback: each test input as both attempts ([[0]] where that does not validate)."""
    sub = {key: [{name: test["input"] for name in ATTEMPTS} for test in task["test"]] for key, task in queries.items()}
    try:
        validate(sub)
        return sub
    except ValueError:
        return {key: [{name: [[0]] for name in ATTEMPTS} for _ in task["test"]] for key, task in queries.items()}


def stop_solver(process, grace=30.0):
    """SIGTERM to its group (his handler writes the submission and exits), SIGKILL after grace, then its session."""
    if process is None:
        return
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(grace)
        except ProcessLookupError:
            pass
        except subprocess.TimeoutExpired:
            print(f"*** havu: solver still running {grace:.0f}s after SIGTERM; killing it", flush=True)
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
    qwen38_server.kill_pids(qwen38_server.session_pids(process.pid), grace=5.0)


def run(args, queries, keys, deadline_epoch, *, validate, write, score):
    """Fallback submission first; then the supervised server and his solver (from args.havu_dir) until it exits or the
    deadline is 60 s away (deadlines). Returns the summary (also output_root/havu_summary.json)."""
    root, t0 = args.output_root.resolve(), time.time()
    total = max(0.0, deadline_epoch - t0)
    write(args.final_submission, fallback(queries, validate))
    summary = dict(harness="havu", status="preparing", tasks=len(keys), havu_dir=str(args.havu_dir),
                   total_s=round(total))
    if not keys:
        summary["status"] = "complete"
        write(root / "havu_summary.json", summary)
        print("*** havu: no task selected; fallback submission only", flush=True)
        return summary
    status_path = root / "supervisor.json"
    (root / "havu").mkdir(parents=True, exist_ok=True)
    summary["code_sha256"], summary["code"] = code_record(args.havu_dir)
    env = solver_env(args, t0, total, keys, len(queries))
    times, kill_s = deadlines(total)
    summary.update(env={k: v for k, v in env.items() if not any(s in k.upper() for s in PRIVATE)}, kill_s=round(kill_s))
    check_python(env, args.havu_dir)
    print(f"*** havu: {len(keys)} tasks, {args.concurrency} workers, context {args.context}; {times}, "
          f"code {args.havu_dir} ({summary['code_sha256'][:12]}, {len(summary['code'])} files)", flush=True)

    def start(n):
        name = f"server_restart{n}" if n else "server"
        return dsv41_vllm.launch(args, name)[0], root / name / "server.log"

    stop, supervisor = threading.Event(), {}

    def supervise():
        try:
            supervisor["code"] = qwen38_server.run_supervisor(
                start, port=args.qwen38_port, model=dsv41_vllm.SERVED_NAME, restarts=args.havu_restarts,
                status_path=status_path, deadline=t0 + times["ARC_STOP_NEW_S"], up_timeout=args.health_timeout_seconds,
                stall_s=args.havu_stall_s, stop=stop, poll_s=float(os.environ.get("ARC_SUPERVISOR_POLL_S", 5)),
                backoff_s=float(os.environ.get("ARC_SUPERVISOR_BACKOFF_S", 10)))
        except BaseException as exc:
            supervisor["error"] = f"{type(exc).__name__}: {exc}"

    sub_path, seen, synced = root / "havu" / "submission.json", None, 0

    def sync():
        nonlocal seen, synced
        try:
            stat = sub_path.stat()
            if (stat.st_mtime_ns, stat.st_size) == seen:
                return
            sub = json.loads(sub_path.read_text())
            validate(sub)
        except FileNotFoundError:
            return
        except (OSError, ValueError) as exc:
            print(f"*** havu: WARNING his submission is not valid yet ({exc}); keeping the last copy", flush=True)
            return
        write(args.final_submission, sub)
        seen, synced = (stat.st_mtime_ns, stat.st_size), synced + 1

    qwen38_server.stop_previous_supervisor(args.qwen38_port)
    thread = threading.Thread(target=supervise, name="dsv41-supervisor", daemon=True)
    thread.start()
    solver, code = None, None
    try:
        solver = subprocess.Popen([sys.executable, "-u", str(args.havu_dir / SOLVER)], env=dict(os.environ, **env),
                                  cwd=root / "havu", stdin=subprocess.DEVNULL, start_new_session=True)
        summary["status"] = "running"
        while (code := solver.poll()) is None:
            sync()
            if time.time() >= t0 + kill_s:
                summary["status"] = "deadline"
                print(f"*** havu: {total - kill_s:.0f} s to the deadline; stopping the solver", flush=True)
                break
            try:
                solver.wait(min(5.0, max(0.1, t0 + kill_s - time.time())))
            except subprocess.TimeoutExpired:
                pass
    except KeyboardInterrupt:  # SIGTERM too (qwen38_pipeline.main); his final write still happens below
        summary["status"] = "interrupted"
    except BaseException as exc:
        summary["status"] = f"error: {type(exc).__name__}: {exc}"
        raise
    finally:
        stop_solver(solver)
        sync()
        stop.set()
        thread.join(300)
        final = json.loads(Path(args.final_submission).read_text())
        sup = qwen38_server.read_status(status_path)
        if summary["status"] == "running":
            summary["status"] = ("complete" if code == 0 and sup.get("state") != "failed" else
                                 "server_failed" if sup.get("state") == "failed" else f"solver_exit_{code}")
        try:
            summary["score"] = score(final)
        except Exception as exc:  # scoring is a report only
            summary["score"] = f"{type(exc).__name__}: {exc}"
        summary.update(solver_exit=solver and solver.returncode, synced=synced, supervisor=sup,
                       supervisor_thread=supervisor, elapsed_s=round(time.time() - t0))
        write(root / "havu_summary.json", summary)
        print(f"*** havu {summary['status']}: submission {args.final_submission} ({synced} updates); "
              f"server starts {len(sup.get('history', []))}, supervisor {sup.get('state')}; score {summary['score']}",
              flush=True)
    return summary

