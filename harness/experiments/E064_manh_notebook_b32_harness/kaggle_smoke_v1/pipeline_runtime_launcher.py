"""Opt-in subprocess launcher; algorithms and adapter format remain unchanged."""

from __future__ import annotations

from dataclasses import asdict
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time

from pipeline_runtime import (
    APP_ROOT,
    RuntimeSpec,
    atomic_json,
    build_runtime_environment,
    model_manifest,
    runtime_fingerprint,
)
from pipeline_runtime_ipc import encode


def create_report_directory(parent):
    parent = Path(parent).expanduser().resolve()
    parent.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix="run-", dir=parent))


def prepare_runtimes(specs, report_dir, *, check_only=False, environment=None):
    """Run serialized source patch checks/applications before concurrent imports."""
    prepared = {}
    for role, spec in specs.items():
        env = build_runtime_environment(
            spec, role, other_specs=specs.values(), environment=environment
        )
        env["CUDA_VISIBLE_DEVICES"] = ""  # This probe never claims a training GPU.
        report = Path(report_dir) / f"{role}-preflight.json"
        log = Path(report_dir) / f"{role}-preflight.log"
        command = [
            spec.python,
            str(APP_ROOT / "pipeline_runtime_worker.py"),
            "--probe",
            "--report",
            str(report),
        ]
        if check_only:
            command.append("--check-only")
        print(
            f"[split runtime] checking {role}: {spec.python}; site_root={spec.site_root}; log={log}",
            flush=True,
        )
        with log.open("w") as handle:
            process = subprocess.Popen(
                command,
                env=env,
                stdout=handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                code = process.wait(timeout=180)
            except BaseException:
                stop_processes([process])
                raise
        if code:
            raise RuntimeError(
                f"{role} runtime preflight exited {code}; {log}\n{log.read_text()[-12000:]}"
            )
        manifest = json.loads(report.read_text())
        if manifest["python_version"][:2] != list(sys.version_info[:2]):
            raise ValueError(
                f"{role} Python minor {manifest['python_version'][:2]} must match coordinator {list(sys.version_info[:2])}"
            )
        prepared[role] = {
            "spec": asdict(spec),
            "manifest": manifest,
            "manifest_path": str(report),
            "environment": build_runtime_environment(
                spec,
                role,
                other_specs=specs.values(),
                fingerprint=runtime_fingerprint(manifest),
                environment=environment,
            ),
        }
        summary = ", ".join(
            f"{key}={manifest['modules'][key]['version']}"
            for key in ("torch", "triton", "transformers", "peft", "vllm")
        )
        print(f"[split runtime] {role}: {summary}", flush=True)
    differences = []
    controlled = (
        ("transformers", "tokenizers", "peft", "unsloth", "unsloth_zoo")
        if {"trainer", "vllm"} <= prepared.keys()
        else ()
    )
    for name in controlled:
        left = prepared["trainer"]["manifest"]["modules"][name]
        right = prepared["vllm"]["manifest"]["modules"][name]
        if any(
            left.get(key) != right.get(key) for key in ("version", "sha256", "sources")
        ):
            differences.append(name)
    if differences:
        print(
            "[split runtime] CONTROL WARNING: role source/version differences in "
            + ", ".join(differences)
            + "; this is not a Torch-only comparison",
            flush=True,
        )
    atomic_json(
        Path(report_dir) / "preflight.json",
        {
            "check_only": check_only,
            "control_differences": differences,
            "runtimes": {
                role: {"spec": item["spec"], "manifest": item["manifest"]}
                for role, item in prepared.items()
            },
        },
    )
    return prepared


def add_replay_runtime_arguments(parser):
    parser.add_argument("--vllm-python", help="Replay using this Python executable")
    parser.add_argument(
        "--vllm-site-root", help="Optional flat target in an isolated venv"
    )
    parser.add_argument("--runtime-report-dir", default="arc_runtime_runs")
    parser.add_argument("--runtime-preflight-only", action="store_true")


def maybe_reexec_replay(script, argv=None):
    """Select replay's environment before its first Torch/application import."""
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--help" in argv or "-h" in argv:
        return
    parser = argparse.ArgumentParser(add_help=False)
    add_replay_runtime_arguments(parser)
    args, remaining = parser.parse_known_args(argv)
    if not (args.vllm_python or args.vllm_site_root or args.runtime_preflight_only):
        return
    spec = RuntimeSpec.create(args.vllm_python, args.vllm_site_root)
    report_dir = create_report_directory(args.runtime_report_dir)
    prepared = prepare_runtimes(
        {"vllm": spec}, report_dir, check_only=args.runtime_preflight_only
    )["vllm"]
    if args.runtime_preflight_only:
        raise SystemExit(0)
    env = dict(prepared["environment"])
    env["ARC_PIPELINE_REPORT_DIR"] = str(report_dir)
    env["ARC_PIPELINE_EXPECTED_MANIFEST"] = prepared["manifest_path"]
    os.execve(spec.python, [spec.python, str(Path(script).resolve()), *remaining], env)


def select_gpu_token(index, environment):
    value = environment.get("CUDA_VISIBLE_DEVICES")
    devices = (
        [part.strip() for part in value.split(",") if part.strip()] if value else None
    )
    if devices is not None and not 0 <= index < len(devices):
        raise ValueError(f"GPU {index} is outside CUDA_VISIBLE_DEVICES={value!r}")
    return devices[index] if devices is not None else str(index)


def _signal_group(process, sig):
    # Each tracked child was launched in its own new session; never signal
    # the notebook, coordinator, or an unrelated process group.
    try:
        os.killpg(process.pid, sig)
    except ProcessLookupError:
        pass


def stop_processes(processes, *, grace=5.0):
    for process in processes:
        _signal_group(process, signal.SIGTERM)
    deadline = time.monotonic() + grace
    while (
        any(process.poll() is None for process in processes)
        and time.monotonic() < deadline
    ):
        time.sleep(0.05)
    for process in processes:
        # Includes surviving EngineCore descendants after their frontend exits.
        _signal_group(process, signal.SIGKILL)
        process.wait(timeout=5)


class SplitRuntimeLauncher:
    def __init__(self, prepared, broker, report_dir, *, environment=None):
        self.prepared = prepared
        self.broker = broker
        self.report_dir = Path(report_dir)
        self.environment = dict(os.environ if environment is None else environment)
        self.credentials = self.report_dir / "ipc-credentials.json"
        atomic_json(
            self.credentials,
            {"address": broker.address, "authkey": broker.authkey.hex()},
            private=True,
        )
        self.children = []
        self.events = []

    def launch(
        self,
        role,
        rank,
        gpu,
        kwargs,
        *,
        target="starter:vllm_pipeline_worker",
        phase="initial",
    ):
        runtime = self.prepared[role]
        name = f"{role}-{rank}-{len(self.children)}"
        job_path = self.report_dir / f"{name}.launch.json"
        error_path = self.report_dir / f"{name}.error.json"
        exit_path = self.report_dir / f"{name}.exit.json"
        atomic_json(
            job_path,
            {
                "schema_version": 1,
                "target": target,
                "role": role,
                "rank": rank,
                "kwargs": encode(kwargs),
                "credentials": str(self.credentials),
                "parent_pid": os.getpid(),
                "error_path": str(error_path),
                "exit_path": str(exit_path),
            },
            private=True,
        )
        env = dict(runtime["environment"])
        # Reflect model/challenge settings resolved by starter after preflight.
        for key in (
            "ARC_MODEL_PATH",
            "ARC_CHALLENGE_PATH",
            "VLLM_BATCH_INVARIANT",
            "VLLM_PLUGINS",
        ):
            if key in self.environment:
                env[key] = self.environment[key]
        env.update(
            {
                "CUDA_VISIBLE_DEVICES": select_gpu_token(gpu, self.environment),
                "ARC_PIPELINE_SELECTED_GPU": str(gpu),
                "ARC_PIPELINE_REPORT_DIR": str(self.report_dir),
                "ARC_PIPELINE_EXPECTED_MANIFEST": runtime["manifest_path"],
            }
        )
        process = subprocess.Popen(
            [
                runtime["spec"]["python"],
                str(APP_ROOT / "pipeline_runtime_worker.py"),
                "--job",
                str(job_path),
            ],
            env=env,
            start_new_session=True,
        )
        record = {
            "process": process,
            "role": role,
            "rank": rank,
            "gpu": gpu,
            "phase": phase,
            "error": error_path,
            "exit": exit_path,
        }
        self.children.append(record)
        self.events.append(
            {
                "event": "started",
                "pid": process.pid,
                "role": role,
                "rank": rank,
                "gpu": gpu,
                "phase": phase,
                "time": time.time(),
            }
        )
        print(
            f"[split runtime] {role} rank={rank} GPU={gpu} pid={process.pid} phase={phase}: {runtime['spec']['python']}",
            flush=True,
        )
        return record

    def run(
        self,
        kwargs,
        *,
        max_gpu_count=4,
        target="starter:vllm_pipeline_worker",
        poll_seconds=0.1,
    ):
        trainers = int(kwargs["trainer_count"])
        initial = int(kwargs["vllm_gpu_count"])
        scale = bool(kwargs["post_training_scale"])
        running = []
        finished_trainers = set()
        future_started = not scale
        succeeded = False
        abort_since = None
        atomic_json(
            self.report_dir / "configuration.json",
            {
                "model": model_manifest(kwargs["model_path"]),
                "worker_options": kwargs["worker_options"],
                "inference_options": kwargs["inference_options"],
                "adapter_ids": kwargs["adapter_ids"],
                "trainer_count": trainers,
                "initial_services": initial,
                "post_training_scale": scale,
                "end_time": kwargs["end_time"],
            },
        )
        try:
            # Starting the serving frontends first does not change the task
            # queue. Their existing startup lock still serializes engine loads.
            for index in range(initial):
                running.append(
                    self.launch(
                        "vllm",
                        trainers + index,
                        max_gpu_count - initial + index,
                        kwargs,
                        target=target,
                    )
                )
            trainer_kwargs = dict(kwargs, defer_post_training_service=scale)
            for rank in range(trainers):
                running.append(
                    self.launch("trainer", rank, rank, trainer_kwargs, target=target)
                )
            while running or not future_started:
                if not self.broker.is_alive():
                    raise RuntimeError(
                        "Split runtime control-state service stopped unexpectedly"
                    )
                for record in tuple(running):
                    code = record["process"].poll()
                    if code is None:
                        continue
                    record["process"].wait()
                    if code != 0 or not record["exit"].exists():
                        detail = (
                            record["error"].read_text()
                            if record["error"].exists()
                            else "No Python traceback (process may have been signalled)"
                        )
                        raise RuntimeError(
                            f"Split runtime {record['role']} rank={record['rank']} pid={record['process'].pid} "
                            f"exited {code}; python={self.prepared[record['role']]['spec']['python']}\n{detail}"
                        )
                    # Reap any residual descendants before reusing this GPU.
                    stop_processes([record["process"]], grace=0)
                    record["cleaned"] = True
                    running.remove(record)
                    if record["role"] == "trainer":
                        finished_trainers.add(record["rank"])
                    self.events.append(
                        {
                            "event": "completed",
                            "pid": record["process"].pid,
                            "role": record["role"],
                            "rank": record["rank"],
                            "time": time.time(),
                        }
                    )
                if not future_started and len(finished_trainers) == trainers:
                    if (
                        kwargs["training_complete_event"].is_set()
                        and kwargs["post_training_rebalance_event"].is_set()
                    ):
                        future_started = True
                        for rank in range(trainers):
                            service = initial + rank
                            with kwargs["service_load_lock"]:
                                pending = int(kwargs["service_outstanding"][service])
                            if pending == 0:
                                self.events.append(
                                    {
                                        "event": "scale_out_skipped_empty",
                                        "rank": rank,
                                        "service": service,
                                    }
                                )
                                continue
                            running.append(
                                self.launch(
                                    "vllm",
                                    rank,
                                    rank,
                                    dict(
                                        kwargs,
                                        service_only=True,
                                        defer_post_training_service=False,
                                    ),
                                    target=target,
                                    phase="post-training",
                                )
                            )
                    elif not running:
                        raise RuntimeError(
                            "All children exited before post-training rebalance completed"
                        )
                if kwargs["abort_event"].is_set():
                    # Let the failing worker publish its original traceback;
                    # do not replace it with a sibling's generic abort message.
                    failures = [
                        record for record in self.children if record["error"].exists()
                    ]
                    if failures:
                        record = failures[0]
                        raise RuntimeError(
                            f"Split runtime {record['role']} rank={record['rank']} failed:\n{record['error'].read_text()}"
                        )
                    if abort_since is None:
                        abort_since = time.monotonic()
                    if not running or time.monotonic() - abort_since >= 3:
                        raise RuntimeError(
                            "Split runtime aborted by trainer/service; see worker logs"
                        )
                if time.time() > float(kwargs["end_time"]) + 30:
                    raise TimeoutError(
                        "Split runtime children did not exit within 30s of the global deadline"
                    )
                if running or not future_started:
                    time.sleep(poll_seconds)
            succeeded = True
        finally:
            if not succeeded:
                try:
                    kwargs["abort_event"].set()
                except (EOFError, OSError):
                    pass  # Preserve the error if the broker itself died.
            stop_processes(
                [
                    record["process"]
                    for record in self.children
                    if not record.get("cleaned")
                ]
            )
            atomic_json(
                self.report_dir / "lifecycle.json",
                {"succeeded": succeeded, "events": self.events},
            )
            self.credentials.unlink(missing_ok=True)
