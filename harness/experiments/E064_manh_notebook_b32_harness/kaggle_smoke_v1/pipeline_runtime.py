"""Import-safe environment selection and provenance for split-runtime workers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import shutil
import site
import sys
import tempfile
import time


APP_ROOT = Path(__file__).resolve().parent
RUNTIME_FLAGS = (
    "--trainer-python",
    "--trainer-site-root",
    "--vllm-python",
    "--vllm-site-root",
)
PACKAGES = {
    "torch": "torch",
    "triton": "triton",
    "vllm": "vllm",
    "transformers": "transformers",
    "tokenizers": "tokenizers",
    "peft": "peft",
    "unsloth": "unsloth",
    "unsloth_zoo": "unsloth_zoo",
    "accelerate": "accelerate",
    "flash_attn": "flash-attn",
    "flashinfer": "flashinfer-python",
}
REQUIRED = {
    "trainer": (
        "torch",
        "triton",
        "transformers",
        "tokenizers",
        "peft",
        "unsloth",
        "unsloth_zoo",
    ),
    "vllm": ("torch", "triton", "transformers", "tokenizers", "vllm"),
}
SOURCE_FILES = {
    "unsloth": ("models/qwen3.py", "models/llama.py", "kernels/fast_lora.py"),
    "unsloth_zoo": (
        "dataset_utils.py", "gradient_checkpointing.py", "patching_utils.py"
    ),
    "vllm": ("v1/worker/gpu_model_runner.py", "v1/worker/gpu/model_runner.py"),
}
_activated = False


def split_runtime_requested(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    return any(
        arg.split("=", 1)[0] in (*RUNTIME_FLAGS, "--runtime-preflight-only")
        for arg in argv
    )


def add_runtime_arguments(parser):
    for role in ("trainer", "vllm"):
        parser.add_argument(
            f"--{role}-python", help=f"Opt-in isolated {role} Python executable"
        )
        parser.add_argument(
            f"--{role}-site-root",
            help=f"Flat pip --target package tree for {role}; requires an isolated venv Python",
        )
    parser.add_argument(
        "--runtime-preflight-only",
        action="store_true",
        help="Check both runtimes without training or patch writes",
    )
    parser.add_argument(
        "--runtime-report-dir",
        default=os.getenv("ARC_RUNTIME_REPORT_DIR", "arc_runtime_runs"),
        help="Parent of per-run runtime manifests (default: arc_runtime_runs)",
    )


@dataclass(frozen=True)
class RuntimeSpec:
    python: str
    site_root: str | None = None

    @classmethod
    def create(cls, python=None, site_root=None):
        selected = str(python or sys.executable)
        executable = shutil.which(selected)
        if executable is None:
            raise ValueError(f"Runtime Python is not executable: {selected}")
        # Do not resolve a venv's Python symlink to its base interpreter.
        executable = os.path.abspath(executable)
        root = str(Path(site_root).expanduser().resolve()) if site_root else None
        if root is not None and not Path(root).is_dir():
            raise ValueError(f"Runtime site root is not a directory: {root}")
        return cls(executable, root)


def runtime_specs_from_args(args):
    if (
        not any(getattr(args, flag[2:].replace("-", "_")) for flag in RUNTIME_FLAGS)
        and not args.runtime_preflight_only
    ):
        return None
    return {
        role: RuntimeSpec.create(
            getattr(args, f"{role}_python"), getattr(args, f"{role}_site_root")
        )
        for role in ("trainer", "vllm")
    }


def validate_split_mode(args, *, model_config):
    """Fail explicitly for paths not covered by the first mixed-runtime gate."""
    unsupported = []
    if args.inference_backend != "vllm" or args.train_only:
        unsupported.append(
            "requires Qwen SFT with --inference-backend vllm (not --train-only)"
        )
    if args.task_rl:
        unsupported.append("--task-rl")
    if args.inference_ramp_after_adapters or args.inference_ramp_gpus:
        unsupported.append("adaptive inference ramp")
    if args.single_task_gpus != 1:
        unsupported.append("multi-GPU training of a single task")
    if args.vllm_gpus < 1:
        unsupported.append("requires at least one initial --vllm-gpus service")
    if args.deepspeed_long_context:
        unsupported.append("--deepspeed-long-context")
    if args.fill_train_to_tokens is not None:
        unsupported.append(
            "--fill-train-to-tokens (prepare the filled challenge before split launch)"
        )
    if model_config.get("model_type") != "qwen3" or model_config.get(
        "unsloth_delta_mhar"
    ):
        unsupported.append("first release supports plain Qwen3 checkpoints only")
    if unsupported:
        raise ValueError("Split runtime does not support: " + "; ".join(unsupported))


def _within(path, root):
    return Path(path).resolve().is_relative_to(Path(root).resolve())


def activate_runtime():
    """Called from sitecustomize in every descendant, before ML imports.

    For flat target installs require a clean venv, avoiding global .pth hooks
    and editable finders executing before we could sanitize sys.path.
    """
    global _activated
    if _activated or not os.environ.get("ARC_PIPELINE_ROLE"):
        return
    root = os.environ.get("ARC_PIPELINE_SITE_ROOT")
    if root:
        if sys.prefix == sys.base_prefix:
            raise RuntimeError(
                "--site-root requires an isolated venv Python, e.g. stable_venv/bin/python; not global python + PYTHONPATH"
            )
        if any(name in sys.modules for name in ("torch", "triton", "vllm", "unsloth")):
            raise RuntimeError(
                "An ML package was imported before split-runtime activation"
            )
        for entry in sys.path:
            if entry and (
                "site-packages" in Path(entry).parts
                or "dist-packages" in Path(entry).parts
            ):
                if not _within(entry, sys.prefix) and not _within(entry, root):
                    raise RuntimeError(
                        f"Global site-packages leaked into isolated runtime: {entry}"
                    )
        site.addsitedir(root)  # Also activates editable plugins and Cutlass .pth.
        sys.path[:] = [root, *(entry for entry in sys.path if entry != root)]
        for entry in sys.path:
            if entry and (
                "site-packages" in Path(entry).parts
                or "dist-packages" in Path(entry).parts
            ):
                if not _within(entry, sys.prefix) and not _within(entry, root):
                    raise RuntimeError(
                        f"A target .pth file added foreign site-packages: {entry}"
                    )
    forbidden = json.loads(os.environ.get("ARC_PIPELINE_OTHER_SITE_ROOTS", "[]"))
    for entry in sys.path:
        if entry and any(_within(entry, other) for other in forbidden):
            raise RuntimeError(
                f"Another role's package tree leaked into sys.path: {entry}"
            )
    _activated = True


def build_runtime_environment(
    spec, role, *, other_specs=(), fingerprint="probe", environment=None
):
    if role not in REQUIRED:
        raise ValueError(f"Unknown pipeline role {role}")
    source_environment = os.environ if environment is None else environment
    env = dict(source_environment)
    forbidden = [
        other.site_root
        for other in other_specs
        if other.site_root and other.site_root != spec.site_root
    ]
    for key in list(env):
        if key.startswith("ARC_PIPELINE_") or key in (
            "PYTHONHOME",
            "PYTHONPATH",
            "ARC_PATCH_SITE_ROOT",
            "ARC_RUNTIME_AUDIT_SITE_ROOT",
            "ARC_TE_CPP_RUNTIME_BOOTSTRAPPED",
            "RANK",
            "LOCAL_RANK",
            "WORLD_SIZE",
            "LOCAL_WORLD_SIZE",
            "MASTER_ADDR",
            "MASTER_PORT",
            "VIRTUAL_ENV",
        ):
            env.pop(key, None)
    for key in ("LD_LIBRARY_PATH", "LD_PRELOAD"):
        parts = env.get(key, "").split(":" if key == "LD_LIBRARY_PATH" else " ")
        for part in filter(None, parts):
            if any(_within(part, root) for root in forbidden) or (
                spec.site_root
                and (
                    "site-packages" in Path(part).parts
                    or "dist-packages" in Path(part).parts
                )
                and not _within(part, spec.site_root)
            ):
                raise ValueError(
                    f"{key} includes a foreign environment library: {part}; choose a role-safe library path"
                )
    bootstrap = APP_ROOT / "pipeline_runtime_bootstrap"
    env.update(
        {
            "PYTHONPATH": os.pathsep.join((str(bootstrap), str(APP_ROOT))),
            "PYTHONNOUSERSITE": "1",
            "PYTHONUNBUFFERED": "1",
            "ARC_PIPELINE_ROLE": role,
            "ARC_PIPELINE_SITE_ROOT": spec.site_root or "",
            "ARC_PIPELINE_OTHER_SITE_ROOTS": json.dumps(forbidden),
        }
    )
    if spec.site_root:
        env["ARC_PATCH_SITE_ROOT"] = spec.site_root
    if role == "vllm":
        env["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    cache_parent = Path(
        source_environment.get(
            "ARC_PIPELINE_CACHE_ROOT",
            os.path.join(tempfile.gettempdir(), "arc_pipeline_cache"),
        )
    )
    if str(cache_parent.resolve()).startswith(
        "/dev/shm/"
    ) or cache_parent.resolve() == Path("/dev/shm"):
        raise ValueError(
            "Use executable storage, not /dev/shm, for ARC_PIPELINE_CACHE_ROOT"
        )
    identity = hashlib.sha256(
        json.dumps(asdict(spec), sort_keys=True).encode()
    ).hexdigest()[:12]
    cache = cache_parent.resolve() / f"{identity}-{fingerprint}" / role
    for key, suffix in (
        ("TRITON_CACHE_DIR", "triton"),
        ("TORCHINDUCTOR_CACHE_DIR", "inductor"),
        ("UNSLOTH_COMPILE_LOCATION", "unsloth"),
        ("VLLM_CACHE_ROOT", "vllm"),
    ):
        env[key] = str(cache / suffix)
    if role == "trainer":
        # Zoo's patch_torch_compile normally discards the Inductor setting.
        # The fork's opt-in setting preserves this explicit scope.
        env["ARC_PIPELINE_INDUCTOR_CACHE_DIR"] = env["TORCHINDUCTOR_CACHE_DIR"]
        env["UNSLOTH_PRESERVE_INDUCTOR_CACHE_DIR"] = "1"
    return env


def verify_trainer_cache_support():
    """Read-only preflight; install the updated Zoo fork, never hotpatch it."""
    from patch_environment import require_file_within, resolve_package_dir

    target = require_file_within(
        resolve_package_dir("unsloth_zoo"), "patching_utils.py"
    )
    if "UNSLOTH_PRESERVE_INDUCTOR_CACHE_DIR" not in target.read_text():
        raise RuntimeError(
            "Split-runtime training requires the updated Unsloth Zoo fork with "
            f"UNSLOTH_PRESERVE_INDUCTOR_CACHE_DIR support; upgrade {target.parent}"
        )


def atomic_json(path, payload, *, private=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        os.chmod(temporary, 0o600 if private else 0o644)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def model_manifest(model_path):
    root = Path(model_path).resolve()
    metadata = (
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "model.safetensors.index.json",
    )
    return {
        "path": str(root),
        "metadata_sha256": {
            name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in metadata
            if (root / name).is_file()
        },
        # Both roles read this same immutable input directory. Do not reread
        # gigabytes of weights just to report file identity at every launch.
        "weight_files": {
            path.name: {
                "bytes": path.stat().st_size,
                "mtime_ns": path.stat().st_mtime_ns,
            }
            for path in sorted(root.glob("*.safetensors"))
        },
    }


def collect_runtime_manifest(role, *, require_packages=True):
    requested_cache = os.environ.get("ARC_PIPELINE_INDUCTOR_CACHE_DIR")
    if role == "trainer" and requested_cache:
        if os.environ.get("TORCHINDUCTOR_CACHE_DIR") != requested_cache:
            raise RuntimeError("Trainer Inductor cache escaped its split-runtime scope")
    modules = {}
    for name, distribution in PACKAGES.items():
        loaded = sys.modules.get(name)
        spec = importlib.util.find_spec(name) if loaded is None else None
        origin = (
            getattr(loaded, "__file__", None)
            if loaded is not None
            else getattr(spec, "origin", None)
        )
        try:
            version = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            version = None
        record = {"path": origin, "version": version, "loaded": loaded is not None}
        if origin and Path(origin).is_file():
            record["sha256"] = hashlib.sha256(Path(origin).read_bytes()).hexdigest()
            record["sources"] = {
                relative: hashlib.sha256(
                    (Path(origin).parent / relative).read_bytes()
                ).hexdigest()
                for relative in SOURCE_FILES.get(name, ())
                if (Path(origin).parent / relative).is_file()
            }
        modules[name] = record
        if require_packages and name in REQUIRED[role] and not origin:
            raise RuntimeError(f"{role} runtime cannot locate required package {name}")
        root = os.environ.get("ARC_PIPELINE_SITE_ROOT")
        if (
            root
            and origin
            and name in ("torch", "triton", "transformers", "tokenizers", "vllm")
        ):
            if not _within(origin, root):
                raise RuntimeError(
                    f"{name} resolved outside selected site root: {origin}"
                )
    maps = Path("/proc/self/maps")
    libraries = (
        sorted(
            {
                line.split()[-1]
                for line in maps.read_text().splitlines()
                if ".so" in line
                and any(
                    word in line
                    for word in (
                        "cuda",
                        "cublas",
                        "cudnn",
                        "nccl",
                        "libstdc++",
                        "torch",
                    )
                )
            }
        )
        if maps.exists()
        else []
    )
    torch = sys.modules.get("torch")
    if torch and modules["torch"]["version"]:
        if (
            str(torch.__version__).split("+", 1)[0]
            != modules["torch"]["version"].split("+", 1)[0]
        ):
            raise RuntimeError(
                "Imported Torch version differs from installed metadata; check for overlapping installations"
            )
    return {
        "role": role,
        "pid": os.getpid(),
        "python": sys.executable,
        "python_version": list(sys.version_info[:3]),
        "sys_path": sys.path,
        "site_root": os.environ.get("ARC_PIPELINE_SITE_ROOT") or None,
        "modules": modules,
        "loaded_libraries": libraries,
        "torch_runtime_version": str(torch.__version__) if torch else None,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "timestamp": time.time(),
        "caches": {
            key: os.environ.get(key)
            for key in (
                "TRITON_CACHE_DIR",
                "TORCHINDUCTOR_CACHE_DIR",
                "UNSLOTH_COMPILE_LOCATION",
                "VLLM_CACHE_ROOT",
            )
        },
    }


def runtime_fingerprint(manifest):
    identity = {key: manifest[key] for key in ("python", "python_version", "site_root")}
    identity["modules"] = {
        name: {key: value for key, value in item.items() if key != "loaded"}
        for name, item in manifest["modules"].items()
    }
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[
        :16
    ]


def verify_runtime_manifest(actual, expected):
    for key in ("python", "python_version", "site_root"):
        if actual[key] != expected[key]:
            raise RuntimeError(
                f"Runtime drift in {key}: {actual[key]!r} != {expected[key]!r}"
            )
    for name in REQUIRED[expected["role"]]:
        for key in ("path", "version", "sha256", "sources"):
            if actual["modules"][name].get(key) != expected["modules"][name].get(key):
                raise RuntimeError(
                    f"Runtime drift in {name}.{key}: imports/source changed after preflight"
                )


class RuntimeManifestWorkerExtension:
    """Named RPC available without vLLM's insecure callable serialization."""

    def arc_runtime_manifest(self):
        return collect_runtime_manifest("vllm")


def configure_engine_runtime(engine_kwargs):
    if not os.environ.get("ARC_PIPELINE_EXPECTED_MANIFEST"):
        return
    extension = "pipeline_runtime.RuntimeManifestWorkerExtension"
    previous = engine_kwargs.get("worker_extension_cls")
    if previous and previous != extension:
        raise RuntimeError("Split runtime cannot replace another vLLM worker extension")
    engine_kwargs["worker_extension_cls"] = extension


def report_engine_runtime(engine):
    expected_path = os.environ.get("ARC_PIPELINE_EXPECTED_MANIFEST")
    if not expected_path:
        return
    expected = json.loads(Path(expected_path).read_text())
    results = engine.collective_rpc("arc_runtime_manifest", timeout=30)
    if not results:
        raise RuntimeError("vLLM returned no EngineCore runtime manifest")
    for result in results:
        verify_runtime_manifest(result, expected)
        atomic_json(
            Path(os.environ["ARC_PIPELINE_REPORT_DIR"])
            / f"engine-{result['pid']}.json",
            result,
        )
    print(
        f"[split runtime] verified {len(results)} EngineCore runtime(s): {expected['python']}",
        flush=True,
    )


def attach_training_runtime_report(trainer, task_key):
    """Observe the real Trainer schedule without changing its configuration."""
    if not os.environ.get("ARC_PIPELINE_REPORT_DIR"):
        return
    from transformers import TrainerCallback

    class RuntimeReport(TrainerCallback):
        def on_train_begin(self, args, state, control, **kwargs):
            manifest = collect_runtime_manifest("trainer")
            expected = json.loads(
                Path(os.environ["ARC_PIPELINE_EXPECTED_MANIFEST"]).read_text()
            )
            verify_runtime_manifest(manifest, expected)
            manifest["training"] = {
                "task": task_key,
                "max_steps": state.max_steps,
                "warmup_steps": args.get_warmup_steps(state.max_steps),
                "learning_rate": args.learning_rate,
                "optimizer": str(args.optim),
                "bf16": args.bf16,
                "fp16": args.fp16,
                "gradient_accumulation_steps": args.gradient_accumulation_steps,
                "per_device_train_batch_size": args.per_device_train_batch_size,
            }
            filename = f"trainer-{os.getpid()}-{task_key}.json"
            atomic_json(
                Path(os.environ["ARC_PIPELINE_REPORT_DIR"]) / filename, manifest
            )

    trainer.add_callback(RuntimeReport())
