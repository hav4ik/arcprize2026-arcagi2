"""Copy an offline checkpoint to local storage before starting CUDA workers.

Use a new destination, preferably under /dev/shm when RAM permits. No tensors,
config fields, or quantization metadata are changed. An interrupted copy is
left under a unique .partial directory, never published as the destination.
"""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import time


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def stage_checkpoint(source, destination):
    source, destination = Path(source).resolve(), Path(destination).absolute()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Use a new staging destination: {destination}")
    if source == destination or source in destination.parents:
        raise ValueError("Staging destination must be outside the source checkpoint")
    if not (source / "config.json").is_file() or not any(source.glob("*.safetensors")):
        raise ValueError(f"Expected an offline Hugging Face safetensors checkpoint: {source}")
    files = []
    for path in sorted(source.rglob("*")):
        if path.is_symlink() or not (path.is_file() or path.is_dir()):
            raise ValueError(f"Only regular checkpoint files/directories are supported: {path}")
        if path.is_file():
            files.append(path)
    total = sum(path.stat().st_size for path in files)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(destination.parent).free < total + 256 * 1024**2:
        raise OSError(f"Insufficient staging space for {total} bytes plus 256 MiB reserve")
    partial = Path(tempfile.mkdtemp(prefix=destination.name + ".partial-", dir=destination.parent))
    started = time.monotonic()
    manifest = {"source": str(source), "destination": str(destination), "files": {}}
    try:
        for path in files:
            before = path.stat()
            target = partial / path.relative_to(source)
            target.parent.mkdir(parents=True, exist_ok=True)
            source_hash = hashlib.sha256()
            copied = 0
            with path.open("rb") as src, target.open("xb") as dst:
                while chunk := src.read(8 * 1024**2):
                    dst.write(chunk)
                    source_hash.update(chunk)
                    copied += len(chunk)
            after = path.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise RuntimeError(f"Source changed during staging: {path}")
            checksum = source_hash.hexdigest()
            if copied != before.st_size or digest(target) != checksum:
                raise RuntimeError(f"Staged file verification failed: {path}")
            manifest["files"][path.relative_to(source).as_posix()] = {
                "bytes": copied, "sha256": checksum,
            }
            print(f"Verified {path.name}: {copied} bytes", flush=True)
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(f"Staging destination appeared during copy: {destination}")
        partial.rename(destination)
    except BaseException:
        print(f"Incomplete staging preserved at {partial}; do not load it", flush=True)
        raise
    manifest["elapsed_seconds"] = time.monotonic() - started
    manifest["total_bytes"] = total
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(stage_checkpoint(args.source, args.destination), indent=2))


if __name__ == "__main__":
    main()
