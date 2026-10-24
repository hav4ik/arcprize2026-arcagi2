#!/usr/bin/env python3
"""E043: compare a Kaggle model instance version's files with the local folder by relative path and size.

  kaggle_verify_model.py <owner>/<model>/<framework>/<instance>/<version> <local dir>

Uses the `kaggle` CLI (token in $KAGGLE_API_TOKEN), all pages. Local files: everything under the dir except the
kaggle metadata JSON and dot-dirs, symlinks followed. Exit 0 only if every local file is on Kaggle with the same size
and no Kaggle safetensors file is above 5 GB.
"""
import csv
import io
import os
import subprocess
import sys
from pathlib import Path

KAGGLE = str(Path(sys.executable).with_name("kaggle"))
SKIP = {"model-instance-metadata.json", "model-metadata.json"}


def remote_files(handle):
    out, token = {}, None
    while True:
        cmd = [KAGGLE, "models", "instances", "versions", "files", handle, "-v", "--page-size", "200"]
        if token:
            cmd += ["--page-token", token]
        p = subprocess.run(cmd, capture_output=True, text=True)
        text = "\n".join(l for l in p.stdout.splitlines() if not l.startswith("Warning"))
        if p.returncode or "404" in p.stdout + p.stderr:
            raise SystemExit(f"listing failed (version not ready?): {(p.stdout + p.stderr).strip()[:200]}")
        token = None
        rows = []
        for line in text.splitlines():
            if line.startswith("Next Page Token"):
                token = line.split("=", 1)[1].strip()
            else:
                rows.append(line)
        for r in csv.DictReader(io.StringIO("\n".join(rows))):
            out[r["name"]] = int(r["size"])
        if not token:
            return out


def main():
    handle, root = sys.argv[1], Path(sys.argv[2])
    local = {}
    for d, dirs, files in os.walk(root, followlinks=True):
        dirs[:] = [x for x in dirs if not x.startswith(".") and x != "__pycache__"]  # the kaggle CLI skips both
        for f in files:
            rel = str((Path(d) / f).relative_to(root))
            if rel not in SKIP:
                local[rel] = os.path.getsize(Path(d) / f)
    remote = remote_files(handle)
    missing = sorted(set(local) - set(remote))
    wrong = sorted(k for k in set(local) & set(remote) if local[k] != remote[k])
    big = sorted(k for k, v in remote.items() if k.endswith(".safetensors") and v > 5_000_000_000)
    print(f"{handle}: {len(remote)} files on Kaggle ({sum(remote.values()) / 1e9:.2f} GB), {len(local)} local "
          f"({sum(local.values()) / 1e9:.2f} GB); missing {missing[:4]} ({len(missing)}), size mismatch {wrong[:4]} "
          f"({len(wrong)}), safetensors above 5 GB {big}")
    ok = not missing and not wrong and not big
    print("KAGGLE-VERIFIED" if ok else "KAGGLE-NOT-VERIFIED")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
