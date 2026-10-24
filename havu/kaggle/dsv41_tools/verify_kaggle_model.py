#!/usr/bin/env python3
"""Compare a Kaggle model version's file list (path + size) with a local manifest (path, size, sha256 per file).

    verify_kaggle_model.py <owner/model/framework/instance/version> <manifest.json>

Uses `kaggle models instances versions files` (KAGGLE_API_TOKEN from the environment), all pages. Reports files
missing on Kaggle, extra files, and size mismatches; exit 1 on any difference. Kaggle does not expose content
hashes, so sha256 is checked before upload (E043 verify_reshard_equiv.py) and the notebook's pre-flight checks
every safetensors file against its header size at run time.
"""
import json
import subprocess
import sys


def kaggle_files(handle):
    out, token = {}, None
    while True:
        cmd = ["kaggle", "models", "instances", "versions",
               "files", handle, "--page-size", "200", "-v"]
        if token:
            cmd += ["--page-token", token]
        r = subprocess.run(cmd, capture_output=True, text=True)
        lines = [l for l in r.stdout.splitlines() if l.strip() and "outdated" not in l]
        if r.returncode != 0 or not lines:
            raise SystemExit(f"kaggle CLI failed: {r.stderr[-500:]}")
        header = lines[0].split(",")
        for l in lines[1:]:
            if l.startswith("Next Page Token"):
                continue
            row = dict(zip(header, l.split(",")))
            out[row.get("name")] = int(row.get("size", -1))
        nxt = [l for l in r.stdout.splitlines() + r.stderr.splitlines() if "Next Page Token" in l]
        if not nxt:
            return out
        token = nxt[-1].split("=", 1)[-1].strip()


def main():
    handle, manifest = sys.argv[1], sys.argv[2]
    m = json.load(open(manifest))
    files = m.get("entries") if isinstance(m.get("entries"), list) else m.get("files", m)
    want = {f["path"]: f["size"] for f in files} if isinstance(files, list) else {k: v["size"] for k, v in files.items()}
    have = kaggle_files(handle)
    missing = sorted(set(want) - set(have))
    extra = sorted(set(have) - set(want))
    wrong = sorted(p for p in set(want) & set(have) if want[p] != have[p])
    print(f"manifest {len(want)} files / {sum(want.values()) / 1e9:.3f} GB; Kaggle {len(have)} files / "
          f"{sum(have.values()) / 1e9:.3f} GB")
    print(f"missing on Kaggle: {len(missing)} {missing[:8]}")
    print(f"extra on Kaggle: {len(extra)} {extra[:8]}")
    print(f"size mismatch: {len(wrong)} {[(p, want[p], have[p]) for p in wrong[:8]]}")
    sys.exit(1 if missing or wrong else 0)


if __name__ == "__main__":
    main()
