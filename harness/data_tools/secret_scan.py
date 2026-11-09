"""Scan files for secrets before they leave this machine. Prints counts and file names only, never values.

Checks every value in SECRETS.yaml and every ENDPOINTS.yaml token, plus key-like patterns (sk-, hf_, KGAT_, ghp_).
.gz files are decompressed and .tgz members are extracted in memory. Exit code 1 if anything matches.

Usage: .venv/bin/python data_tools/secret_scan.py FILE_LIST   (one path per line; '-' reads stdin)
"""
import gzip
import io
import re
import sys
import tarfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
# Values that are not secret even though they are stored next to secrets.
NOT_SECRET_KEYS = {"huggingface_org_name"}
PATTERN = re.compile(rb"(sk-[A-Za-z0-9]{20,}|hf_[A-Za-z0-9]{20,}|KGAT_[A-Za-z0-9]{10,}|ghp_[A-Za-z0-9]{20,})")


def secret_values():
    vals = [str(v) for k, v in yaml.safe_load((ROOT / "SECRETS.yaml").read_text()).items()
            if k not in NOT_SECRET_KEYS and v and len(str(v)) >= 12]
    ep_path = ROOT / "ENDPOINTS.yaml"
    if ep_path.exists():
        ep = yaml.safe_load(ep_path.read_text()) or []
        items = ep if isinstance(ep, list) else (ep.get("endpoints") or list(ep.values()))
        vals += [str(e["token"]) for e in items if isinstance(e, dict) and e.get("token") and len(str(e["token"])) >= 8]
    return [v.encode() for v in vals]


def blobs(path):
    data = Path(path).read_bytes()
    if path.endswith((".tgz", ".tar.gz")):
        with tarfile.open(fileobj=io.BytesIO(data)) as t:
            for m in t.getmembers():
                if m.isfile():
                    yield f"{path}::{m.name}", t.extractfile(m).read()
    elif path.endswith(".gz"):
        yield path, gzip.decompress(data)
    else:
        yield path, data


def main():
    src = sys.stdin if sys.argv[1] == "-" else open(sys.argv[1])
    files = [line.strip() for line in src if line.strip()]
    secrets = secret_values()
    hits = {}
    for f in files:
        if not Path(f).is_file():
            continue
        for name, b in blobs(f):
            n = sum(b.count(s) for s in secrets) + len(PATTERN.findall(b))
            if n:
                hits[name] = n
    print(f"secret scan: {len(files)} files, {len(secrets)} secret values checked, {len(hits)} files with hits")
    for name, n in hits.items():
        print(f"  HIT {name} ({n})")
    sys.exit(1 if hits else 0)


if __name__ == "__main__":
    main()
