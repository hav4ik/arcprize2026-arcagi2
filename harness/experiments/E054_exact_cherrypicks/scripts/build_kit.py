#!/usr/bin/env python3
"""E054: build the overlay kit files (deterministic, paths relative to site-packages).

Inputs: the current kit overlay (kaggle/dsv41_tools/kit/vllm_overlay_rw.{tgz,sha256}, tgz sha256 377d83e0...)
and the two E054 files under overlay/. Outputs in kit/:
  vllm_overlay_e054_add.tgz / .sha256   the two E054 files only (the overlay addition)
  vllm_overlay_rw.tgz / .sha256         drop-in replacement for the kit's pair: the kit's 12 members in
                                        the kit's order, the 2 E054 members replaced, sha list in the
                                        kit's order with those 2 lines replaced
Both tarballs: gzip mtime 0, fixed member mtime/owner/mode, so a rebuild gives the same sha256.
Refuses to build if the kit pair is not the expected version (377d83e0) or a replaced member is not the
expected base file (dsv41_indexer.py 3da076e1, engram.py 5f1e7230).

  python3 scripts/build_kit.py            # build + self-check
  python3 scripts/build_kit.py --check    # verify the committed kit/ files only
"""

import argparse
import calendar
import gzip
import hashlib
import io
import sys
import tarfile
from pathlib import Path

E054 = Path(__file__).resolve().parents[1]
ROOT = E054.parents[1]
BASE_TGZ = ROOT / "kaggle/dsv41_tools/kit/vllm_overlay_rw.tgz"
BASE_SHA_FILE = ROOT / "kaggle/dsv41_tools/kit/vllm_overlay_rw.sha256"
BASE_TGZ_SHA = "377d83e0942ef32133545beb8ff7d945039ea5fd061954546b4b046be84fea6e"
FILES = {  # member -> sha256 of the base (kit) version it replaces
    "vllm/model_executor/kernels/attention/dsa/dsv41_indexer.py":
        "3da076e17cb043c57c75f660b0f1b1d2fd15a5d431fb1d7730d192520e629d2f",
    "vllm/models/deepseek_v4_1/common/engram.py":
        "5f1e7230e52f6904c4de834fe97cc57160fedbc30c250968788aab42bf5e9561",
}
MTIME = calendar.timegm((2026, 10, 25, 0, 0, 0))  # fixed member mtime (2026-10-25 00:00 UTC)
OUT = E054 / "kit"


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def write_tgz(path: Path, members: list[tuple[tarfile.TarInfo, bytes]]) -> None:
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w", format=tarfile.GNU_FORMAT) as tf:
        for info, data in members:
            tf.addfile(info, io.BytesIO(data))
    buf = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=buf, mtime=0, compresslevel=9) as gz:
        gz.write(raw.getvalue())
    path.write_bytes(buf.getvalue())


def info(name: str, size: int, like: tarfile.TarInfo | None = None) -> tarfile.TarInfo:
    ti = tarfile.TarInfo(name)
    ti.size = size
    ti.mtime = MTIME
    ti.mode = like.mode if like is not None else 0o644
    ti.uid = ti.gid = 0
    ti.uname = ti.gname = "root"
    return ti


def read_base():
    blob = BASE_TGZ.read_bytes()
    if sha(blob) != BASE_TGZ_SHA:
        sys.exit(f"REFUSING: {BASE_TGZ} is not the expected kit overlay ({BASE_TGZ_SHA[:8]})")
    members = []
    with tarfile.open(fileobj=io.BytesIO(blob)) as tf:
        for m in tf.getmembers():
            if m.isfile():
                members.append((m, tf.extractfile(m).read()))
    lines = [ln.split() for ln in BASE_SHA_FILE.read_text().splitlines() if ln.strip()]
    listed = {rel: h for h, rel in lines}
    for m, data in members:
        if listed.get(m.name) != sha(data):
            sys.exit(f"REFUSING: kit sha list and tarball disagree on {m.name}")
    for rel, h in FILES.items():
        if listed.get(rel) != h:
            sys.exit(f"REFUSING: kit {rel} is {listed.get(rel, 'missing')[:8]}, E054 targets {h[:8]}")
    return members, lines


def build():
    members, lines = read_base()
    new = {rel: (E054 / "overlay" / rel).read_bytes() for rel in FILES}
    OUT.mkdir(exist_ok=True)
    add = [(info(rel, len(data)), data) for rel, data in new.items()]
    write_tgz(OUT / "vllm_overlay_e054_add.tgz", add)
    (OUT / "vllm_overlay_e054_add.sha256").write_text("".join(f"{sha(d)}  {rel}\n" for rel, d in new.items()))
    full = [(info(m.name, len(new[m.name]), m), new[m.name]) if m.name in new else (info(m.name, len(d), m), d)
            for m, d in members]
    write_tgz(OUT / "vllm_overlay_rw.tgz", full)
    (OUT / "vllm_overlay_rw.sha256").write_text(
        "".join(f"{sha(new[rel]) if rel in new else h}  {rel}\n" for h, rel in lines))
    check()


def check():
    ok = True
    for stem in ("vllm_overlay_e054_add", "vllm_overlay_rw"):
        tgz, lst = OUT / f"{stem}.tgz", OUT / f"{stem}.sha256"
        rows = [ln.split() for ln in lst.read_text().splitlines() if ln.strip()]
        with tarfile.open(tgz) as tf:
            got = {m.name: sha(tf.extractfile(m).read()) for m in tf.getmembers() if m.isfile()}
        good = len(rows) == len(got) and all(got.get(rel) == h for h, rel in rows)
        for h, rel in rows:
            if rel in FILES and h != sha((E054 / "overlay" / rel).read_bytes()):
                good = False
        ok &= good
        print(f"{'OK ' if good else 'BAD'} {tgz.name} sha256 {sha(tgz.read_bytes())}  ({len(rows)} files)")
        for h, rel in rows:
            print(f"    {h[:8]}  {rel}")
    if not ok:
        sys.exit("kit check failed")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true")
    (check if ap.parse_args().check else build)()
