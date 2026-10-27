#!/usr/bin/env python3
"""E056 (candidate B): build the vLLM overlay pair = the kit's overlay + E054 #156 part 1 + E051 KV CPU offload.

Adapted from experiments/E054_exact_cherrypicks/scripts/build_kit.py (deterministic, paths relative to site-packages).

Inputs (each sha256-pinned; the build refuses on any other version):
  kaggle/dsv41_tools/kit/vllm_overlay_rw.{tgz,sha256}   the current kit overlay, tgz 377d83e0 (12 members)
  E054 overlay/.../dsv41_indexer.py                      #156 part 1 (9f13485b), replaces the kit's 3da076e1
  E051 overlay/vllm/... (4 files)                        KV CPU offload, the shas gpu/install_overlay.sh checks
Output in kit/ (a drop-in for the kit's pair; cell 2 of the notebook installs it unchanged):
  vllm_overlay_rw.tgz      the kit's 12 members in the kit's order with dsv41_indexer.py replaced, then the 4 E051
                           files appended as new members (none of them is a kit member; if one were, it would be
                           replaced in place instead and the build says so)
  vllm_overlay_rw.sha256   the kit's 12 lines (indexer line replaced) + 4 new lines, same "<sha>  <path>" format
Both are deterministic: gzip mtime 0, GNU tar, fixed member mtime/owner (kit members keep their mode, new ones 0644).
Engram (#58678, E054 NO-GO) is NOT included: engram.py stays the kit version 5f1e7230.

  python3 scripts/build_kit.py            # build + check
  python3 scripts/build_kit.py --check    # verify the committed kit/ files (members, shas, sources, byte rebuild)
"""

import argparse
import calendar
import gzip
import hashlib
import io
import sys
import tarfile
from pathlib import Path

E056 = Path(__file__).resolve().parents[1]
ROOT = E056.parents[1]
BASE_TGZ = ROOT / "kaggle/dsv41_tools/kit/vllm_overlay_rw.tgz"
BASE_SHA_FILE = ROOT / "kaggle/dsv41_tools/kit/vllm_overlay_rw.sha256"
BASE_TGZ_SHA = "377d83e0942ef32133545beb8ff7d945039ea5fd061954546b4b046be84fea6e"
E054_OV = ROOT / "experiments/E054_exact_cherrypicks/overlay"
E051_OV = ROOT / "experiments/E051_kv_offload/overlay"
# member -> (source file, sha256 of the source, sha256 of the kit member it replaces or None if new to the kit)
FILES = {
    "vllm/model_executor/kernels/attention/dsa/dsv41_indexer.py": (
        E054_OV, "9f13485bd59a62713cc1dc6e1678928986b77bcd6d0e5d0912ca8b67f55f1ed5",
        "3da076e17cb043c57c75f660b0f1b1d2fd15a5d431fb1d7730d192520e629d2f"),
    # E051, in gpu/install_overlay.sh's order; shas = install_overlay.sh's "sha256(overlay)" column
    "vllm/distributed/kv_transfer/kv_connector/v1/offloading/config.py": (
        E051_OV, "664b4b0118e64194c62fe99658a7f84000fe4b2264b956228006783278e09c51", None),
    "vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py": (
        E051_OV, "dffd7d86e990fe45a27fba017f53befee4ea52f281407cb608f24168a54034a6", None),
    "vllm/v1/kv_offload/cpu/manager.py": (
        E051_OV, "eda86f25e7eb20a8b4af686dce0e5d005b5ce3687557914293f400d0475d7dfd", None),
    "vllm/v1/kv_offload/cpu/spec.py": (
        E051_OV, "a0dbbafc1e234f3bb937b38fe9dac6b3d1e7c48003ba0015e0f1ab8d717e785f", None),
}
MTIME = calendar.timegm((2026, 10, 25, 0, 0, 0))  # fixed member mtime (2026-10-25 00:00 UTC), as E054
OUT = E056 / "kit"


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def tgz_bytes(members: list[tuple[tarfile.TarInfo, bytes]]) -> bytes:
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w", format=tarfile.GNU_FORMAT) as tf:
        for info, data in members:
            tf.addfile(info, io.BytesIO(data))
    buf = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=buf, mtime=0, compresslevel=9) as gz:
        gz.write(raw.getvalue())
    return buf.getvalue()


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
    if [m.name for m, _ in members] != [rel for _, rel in lines]:
        sys.exit("REFUSING: kit sha list and tarball list different members or orders")
    for m, data in members:
        if listed.get(m.name) != sha(data):
            sys.exit(f"REFUSING: kit sha list and tarball disagree on {m.name}")
    for rel, (_, _, base) in FILES.items():
        if base is not None and listed.get(rel) != base:
            sys.exit(f"REFUSING: kit {rel} is {listed.get(rel, 'missing')[:8]}, E056 replaces {base[:8]}")
        if base is None and rel in listed:
            print(f"NOTE: {rel} is already a kit member ({listed[rel][:8]}): replaced in place, not appended")
    return members, lines


def sources():
    new = {}
    for rel, (ov, want, _) in FILES.items():
        data = (ov / rel).read_bytes()
        if sha(data) != want:
            sys.exit(f"REFUSING: source {ov / rel} is {sha(data)[:8]}, expected {want[:8]}")
        new[rel] = data
    return new


def assemble():
    """(tgz bytes, sha-list text, [(member, sha, status)]) from the pinned inputs."""
    members, lines = read_base()
    new = sources()
    out, rows, kit_names = [], [], [m.name for m, _ in members]
    for m, d in members:
        if m.name in new:
            out.append((info(m.name, len(new[m.name]), m), new[m.name]))
            rows.append((m.name, sha(new[m.name]), f"replaced (kit {sha(d)[:8]})"))
        else:
            out.append((info(m.name, len(d), m), d))
            rows.append((m.name, sha(d), "kit"))
    for rel, data in new.items():
        if rel not in kit_names:
            out.append((info(rel, len(data)), data))
            rows.append((rel, sha(data), "appended (new to the kit)"))
    text = "".join(f"{h}  {rel}\n" for rel, h, _ in rows)
    return tgz_bytes(out), text, rows


def build():
    blob, text, _ = assemble()
    OUT.mkdir(exist_ok=True)
    (OUT / "vllm_overlay_rw.tgz").write_bytes(blob)
    (OUT / "vllm_overlay_rw.sha256").write_text(text)
    check()


def check():
    blob, text, rows = assemble()
    tgz, lst = OUT / "vllm_overlay_rw.tgz", OUT / "vllm_overlay_rw.sha256"
    got_blob, got_text = tgz.read_bytes(), lst.read_text()
    listed = [ln.split() for ln in got_text.splitlines() if ln.strip()]
    with tarfile.open(fileobj=io.BytesIO(got_blob)) as tf:
        got = [(m.name, sha(tf.extractfile(m).read())) for m in tf.getmembers() if m.isfile()]
    ok = {
        "rebuild is byte-identical (tgz)": got_blob == blob,
        "rebuild is byte-identical (sha list)": got_text == text,
        "tarball members == sha list, same order": got == [(rel, h) for h, rel in listed],
        "16 members": len(got) == 16,
        "engram.py is the kit version 5f1e7230": dict(got).get("vllm/models/deepseek_v4_1/common/engram.py", "")
        .startswith("5f1e7230"),
    }
    for name, good in ok.items():
        print(f"{'ok ' if good else 'BAD'} {name}")
    print(f"vllm_overlay_rw.tgz    sha256 {sha(got_blob)}  ({len(got)} files)")
    print(f"vllm_overlay_rw.sha256 sha256 {sha(got_text.encode())}")
    for rel, h, status in rows:
        print(f"    {h[:8]}  {rel}  [{status}]")
    if not all(ok.values()):
        sys.exit("kit check failed")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true")
    (check if ap.parse_args().check else build)()
