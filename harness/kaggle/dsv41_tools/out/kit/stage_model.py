#!/usr/bin/env python3
"""Stage the model checkpoint for the vLLM load, in the background, while the notebook installs vLLM.

Usage: stage_model.py <src dir> <dst dir> <marker file> <mode: copy | warm | none> [threads]

- copy: copy every file (symlinks resolved) to <dst> with N threads, largest first, then check sizes. Kaggle's
  /kaggle/input is a lazily fetched mount, and 125-131 GB of weights cannot stay in the page cache next to the server's
  ~100 GiB of pinned Engram and processes (173 GiB cgroup), so a page-cache warm-up alone would be re-read slowly at
  load. The local disk (/tmp, ~1 TB free) keeps the copy. Falls back to `warm` if the disk lacks space.
- warm: read every file once (64 MiB chunks) with N threads, to fetch the mount's blocks before the load.
- none: nothing.
Before writing "ok", a pre-flight checks the directory the server will load (E043 audit: vLLM skips index files
missing on disk and the fp8 path has no "weights not initialized" check, so a partial checkpoint loads silently):
config.json and tokenizer.json exist; every file named in model.safetensors.index.json and every gptq/layer_XX
(XX < num_hidden_layers) exists with exactly the size its safetensors header implies (8 + header + last data offset).
The marker gets one line: "ok <dir to load from>", "fail preflight: <what>" (the solver then keeps the fallback
submission), or "fail <reason>" (copy failed; the solver loads from <src>).
"""
import json
import os
import shutil
import struct
import sys
import time
from concurrent.futures import ThreadPoolExecutor

CHUNK = 64 << 20


def files(src):
    out = []
    for root, _, names in os.walk(src, followlinks=True):
        if "/.cache" in root or "/." in root[len(src):]:
            continue
        for n in names:
            p = os.path.join(root, n)
            out.append((os.path.relpath(p, src), os.path.getsize(p)))
    return sorted(out, key=lambda x: -x[1])


def copy_one(src, dst, rel):
    os.makedirs(os.path.dirname(os.path.join(dst, rel)), exist_ok=True)
    tmp = os.path.join(dst, rel) + ".partial"
    with open(os.path.join(src, rel), "rb") as fi, open(tmp, "wb") as fo:
        while True:
            b = fi.read(CHUNK)
            if not b:
                break
            fo.write(b)
    os.replace(tmp, os.path.join(dst, rel))


def warm_one(src, _dst, rel):
    with open(os.path.join(src, rel), "rb") as fi:
        while fi.read(CHUNK):
            pass


def expected_size(path):
    """File size implied by a safetensors header: 8 + header length + the largest tensor end offset."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
    ends = [v["data_offsets"][1] for k, v in hdr.items() if k != "__metadata__"]
    return 8 + n + (max(ends) if ends else 0)


def preflight(d):
    """None if the checkpoint at d looks complete, else a short description of what is wrong."""
    for name in ("config.json", "tokenizer.json", "model.safetensors.index.json"):
        if not os.path.isfile(os.path.join(d, name)):
            return f"missing {name}"
    files = sorted(set(json.load(open(os.path.join(d, "model.safetensors.index.json")))["weight_map"].values()))
    cfg = json.load(open(os.path.join(d, "config.json")))
    layers = cfg.get("num_hidden_layers") or (cfg.get("text_config") or {}).get("num_hidden_layers") or 0
    if not layers:
        return "config.json has no num_hidden_layers (top level or text_config)"
    if os.path.isfile(os.path.join(d, "e043_manifest.json")) and not os.path.isdir(os.path.join(d, "gptq")):
        return "gptq/ missing (an E043 export needs it; the experts would load uninitialized)"
    if os.path.isdir(os.path.join(d, "gptq")):
        files += [f"gptq/layer_{i:02d}.safetensors" for i in range(layers)]
    bad = []
    for rel in files:
        p = os.path.join(d, rel)
        try:
            if not os.path.isfile(p):
                bad.append(f"{rel} missing")
            elif os.path.getsize(p) != expected_size(p):
                bad.append(f"{rel} size {os.path.getsize(p)} != {expected_size(p)}")
        except Exception as exc:
            bad.append(f"{rel} unreadable ({type(exc).__name__})")
    return None if not bad else f"{len(bad)} of {len(files)} files bad: {'; '.join(bad[:4])}"


def main():
    src, dst, marker, mode = sys.argv[1:5]
    threads = int(sys.argv[5]) if len(sys.argv) > 5 else 16
    t0 = time.time()
    try:
        if mode == "none":
            bad = preflight(src)
            open(marker, "w").write(f"fail preflight: {bad}\n" if bad else f"ok {src}\n")
            print(f"preflight {'FAILED: ' + bad if bad else 'ok'}", flush=True)
            return
        fl = files(src)
        total = sum(s for _, s in fl)
        if mode == "warm":
            # Pre-flight first and release the solver at once; the warm-up then runs alongside the server load.
            bad = preflight(src)
            open(marker, "w").write(f"fail preflight: {bad}\n" if bad else f"ok {src}\n")
            print(f"preflight {'FAILED: ' + bad if bad else 'ok'}; warming {total / 2**30:.1f} GiB", flush=True)
            if bad:
                return
        if mode == "copy":
            os.makedirs(dst, exist_ok=True)
            free = shutil.disk_usage(dst).free
            if free < total + (10 << 30):
                print(f"copy: {total / 2**30:.1f} GiB needed, {free / 2**30:.1f} GiB free; warming instead", flush=True)
                mode = "warm"
        fn = copy_one if mode == "copy" else warm_one
        with ThreadPoolExecutor(threads) as ex:
            for i, _ in enumerate(ex.map(lambda x: fn(src, dst, x[0]), fl), 1):
                if i % 10 == 0 or i == len(fl):
                    el = time.time() - t0
                    print(f"{mode}: {i}/{len(fl)} files, {el:.0f}s", flush=True)
        if mode == "copy":
            bad = [r for r, s in fl if os.path.getsize(os.path.join(dst, r)) != s]
            if bad:
                raise RuntimeError(f"size mismatch: {bad[:3]}")
        el = time.time() - t0
        print(f"{mode} done: {total / 2**30:.1f} GiB in {el:.0f}s ({total / 2**20 / max(el, 1):.0f} MiB/s)", flush=True)
        if mode == "warm":
            return  # marker already written before the warm-up
        target = dst if mode == "copy" else src
        bad = preflight(target)
        if bad:
            print(f"preflight FAILED: {bad}", flush=True)
            open(marker, "w").write(f"fail preflight: {bad}\n")
            return
        print(f"preflight ok: {target}", flush=True)
        open(marker, "w").write(f"ok {target}\n")
    except Exception as exc:
        open(marker, "w").write(f"fail {type(exc).__name__}: {exc}\n")
        raise


if __name__ == "__main__":
    main()
