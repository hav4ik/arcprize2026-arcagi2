"""Inputs of the pruned DeepSeek-V4.1-Flash-1017 llama-server on 4x L4, with no experiment-folder imports.

Run it with python3 -I: it reads the dspk-41-imp dataset, as data only (JSON, and npz with allow_pickle=False).

- model:   a folder of links, parts 1-5 of the split GGUF and the IQ4_NL engram table as part 6
           (llama.cpp needs LLAMA_SPLIT_ACCEPT_UNSTAMPED=1 for it); for the GSQ sets (q2-x), all parts and the
           engram sidecar under the name of the model's deepseek41.engram.file
- experts: the LLAMA_DSV41_EXPERT_KEEP file, the 288 of 384 routed experts per layer with the highest notool + tool
           importance (checksum pinned to the measured mask)
- engram:  75% of each layer's rows in hot-set files for LLAMA_DSV41_ENGRAM_LOCAL with LLAMA_DSV41_ENGRAM_PRUNE=1
           (the other rows read as zeros), hottest first: 16 GiB and 7 GiB in /dev/shm, the rest (54.25 GiB) on
           local disk in /kaggle/tmp

prepare() writes <prep dir>/prep.json last; a later call with the same inputs reuses its outputs. The hot-set files
take one pass over the 110 GB table.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import re
import struct
import sys
import threading
import time


# Kaggle mounts these under other folders in other sessions; find_input looks the name up under INPUT_ROOT.
INPUT_ROOT = Path("/kaggle/input")
MODEL = ("/mnt/deepseek-q2/"
         "DeepSeek-V4.1-Flash-1017-UD-IQ2_XXS-00001-of-00006.gguf")
ENGRAM_TABLE = ("/mnt/deepseek-engram-q4/"
                "DeepSeek-V4.1-Flash-1017-engram-IQ4_NL.gguf_file")
IMP_DIR = "/kaggle/input/dspk-41-imp"
PREP_DIR = "/kaggle/tmp/dsv41_prep"
ENGRAM_DIR = "/dev/shm/dsv41_engram"
ENGRAM_DIRS = (ENGRAM_DIR, ENGRAM_DIR, "/kaggle/tmp/dsv41_engram")  # tier k goes to the k-th, the rest to the last
ENGRAM_TENSOR = "per_layer_token_embd.weight"
# The GSQ sidecar keeps the same rows as one table per Engram layer, layer 1's then layer 14's.
ENGRAM_LAYER_TENSOR = re.compile(r"blk\.(\d+)\.engram_embd\.weight")
ENGRAM_KEEP = 0.75  # share of each layer's rows, or the name of a notool mask such as engram_top48gib
ENGRAM_TIERS = (16, 7)  # GiB of the first tiers; the last takes the rest
# Tiers rank the kept rows by lookups per task, (1 - ENGRAM_TIER_TOOL) of the notool workload and ENGRAM_TIER_TOOL of
# the tool one, layer 1's ENGRAM_TIER_FIRST times: its reads wait on ~7 ms of the step, those of layer 14 on 13 layers.
ENGRAM_TIER_TOOL = 0.5
ENGRAM_TIER_FIRST = 4.0
# The shared 1017 table: layer 1 reads rows [0, 384006168), layer 14 the rest (deepseek41.ple.head_offsets).
ENGRAM_OFFSETS = (0, 384006168)
ENGRAM_ROWS = 768022850
# Lookups of one task: 7k prompt tokens at the prefill rate and 64k decoded tokens at the decode rate.
TASK_PROMPT, TASK_DECODE = 7, 64
N_LAYER, N_EXPERT, N_KEEP = 40, 384, 288
KEEP_SHA256_PREFIX = "d5b5d0ed88521496"  # keep_e288_comb.u8 of the measured runs
SPLIT = re.compile(r"-(\d{5})-of-(\d{5})\.gguf$")
# ggml type: (elements per block, bytes per block)
GGML_BLOCKS = {0: (1, 4), 1: (1, 2), 8: (32, 34), 12: (256, 144), 13: (256, 176), 14: (256, 210),
               20: (32, 18), 23: (256, 136), 30: (1, 2)}
VERSION = 1
SYSFS_CPU = Path("/sys/devices/system/cpu")


def gguf_tensors(path):
    """{name: (shape, ggml type, absolute data offset)} and the scalar KVs of a GGUF file."""
    sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
    formats = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}
    kv, tensors = {}, {}
    with open(path, "rb") as f:
        if f.read(4) != b"GGUF":
            raise ValueError(f"Not a GGUF file: {path}")
        _, n_tensors, n_kv = struct.unpack("<IQQ", f.read(20))

        def string():
            n, = struct.unpack("<Q", f.read(8))
            return f.read(n).decode("utf-8", "replace")

        def skip_array():
            elem, count = struct.unpack("<IQ", f.read(12))
            if elem == 8:
                for _ in range(count):
                    string()
            elif elem == 9:
                for _ in range(count):
                    skip_array()
            else:
                f.seek(count * sizes[elem], 1)

        for _ in range(n_kv):
            key = string()
            kind, = struct.unpack("<I", f.read(4))
            if kind == 9:
                skip_array()
            else:
                kv[key] = string() if kind == 8 else struct.unpack("<" + formats[kind], f.read(sizes[kind]))[0]
        for _ in range(n_tensors):
            name = string()
            n_dims, = struct.unpack("<I", f.read(4))
            shape = struct.unpack(f"<{n_dims}Q", f.read(8 * n_dims))
            kind, offset = struct.unpack("<IQ", f.read(12))
            tensors[name] = (shape, kind, offset)
        align = int(kv.get("general.alignment", 32))
        start = -(-f.tell() // align) * align
    return {k: (shape, kind, start + offset) for k, (shape, kind, offset) in tensors.items()}, kv


def table_layout(path, name=ENGRAM_TENSOR):
    """(rows, bytes per row, data offset) of the engram table, or of the per-layer tables of a GSQ sidecar, which
    follow each other in the file and read as one."""
    tensors, _ = gguf_tensors(path)
    if name in tensors:
        names = [name]
    else:
        names = sorted((n for n in tensors if ENGRAM_LAYER_TENSOR.fullmatch(n)),
                       key=lambda n: int(ENGRAM_LAYER_TENSOR.fullmatch(n).group(1)))
        if not names:
            raise ValueError(f"{path} has no tensor {name} and no per-layer engram tables")
    rows = row_bytes = offset = None
    for n in names:
        shape, kind, start = tensors[n]
        if kind not in GGML_BLOCKS or len(shape) != 2 or shape[0] % GGML_BLOCKS[kind][0]:
            raise ValueError(f"{n}: unsupported ggml type {kind} or shape {shape}")
        block, size = GGML_BLOCKS[kind]
        if rows is None:
            rows, row_bytes, offset = 0, shape[0] // block * size, start
        if shape[0] // block * size != row_bytes or start != offset + rows * row_bytes:
            raise ValueError(f"{path}: {n} does not follow the previous engram table")
        rows += shape[1]
    if offset + rows * row_bytes > os.path.getsize(path):
        raise ValueError(f"{path} is shorter than its engram tables")
    return rows, row_bytes, offset


def shards(model):
    """The split parts of the model folder (or of its first part), by index; part 6 may be missing."""
    model = Path(model)
    first = model if model.is_file() else next(iter(sorted(p for p in model.glob("*-00001-of-*.gguf"))), None)
    match = SPLIT.search(first.name) if first is not None else None
    if match is None:
        raise ValueError(f"No first split part (*-00001-of-NNNNN.gguf) in {model}")
    stem, count = first.name[:match.start()], int(match.group(2))
    return {i: first.with_name(f"{stem}-{i:05d}-of-{count:05d}.gguf") for i in range(1, count + 1)}


def link_model(model, table, out):
    """A folder of links to the model's parts, with the engram table as the last part, or for a GSQ set, which names
    its engram sidecar (deepseek41.engram.file), all its parts and the table under that name."""
    parts = shards(model)
    sidecar = gguf_tensors(parts[1])[1].get("deepseek41.engram.file")
    if sidecar is not None and (not sidecar or Path(sidecar).name != sidecar or sidecar in (".", "..")):
        raise ValueError(f"{parts[1]}: deepseek41.engram.file {sidecar!r} is not a file name")
    last = None if sidecar else max(parts)
    out.mkdir(parents=True, exist_ok=True)
    for stale in out.iterdir():
        if stale.is_symlink():
            stale.unlink()
    links = {}
    sources = [(parts[i].name, Path(table) if i == last else source) for i, source in parts.items()]
    if sidecar:
        sources.append((sidecar, Path(table)))
    for name, source in sources:
        if not source.is_file():
            raise ValueError(f"Missing model part {name}: {source}")
        link = out / name
        link.symlink_to(source.resolve())
        links[link.name] = dict(source=str(source.resolve()), bytes=source.stat().st_size)
    return out / parts[1].name, links


def expert_keep(imp, out):
    """LLAMA_DSV41_EXPERT_KEEP: per layer the N_KEEP experts of the summed, normalized notool and tool scores."""
    import numpy as np
    scores = []
    for kind in ("notool", "tool"):
        layers = json.loads((imp / "experts" / kind / "scores.json").read_text())["layers"]
        if len(layers) != N_LAYER:
            raise ValueError(f"experts/{kind}/scores.json has {len(layers)} layers, expected {N_LAYER}")
        scores.append(layers)
    keep = np.zeros((N_LAYER, N_EXPERT), np.uint8)
    for i, (notool, tool) in enumerate(zip(*scores)):
        a = np.asarray(notool["recommended_score"], np.float64)
        b = np.asarray(tool["recommended_score"], np.float64)
        if a.shape != (N_EXPERT,) or b.shape != (N_EXPERT,):
            raise ValueError(f"layer {i}: expected {N_EXPERT} expert scores")
        keep[i, np.argsort(-(a / a.sum() + b / b.sum()), kind="stable")[:N_KEEP]] = 1
    data = keep.tobytes()
    digest = hashlib.sha256(data).hexdigest()
    if not digest.startswith(KEEP_SHA256_PREFIX):
        raise ValueError(f"Expert keep mask {digest[:16]} is not the measured {KEEP_SHA256_PREFIX}")
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, out)
    return digest


def task_weights(counts):
    """Per-lookup weights of the prompt and generated counts that give the lookups of one task."""
    import numpy as np
    ntok = np.asarray(counts["ntok"], dtype=np.float64).reshape(-1)
    n_prompt, n_decode = (ntok[0], ntok[1]) if ntok.size >= 2 else (ntok[0], ntok[0])
    return TASK_PROMPT / n_prompt, TASK_DECODE / n_decode


def top_rows(score, n):
    """Indices (ascending) of the n highest scores, ties to the lower index as in a stable sort."""
    import numpy as np
    n = max(0, min(int(n), len(score)))
    if n in (0, len(score)):
        return np.arange(n, dtype=np.int64)
    kth = np.partition(score, len(score) - n)[len(score) - n]
    above = np.flatnonzero(score > kth)
    return np.sort(np.concatenate([above, np.flatnonzero(score == kth)[:n - len(above)]]))


def keep_rows(scores, n_rows, n):
    """n of n_rows rows (ascending): by each score in turn the highest rows it saw (score > 0) that are not yet kept,
    then the rest spread evenly over the rows, since a hashed n-gram row no score saw is as likely as any other."""
    import numpy as np
    kept = np.zeros(n_rows, dtype=bool)
    left = min(int(n), n_rows)
    for score in scores:
        seen = np.flatnonzero(score > 0)
        seen = seen[~kept[seen]]
        if len(seen) > left:
            seen = seen[top_rows(score[seen], left)]
        kept[seen] = True
        left -= len(seen)
    if left:
        rest = np.flatnonzero(~kept)
        kept[rest[np.arange(left, dtype=np.int64) * len(rest) // left]] = True
    return np.flatnonzero(kept)


def engram_tiers(imp, keep, gib, row_bytes, table_rows, tool_share=ENGRAM_TIER_TOOL, first=ENGRAM_TIER_FIRST):
    """The kept rows (global table rows, ascending) per tier, hottest first, and the tier summary.

    keep is the share of each layer's rows to keep, or the name of a notool mask of the dataset. A share keeps the rows
    the notool calibration looked up (by lookups per task), then of the others those the tool workload looked up, then
    an even spread of the rest. Tiers rank the kept rows by their lookups per task, tool_share of them of the tool
    workload, layer 1's first times.
    """
    import numpy as np
    counts = np.load(imp / "engram" / "notool" / "counts.npz", allow_pickle=False)
    w0, w1 = task_weights(counts)
    if isinstance(keep, str):
        mask = np.load(imp / "engram" / "notool" / "masks" / f"{keep}.npz", allow_pickle=False)
    tool = np.load(imp / "engram" / "tool" / "counts.npz", allow_pickle=False)
    t0, t1 = task_weights(tool)
    rows, hot, rank, looked = [], [], [], np.zeros(4)
    ends = (*ENGRAM_OFFSETS[1:], table_rows)
    for layer, (start, end) in enumerate(zip(ENGRAM_OFFSETS, ends)):
        r0 = counts[f"l{layer}_r0"].astype(np.float32)
        r1 = counts[f"l{layer}_r1"].astype(np.float32)
        if len(r0) > end - start or len(r1) != len(r0):
            raise ValueError(f"engram counts of layer {layer} hold {len(r0)} rows, the table {end - start}")
        q0 = tool[f"l{layer}_r0"].astype(np.float32)
        q1 = tool[f"l{layer}_r1"].astype(np.float32)
        if len(q0) > end - start or len(q1) != len(q0):
            raise ValueError(f"tool engram counts of layer {layer} hold {len(q0)} rows, the table {end - start}")
        looked += [a.sum(dtype=np.float64) for a in (r0, r1, q0, q1)]
        if isinstance(keep, str):
            idx = np.flatnonzero(np.unpackbits(mask[f"layer{layer}"], bitorder="little")[:len(r0)])
        else:
            idx = keep_rows((r0 * w0 + r1 * w1, q0 * t0 + q1 * t1), end - start, round(keep * (end - start)))
        # counts of the rows past the end of a count array are 0
        pair = np.zeros((len(idx), 4), dtype=np.float32)
        for c, a in enumerate((r0, r1, q0, q1)):
            on = idx[idx < len(a)]
            pair[:len(on), c] = a[on]
        rows.append(idx.astype(np.int64) + start)
        hot.append(pair)
        rank.append(((1 - tool_share) * (pair[:, 0] * w0 + pair[:, 1] * w1)
                     + tool_share * (pair[:, 2] * t0 + pair[:, 3] * t1)) * (first if layer == 0 else 1))
        del r0, r1, q0, q1, idx, on
    rows, hot = np.concatenate(rows), np.concatenate(hot)
    # Tier k is the rows among the top sum(gib[:k + 1]) by score and not in the tiers before it.
    score = np.concatenate(rank)
    del rank
    tier = np.full(len(rows), len(gib), dtype=np.int8)
    ends = np.minimum(np.cumsum([int(g * 2**30 // row_bytes) for g in gib]), len(rows))
    for k in reversed(range(len(gib))):
        tier[top_rows(score, ends[k])] = k
    del score
    tiers, summary = [], []
    for k in range(len(gib) + 1):
        sel = tier == k
        n = int(sel.sum())
        if n == 0:
            continue
        tiers.append(rows[sel])
        share = hot[sel].astype(np.float64).sum(0) / looked  # float32 sums along the rows lose most of the counts
        summary.append(dict(rows=n, gib=round(n * row_bytes / 2**30, 2),
                            prompt_share=round(float(share[0]), 4), decode_share=round(float(share[1]), 4),
                            tool_prompt_share=round(float(share[2]), 4), tool_decode_share=round(float(share[3]), 4)))
    return tiers, summary


def write_hotsets(table, tiers, outs, table_rows, row_bytes, base, threads=48, chunk_mib=48):
    """One ordered pass over the table: each tier's rows (ascending, disjoint) into its hot-set file.

    Format: "DSV41EHS", u32 1, u32 row bytes, u64 table rows, u64 n, the bitmap (ceil(table rows / 64) u64), the n
    rows. Many handles read large requests (on FUSE a request costs ~55 ms whatever its size, and one handle's reads
    are serialized). Each file is written as .tmp and renamed when complete.
    """
    import numpy as np
    words = (table_rows + 63) // 64
    owner = np.full(table_rows, -1, dtype=np.int8)
    files = []
    for k, (rows, out) in enumerate(zip(tiers, outs)):
        if len(rows) == 0 or rows[0] < 0 or rows[-1] >= table_rows or (np.diff(rows) <= 0).any():
            raise ValueError(f"tier {k}: rows must be ascending, distinct and inside the table")
        if (owner[rows] >= 0).any():
            raise ValueError(f"tier {k}: rows shared with another tier")
        owner[rows] = k
        bits = np.zeros(words * 64, dtype=bool)
        bits[rows] = True
        f = open(str(out) + ".tmp", "wb")
        f.write(b"DSV41EHS" + struct.pack("<IIQQ", 1, row_bytes, table_rows, len(rows)))
        f.write(np.packbits(bits, bitorder="little").tobytes())
        del bits
        files.append(dict(out=out, f=f, n=len(rows), written=0))
    per = (chunk_mib << 20) // row_bytes  # rows per chunk
    sub = max(1, per // threads)
    local, handles = threading.local(), []

    def read_range(a, b):
        fd = getattr(local, "fd", None)
        if fd is None:
            fd = local.fd = os.open(table, os.O_RDONLY)
            handles.append(fd)
        want, buf = (b - a) * row_bytes, bytearray()
        while len(buf) < want:
            got = os.pread(fd, want - len(buf), base + a * row_bytes + len(buf))
            if not got:
                raise OSError(f"Short read of {table}")
            buf += got
        return bytes(buf)

    t0 = time.monotonic()
    with ThreadPoolExecutor(threads) as pool:
        def submit(a):
            b = min(a + per, table_rows)
            return a, b, [pool.submit(read_range, s, min(s + sub, b)) for s in range(a, b, sub)]
        pending = [submit(a) for a in range(0, min(table_rows, 2 * per), per)]
        a_next = 2 * per
        while pending:
            a, b, futures = pending.pop(0)
            if a_next < table_rows:
                pending.append(submit(a_next))
                a_next += per
            data = np.frombuffer(b"".join(f.result() for f in futures), dtype=np.uint8).reshape(b - a, row_bytes)
            own = owner[a:b]
            for k, tier in enumerate(files):
                sel = data[own == k]
                if len(sel):
                    tier["f"].write(sel.tobytes())
                    tier["written"] += len(sel)
            if (a // per) % 400 == 0:
                print(f"engram hot sets: {b / table_rows:6.1%}, {b * row_bytes / 2**30 / (time.monotonic() - t0):.2f} "
                      "GiB/s", flush=True)
    for fd in handles:
        os.close(fd)
    for tier in files:
        tier["f"].flush()
        os.fsync(tier["f"].fileno())
        tier["f"].close()
        if tier["written"] != tier["n"]:
            raise OSError(f"{tier['out']}: wrote {tier['written']} rows, expected {tier['n']}")
        os.replace(str(tier["out"]) + ".tmp", tier["out"])
    return time.monotonic() - t0


def find_input(path, depth=6):
    """path when it exists, else the one path of the same name within depth levels of INPUT_ROOT."""
    path = Path(path)
    if path.exists() or not path.is_relative_to(INPUT_ROOT):
        return path
    found = [p for d in range(depth) for p in sorted(INPUT_ROOT.glob("/".join(["*"] * d + [path.name])))]
    if len(found) != 1:
        raise FileNotFoundError(f"{path} is missing, and {INPUT_ROOT} holds {len(found)} paths named {path.name}")
    print(f"{path.name}: using {found[0]}", flush=True)
    return found[0]


def stamp(path):
    st = Path(path).stat()
    return dict(path=str(Path(path).resolve()), bytes=st.st_size, mtime_ns=st.st_mtime_ns)


def prepare(model=MODEL, table=ENGRAM_TABLE, imp=IMP_DIR, prep_dir=PREP_DIR, engram_dirs=ENGRAM_DIRS,
            tiers=ENGRAM_TIERS, keep=ENGRAM_KEEP, threads=48):
    """Build (or reuse) the model links, the expert keep file and the engram hot sets; returns prep.json."""
    model, table, imp = find_input(model), find_input(table), find_input(imp)
    prep_dir = Path(prep_dir)
    prep_dir.mkdir(parents=True, exist_ok=True)
    table_rows, row_bytes, base = table_layout(table)
    if table_rows != ENGRAM_ROWS:
        raise ValueError(f"{table}: {table_rows} engram rows, expected the 1017 table's {ENGRAM_ROWS}")
    imp_files = [imp / "experts" / k / "scores.json" for k in ("notool", "tool")]
    imp_files += [imp / "engram" / kind / "counts.npz" for kind in ("notool", "tool")]
    if isinstance(keep, str):
        imp_files.append(imp / "engram" / "notool" / "masks" / f"{keep}.npz")
    elif not 0 < keep <= 1:
        raise ValueError(f"engram keep {keep!r} is neither a share in (0, 1] nor a mask name")
    inputs = dict(version=VERSION, table=stamp(table), imp=[stamp(p) for p in imp_files], keep=keep,
                  tiers=list(tiers), rank=[ENGRAM_TIER_TOOL, ENGRAM_TIER_FIRST], task=[TASK_PROMPT, TASK_DECODE],
                  engram_dirs=[str(d) for d in engram_dirs])
    manifest_path = prep_dir / "prep.json"
    previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else None

    model_path, links = link_model(model, table, prep_dir / "model")
    keep_file = prep_dir / "expert_keep_e288.u8"
    keep_sha256 = expert_keep(imp, keep_file)

    n_tiers = len(tiers) + 1
    outs = [Path(engram_dirs[min(k, len(engram_dirs) - 1)]) / f"t{k}.ehs" for k in range(n_tiers)]
    reuse = (previous is not None and previous.get("inputs") == inputs
             and all(out.is_file() and out.stat().st_size == t["file_bytes"]
                     for out, t in zip(outs, previous.get("engram", {}).get("tiers", []))))
    if reuse:
        engram = previous["engram"]
        print(f"engram hot sets: reusing {', '.join(map(str, outs))}", flush=True)
    else:
        if manifest_path.exists():
            manifest_path.unlink()
        # the hot sets of other inputs, which may be in other folders, before the new ones take the space
        for t in (previous or {}).get("engram", {}).get("tiers", []):
            for stale in (Path(t["file"]), Path(t["file"] + ".tmp")):
                if stale.name.endswith((".ehs", ".ehs.tmp")) and stale.exists():
                    stale.unlink()
        rows, summary = engram_tiers(imp, keep, tiers, row_bytes, table_rows, ENGRAM_TIER_TOOL, ENGRAM_TIER_FIRST)
        outs = outs[:len(rows)]
        header = 32 + 8 * ((table_rows + 63) // 64)
        need = {}
        for out, r in zip(outs, rows):
            out.parent.mkdir(parents=True, exist_ok=True)
            need[out.parent] = need.get(out.parent, 0) + header + len(r) * row_bytes
            for stale in (out, Path(str(out) + ".tmp")):
                if stale.exists():
                    stale.unlink()
        for folder, n in need.items():
            st = os.statvfs(folder)
            if st.f_bavail * st.f_frsize < n + 2**30:
                raise ValueError(f"{folder} has {st.f_bavail * st.f_frsize / 2**30:.1f} GiB free, the hot sets "
                                 f"need {n / 2**30:.1f} GiB")
        for t, s in zip(summary, rows):
            print(f"engram tier: {t['rows']} rows, {t['gib']} GiB, prompt lookups {t['prompt_share']:.2%}, "
                  f"decode lookups {t['decode_share']:.2%}, tool prompt {t['tool_prompt_share']:.2%}, "
                  f"tool decode {t['tool_decode_share']:.2%}", flush=True)
        seconds = write_hotsets(table, rows, outs, table_rows, row_bytes, base, threads=threads)
        del rows
        for t, out in zip(summary, outs):
            t.update(file=str(out), file_bytes=out.stat().st_size)
        engram = dict(tiers=summary, rows=sum(t["rows"] for t in summary), row_bytes=row_bytes,
                      table_rows=table_rows, build_seconds=round(seconds, 1))
    result = dict(inputs=inputs, model=str(model_path), links=links,
                  expert_keep=dict(file=str(keep_file), sha256=keep_sha256, kept=N_KEEP, experts=N_EXPERT),
                  engram=engram, engram_local=",".join(f"*:hot:{t['file']}" for t in engram["tiers"]))
    tmp = manifest_path.with_name("prep.json.tmp")
    tmp.write_text(json.dumps(result, indent=2) + "\n")
    os.replace(tmp, manifest_path)
    return result


def cpu_list(text):
    cpus = set()
    for part in filter(None, text.strip().split(",")):
        first, _, last = part.partition("-")
        if not first.isdigit() or not (last or first).isdigit() or int(first) > int(last or first):
            raise ValueError(f"Invalid CPU list entry {part!r}")
        cpus.update(range(int(first), int(last or first) + 1))
    return cpus


def cpu_ranges(cpus):
    out, cpus = [], sorted(cpus)
    for c in cpus:
        if out and c == out[-1][1] + 1:
            out[-1][1] = c
        else:
            out.append([c, c])
    return ",".join(f"{a}-{b}" if b > a else str(a) for a, b in out)


def cpu_cores(cpus):
    """The physical cores of the available CPUs, by first thread: (all its threads, the available ones)."""
    cores = {}
    for c in cpus:
        try:
            path = SYSFS_CPU / f"cpu{c}" / "topology" / "thread_siblings_list"
            siblings = frozenset(cpu_list(path.read_text()))
        except (OSError, ValueError):
            siblings = frozenset([c])
        cores.setdefault(siblings, set()).add(c)
    return sorted(((sorted(k), sorted(v)) for k, v in cores.items()), key=lambda core: core[0][0])


def cpu_layout(cpus=None, devices=4, ar_threads=4, max_group_threads=10):
    """The CPU pinning of the measured profile, from this machine's topology instead of fixed ids.

    GGML_CUDA_AR_CPU_CORE=c pins all-reduce thread t to CPU c+t: the first threads of the last cores whose ids follow
    each other. The engram workers get a spare thread of those cores, and the CPU expert jobs of each device a group
    of the whole cores that are left, at most 10 threads (on 4x L4 with 24 cores x 2: 0-4,24-28/.../15-19,39-43,
    all-reduce 20-23, engram 44).
    """
    cpus = set(os.sched_getaffinity(0) if cpus is None else cpus)
    cores = cpu_cores(cpus)
    first = {c: core for core in cores for c in core[1] if c == core[0][0]}
    ar = next((c for c in sorted(first, reverse=True) if all(c + t in first for t in range(ar_threads))), None)
    ar_cores = [first[ar + t] for t in range(ar_threads)] if ar is not None else []
    used = {id(core) for core in ar_cores}
    whole = [core for core in cores if id(core) not in used and core[0] == core[1]]
    per_core = len(whole[0][1]) if whole else 1
    n = min(len(whole) // devices, max(1, max_group_threads // per_core))
    layout = {}
    if n:
        groups = [sorted(c for core in whole[g * n:(g + 1) * n] for c in core[1]) for g in range(devices)]
        layout["GGML_CUDA_EXPERT_CACHE_CPU_HYBRID_CORES"] = "/".join(cpu_ranges(g) for g in groups)
        layout["GGML_CUDA_EXPERT_CACHE_CPU_HYBRID_THREADS"] = str(n * per_core)
        pinned = {c for g in groups for c in g}
    else:
        layout["GGML_CUDA_EXPERT_CACHE_CPU_HYBRID_THREADS"] = str(max_group_threads)
        pinned = set()
    if ar is not None:
        layout["GGML_CUDA_AR_CPU_CORE"] = str(ar)
        pinned |= set(range(ar, ar + ar_threads))
    spare = [c for core in ar_cores for c in core[1] if c not in pinned] or sorted(cpus - pinned, reverse=True)
    if spare:
        layout["LLAMA_DSV41_ENGRAM_CPUS"] = str(spare[0])
    return layout


def keep_arg(text):
    try:
        return float(text)
    except ValueError:
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", text):
            raise argparse.ArgumentTypeError(f"not a share or a mask name: {text!r}") from None
        return text


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=MODEL)
    p.add_argument("--engram-table", default=ENGRAM_TABLE)
    p.add_argument("--imp-dir", default=IMP_DIR)
    p.add_argument("--prep-dir", default=PREP_DIR)
    p.add_argument("--engram-dirs", default=",".join(ENGRAM_DIRS),
                   help="Tier k goes to the k-th folder (the last for the rest).")
    p.add_argument("--engram-tiers", default=",".join(map(str, ENGRAM_TIERS)))
    p.add_argument("--engram-keep", type=keep_arg, default=ENGRAM_KEEP,
                   help="Share of each layer's engram rows to keep, or the name of a notool mask of the dataset.")
    p.add_argument("--threads", type=int, default=48)
    a = p.parse_args()
    result = prepare(a.model, a.engram_table, a.imp_dir, a.prep_dir, a.engram_dirs.split(","),
                     [float(x) for x in a.engram_tiers.split(",") if x], a.engram_keep, a.threads)
    print(f"prep: {Path(a.prep_dir) / 'prep.json'} ({result['engram']['rows']} engram rows)", flush=True)


if __name__ == "__main__":
    sys.exit(main())
