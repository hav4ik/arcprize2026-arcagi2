#!/usr/bin/env python3
# NOT USED (E064, 2026-10-28: superseded by B32-M; kept for reference)
"""E064: Manh's Kaggle server notebook (nguyennguyen599/arc2-2026-harness-deepseek: his CUDA 13 + vLLM install and
server, http://127.0.0.1:18084/v1) + B32's harness on top, in external-server mode (pinned_manh/).

Manh's cells are kept verbatim, except his cell that cats a hardcoded old server log and his commented-out pipeline
cell. Our cells (cells/manh_m*.py) follow: M1 settings + clock + fallback submission + the kit (embedded base64,
sha256-checked; no dataset to share), M2 the server contract check, M3 the solver, M4 the submission check.

Usage: build_manh_stack.py --manh-nb DIR (kernels pull of his notebook) [--slug arc2-2026-manh-b32] [--owner chankhavu]
Writes out/nb_<slug>/<slug>.ipynb + kernel-metadata.json. Nothing is pushed: every Kaggle run needs the user's OK.
"""
import argparse
import base64
import gzip
import hashlib
import io
import json
import tarfile
from pathlib import Path

from build_frozen_mt import KIT_TGZ, PINS_MTIME, check_imports

HERE = Path(__file__).resolve().parent
PINNED = HERE / "pinned_manh"
CELLS = [HERE / f"cells/{n}" for n in ("manh_m1_setup.py", "manh_m2_contract.py", "manh_m3_solver.py",
                                         "manh_m4_final.py")]
# B32's pinned files at their kit paths (pinned_manh: + external mode, + `reasoning` on re-render)
PINS = {"solver_dsv41.py": "solver_dsv41.py", "repo/evals/run_deepseek_tools.py": "run_deepseek_tools.py",
        "repo/evals/train_check.py": "train_check.py", "repo/evals/auto_repair.py": "auto_repair.py",
        "repo/evals/auto_repair_names.json": "auto_repair_names.json"}
SKIP = {"MANIFEST.json", "stage_model.py"}  # staging is Manh's (the model mounts read-only)


def manh_cells(nb):
    """Manh's cells to keep: drop the hardcoded server-log cat and the commented-out full pipeline cell."""
    keep = []
    for c in nb["cells"]:
        src = "".join(c["source"])
        if src.startswith("!cat /kaggle/working/dsv41_server_"):
            continue
        if src.strip() and all(l.lstrip().startswith("#") for l in src.splitlines() if l.strip()):
            continue
        keep.append(c)
    srcs = ["".join(c["source"]) for c in keep]
    assert any("global_end_time" in s for s in srcs), "Manh's clock cell is missing"
    assert any("dsv41_vllm.py" in s and "--qwen38-port 18084" in s for s in srcs), "Manh's server cell is missing"
    return keep


def build_kit():
    """The kit tarball (deterministic) + {path: sha256}: the Version 5 kit's harness tree with B32's pinned files."""
    files = {}
    with tarfile.open(KIT_TGZ) as tf:
        for m in tf.getmembers():
            name = m.name.removeprefix("./")
            if m.isfile() and name not in SKIP and not name.startswith("vllm_overlay"):
                files[name] = tf.extractfile(m).read()
    for arc, src in PINS.items():
        files[arc] = (PINNED / src).read_bytes()
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w", format=tarfile.USTAR_FORMAT) as tf:
        for arc in sorted(files):
            info = tarfile.TarInfo(arc)
            info.size, info.mtime, info.mode, info.uid, info.gid, info.uname, info.gname = (
                len(files[arc]), PINS_MTIME, 0o644, 0, 0, "", "")
            tf.addfile(info, io.BytesIO(files[arc]))
    gz = io.BytesIO()
    with gzip.GzipFile(fileobj=gz, mode="wb", mtime=0, filename="") as g:
        g.write(raw.getvalue())
    return gz.getvalue(), {arc: hashlib.sha256(b).hexdigest() for arc, b in sorted(files.items())}


def code(src):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": src.splitlines(keepends=True)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manh-nb", type=Path, required=True, help="folder of `kaggle kernels pull -m` of his notebook")
    ap.add_argument("--slug", default="arc2-2026-manh-b32")
    ap.add_argument("--owner", default="chankhavu")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    meta = json.load(open(a.manh_nb / "kernel-metadata.json"))
    nb = json.load(open(a.manh_nb / meta["code_file"]))
    bad = check_imports({arc: PINNED / src for arc, src in PINS.items()})
    assert not bad, f"pinned files import names the kit lacks: {bad}"
    tgz, sha = build_kit()
    cells = manh_cells(nb)
    m1 = CELLS[0].read_text()
    assert m1.count('"__KIT_B64__"') == 1 and m1.count('"""__KIT_SHA__"""') == 1
    m1 = m1.replace('"__KIT_B64__"', '"' + base64.b64encode(tgz).decode() + '"').replace(
        '"""__KIT_SHA__"""', '"""' + json.dumps(sha, sort_keys=True) + '"""')
    nb["cells"] = cells + [code(m1)] + [code(p.read_text()) for p in CELLS[1:]]
    out = a.out or HERE / f"out/nb_{a.slug}"
    out.mkdir(parents=True, exist_ok=True)
    json.dump(nb, open(out / f"{a.slug}.ipynb", "w"), indent=1, ensure_ascii=False)
    for k in ("id_no",):
        meta.pop(k, None)
    meta.update(id=f"{a.owner}/{a.slug}", title=a.slug, code_file=f"{a.slug}.ipynb", is_private=True,
                enable_gpu=True, enable_internet=False, machine_shape="NvidiaL4")
    assert "arc-prize-2026-arc-agi-2" in meta["competition_sources"], "4xL4 needs the competition attached"
    json.dump(meta, open(out / "kernel-metadata.json", "w"), indent=1)
    (out / "KIT_SHA256.json").write_text(json.dumps({"tgz": hashlib.sha256(tgz).hexdigest(), "files": sha}, indent=1))
    print(out, f"{len(cells)} of Manh's cells + 4 ours; kit {len(tgz) / 1e3:.0f} kB, {len(sha)} files;",
          {f: h[:8] for f, h in sha.items() if f in PINS}, {k: meta[k] for k in ("id", "dataset_sources",
                                                                                  "model_sources", "docker_image")})


if __name__ == "__main__":
    main()
