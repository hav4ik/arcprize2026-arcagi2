#!/usr/bin/env python3
"""V12 + images (2026-10-30): Manh's V12 notebook (scriptVersionId 363638159, the user's gold standard: replica 80.83,
0 events) with the E058 image switch and nothing else changed in its behaviour.

Changes, all listed line by line in out/nb_v12_img/README.md:
- cell 00: ARC_GRID_FORMAT=text+image, MANH_VISION_MAX_IMAGES=32 (V12: 16; ARC's most is 22 images per prompt), and
  V12IMG_MODEL_SRC, the one model-input setting ("" = the attached Kaggle Model);
- cell 0b: the pinned files of pinned_v12img/ (V12's solver 11bfaaaa + the image switch + recovery R1 / R2; V12's
  harness 882c817f + the minimal E058 text+image code + the R2 turn-replay hook; arc_harness/grid_image.py 18ed78fb; V12's train_check / auto-repair unchanged), their
  sha256, ARC_GRID_FORMAT / MANH_VISION_MAX_IMAGES in the settings check, the image features in the feature check;
- cell 1b: the integrity manifest of the image model (E066 candidate 1's sha256 manifest; --integrity-manifest);
- cell 2m: --disable-chunked-mm-input when ARC_GRID_FORMAT=text+image (each image span prefilled whole, as
  DeepSeek's reference and the image model's calibration); --watermark-reload-only (Manh, 2026-10-30; text and image);
- new cell 2i (commit runs only): the image path's smoke checks in the background (cells/v12img_smoke.py);
- new cell 4i (commit runs only): their verdict (cells/v12img_verdict.py);
- the markdown cell: a V12 + images note on top;
- metadata: our slug, the image model as the model input (--model-source: E066 candidate 1's Kaggle Model);
  V12's datasets, docker image, machine;
- the AGENTS.md rule (Manh, 2026-10-30): V12's recursive /kaggle/input lookups become the bounded ones of
  input_paths.py (they find the same files on Kaggle's and the pods' layouts; --keep-v12-lookups leaves them).
Usage: build_v12_img.py [--v12 DIR] [--out DIR] [--model-source SRC] [--model-dir DIR] [--integrity-manifest F]
Nothing is pushed.
"""
import argparse
import base64
import copy
import gzip
import hashlib
import io
import json
import tarfile
from pathlib import Path

import input_paths as IP

HERE = Path(__file__).resolve().parent
V12_DIR = HERE / "out/manh_v12_363638159"
V12_SHA = "c47be5a819ad63a09c6ee831bff7f9770e070cd91fe04a27865f6ddd6e7ec42f"
P = HERE / "pinned_v12img"
# overlay arcname -> source; V12's five files first, in V12's order, then the image renderer
FILES = {"solver_dsv41.py": P / "solver_dsv41.py", "repo/evals/run_deepseek_tools.py": P / "run_deepseek_tools.py",
         "repo/evals/train_check.py": P / "train_check.py", "repo/evals/auto_repair.py": P / "auto_repair.py",
         "repo/evals/auto_repair_names.json": P / "auto_repair_names.json",
         "repo/evals/arc_harness/grid_image.py": P / "arc_harness/grid_image.py"}
# V12's own pins of the four files this build leaves unchanged (sha256 prefixes)
UNCHANGED = {"repo/evals/train_check.py": "8fe36175", "repo/evals/auto_repair.py": "2292369c",
             "repo/evals/auto_repair_names.json": "571f47eb"}
NEED_EXTRA = {"solver_dsv41.py": ['GRID_FORMAT = E("ARC_GRID_FORMAT"', "MM_KEYS"],
              "repo/evals/run_deepseek_tools.py": ["def chat_context_cap", "def text_image_content"],
              "repo/evals/arc_harness/grid_image.py": ["def aligned_cell"]}
EXPECT_EXTRA = {"ARC_GRID_FORMAT": "text+image", "MANH_VISION_MAX_IMAGES": "32"}
# E066 candidate 1 (uniform E128 2-bit, vision tower kept, E128's 16 Engram row-chunk files; 137 files, 132.06 GB),
# private under arazzd, shared with chankhavu; cell 1b checks it against c1_gptq_sha256_manifest.json
MODEL_SOURCE = "arazzd/deepseek-v41-flash-1017-arc-tool-img-e128-2bit/PyTorch/default/1"
MANIFEST = HERE / "out/nb_v12_img/model_manifest_c1.json"  # c1_gptq_sha256_manifest.json in cell 1b's format
# the causal re-prune (HF chankhavu/DeepSeek-V4.1-Flash-1017-arc-tool-img-E128-2bit-causal@209c810a74 + E128's 16 Engram
# files), predicted from the Hub's sha256 listing; not on Kaggle yet
CAUSAL_MANIFEST = HERE / "out/nb_v12_img/model_manifest_causal_predicted.json"
CELL00_WHICH = (
    "# V12 + images: WHICH image model (one setting). \"c1\" = arazzd/deepseek-v41-flash-1017-arc-tool-img-e128-2bit (E066\n"
    "# candidate 1, on Kaggle); \"causal\" = the causal re-prune (HF chankhavu/DeepSeek-V4.1-Flash-1017-arc-tool-img-E128-\n"
    "# 2bit-causal + E128's 16 Engram files). To switch, change this AND the model_sources line of kernel-metadata.json;\n"
    "# cell 1b then checks that model's manifest.\n"
    "V12IMG_MODEL = \"c1\"\n"
    "os.environ.setdefault(\"V12IMG_MODEL\", V12IMG_MODEL)\n")
SLUG = "dsv41-tools-v12-img"
SMOKE_CELL, VERDICT_CELL = HERE / "cells/v12img_smoke.py", HERE / "cells/v12img_verdict.py"

MARKDOWN_HEAD = """\
# V12 + images

Manh's V12 (`nguyennguyen599/dsv41-tools-submit-b32m-external-server`, scriptVersionId 363638159) with the E058 image
harness switched on and nothing else changed in its behaviour (`build_v12_img.py`): `ARC_GRID_FORMAT=text+image` (one
token-aligned image per grid next to the unchanged JSON prompt, chat path instead of `--token-exact`), 32 images per
prompt, `--disable-chunked-mm-input`, the image E128 model (E066 candidate 1), and commit-run image checks (cells 2i,
4i).
`ARC_GRID_FORMAT=json` gives V12's text harness exactly.

"""
CELL00_HEAD = ("# V12 + IMAGES (build_v12_img.py, 2026-10-30): Manh's V12 (363638159) + the image switch: ARC_GRID_FORMAT,\n"
               "# MANH_VISION_MAX_IMAGES and the model input (V12IMG_MODEL_SRC) below; everything else is V12's.\n")
CELL00_SETTINGS_OLD = '               ("ARC_MAX_SERVER_RESTARTS", "3")):\n'
CELL00_SETTINGS_NEW = (
    '               ("ARC_MAX_SERVER_RESTARTS", "3"),\n'
    '               ("ARC_GRID_FORMAT", "text+image"),  # V12 + images: the E058 image harness (json = V12\'s text)\n'
    '               ("MANH_VISION_MAX_IMAGES", "32")):  # V12 + images: images per prompt (cell 2m); ARC needs <= 22\n')
CELL00_LOOP_END = "    os.environ.setdefault(_k, _v)\n"
CELL00_MODEL = (
    "# V12 + images: the model input. \"\" = the attached Kaggle Model (cell 0: the dir holding e043_manifest.json); a pod\n"
    "# run may name its model dir here instead (the E066 candidate 1 files, 137, as cell 1b lists them).\n"
    "V12IMG_MODEL_SRC = __MODEL_DIR__\n"
    "if V12IMG_MODEL_SRC:\n"
    "    os.environ.setdefault(\"ARC_MODEL_SRC\", V12IMG_MODEL_SRC)\n")
CELL00_PRINT = ('print("V12 + images settings:", {k: os.environ.get(k) for k in ("ARC_GRID_FORMAT", "MANH_VISION_MAX_IMAGES",\n'
                '                                                         "MANH_VISION", "ARC_MODEL_SRC")}, flush=True)\n')
CELL0B_NOTE = ("# V12 + images: the pinned files are pinned_v12img/'s (V12's solver + the image switch, V12's harness + the\n"
               "# minimal E058 text+image code, arc_harness/grid_image.py for the images; train_check / auto-repair are V12's).\n")
CELL2M_ANCHOR = '                "--vllm-arg=--prefix-cache-retention-interval=128"]\n'
# Manh (2026-10-30): the 0.1 watermark applies only to preempted requests and KV-connector / offload reloads (fork
# SchedulerConfig.watermark_reload_only, in his g22df163d2d wheels; not in g8cf0c0664b); text and image alike
CELL2M_WATERMARK = '"--vllm-arg=--watermark=0.1", '
CELL2M_WATERMARK_NEW = '"--vllm-arg=--watermark=0.1", "--vllm-arg=--watermark-reload-only", '
CELL2M_ADD = (
    '        if os.environ.get("ARC_GRID_FORMAT") == "text+image":  # V12 + images: each image span prefilled whole\n'
    '            _cmd.append("--vllm-arg=--disable-chunked-mm-input")  # (DeepSeek\'s reference; the model\'s calibration)\n')


def src(c):
    s = c["source"]
    return s if isinstance(s, str) else "".join(s)


def set_src(c, text):
    c["source"] = text


def code(text):
    return {"cell_type": "code", "source": text, "metadata": {"trusted": True}, "outputs": [], "execution_count": None}


def find(cells, prefix):
    hits = [i for i, c in enumerate(cells) if src(c).startswith(prefix)]
    assert len(hits) == 1, f"cell {prefix!r}: {len(hits)} matches"
    return hits[0]


def replace_once(text, old, new):
    assert text.count(old) == 1, f"expected once: {old[:80]!r} ({text.count(old)})"
    return text.replace(old, new)


def sha(b):
    return hashlib.sha256(b).hexdigest()


def overlay_b64():
    """The cell 0b overlay: a deterministic tar.gz (fixed order, metadata and gzip mtime) of FILES."""
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w", format=tarfile.USTAR_FORMAT) as tf:
        for arc, path in FILES.items():
            data = path.read_bytes()
            info = tarfile.TarInfo(arc)
            info.size, info.mode, info.mtime, info.uid, info.gid = len(data), 0o644, 0, 0, 0
            tf.addfile(info, io.BytesIO(data))
    gz = io.BytesIO()
    with gzip.GzipFile(fileobj=gz, mode="wb", mtime=0, filename="") as g:
        g.write(raw.getvalue())
    return base64.b64encode(gz.getvalue()).decode()


def v12_overlay(cell0b):
    """V12's own cell 0b overlay: {arcname: bytes}, checked against its _MT_SHA."""
    b64 = cell0b.split('_MT = "', 1)[1].split('"', 1)[0]
    pins = json.loads(cell0b.split("_MT_SHA = ", 1)[1].split("\n", 1)[0])
    out = {}
    with tarfile.open(fileobj=io.BytesIO(base64.b64decode(b64))) as tf:
        for m in tf.getmembers():
            out[m.name] = tf.extractfile(m).read()
    assert {k: sha(v) for k, v in out.items()} == pins, "V12's overlay does not match its own _MT_SHA"
    return out


def json_line(cell, name):
    return json.loads(cell.split(f"\n{name} = ", 1)[1].split("\n", 1)[0])


def set_json_line(cell, name, value):
    old = f"\n{name} = " + cell.split(f"\n{name} = ", 1)[1].split("\n", 1)[0] + "\n"
    return replace_once(cell, old, f"\n{name} = {json.dumps(value)}\n")


def build(v12_dir=V12_DIR, model_source=MODEL_SOURCE, model_dir="", integrity=None, keep_v12_lookups=False,
          slug=SLUG, causal=None):
    meta0 = json.loads((v12_dir / "kernel-metadata.json").read_text())
    raw = (v12_dir / meta0["code_file"]).read_bytes()
    assert sha(raw) == V12_SHA, f"not V12's notebook: {sha(raw)}"
    v12 = json.loads(raw)
    nb = copy.deepcopy(v12)
    cells = nb["cells"]

    # markdown
    assert cells[0]["cell_type"] == "markdown"
    set_src(cells[0], MARKDOWN_HEAD + src(cells[0]))
    # cell 00
    i = find(cells, "# B32-M + MANH'S SERVER")
    t = CELL00_HEAD + src(cells[i])
    t = replace_once(t, CELL00_SETTINGS_OLD, CELL00_SETTINGS_NEW)
    t = replace_once(t, CELL00_LOOP_END, CELL00_LOOP_END + CELL00_MODEL.replace("__MODEL_DIR__", json.dumps(model_dir))
                     + (CELL00_WHICH if causal else ""))
    t = t + ("" if t.endswith("\n") else "\n") + CELL00_PRINT
    set_src(cells[i], t)
    # cell 0b
    i = find(cells, "# Cell 0b (E046, frozen inputs)")
    t = src(cells[i])
    base = v12_overlay(t)
    for arc, prefix in UNCHANGED.items():
        assert sha(base[arc]).startswith(prefix) and FILES[arc].read_bytes() == base[arc], arc
    assert sha(base["solver_dsv41.py"]).startswith("11bfaaaa")
    assert sha(base["repo/evals/run_deepseek_tools.py"]).startswith("882c817f")
    t = replace_once(t, "\nimport base64, hashlib, io, shutil, tarfile\n",
                     "\n" + CELL0B_NOTE + "import base64, hashlib, io, shutil, tarfile\n")
    old_mt = '_MT = "' + t.split('_MT = "', 1)[1].split('"', 1)[0] + '"'
    t = replace_once(t, old_mt, f'_MT = "{overlay_b64()}"')
    t = set_json_line(t, "_MT_SHA", {arc: sha(p.read_bytes()) for arc, p in FILES.items()})
    expect = json_line(t, "_MT_EXPECT")
    t = set_json_line(t, "_MT_EXPECT", {**expect, **EXPECT_EXTRA})
    need = json_line(t, "_MT_NEED")
    for f, needs in NEED_EXTRA.items():
        need[f] = need.get(f, []) + [n for n in needs if n not in need.get(f, [])]
    t = set_json_line(t, "_MT_NEED", need)
    assert json_line(t, "_MT_NEW") == ["repo/evals/auto_repair.py", "repo/evals/auto_repair_names.json"]
    set_src(cells[i], t)
    # cell 1b
    if integrity:
        i = find(cells, "# Cell 1b (smoke, commit runs only)")
        t = src(cells[i])
        head = 'INTEGRITY = json.loads(r"""'
        old_blob = t.split(head, 1)[1].split('""")', 1)[0]
        def blob(path):
            man = json.loads(Path(path).read_text())
            assert set(man) == {"model"} and all(isinstance(v, list) and len(v) == 2 for v in man["model"].values())
            return json.dumps({"model": dict(sorted(man["model"].items()))})
        if causal:  # one setting (cell 00's V12IMG_MODEL) picks the manifest of the attached model
            new = ('INTEGRITY_BY_MODEL = {"c1": json.loads(r"""' + blob(integrity) + '"""),\n'
                   '                      "causal": json.loads(r"""' + blob(causal) + '""")}  # V12 + images\n'
                   'INTEGRITY = INTEGRITY_BY_MODEL[os.environ.get("V12IMG_MODEL", "c1")]')
            set_src(cells[i], replace_once(t, head + old_blob + '""")', new))
        else:
            set_src(cells[i], replace_once(t, head + old_blob + '""")', head + blob(integrity) + '""")'))
    # cell 2m
    i = find(cells, "# Cell 2m (B32-M + Manh's server)")
    t = replace_once(src(cells[i]), CELL2M_ANCHOR, CELL2M_ANCHOR + CELL2M_ADD)
    t = replace_once(t, CELL2M_WATERMARK, CELL2M_WATERMARK_NEW)
    assert '"VLLM_GPU_DRIVER_RESERVE_MB": "256",' in t  # Manh's second ask: already V12's
    set_src(cells[i], t)
    # new cells 2i (after 2c) and 4i (last)
    i = find(cells, "# Cell 2c (B32-M smoke")
    cells.insert(i + 1, code(SMOKE_CELL.read_text()))
    assert src(cells[-1]).startswith("# B32-M smoke verdict")
    cells.append(code(VERDICT_CELL.read_text()))

    meta = {k: v for k, v in meta0.items() if k != "id_no"}
    meta.update(id=f"chankhavu/{slug}", title=slug, code_file=f"{slug}.ipynb", model_sources=[model_source],
                is_private=True, enable_internet=False)
    if not keep_v12_lookups:
        IP.finish(nb, meta)
        for c in cells:  # back to Kaggle's one-string sources (finish() splits them into lines)
            c["source"] = src(c)
    return nb, meta, v12, meta0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--v12", type=Path, default=V12_DIR)
    ap.add_argument("--out", type=Path, default=HERE / "out/nb_v12_img")
    ap.add_argument("--slug", default=SLUG)
    ap.add_argument("--model-source", default=MODEL_SOURCE,
                    help="kernel-metadata model_sources entry (owner/model/framework/variation/version)")
    ap.add_argument("--model-dir", default="", help="cell 00's V12IMG_MODEL_SRC (a staged dir); '' = the attached model")
    ap.add_argument("--integrity-manifest", type=Path,
                    help="cell 1b's INTEGRITY ({'model': {rel: [size, sha256]}}); default: model_manifest_c1.json")
    ap.add_argument("--docker-latest", action="store_true",
                    help="Kaggle's latest docker image instead of V12's pin (Manh's private BYOD image, which another "
                         "account may not be able to use)")
    ap.add_argument("--keep-v12-lookups", action="store_true",
                    help="leave V12's recursive /kaggle/input lookups (breaks the AGENTS.md rule; pod A/B only)")
    a = ap.parse_args()
    integrity = a.integrity_manifest or MANIFEST
    nb, meta, _v12, _m = build(a.v12, a.model_source, a.model_dir, integrity if integrity.exists() else None,
                               a.keep_v12_lookups, a.slug, CAUSAL_MANIFEST if CAUSAL_MANIFEST.exists() else None)
    if a.docker_latest:
        meta.pop("docker_image", None)
        meta["docker_image_pinning_type"] = "latest"
    a.out.mkdir(parents=True, exist_ok=True)
    text = json.dumps(nb, ensure_ascii=False, separators=(",", ":"))
    (a.out / meta["code_file"]).write_text(text)
    (a.out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"built {a.out / meta['code_file']} (sha256 {sha(text.encode())[:12]}): {len(nb['cells'])} cells; pins",
          {arc: sha(p.read_bytes())[:8] for arc, p in FILES.items()}, "| model", meta["model_sources"],
          "| integrity", integrity if integrity.exists() else "V12's (unchanged)")


if __name__ == "__main__":
    main()
