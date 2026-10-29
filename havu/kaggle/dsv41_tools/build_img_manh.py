#!/usr/bin/env python3
"""E066 image candidate on Manh's own notebook (2026-10-29): his "B32-M + Manh's production vLLM server" notebook
(nguyennguyen599/dsv41-tools-submit-b32m-external-server, pulled with `kaggle kernels pull -m`) is the source of truth
for serving: his install cells, his production server cell 2m (dsv41_vllm.py, his flags and env), his solver cell 3
attaching through ARC_EXTERNAL_SERVER. On top, only what the E066 image candidate needs:
- cell 00: ARC_GRID_FORMAT=text+image (our harness's image switch) and MANH_VISION_MAX_IMAGES=32 (his cell 2m reads
  it; his launcher accepts 2-32 images per prompt, the harness sends one per grid, ARC's worst case ~24);
- cell 0b: our pinned files (pinned_img/: B32's solver + S25 = his S23 ARC_EXTERNAL_SERVER + the image switch; the
  live harness frozen 2026-10-29 with E058 text+image and the chat-path --context-limit; the current
  arc_harness/grid_image.py; B32's train_check and auto-repair), his settings checks + ARC_GRID_FORMAT;
- cell 2r: our multi-variant, sha256-guarded vLLM patch (per-layer routed-expert counts in model.py + the router's
  expert-count whitelist in dsv4_topk.py; for his g5e1bccd07c wheel also the Engram row chunks) instead of his
  row-chunk-only cell (a newer wheel of his already has row chunks);
- cell 2c: the API contract check also requires parsed chat tool calls and an accepted image;
- metadata: our slug and the E066 Kaggle Model; his datasets and docker image.
Usage: build_img_manh.py --manh-nb DIR --model-source owner/model/framework/instance/version [--slug S] [--out DIR]
Nothing is pushed.
"""
import argparse
import base64
import hashlib
import json
import re
from pathlib import Path

import build_frozen_mt as B

HERE = Path(__file__).resolve().parent
P = HERE / "pinned_img"
PATCH_CELL = HERE / "cells/img_patches.py"
PATCH_DIFFS = {"MODEL5E1B": HERE / "b32m_patches/dynamic_count_rowchunk_model.diff",
               "MODELF506": HERE / "b32m_patches/dynamic_count_model_f506.diff",
               "TOPK": HERE / "b32m_patches/dynamic_count_dsv4_topk.diff"}
CONTRACT = HERE / "cells/b32m_contract.py"
FILES = {"solver_dsv41.py": P / "solver_dsv41.py", "repo/evals/run_deepseek_tools.py": P / "run_deepseek_tools.py",
         "repo/evals/arc_harness/grid_image.py": P / "arc_harness/grid_image.py",
         "repo/evals/train_check.py": P / "train_check.py", "repo/evals/auto_repair.py": P / "auto_repair.py",
         "repo/evals/auto_repair_names.json": P / "auto_repair_names.json"}
NEED_EXTRA = {"solver_dsv41.py": ['GRID_FORMAT = E("ARC_GRID_FORMAT"', 'EXTERNAL = E("ARC_EXTERNAL_SERVER"'],
              "repo/evals/run_deepseek_tools.py": ["def chat_context_cap", "def text_image_content"],
              "repo/evals/arc_harness/grid_image.py": ["def aligned_cell"]}


def src(c):
    return "".join(c["source"])


def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": text.splitlines(keepends=True)}


def replace_once(text, old, new):
    assert text.count(old) == 1, f"expected once: {old[:80]!r} ({text.count(old)})"
    return text.replace(old, new)


def find(cells, prefix):
    hits = [i for i, c in enumerate(cells) if src(c).startswith(prefix)]
    assert len(hits) == 1, f"cell {prefix!r}: {len(hits)} matches"
    return hits[0]


def json_line(cell, name):
    return json.loads(cell.split(f"{name} = ", 1)[1].split("\n", 1)[0])


def patch_cell():
    text = PATCH_CELL.read_text()
    for key, path in PATCH_DIFFS.items():
        diff = path.read_bytes()
        text = replace_once(text, f"__{key}_DIFF_B64__", base64.b64encode(diff).decode())
        text = replace_once(text, f"__{key}_DIFF_SHA__", hashlib.sha256(diff).hexdigest())
    return text


def contract_image_block():
    """Our contract cell's E066 block (required chat tool calls + one accepted image), to graft onto his cell."""
    text = CONTRACT.read_text()
    start = text.index('    if os.environ.get("ARC_GRID_FORMAT") == "text+image":')
    end = text.index('    return {**res, "fail": fail, "done": True}\n', start)
    return text[start:end]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manh-nb", type=Path, required=True, help="`kaggle kernels pull -m` folder of his notebook")
    ap.add_argument("--model-source", required=True)
    ap.add_argument("--slug", default="dsv41-tools-submit-img-manh")
    ap.add_argument("--vision-max-images", type=int, default=32)
    ap.add_argument("--commit", choices=["light", "load"], default="light",
                    help="commit run: light = 2 smallest tasks (00576224, 25ff71a9 with 2 test inputs), ends when they are done, "
                         "deadline cap ~45 min (saves quota; load testing is done on pods); load = his 80-task load test")
    ap.add_argument("--model-manifest", type=Path, required=True,
                    help="the E066 export's path/size/sha256 manifest: commit-run input integrity (cell 1b)")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    meta = json.load(open(a.manh_nb / "kernel-metadata.json"))
    nb = json.load(open(a.manh_nb / meta["code_file"]))
    cells = nb["cells"]

    # cell 00: settings (os.environ.setdefault tuples)
    i = find(cells, "# B32-M + MANH'S SERVER")
    s0 = replace_once(src(cells[i]), '               ("ARC_PORT", "18084"),\n',
                      '               ("ARC_PORT", "18084"),\n'
                      '               ("ARC_GRID_FORMAT", "text+image"),  # E066: the E058 image harness\n'
                      f'               ("MANH_VISION_MAX_IMAGES", "{a.vision_max_images}"),  # E066: one image per grid\n'
                      '               ("MANH_VISION", "1"),  # E066: his vision recipe (images), not --language-model-only\n')
    s0 = (f"# E066 image candidate (build_img_manh.py, 2026-10-29): Manh's notebook + our image harness and the E066\n"
          f"# checkpoint {a.model_source} (per-layer expert counts: cell 2r).\n" + s0)
    if a.commit == "light":  # commit runs only (the competition rerun never reads these)
        start = s0.index('    os.environ["B32M_LOAD_TEST"] = "1"\n')
        end = s0.index('    print("B32-M LOAD TEST: eval file", _lf[:1], flush=True)\n')
        end += len('    print("B32-M LOAD TEST: eval file", _lf[:1], flush=True)\n')
        s0 = (s0[:start] + '    os.environ.setdefault("ARC_TASK_IDS", "00576224,25ff71a9")  # E066 light smoke: the 2 smallest tasks\n'
              '    print("E066 LIGHT SMOKE: tasks", os.environ["ARC_TASK_IDS"], flush=True)\n' + s0[end:])
        for k, old_v, new_v in (("ARC_STOP_NEW_S", "2280", "2700"), ("ARC_FORCE_AT_S", "2400", "2820"),
                                ("ARC_HARD_STOP_S", "2460", "2880"), ("ARC_KILL_AT_S", "2550", "2970")):
            s0 = replace_once(s0, f'os.environ.setdefault("{k}", "{old_v}")', f'os.environ.setdefault("{k}", "{new_v}")')
        s0 = s0.replace("    # smoke within 43 min of the first cell", "    # E066 light smoke: capped at ~50 min of the first cell (ends when the 2 tasks are done)")
    cells[i] = code(s0)

    # cell 0b: our pins, his checks + ours
    j = find(cells, "# Cell 0b (E046, frozen inputs)")
    his = src(cells[j])
    expect = json_line(his, "_MT_EXPECT")
    expect["ARC_GRID_FORMAT"] = "text+image"
    expect["MANH_VISION"] = "1"
    need = json_line(his, "_MT_NEED")
    for f, needs in NEED_EXTRA.items():
        need.setdefault(f, [])
        need[f] += [n for n in needs if n not in need[f]]
    new = json_line(his, "_MT_NEW")
    bad = B.check_imports(FILES)
    assert not bad, f"pinned files import names the notebook would not have: {bad}"
    cell0b, sha = B.code_cell(FILES, expect, new, need)
    cells[j] = code(cell0b)

    # cell 2r: our multi-variant vLLM patch. His v4 had a row-chunk-only 2r (replaced); from his v5 on, his wheels load
    # row chunks themselves and 2r is gone, so ours goes right after his install cells (before cell 2's env record)
    hits = [i for i, c in enumerate(cells) if src(c).startswith("# Cell 2r ")]
    if hits:
        cells[hits[0]] = code(patch_cell())
    else:
        k = find(cells, "# Cell 2 (B32-M): Manh's environment")
        assert src(cells[k - 1]).startswith("# Manh's cell 5"), "expected his install cells right before cell 2"
        cells.insert(k, code(patch_cell()))

    # cell 2c: the contract also needs chat tool calls and images
    m = find(cells, "# Cell 2c ")
    c2 = src(cells[m])
    anchor = '    return {**res, "fail": fail, "done": True}\n'
    assert c2.count(anchor) == 1 and "ARC_GRID_FORMAT" not in c2
    c2 = c2.replace(anchor, contract_image_block() + anchor)
    # the chat tool-call probe is fatal with images: give a 2-bit model room to finish thinking before its call
    c2 = replace_once(c2, '"temperature": 0,\n                                          "max_tokens": 1500, "chat_template_kwargs": kw})',
                      '"temperature": 0,\n                                          "max_tokens": 4000, "chat_template_kwargs": kw})')
    if "import sys" not in c2 and ", sys," not in c2:
        c2 = replace_once(c2, "import re, threading, urllib.request", "import re, sys, threading, urllib.request")
    cells[m] = code(c2)

    # cell 1b: commit-run input integrity against the E066 export's manifest (his cell checks E128's)
    q = find(cells, "# Cell 1b ")
    cells[q] = code(B.integrity_cell(a.model_manifest))

    # cell 4b report: the external solver logs "server up after Ns (external); ..."; his server's log is
    # MANH_SERVER_LOG; with images, a contract failure (tool calls not parsed, image dropped) fails the commit
    r = find(cells, "# Cell 4b ")
    rep = src(cells[r])
    rep = replace_once(rep, 'r"server up after (\\d+)s; (.*)"', 'r"server up after (\\d+)s(?: \\(external\\))?; (.*)"')
    assert rep.count('open(f"{_w}/dsv41_logs/server.log", errors="replace")') == 2
    rep = rep.replace('open(f"{_w}/dsv41_logs/server.log", errors="replace")',
                      'open(globals().get("MANH_SERVER_LOG") or f"{_w}/dsv41_logs/server.log", errors="replace")')
    for fatal in ('("server never", "environment", "row-chunk")', '("server never", "environment")'):  # his v4 / v5
        old = f'_fatal = [x for x in _b32m_bad if x.startswith({fatal})]'
        if rep.count(old) == 1:
            rep = rep.replace(old, f'_fatal = [x for x in _b32m_bad if x.startswith({fatal}\n'
                              '                  + (("contract",) if os.environ.get("ARC_GRID_FORMAT") == "text+image" else ()))]')
            break
    else:
        raise AssertionError("his cell 4b: the _fatal line was not found")
    cells[r] = code(rep)

    nb["cells"] = cells
    out = a.out or HERE / f"out/nb_{a.slug}"
    out.mkdir(parents=True, exist_ok=True)
    json.dump(nb, open(out / f"{a.slug}.ipynb", "w"), indent=1, ensure_ascii=False)
    meta.update(id=f"chankhavu/{a.slug}", title=a.slug, code_file=f"{a.slug}.ipynb", is_private=True,
                model_sources=[a.model_source])
    meta.pop("id_no", None)
    # docker: Kaggle's latest image (the user, 2026-10-29), as B32-M V2/V3 ran; his pinned digest is unverified here
    meta.pop("docker_image", None)
    meta["docker_image_pinning_type"] = "latest"
    assert "arc-prize-2026-arc-agi-2" in meta["competition_sources"], "4xL4 needs the competition attached"
    json.dump(meta, open(out / "kernel-metadata.json", "w"), indent=1)
    print(out, len(cells), "cells;", {f: h[:8] for f, h in sha.items()},
          {k: meta.get(k) for k in ("id", "model_sources", "dataset_sources", "docker_image_pinning_type")})


if __name__ == "__main__":
    main()
