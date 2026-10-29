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


# cell 4b additions (inside his `if not RERUN` block, 4-space indent; _w = WORK, _sl = solver.log text, json/os imported)
E066_REPORT_CHECKS = """\
    try:  # E066: cell 2r's patch outcome (fatal unless patched / already_fixed)
        _pr = json.load(open(f"{_w}/e066_vllm_patch.json"))
        print("E066 vLLM patch:", [(os.path.basename(x.get("file", "?")), x.get("action")) for x in _pr], flush=True)
        _b32m_bad += [f"environment: E066 vLLM patch {os.path.basename(x.get('file', '?'))}: {x.get('action')}"
                      for x in _pr if x.get("action") not in ("patched", "already_fixed")]
    except Exception as exc:
        _b32m_bad.append(f"environment: e066_vllm_patch.json: {type(exc).__name__}: {exc}")
    try:  # E066: cell 1b's names-and-sizes pass over every model file
        _ms = json.load(open(f"{_w}/smoke_integrity.json")).get("model_stat")
        print("E066 model files (names, sizes):", _ms, flush=True)
        if not _ms or _ms["missing"] or _ms["size_mismatch"]:
            _b32m_bad.append(f"environment: model files {_ms}")
    except Exception as exc:
        _b32m_bad.append(f"environment: smoke_integrity.json: {type(exc).__name__}: {exc}")
"""
E066_LIGHT_CHECKS = """\
    if os.environ.get("E066_LIGHT_SMOKE") == "1":  # E066 light smoke: no crash, real answers, a valid submission
        _b32m_bad += [f"solver.log: {x}" for x in ("server died", "server exited during startup", "solver crashed",
                                                  "server not up after") if x in _sl]
        try:
            _att = [json.loads(x) for x in open(f"{_w}/dsv41_logs/attempts.jsonl") if x.strip()]
            _ans = [x for x in _att if x.get("pred") and not x.get("error")]
            print(f"E066 light smoke: {len(_att)} attempt records, {len(_ans)} with an answer,",
                  sum(bool(x.get("correct")) for x in _att), "correct", flush=True)
            if not _ans:
                _b32m_bad.append(f"no attempt produced an answer ({len(_att)} attempt records)")
        except Exception as exc:
            _b32m_bad.append(f"attempts.jsonl: {type(exc).__name__}: {exc}")
        try:
            _sub = json.load(open(f"{_w}/submission.json"))
            assert isinstance(_sub, dict) and _sub, "empty"
        except Exception as exc:
            _b32m_bad.append(f"no valid submission.json ({type(exc).__name__}: {exc})")
"""
WAIT_CELL = """\
# Cell 3w (E066, commit runs only): cell 2c's API contract runs in a background thread and needs the server. The light
# smoke's 2 tasks can finish before it does, and cell 4 then stops the server, so wait for it here (at most 15 min).
import os, time as _wt
if not RERUN and "_ct_thread" in globals():
    _wt0 = _wt.time()
    _ct_thread.join(timeout=900)
    print("E066: contract thread", "finished" if not _ct_thread.is_alive() else "STILL RUNNING after 900 s",
          f"(waited {_wt.time() - _wt0:.0f} s)", flush=True)
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manh-nb", type=Path, required=True, help="`kaggle kernels pull -m` folder of his notebook")
    ap.add_argument("--model-source", required=True)
    ap.add_argument("--slug", default="dsv41-tools-submit-img-manh")
    ap.add_argument("--vision-max-images", type=int, default=32)
    ap.add_argument("--commit", choices=["light", "load"], default="light",
                    help="commit run: light = 2 smallest tasks (00576224, 25ff71a9 with 2 test inputs), ends when they are done, "
                         "deadline cap ~75 min (saves quota; load testing is done on pods); load = his 80-task load test")
    ap.add_argument("--max-num-seqs", type=int, default=64,
                    help="his cell 2m's MANH_SEQS (--concurrency = vLLM max-num-seqs) and our ARC_WORKERS. 64 = Manh, "
                         "2026-10-29 ~21:00 UTC: the value he ran on Kaggle (his v6 cell says 80); his line is the only "
                         "edit to cell 2m")
    ap.add_argument("--kv-offload-gib", default=None,
                    help="his --kv-offload-gib-per-rank (v6: 12). Host RAM: his engine ~144 GiB + our solver/sandboxes "
                         "~17 GiB vs Kaggle's 173 GiB cgroup (2026-10-29); only with Manh's OK")
    ap.add_argument("--gpu-util", default=None, help="his --gpu-memory-utilization (v6: 0.94); only with Manh's OK")
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
              '    os.environ.setdefault("E066_LIGHT_SMOKE", "1")  # cells 3w and 4b: the light smoke\'s checks\n'
              '    print("E066 LIGHT SMOKE: tasks", os.environ["ARC_TASK_IDS"], flush=True)\n' + s0[end:])
        # 70 min: his cell 2m waits for the server only until ARC_STOP_NEW_S. Pod 3 (2026-10-29) needed 20.6 min for one
        # rank's Engram load under use_thp on a fragmented host, so 45 min could fail a good commit run. A healthy run
        # still ends as soon as both tasks are done.
        for k, old_v, new_v in (("ARC_STOP_NEW_S", "2280", "4200"), ("ARC_FORCE_AT_S", "2400", "4320"),
                                ("ARC_HARD_STOP_S", "2460", "4380"), ("ARC_KILL_AT_S", "2550", "4470")):
            s0 = replace_once(s0, f'os.environ.setdefault("{k}", "{old_v}")', f'os.environ.setdefault("{k}", "{new_v}")')
        line = next(l for l in s0.splitlines() if l.startswith("    # smoke within 43 min of the first cell"))
        s0 = replace_once(s0, line, "    # E066 light smoke: the solver ends once the 2 tasks are done (model load, contract, generation, final "
                                    "write); the deadlines below only cap it at ~75 min of the first cell")
    s0 = replace_once(s0, '(("ARC_WORKERS", "80"),', f'(("ARC_WORKERS", "{a.max_num_seqs}"),')
    cells[i] = code(s0)

    # cell 2m: his server cell, byte-identical except his max-num-seqs line (Manh asked for 64 on 2026-10-29)
    k2 = find(cells, "# Cell 2m (B32-M + Manh's server)")
    c2m = src(cells[k2])
    line = next(l for l in c2m.splitlines() if l.startswith("MANH_CONTEXT, MANH_SEQS = "))
    ctx, seqs = (int(x) for x in line.split("=")[1].split(","))
    if seqs != a.max_num_seqs:
        c2m = replace_once(c2m, line, f"MANH_CONTEXT, MANH_SEQS = {ctx}, {a.max_num_seqs}  # E066 build: Manh's {seqs} -> "
                                      f"{a.max_num_seqs} (his request, the value he ran on Kaggle)")
    for flag, val in (("--kv-offload-gib-per-rank", a.kv_offload_gib), ("--gpu-memory-utilization", a.gpu_util)):
        if val is not None:
            m_ = re.search(rf'"{flag}", "([0-9.]+)"', c2m)
            assert m_ and c2m.count(m_.group(0)) == 1, flag
            c2m = c2m.replace(m_.group(0), f'"{flag}", "{val}"')
    cells[k2] = code(c2m)

    # cell 0b: our pins, his checks + ours
    j = find(cells, "# Cell 0b (E046, frozen inputs)")
    his = src(cells[j])
    expect = json_line(his, "_MT_EXPECT")
    expect["ARC_GRID_FORMAT"] = "text+image"
    expect["MANH_VISION"] = "1"
    assert expect.get("ARC_WORKERS") == "80", expect.get("ARC_WORKERS")  # his check pins the settings' worker count
    expect["ARC_WORKERS"] = str(a.max_num_seqs)  # else the rerun's 0b check falls back to v3 settings (36 workers)
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
    c2 = replace_once(c2, "    threading.Thread(target=contract_thread, daemon=True).start()\n",
                      "    _ct_thread = threading.Thread(target=contract_thread, daemon=True)  # E066: cell 3w waits for it\n"
                      "    _ct_thread.start()\n")
    if "import sys" not in c2 and ", sys," not in c2:
        c2 = replace_once(c2, "import re, threading, urllib.request", "import re, sys, threading, urllib.request")
    cells[m] = code(c2)

    # cell 1b: commit-run input integrity against the E066 export's manifest (his cell checks E128's)
    q = find(cells, "# Cell 1b ")
    ic = B.integrity_cell(a.model_manifest)
    man = json.load(open(a.model_manifest))
    ic = re.sub(r"every file of the model \(\d+ files, \d+ GB\)",
                f"every file of the model ({man['files']} files, {man['total_bytes'] / 1e9:.0f} GB)", ic)
    ic = replace_once(ic, '        integrity_save(state)\n        log = f"{WORK}/dsv41_logs/solver.log"\n',
                      '        integrity_save(state)\n'
                      '        # E066: names and sizes of every model file at once (seconds): the sha256 pass starts at server up and\n'
                      '        # rarely finishes inside a light smoke, so cell 4b fails the commit run on a missing or short file\n'
                      '        stat = {"files": len(INTEGRITY["model"]), "missing": [], "size_mismatch": []}\n'
                      '        for rel, (size, _digest) in INTEGRITY["model"].items():\n'
                      '            p = os.path.join(model_dir, rel)\n'
                      '            if not os.path.exists(p):\n'
                      '                stat["missing"].append(rel)\n'
                      '            elif os.path.getsize(p) != size:\n'
                      '                stat["size_mismatch"].append(rel)\n'
                      '        state["model_stat"] = stat\n'
                      '        integrity_save(state)\n'
                      '        log = f"{WORK}/dsv41_logs/solver.log"\n')
    cells[q] = code(ic)

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
    load_if = '    if os.environ.get("B32M_LOAD_TEST") == "1":  # load test'
    assert rep.count(load_if) == 1
    rep = rep.replace(load_if, E066_REPORT_CHECKS + load_if)
    tail = "    B32M_BAD = _b32m_bad  # the verdict cell at the end raises on it"
    assert rep.count(tail) == 1
    rep = rep.replace(tail, E066_LIGHT_CHECKS + tail)
    cells[r] = code(rep)

    # cell 3w (new): wait for the contract thread before his cell 4 stops the server
    w = find(cells, "# Cell 3: the solver")
    cells.insert(w + 1, code(WAIT_CELL))

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
