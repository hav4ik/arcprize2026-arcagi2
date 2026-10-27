#!/usr/bin/env python3
"""E056: install the vLLM overlay into a venv through a generated notebook's own cells (the real packaging path).

In one process and in the notebook's order this runs, verbatim (code taken from the .ipynb, not copied by hand):
  cell 00  settings (incl. the PYTORCH_CUDA_ALLOC_CONF fix when --server-extra carries --kv-transfer-config)
  cell 0   clock, paths, fallback submission, kit unpack + MANIFEST check
  cell 0b  pinned solver/harness + the replacement overlay pair over the kit; sha, feature and setting checks;
           commit run: raises on failure; rerun: restores the kit's *.v3orig files
  cell 2   only the overlay block: extract KIT/vllm_overlay_rw.tgz into site-packages, assert every sha line.
           The wheel-set install before it is skipped (the target venv already has the fork wheel set) and so are
           the CUDA shim and import checks after it.
Cell 1 (model staging) is skipped. Afterwards every line of KIT/vllm_overlay_rw.sha256 is checked again here and a
JSON summary is printed as the last line.

  python3 -I notebook_install.py --notebook NB.ipynb --input DIR --scratch DIR --site-packages SP [--rerun]
         [--env K=V ...]
--input: a dir holding the Kaggle kit tarball (dsv41_kit.tgz, e.g. kaggle/dsv41_tools/kaggle_kit_v5/) and an
arc-agi_test_challenges.json (cell 0 finds both with a ** glob). The process env is reduced to PATH/HOME/LANG plus
the cells' own variables (no ARC_*/KAGGLE_* from outside unless given with --env), so nothing else leaks in.
"""
import argparse
import hashlib
import io
import json
import os
import sys
import textwrap
from contextlib import redirect_stdout

OV_START = '    with tarfile.open(f"{KIT}/vllm_overlay_rw.tgz") as tf:\n'
OV_END = '    print("vLLM overlay installed and checked", flush=True)\n'


def cells_by_prefix(nb):
    out = {}
    for c in nb["cells"]:
        s = "".join(c["source"])
        for key, prefix in (("00", "# Cell 00"), ("0", "# Cell 0:"), ("0b", "# Cell 0b"), ("2", "# Cell 2:")):
            if s.startswith(prefix):
                assert key not in out, f"two cells start with {prefix!r}"
                out[key] = s
    missing = {"00", "0", "0b", "2"} - set(out)
    assert not missing, f"cells not found: {missing}"
    return out


def overlay_block(cell2):
    """Cell 2's own imports + its overlay block, dedented (the block sits inside `if ...: try:`)."""
    imports = next(ln for ln in cell2.splitlines(keepends=True) if ln.startswith("import hashlib"))
    i, j = cell2.index(OV_START), cell2.index(OV_END) + len(OV_END)
    return imports + textwrap.dedent(cell2[i:j])


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--notebook", required=True)
    p.add_argument("--input", required=True)
    p.add_argument("--scratch", required=True)
    p.add_argument("--site-packages", required=True)
    p.add_argument("--rerun", action="store_true", help="act as the competition rerun (KAGGLE_IS_COMPETITION_RERUN=1)")
    p.add_argument("--env", action="append", default=[], help="extra K=V for the notebook process (tests)")
    a = p.parse_args()
    cells = cells_by_prefix(json.load(open(a.notebook)))
    block = overlay_block(cells["2"])
    keep = {k: os.environ[k] for k in ("PATH", "HOME", "LANG", "TMPDIR") if k in os.environ}
    os.environ.clear()
    os.environ.update(keep)
    os.makedirs(a.scratch, exist_ok=True)
    os.environ.update(ARC_INPUT_ROOT=a.input, ARC_OUT=os.path.join(a.scratch, "working"),
                      ARC_SCRATCH=os.path.join(a.scratch, "dsv41"))
    if a.rerun:
        os.environ["KAGGLE_IS_COMPETITION_RERUN"] = "1"
    for kv in a.env:
        k, v = kv.split("=", 1)
        os.environ[k] = v
    ns = {"__name__": "__notebook__"}
    log = io.StringIO()
    summary = {"notebook": os.path.basename(a.notebook), "rerun": a.rerun, "site_packages": a.site_packages}
    try:
        with redirect_stdout(log):
            exec(compile(cells["00"], "cell00", "exec"), ns)
            exec(compile(cells["0"], "cell0", "exec"), ns)
            assert ns.get("KIT") and ns.get("MODE") != "fallback", f"cell 0: KIT={ns.get('KIT')} MODE={ns.get('MODE')}"
            exec(compile(cells["0b"], "cell0b", "exec"), ns)
            ns["SP"] = a.site_packages
            exec(compile(block, "cell2_overlay", "exec"), ns)
    except Exception as exc:
        summary["error"] = f"{type(exc).__name__}: {exc}"
    sys.stdout.write(log.getvalue())
    kit = ns.get("KIT")
    if kit and "error" not in summary:
        rows = [ln.split() for ln in open(f"{kit}/vllm_overlay_rw.sha256") if ln.strip()]
        bad = [rel for h, rel in rows
               if hashlib.sha256(open(f"{a.site_packages}/{rel}", "rb").read()).hexdigest() != h]
        summary.update(
            kit_overlay_tgz=hashlib.sha256(open(f"{kit}/vllm_overlay_rw.tgz", "rb").read()).hexdigest(),
            kit_overlay_sha256=hashlib.sha256(open(f"{kit}/vllm_overlay_rw.sha256", "rb").read()).hexdigest(),
            files=len(rows), files_bad=bad, cell0b_ok="E046 CHECK OK" in log.getvalue(),
            fallback="E046 FALLBACK" in log.getvalue(),
            server_extra_has_kv_transfer="--kv-transfer-config" in os.environ.get("ARC_SERVER_EXTRA_ARGS", ""),
            alloc_conf=os.environ.get("PYTORCH_CUDA_ALLOC_CONF"))
    print("E056 INSTALL SUMMARY " + json.dumps(summary), flush=True)
    sys.exit(0 if "error" not in summary and not summary.get("files_bad") else 1)


if __name__ == "__main__":
    main()
