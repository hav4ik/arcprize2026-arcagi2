# Cell 0b (frozen inputs from the attached pins dataset). The candidate's pinned solver, harness, helper modules and
# vLLM overlay come from a small private dataset (dsv41_pins.tgz + SHA256SUMS) built by
# `build_frozen_mt.py --pins-dataset`; the kit and model inputs stay Version 5's. After cell 0 has unpacked and checked
# the kit, this cell backs up the kit files the pins replace, installs the pins, and checks hashes, required features
# and settings.
# Commit run ("Save & Run All"): any failure RAISES, so the commit errors and the log shows "E046 CHECK FAILED".
# Competition rerun: the same failure restores the kit's v3 files and runs per pair at 36 workers (a scored submission
# beats an error); the rerun path never raises (S10). Uses cell 0's KIT, MODE, RERUN, INPUT and find_one.
import hashlib
import json
import os
import shutil
import tarfile

PINS = json.loads(r"""__PINS__""")  # filled in by build_frozen_mt.py: hashes, expected settings, feature checks


def pins_sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def pins_install(kit, archive, pins, state):
    """Back up each kit file the pins replace (files new to the kit have no v3 original), then extract the pins over
    the kit. `state` records the backups as they are made and whether extraction started, so a failure part-way
    leaves the fallback exactly what it needs (S10: a backup counts only once its copy exists)."""
    for member in pins["sha"]:
        if member not in pins["new"]:
            shutil.copy2(f"{kit}/{member}", f"{kit}/{member}.v3orig")
            state["backups"][member] = f"{kit}/{member}.v3orig"
    state["extracted"] = True
    with tarfile.open(archive) as tf:
        tf.extractall(kit)


def pins_check(kit, pins):
    """Raise AssertionError at the first mismatch: member hashes, the features every candidate's solver and harness
    need, the features its opt-in settings rely on, and the settings themselves."""
    for member, sha in pins["sha"].items():
        assert pins_sha256(f"{kit}/{member}") == sha, f"pinned file mismatch: {member}"
    with open(f"{kit}/solver_dsv41.py") as f:
        solver = f.read()
    with open(f"{kit}/repo/evals/run_deepseek_tools.py") as f:
        harness = f.read()
    for need in pins["solver_need"]:
        assert need in solver, f"solver lacks {need}"
    for need in pins["harness_need"]:
        assert need in harness, f"harness lacks {need}"
    for member, needs in pins["opt_need"].items():
        with open(f"{kit}/{member}") as f:
            text = f.read()
        for need in needs:
            assert need in text, f"{member} lacks {need}"
    for key, value in pins["expect"].items():
        assert os.environ.get(key) == value, f"{key}={os.environ.get(key)!r}, expected {value!r}"


def pins_fallback(kit, state, pins):
    """Competition rerun after a failed check: the kit's v3 files back (when the pins were extracted at all), Version
    5's settings (per pair, 36 workers, its server flags, every opt-in lever dropped). Never raises; returns what could
    not be restored."""
    kept = []
    try:
        if state["extracted"]:
            for member in pins["sha"]:
                if member in pins["new"]:
                    continue  # new to the kit: the v3 code never imports it
                try:
                    shutil.copy2(state["backups"][member], f"{kit}/{member}.restore")
                    os.replace(f"{kit}/{member}.restore", f"{kit}/{member}")
                except Exception as exc:
                    kept.append(f"{member} ({type(exc).__name__})")
        # (a pinned solver kept by a failed restore would pass the opt-in harness flags to the v3 harness, which exits)
        for key in pins["fallback_drop"]:
            os.environ.pop(key, None)
        os.environ["ARC_WORKERS"] = "36"
        os.environ["ARC_SERVER_EXTRA_ARGS"] = pins["fallback_server_extra"]  # no flag that needs the new overlay
    except Exception as exc:
        kept.append(f"{type(exc).__name__}: {exc}")
    return kept


if MODE != "fallback":
    _pins_state, _pins_why = {"backups": {}, "extracted": False}, ""
    try:
        assert KIT, "kit not unpacked"
        _pins_archive = find_one(f"{INPUT}/**/{PINS['archive']}")
        assert _pins_archive, f"{PINS['archive']} is not attached (dataset {PINS['dataset']})"
        assert pins_sha256(_pins_archive) == PINS["archive_sha"], f"{_pins_archive} is not the build's archive"
        pins_install(KIT, _pins_archive, PINS, _pins_state)
        pins_check(KIT, PINS)
    except Exception as exc:
        _pins_why = f"{type(exc).__name__}: {exc}"
    if not _pins_why:
        print("E046 CHECK OK: pinned files installed and checked:", {m: h[:8] for m, h in PINS["sha"].items()},
              PINS["expect"], flush=True)
    else:
        print("=" * 100, f"E046 CHECK FAILED: {_pins_why}", "=" * 100, sep="\n", flush=True)
        if not RERUN:
            raise RuntimeError(f"E046 CHECK FAILED (commit run): {_pins_why}")
        _pins_kept = pins_fallback(KIT, _pins_state, PINS)
        print("E046 FALLBACK: the kit's v3 code, per pair, 36 workers"
              + (f"; not restored, kept as is: {', '.join(_pins_kept)}" if _pins_kept else ""), flush=True)
