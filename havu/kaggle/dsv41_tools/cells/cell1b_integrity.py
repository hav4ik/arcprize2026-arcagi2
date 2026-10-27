# Cell 1b (smoke, commit runs only): input integrity, in a background thread so it costs no wall time. Checks every
# input the competition rerun reads: the competition JSON files parse; the kit (already sha256-checked by cell 0); the
# offline wheel set against its SHA256SUMS when it ships one; and every file of the model (136 files, 131 GB) by size
# and sha256 against the manifest made before upload. The model is hashed once the server is up, so it does not slow
# the load. Results go to /kaggle/working/smoke_integrity.json; the report cell prints them and fails the commit run on
# any mismatch. The competition rerun skips this cell. Uses cell 0's T0, INPUT, WORK, KIT, MODEL_SRC, MODE, RERUN and
# find_one.
import hashlib
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

INTEGRITY = json.loads(r"""__INTEGRITY__""")  # filled in by build_frozen_mt.py: {"model": {path: [size, sha256]}}
INTEGRITY_OUT = f"{WORK}/smoke_integrity.json"


def integrity_save(state):
    with open(INTEGRITY_OUT + ".tmp", "w") as f:
        json.dump(state, f, indent=1)
    os.replace(INTEGRITY_OUT + ".tmp", INTEGRITY_OUT)


def integrity_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(16 << 20)
            if not block:
                return h.hexdigest()
            h.update(block)


def integrity_worker(state, model_dir, threads=8, wait_s=1800):
    try:
        state["competition"] = {}
        comp_files = sorted({p for p in (find_one(f"{INPUT}/**/arc-agi_test_challenges.json"),) if p})
        comp_dir = os.path.dirname(comp_files[0]) if comp_files else None
        for name in sorted(os.listdir(comp_dir)) if comp_dir else []:
            if name.endswith(".json"):
                with open(os.path.join(comp_dir, name)) as f:
                    state["competition"][name] = len(json.load(f))
        state["kit"] = "sha256-checked by cell 0 (MANIFEST.json)" if KIT else "MISSING"
        wheel = find_one(f"{INPUT}/**/vllm-0.28.1rc1*.whl")
        wdir = os.path.dirname(wheel) if wheel else None
        sums = os.path.join(wdir, "SHA256SUMS") if wdir else None
        if sums and os.path.exists(sums):
            bad = []
            lines = [l.split() for l in open(sums) if l.strip()]
            for digest, name in ((l[0], l[-1].lstrip("*")) for l in lines):
                p = os.path.join(wdir, name)
                if not os.path.exists(p) or integrity_sha256(p) != digest:
                    bad.append(name)
            state["wheels"] = {"checked": len(lines), "bad": bad}
        else:
            state["wheels"] = {"files": len(os.listdir(wdir)) if wdir else 0, "note": "no SHA256SUMS shipped; uv installs "
                               "from the pinned requirements.lock and the import check in cell 2 runs"}
        integrity_save(state)
        log = f"{WORK}/dsv41_logs/solver.log"
        while time.time() - T0 < wait_s:  # hash after the server is up, so the load does not compete for the mount
            if os.path.exists(log) and "server up" in open(log, errors="replace").read():
                break
            time.sleep(15)
        files = INTEGRITY["model"]
        res = {"files": len(files), "checked": 0, "missing": [], "size_mismatch": [], "sha_mismatch": [], "bytes": 0,
               "started_min": round((time.time() - T0) / 60, 1)}
        state["model"] = res
        t0 = time.time()

        def check(rel):
            size, digest = files[rel]
            p = os.path.join(model_dir, rel)
            if not os.path.exists(p):
                return rel, "missing", 0
            if os.path.getsize(p) != size:
                return rel, "size_mismatch", 0
            return rel, ("ok" if integrity_sha256(p) == digest else "sha_mismatch"), size

        with ThreadPoolExecutor(threads) as ex:
            for rel, verdict, n in ex.map(check, sorted(files, key=lambda r: -files[r][0])):
                res["checked"] += 1
                res["bytes"] += n
                if verdict != "ok":
                    res[verdict].append(rel)
                if res["checked"] % 10 == 0:
                    res["minutes"] = round((time.time() - t0) / 60, 1)
                    integrity_save(state)
        res["minutes"] = round((time.time() - t0) / 60, 1)
        state["done"] = True
    except Exception as exc:
        state["error"] = f"{type(exc).__name__}: {exc}"
    integrity_save(state)


if not RERUN and MODE != "fallback":
    _integrity_state = {"started_min": round((time.time() - T0) / 60, 1), "model_dir": MODEL_SRC}
    integrity_save(_integrity_state)
    threading.Thread(target=integrity_worker, args=(_integrity_state, MODEL_SRC or ""), daemon=True).start()
    print(f"input integrity check started in the background ({len(INTEGRITY['model'])} model files; hashed after the "
          "server is up)", flush=True)
