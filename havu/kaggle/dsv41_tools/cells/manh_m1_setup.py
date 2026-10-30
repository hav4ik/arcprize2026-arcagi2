# NOT USED (E064, 2026-10-28: superseded by B32-M; kept for reference)
# Cell M1 (E064): B32's harness on Manh's server. Clock, settings, fallback submission, our kit.
# Manh's cells above installed his vLLM and started the server (http://127.0.0.1:18084/v1, model dsv41). Nothing
# below installs Python packages or starts, restarts or kills a server: the solver runs in external-server mode
# (ARC_EXTERNAL_SERVER) with B32's settings. The kit (B32's pinned solver + harness + train_check + auto-repair,
# with the external mode and the `reasoning` re-render fix) rides in this cell, sha256-checked.
# Commit run ("Save & Run All"): a smoke (3 tasks, deadlines from server-up); any failure in our cells RAISES.
# Competition rerun: never raises; a failure keeps the fallback submission.
import base64, hashlib, io, json, os, sys, tarfile, time
RERUN = bool(os.environ.get("KAGGLE_IS_COMPETITION_RERUN"))
MODE = "submit" if RERUN else os.environ.get("ARC_COMMIT_MODE", "smoke")
# The notebook started at Manh's cell 0: global_end_time = start + 12 h - 600 s.
try:
    T0 = float(global_end_time) - 12 * 3600 + 600  # noqa: F821 (defined in Manh's cell 0)
except NameError:
    T0 = time.time()
os.environ["ARC_T0"] = str(T0)
WORK = "/kaggle/working"
os.makedirs(WORK, exist_ok=True)
_C = [p for p in ("/kaggle/input/arc-prize-2026-arc-agi-2/arc-agi_test_challenges.json",
                  "/kaggle/input/competitions/arc-prize-2026-arc-agi-2/arc-agi_test_challenges.json") if os.path.exists(p)]
if not _C:
    import glob
    _C = sorted({p for d in ("", "*/", "*/*/", "datasets/*/*/")  # fixed depths, never a recursive scan
                 for p in glob.glob(f"/kaggle/input/{d}arc-agi_test_challenges.json")})
CHALLENGES = os.environ.get("ARC_CHALLENGES") or _C[0]
_tasks = json.load(open(CHALLENGES))
_fb = {t: [{"attempt_1": x["input"], "attempt_2": x["input"]} for x in v["test"]] for t, v in _tasks.items()}
with open(f"{WORK}/submission.json.tmp", "w") as f:
    json.dump(_fb, f)
os.replace(f"{WORK}/submission.json.tmp", f"{WORK}/submission.json")
print(f"mode {MODE}; fallback submission for {len(_tasks)} tasks written at +{time.time() - T0:.0f}s", flush=True)

_KIT_B64 = "__KIT_B64__"
_KIT_SHA = json.loads("""__KIT_SHA__""")
KIT = "/tmp/arc_kit"
try:
    with tarfile.open(fileobj=io.BytesIO(base64.b64decode(_KIT_B64))) as _tf:
        _tf.extractall(KIT)
    _bad = [f for f, h in _KIT_SHA.items() if hashlib.sha256(open(f"{KIT}/{f}", "rb").read()).hexdigest() != h]
    assert not _bad, f"kit files differ: {_bad[:5]}"
    _sv = open(f"{KIT}/solver_dsv41.py").read()
    assert "ARC_EXTERNAL_SERVER" in _sv and "def wait_external" in _sv, "solver lacks the external-server mode"
    print("E064 KIT OK:", {f: h[:8] for f, h in _KIT_SHA.items() if not f.startswith("repo/evals/arc_harness")},
          flush=True)
except Exception as exc:
    print("=" * 100, f"E064 KIT FAILED: {type(exc).__name__}: {exc}", "=" * 100, sep="\n", flush=True)
    if not RERUN:
        raise
    MODE = "fallback"

# B32's settings (CANDIDATE_B32.md); the context limit follows the server's max_model_len (Manh's --context).
BASE_ENV = dict(os.environ)
for _k, _v in (("ARC_EXTERNAL_SERVER", "http://127.0.0.1:18084/v1"), ("ARC_SERVED_MODEL", "dsv41"),
               ("ARC_WORKERS", "56"), ("ARC_MULTI_TEST", "1"), ("ARC_MULTI_BUDGET", "0.5"),
               ("ARC_CONTEXT_LIMIT", "163840"), ("ARC_KV_ADMIT_MAX", "0.85"), ("ARC_KV_ADMIT_MIN_WORKERS", "16"),
               ("ARC_ADMIT_MAX_WAITING", "1"), ("ARC_AUTO_REPAIR", "1"), ("ARC_MODE", "submit"),
               # Kaggle default deadlines from the notebook start (B32): stop-new 10h50, force 11h10, hard 11h30;
               # the kill switch (cell M3) at 11h45, before Manh's global_end_time (11h50)
               ("ARC_STOP_NEW_S", "39000"), ("ARC_FORCE_AT_S", "40200"), ("ARC_HARD_STOP_S", "41400"),
               ("ARC_KILL_AT_S", "42300"), ("ARC_SERVER_UP_TIMEOUT_S", "7200"),
               ("ARC_OUT", WORK), ("ARC_KIT", KIT), ("ARC_CHALLENGES", CHALLENGES), ("ARC_TOOL_PYTHON", sys.executable),
               ("ARC_SCRATCH", "/tmp/arc_scratch")):
    BASE_ENV.setdefault(_k, _v)
os.makedirs(BASE_ENV["ARC_SCRATCH"], exist_ok=True)
BASE_ENV.setdefault("ARC_MODEL_DIR", "/kaggle/input/models/chankhavu/dsv41-flash-1017-arc-tool-e128-2bit/pytorch/default/1")
if MODE == "smoke":
    BASE_ENV.setdefault("ARC_TASK_IDS", "25ff71a9,27a28665,00576224")
SERVER = BASE_ENV["ARC_EXTERNAL_SERVER"].rstrip("/").removesuffix("/v1")
print("E064 settings:", {k: BASE_ENV[k] for k in sorted(BASE_ENV) if k.startswith("ARC_")}, flush=True)
