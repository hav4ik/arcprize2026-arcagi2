# Cell 2r (B32-M plan B): the Engram row-chunk loader for the Kaggle copy of E128, patched into Manh's installed vLLM.
# His Kaggle wheel vllm-0.32.1rc1.dev0+g5e1bccd07c cannot load the Kaggle Model copy (Engram tables stored as
# `layers.{1,14}.engram.embed.{weight,scale}.rows_<start>`: KeyError 'layers.1.engram.embed.weight.rows_0'). His fix
# is on his GitHub main as a6c3ee8c2b (our E062 patch 01). This cell applies exactly that commit's diff (embedded:
# internet is off) to vllm/models/deepseek_v41/nvidia/model.py, guarded by sha256:
# - fixed hash already (his new wheel): nothing to do;
# - base hash (5e1bccd07c): apply, check the result has the fixed hash, drop the file's __pycache__ entry;
# - any other hash: commit runs RAISE; the competition rerun leaves the file as is and goes on (as B32's v3 fallback:
#   a scored submission beats an error). The path taken is recorded in b32m_env.json (cell 2).
import base64, glob, hashlib, importlib.metadata as _md, re

ROWCHUNK_BASE = "ead1e087c29b7d9ef76313e9849cffecfb01bec50d8625d4724527af853d10dd"   # 5e1bccd07c
ROWCHUNK_FIXED = "b2a04fc2a251d570f1a7bf901f55da016ee060264c2697070dc2ce3f3e614ee4"  # a6c3ee8c2b
ROWCHUNK_DIFF_SHA = "__DIFF_SHA__"
ROWCHUNK_DIFF = base64.b64decode("__DIFF_B64__").decode()


def apply_unified(text, diff):
    """Apply a one-file unified diff to `text`; raises ValueError when a context or removed line does not match."""
    src, out, i = text.splitlines(keepends=True), [], 0
    lines = diff.splitlines(keepends=True)
    k = 0
    while k < len(lines):
        m = re.match(r"^@@ -(\d+)(?:,(\d+))? \+\d+(?:,\d+)? @@", lines[k])
        k += 1
        if not m:
            continue
        start = int(m.group(1)) - 1 if m.group(2) != "0" else int(m.group(1))
        if start < i:
            raise ValueError("overlapping hunks")
        out += src[i:start]
        i = start
        while k < len(lines) and not lines[k].startswith("@@"):
            tag, body = lines[k][:1], lines[k][1:]
            if tag in (" ", "-"):
                if i >= len(src) or src[i] != body:
                    raise ValueError(f"hunk mismatch at line {i + 1}: {body[:60]!r}")
                if tag == " ":
                    out.append(body)
                i += 1
            elif tag == "+":
                out.append(body)
            elif tag != "\\":
                break
            k += 1
    return "".join(out + src[i:])


def rowchunk_patch(path, rerun, base=ROWCHUNK_BASE, fixed=ROWCHUNK_FIXED, diff=ROWCHUNK_DIFF):
    """-> record {path, before, after, action}; raises in commit runs on an unknown file or a failed patch."""
    rec = {"file": str(path), "base": base[:16], "fixed": fixed[:16]}
    try:
        data = open(path, "rb").read()
        rec["before"] = hashlib.sha256(data).hexdigest()
        if rec["before"] == fixed:
            rec["action"] = "already_fixed"
            return rec
        if rec["before"] != base:
            raise ValueError(f"unknown model.py (sha256 {rec['before'][:16]}), neither base nor fixed")
        new = apply_unified(data.decode(), diff).encode()
        if hashlib.sha256(new).hexdigest() != fixed:
            raise ValueError("patched file does not have the fixed hash")
        tmp = f"{path}.b32m_tmp"
        with open(tmp, "wb") as f:
            f.write(new)
        os.replace(tmp, path)
        rec["after"] = hashlib.sha256(open(path, "rb").read()).hexdigest()
        stem = os.path.splitext(os.path.basename(path))[0]
        rec["pyc_removed"] = []
        for pyc in glob.glob(os.path.join(os.path.dirname(path), "__pycache__", f"{stem}.*.pyc")):
            os.remove(pyc)
            rec["pyc_removed"].append(os.path.basename(pyc))
        rec["action"] = "patched"
        return rec
    except Exception as exc:
        rec["action"] = "failed"
        rec["error"] = f"{type(exc).__name__}: {exc}"
        if not rerun:
            raise RuntimeError(f"B32-M ROW-CHUNK PATCH FAILED (commit run): {rec}") from exc
        rec["action"] = "rerun_left_as_is"
        return rec


ROWCHUNK = {"action": "skipped (mode fallback)"}
if MODE != "fallback":
    assert hashlib.sha256(ROWCHUNK_DIFF.encode()).hexdigest() == ROWCHUNK_DIFF_SHA, "embedded diff corrupted"
    try:
        _mp = str(_md.distribution("vllm").locate_file("vllm/models/deepseek_v41/nvidia/model.py"))
    except Exception as exc:
        _mp = f"<vllm not installed: {type(exc).__name__}>"
    ROWCHUNK = rowchunk_patch(_mp, RERUN)
    print(("B32-M ROW-CHUNK " + ROWCHUNK["action"].upper()).replace("_", " "), ROWCHUNK, flush=True)
    if ROWCHUNK["action"] == "rerun_left_as_is":
        print("=" * 100, "B32-M ROW-CHUNK FALLBACK (rerun): model.py left as is; the Kaggle copy may not load and the "
              "fallback submission would stand", "=" * 100, sep="\n", flush=True)
