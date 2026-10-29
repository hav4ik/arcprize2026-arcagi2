# Cell 2r (E066 image candidate): two runtime patches into Manh's installed vLLM (wheels g5e1bccd07c .. g71dca8bc24), embedded
# (internet is off) and sha256-guarded:
# - vllm/models/deepseek_v41/nvidia/model.py: his Engram row-chunk fix a6c3ee8c2b (the Kaggle Models store Engram as
#   row chunks) plus per-layer routed-expert counts (`n_routed_experts_per_layer`, branch dynamic-count-moe b3a3f642a9);
# - vllm/model_executor/layers/fused_moe/router/dsv4_topk.py: the Triton router's expert-count whitelist widened from
#   (128, 256, 384) to 6..512 (the CUDA fallback rejects per-layer counts such as 72 or 152).
# Per file: target hash already -> nothing to do; base hash (5e1bccd07c) -> apply, check the target hash, drop the
# file's __pycache__ entry; any other hash -> commit runs RAISE; the competition rerun leaves the file as is (a
# per-layer model then fails to load and the fallback submission stands). Records go to b32m_env.json (cell 2).
import base64, glob, hashlib, importlib.metadata as _md, re

IMG_PATCHES = [  # per file: the known wheels' originals -> our target (row chunks are in his wheels from f506afa210)
    {"rel": "vllm/models/deepseek_v41/nvidia/model.py", "variants": [
        {"wheel": "g5e1bccd07c", "base": "ead1e087c29b7d9ef76313e9849cffecfb01bec50d8625d4724527af853d10dd",
         "fixed": "75f3725adc5d131ece87a0334774683e41a027e77c37ecec4ff62a61f1ad7f42",
         "diff_sha": "__MODEL5E1B_DIFF_SHA__", "diff": base64.b64decode("__MODEL5E1B_DIFF_B64__").decode()},
        {"wheel": "gf506afa210 / g71dca8bc24", "base": "029aa62e383332aad37836866f4aebe8859a277977ee368b44f6f1dd8cc8ad64",
         "fixed": "ad4ac0fd9bc604feb43afc7cc6052cc749b914653d731020e8f28074a078bd37",
         "diff_sha": "__MODELF506_DIFF_SHA__", "diff": base64.b64decode("__MODELF506_DIFF_B64__").decode()}]},
    {"rel": "vllm/model_executor/layers/fused_moe/router/dsv4_topk.py", "variants": [
        {"wheel": "g5e1bccd07c .. g71dca8bc24", "base": "7e1e101ab2fb266bef34ac5826749a9803add4cde84eeb68093882948a14c50d",
         "fixed": "ab32ea7e2f3a69767d996cefb3840cca18c0bbc926a98616534be04c32c6d005",
         "diff_sha": "__TOPK_DIFF_SHA__", "diff": base64.b64decode("__TOPK_DIFF_B64__").decode()}]},
]


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


def guarded_patch(path, rerun, variants):
    """-> record {file, before, after, action, wheel}; the variant whose base hash matches is applied; a file already at
    a target hash is left alone; raises in commit runs on an unknown file or a failed patch."""
    rec = {"file": str(path), "bases": [v["base"][:16] for v in variants]}
    try:
        data = open(path, "rb").read()
        rec["before"] = hashlib.sha256(data).hexdigest()
        if rec["before"] in {v["fixed"] for v in variants}:
            rec["action"] = "already_fixed"
            return rec
        v = next((v for v in variants if v["base"] == rec["before"]), None)
        if v is None:
            raise ValueError(f"unknown file (sha256 {rec['before'][:16]}), not a known wheel's original")
        rec["wheel"] = v["wheel"]
        new = apply_unified(data.decode(), v["diff"]).encode()
        if hashlib.sha256(new).hexdigest() != v["fixed"]:
            raise ValueError("patched file does not have the target hash")
        tmp = f"{path}.img_tmp"
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
            raise RuntimeError(f"E066 VLLM PATCH FAILED (commit run): {rec}") from exc
        rec["action"] = "rerun_left_as_is"
        return rec


IMG_PATCH_RECORDS = []
ROWCHUNK = {"action": "skipped (mode fallback)"}
if MODE != "fallback":
    for _p in IMG_PATCHES:
        for _v in _p["variants"]:
            assert hashlib.sha256(_v["diff"].encode()).hexdigest() == _v["diff_sha"], f"embedded diff corrupted: {_p['rel']}"
        try:
            _fp = str(_md.distribution("vllm").locate_file(_p["rel"]))
        except Exception as exc:
            _fp = f"<vllm not installed: {type(exc).__name__}>"
        _rec = guarded_patch(_fp, RERUN, _p["variants"])
        IMG_PATCH_RECORDS.append(_rec)
        print(("E066 VLLM PATCH " + _rec["action"].upper()).replace("_", " "), _rec, flush=True)
        if _rec["action"] == "rerun_left_as_is":
            print("=" * 100, f"E066 VLLM PATCH FALLBACK (rerun): {_p['rel']} left as is; a per-layer model may not "
                  "load and the fallback submission would stand", "=" * 100, sep="\n", flush=True)
    ROWCHUNK = {**IMG_PATCH_RECORDS[0], "all": IMG_PATCH_RECORDS}
