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
ROWCHUNK_DIFF_SHA = "14032fab8f265402176a1d1c6cc6b57d5ae05a13039958bf383f554e0b519ac0"
ROWCHUNK_DIFF = base64.b64decode("ZGlmZiAtLWdpdCBhL3ZsbG0vbW9kZWxzL2RlZXBzZWVrX3Y0MS9udmlkaWEvbW9kZWwucHkgYi92bGxtL21vZGVscy9kZWVwc2Vla192NDEvbnZpZGlhL21vZGVsLnB5CmluZGV4IDgxZTg0MzMuLmUxYTNmMTYgMTAwNjQ0Ci0tLSBhL3ZsbG0vbW9kZWxzL2RlZXBzZWVrX3Y0MS9udmlkaWEvbW9kZWwucHkKKysrIGIvdmxsbS9tb2RlbHMvZGVlcHNlZWtfdjQxL252aWRpYS9tb2RlbC5weQpAQCAtMTYxMCw2ICsxNjEwLDY3IEBAIGRlZiBfbGluZWFyX3NjYWxlX3BhcmFtX25hbWUodmxsbV9jb25maWc6IFZsbG1Db25maWcsIGV4cGVydF9kdHlwZTogc3RyKSAtPiBzdHI6CiAgICAgcmV0dXJuICJ3ZWlnaHRfc2NhbGUiIGlmIHVzZV9teGZwOCBlbHNlICJ3ZWlnaHRfc2NhbGVfaW52IgogCiAKK19FTkdSQU1fUk9XX0NIVU5LID0gcmUuY29tcGlsZShyIl4oLipcLmVuZ3JhbSlcLmVtYmVkXC4od2VpZ2h0fHNjYWxlKVwucm93c18oXGQrKSQiKQorCisKK2RlZiBfZW5ncmFtX3Jvd19jaHVua3MoCisgICAgd2VpZ2h0czogSXRlcmFibGVbdHVwbGVbc3RyLCB0b3JjaC5UZW5zb3JdXSwKKyAgICBtYXBwZXI6IFdlaWdodHNNYXBwZXIsCisgICAgcGFyYW1zOiBkaWN0W3N0ciwgbm4uUGFyYW1ldGVyXSwKKyAgICBsb2FkZWQ6IHNldFtzdHJdLAorKSAtPiBJdGVyYWJsZVt0dXBsZVtzdHIsIHRvcmNoLlRlbnNvcl1dOgorICAgICIiIkxvYWQgRW5ncmFtIHRhYmxlcyBzdG9yZWQgYXMgcm93IGNodW5rcy4KKworICAgIEV4cG9ydHMgd2l0aCBhIGZpbGUgc2l6ZSBsaW1pdCBzdG9yZSBlYWNoIHRhYmxlIGFzCisgICAgYGA8bGF5ZXI+LmVuZ3JhbS5lbWJlZC57d2VpZ2h0LHNjYWxlfS5yb3dzXzxzdGFydD5gYC4gRWFjaCBjaHVuaydzIG92ZXJsYXAKKyAgICB3aXRoIHRoaXMgcmFuaydzIHJvd3MgaXMgY29waWVkIGludG8gdGhlIHRhYmxlOyBvdGhlciB3ZWlnaHRzIHBhc3MKKyAgICB0aHJvdWdoLiBBIHRhYmxlIHRoYXQgcmVjZWl2ZXMgYW55IGNodW5rIG11c3QgaGF2ZSB0aGlzIHJhbmsncyByb3dzCisgICAgY292ZXJlZCBleGFjdGx5IG9uY2UuCisgICAgIiIiCisgICAgc3BhbnM6IGRpY3Rbc3RyLCBsaXN0W3R1cGxlW2ludCwgaW50XV1dID0ge30KKyAgICBmb3IgbmFtZSwgdGVuc29yIGluIHdlaWdodHM6CisgICAgICAgIG1hdGNoID0gX0VOR1JBTV9ST1dfQ0hVTksubWF0Y2gobmFtZSkKKyAgICAgICAgaWYgbWF0Y2ggaXMgTm9uZToKKyAgICAgICAgICAgIHlpZWxkIG5hbWUsIHRlbnNvcgorICAgICAgICAgICAgY29udGludWUKKyAgICAgICAgIyBNYXAgb25seSB0aGUgbW9kdWxlIHByZWZpeDogdGhlIFZMIHdyYXBwZXIgaGFuZHMgaXRzIGNoaWxkIG5hbWVzIHRoYXQKKyAgICAgICAgIyBhcmUgYWxyZWFkeSBtYXBwZWQsIGFuZCB0aGUgc3VmZml4IHJ1bGVzIGRvIG5vdCBzZWUgYGAucm93c188c3RhcnQ+YGAuCisgICAgICAgIG1vZHVsZSA9IG1hcHBlci5tYXBfbmFtZShtYXRjaC5ncm91cCgxKSkKKyAgICAgICAgaWYgbW9kdWxlIGlzIE5vbmU6CisgICAgICAgICAgICBjb250aW51ZQorICAgICAgICBsZWFmID0gIndlaWdodCIgaWYgbWF0Y2guZ3JvdXAoMikgPT0gIndlaWdodCIgZWxzZSAid2VpZ2h0X3NjYWxlX2ludiIKKyAgICAgICAgcG5hbWUgPSBmInttb2R1bGV9LmVtYmVkX3Rva2Vucy57bGVhZn0iCisgICAgICAgIHBhcmFtID0gcGFyYW1zLmdldChwbmFtZSkKKyAgICAgICAgaWYgcGFyYW0gaXMgTm9uZToKKyAgICAgICAgICAgIGlmIGFueShrZXkuc3RhcnRzd2l0aChmInttb2R1bGV9LiIpIGZvciBrZXkgaW4gcGFyYW1zKToKKyAgICAgICAgICAgICAgICByYWlzZSBWYWx1ZUVycm9yKGYiRW5ncmFtIHJvdyBjaHVuayB7bmFtZX0gbWF0Y2hlcyBubyBwYXJhbWV0ZXIiKQorICAgICAgICAgICAgY29udGludWUgICMgQSBsYXllciBvZiBhbm90aGVyIHBpcGVsaW5lLXBhcmFsbGVsIHJhbmsuCisgICAgICAgIGlmIHRlbnNvci5kdHlwZSA9PSB0b3JjaC5mbG9hdDhfZThtMGZudToKKyAgICAgICAgICAgIHRlbnNvciA9IHRlbnNvci52aWV3KHRvcmNoLnVpbnQ4KQorICAgICAgICBzdGFydCwgbG8gPSBpbnQobWF0Y2guZ3JvdXAoMykpLCBwYXJhbS5lbmdyYW1fdm9jYWJfc3RhcnQKKyAgICAgICAgYSA9IG1heChzdGFydCwgbG8pCisgICAgICAgIGIgPSBtaW4oc3RhcnQgKyB0ZW5zb3Iuc2hhcGVbMF0sIGxvICsgcGFyYW0uc2hhcGVbMF0pCisgICAgICAgIHJvd3MgPSBzcGFucy5zZXRkZWZhdWx0KHBuYW1lLCBbXSkKKyAgICAgICAgaWYgYSA8IGI6CisgICAgICAgICAgICBwYXJhbS5kYXRhW2EgLSBsbyA6IGIgLSBsb10uY29weV8odGVuc29yW2EgLSBzdGFydCA6IGIgLSBzdGFydF0pCisgICAgICAgICAgICByb3dzLmFwcGVuZCgoYSAtIGxvLCBiIC0gbG8pKQorICAgICAgICBsb2FkZWQuYWRkKHBuYW1lKQorICAgIGZvciBwbmFtZSwgcm93cyBpbiBzcGFucy5pdGVtcygpOgorICAgICAgICBlbmQgPSAwCisgICAgICAgIGZvciBhLCBiIGluIHNvcnRlZChyb3dzKToKKyAgICAgICAgICAgIGlmIGEgIT0gZW5kOgorICAgICAgICAgICAgICAgIHJhaXNlIFZhbHVlRXJyb3IoCisgICAgICAgICAgICAgICAgICAgIGYiRW5ncmFtIHJvdyBjaHVua3Mgb2Yge3BuYW1lfTogcmFuayByb3dzICIKKyAgICAgICAgICAgICAgICAgICAgZiJ7bWluKGEsIGVuZCl9Li57bWF4KGEsIGVuZCl9IG1pc3Npbmcgb3IgZHVwbGljYXRlZCIKKyAgICAgICAgICAgICAgICApCisgICAgICAgICAgICBlbmQgPSBiCisgICAgICAgIGlmIGVuZCAhPSBwYXJhbXNbcG5hbWVdLnNoYXBlWzBdOgorICAgICAgICAgICAgcmFpc2UgVmFsdWVFcnJvcigKKyAgICAgICAgICAgICAgICBmIkVuZ3JhbSByb3cgY2h1bmtzIGNvdmVyIHtlbmR9IG9mICIKKyAgICAgICAgICAgICAgICBmIntwYXJhbXNbcG5hbWVdLnNoYXBlWzBdfSByYW5rIHJvd3Mgb2Yge3BuYW1lfSIKKyAgICAgICAgICAgICkKKworCiBkZWYgX21ha2VfZGVlcHNlZWtfdjRfd2VpZ2h0c19tYXBwZXIoCiAgICAgZXhwZXJ0X2R0eXBlOiBzdHIsIGxpbmVhcl9zY2FsZV9uYW1lOiBzdHIgPSAid2VpZ2h0X3NjYWxlX2ludiIKICkgLT4gV2VpZ2h0c01hcHBlcjoKQEAgLTE4NTgsOCArMTkxOSwxNiBAQCBjbGFzcyBEZWVwc2Vla1Y0MUxMTUZvckNhdXNhbExNKAogICAgICAgICBpZiBncHRxX2ZpbGVzOgogICAgICAgICAgICAgIyBBbHJlYWR5IGluIHZMTE0gbmFtaW5nLCB3aGljaCBib3RoIHRoZSB0ZXh0IGFuZCBWTCBtYXBwZXJzIGtlZXAuCiAgICAgICAgICAgICB3ZWlnaHRzID0gY2hhaW4od2VpZ2h0cywgaXRlcl9leHBlcnRfZ3B0cV93ZWlnaHRzKGdwdHFfZmlsZXMsICJtb2RlbC4iKSkKKyAgICAgICAgZW5ncmFtX2NodW5rczogc2V0W3N0cl0gPSBzZXQoKQorICAgICAgICB3ZWlnaHRzID0gX2VuZ3JhbV9yb3dfY2h1bmtzKAorICAgICAgICAgICAgd2VpZ2h0cywKKyAgICAgICAgICAgIHNlbGYuaGZfdG9fdmxsbV9tYXBwZXIsCisgICAgICAgICAgICBkaWN0KHNlbGYubmFtZWRfcGFyYW1ldGVycygpKSwKKyAgICAgICAgICAgIGVuZ3JhbV9jaHVua3MsCisgICAgICAgICkKICAgICAgICAgbG9hZGVyID0gQXV0b1dlaWdodHNMb2FkZXIoc2VsZikKICAgICAgICAgbG9hZGVkX3BhcmFtcyA9IGxvYWRlci5sb2FkX3dlaWdodHMod2VpZ2h0cywgbWFwcGVyPXNlbGYuaGZfdG9fdmxsbV9tYXBwZXIpCisgICAgICAgIGxvYWRlZF9wYXJhbXMgfD0gZW5ncmFtX2NodW5rcwogICAgICAgICBzZWxmLnByb2Nlc3Nfd2VpZ2h0c19hZnRlcl9sb2FkaW5nKCkKICAgICAgICAgaWYgY29uZmlnLmVuZ3JhbV9jb25maWcgYW5kIGNvbmZpZy5lbmdyYW1fY29uZmlnLnVzZV90aHA6CiAgICAgICAgICAgICAjIExvYWRpbmcgd2VpZ2h0cyByZWZpbGxzIHRoZSBmaWxlIGNhY2hlOyByZWxlYXNlIGl0IGJlZm9yZSBNQURWX0NPTExBUFNFLgo=").decode()


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
