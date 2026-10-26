"""E054: what cell 0b would run to put the E054 files into the kit overlay before cell 2 installs it.

Not wired into kaggle/dsv41_tools/build_frozen_mt.py (README "Kaggle packaging" describes the builder change).
The function body is self-contained (stdlib only) so the builder can paste it into cell 0b's source, with
_E054_ADD = base64 of kit/vllm_overlay_e054_add.tgz and _E054_SHA = the {path: sha256} of its .sha256 file.

Cell 2 then runs unchanged: it extracts KIT/vllm_overlay_rw.tgz into site-packages and checks every line of
KIT/vllm_overlay_rw.sha256. A gzip tar cannot be appended in place, so the tarball is rebuilt: the kit's
members in order, the E054 members replaced (or appended if new), the sha list rewritten the same way.
On failure in the competition rerun, restore_kit_overlay() puts the *.v3orig copies back (V6 behaviour).
tests/test_cell0b_sim.py runs both against the real kit files and cell 2's install code.
"""

KIT_OVERLAY_SHA = "377d83e0942ef32133545beb8ff7d945039ea5fd061954546b4b046be84fea6e"


def e054_overlay_into_kit(kit, add_tgz, add_sha, base_sha=KIT_OVERLAY_SHA):
    import hashlib
    import io
    import os
    import shutil
    import tarfile

    tgz, lst = f"{kit}/vllm_overlay_rw.tgz", f"{kit}/vllm_overlay_rw.sha256"
    with open(tgz, "rb") as fh:
        h = hashlib.sha256(fh.read()).hexdigest()
    assert h == base_sha, f"kit overlay is {h[:8]}, E054 was built against {base_sha[:8]}"
    new = {}
    with tarfile.open(fileobj=io.BytesIO(add_tgz)) as tf:
        for m in tf.getmembers():
            if m.isfile():
                data = tf.extractfile(m).read()
                assert hashlib.sha256(data).hexdigest() == add_sha[m.name], f"E054 member {m.name} corrupt"
                new[m.name] = (m, data)
    assert set(new) == set(add_sha), "E054 tarball and sha list disagree"
    for f in (tgz, lst):
        if not os.path.exists(f + ".v3orig"):
            shutil.copy2(f, f + ".v3orig")
    out = io.BytesIO()
    seen = set()
    with tarfile.open(tgz + ".v3orig") as src, tarfile.open(fileobj=out, mode="w:gz") as dst:
        for m in src.getmembers():
            if m.name in new:
                mi, data = new[m.name]
                dst.addfile(mi, io.BytesIO(data))
                seen.add(m.name)
            else:
                dst.addfile(m, src.extractfile(m) if m.isfile() else None)
        for name in sorted(set(new) - seen):
            mi, data = new[name]
            dst.addfile(mi, io.BytesIO(data))
    with open(tgz + ".e054", "wb") as fh:
        fh.write(out.getvalue())
    os.replace(tgz + ".e054", tgz)
    with open(lst + ".v3orig") as fh:
        rows = [ln.split() for ln in fh if ln.strip()]
    listed = {rel for _, rel in rows}
    text = "".join(f"{add_sha.get(rel, h)}  {rel}\n" for h, rel in rows)
    text += "".join(f"{add_sha[rel]}  {rel}\n" for rel in sorted(set(add_sha) - listed))
    with open(lst + ".e054", "w") as fh:
        fh.write(text)
    os.replace(lst + ".e054", lst)
    return {rel: s[:8] for rel, s in add_sha.items()}


def restore_kit_overlay(kit):
    import os
    import shutil

    for f in (f"{kit}/vllm_overlay_rw.tgz", f"{kit}/vllm_overlay_rw.sha256"):
        if os.path.exists(f + ".v3orig"):
            shutil.copy2(f + ".v3orig", f + ".restore")
            os.replace(f + ".restore", f)
