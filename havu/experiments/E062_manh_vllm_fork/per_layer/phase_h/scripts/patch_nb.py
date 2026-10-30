"""Patch V12's cell 1b integrity manifest: replace the embedded INTEGRITY JSON ({"model": {path: [size, sha256]}}) by
the staged model dir's manifest, as a raw-text replacement of that one JSON string in the .ipynb (every other byte
unchanged). Checks: the old blob occurs exactly once; the result parses; only cell 6's source differs, and only in the
INTEGRITY line; regenerating the E128 blob from its own manifest reproduces the original text (same format).
usage: patch_nb.py IN.ipynb NEW_MANIFEST.json OUT.ipynb"""
import difflib, hashlib, json, sys
src, man, dst = sys.argv[1:4]
raw = open(src, encoding="utf-8").read()
nb = json.loads(raw)
cell = "".join(nb["cells"][6]["source"])
a = cell.index('INTEGRITY = json.loads(r"""') + len('INTEGRITY = json.loads(r"""'); b = cell.index('""")', a)
old_blob = cell[a:b]
old = json.loads(old_blob)
assert json.dumps(old) == old_blob, "format: json.dumps of the old manifest does not reproduce the embedded text"
new = json.load(open(man))
assert set(new) == {"model"} and all(isinstance(v, list) and len(v) == 2 for v in new["model"].values())
new_blob = json.dumps({"model": dict(sorted(new["model"].items()))})
assert list(old["model"]) == sorted(old["model"]), "the original is sorted by path"
esc_old, esc_new = json.dumps(old_blob)[1:-1], json.dumps(new_blob)[1:-1]
assert raw.count(esc_old) == 1, f"old blob occurs {raw.count(esc_old)} times"
out = raw.replace(esc_old, esc_new)
nb2 = json.loads(out)
diffs = [i for i, (c1, c2) in enumerate(zip(nb["cells"], nb2["cells"])) if c1 != c2]
src_lines = lambda c: "".join(c["source"]).splitlines(keepends=True)
l1, l2 = src_lines(nb["cells"][6]), src_lines(nb2["cells"][6])
changed = [i for i, (x, y) in enumerate(zip(l1, l2)) if x != y]
if new_blob == old_blob:  # self-test: the identity patch must reproduce the original bytes
    assert diffs == [] and out == raw
    changed = [next(i for i, x in enumerate(l1) if x.startswith("INTEGRITY = json.loads"))]
else:
    assert diffs == [6] and len(nb["cells"]) == len(nb2["cells"]), diffs
    assert len(l1) == len(l2) and len(changed) == 1 and l1[changed[0]].startswith("INTEGRITY = json.loads"), changed
open(dst, "w", encoding="utf-8").write(out)
h = lambda s: hashlib.sha256(s.encode()).hexdigest()
print(f"patched {dst}: sha256 {h(out)} (was {h(raw)}); cell 6 line {changed[0]}: {len(old['model'])} -> {len(new['model'])} files, "
      f"{sum(v[0] for v in old['model'].values()) / 1e9:.2f} -> {sum(v[0] for v in new['model'].values()) / 1e9:.2f} GB")
om, nm = old["model"], new["model"]
print("files only in the old manifest:", sorted(set(om) - set(nm)))
print("files only in the new manifest:", sorted(set(nm) - set(om)))
print("same path, different bytes:", sorted(k for k in set(om) & set(nm) if om[k] != nm[k]))
print("same path and bytes:", len([k for k in set(om) & set(nm) if om[k] == nm[k]]))
