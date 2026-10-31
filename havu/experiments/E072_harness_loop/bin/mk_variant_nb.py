"""mk_variant_nb.py <notebook in> <'K=V K=V' job env> <notebook out>: makes the job's notebook. Cell 0b checks `_MT_EXPECT`
(os.environ == expected) and, on a mismatch in a rerun, falls back to the kit's v3 solver (a crash path on Manh's stack, R10).
So when the job env changes a key that _MT_EXPECT lists, the dict entry is patched to the new value (exact string replacement,
once) and nothing else changes. Verified statically: every _MT_EXPECT entry must then equal the effective env (job env, else the
notebook's own expectation). Prints JSON {sha_in, sha_out, patched:{key:[old,new]}}. Exit 1 on any inconsistency."""
import hashlib, json, re, sys
src, envs, dst = sys.argv[1:4]
extra = json.loads(sys.argv[4]) if len(sys.argv) > 4 and sys.argv[4] else []  # [{"old":..,"new":..}]: exact string patches, each must occur once in the notebook source
raw = open(src, "rb").read(); nb = json.loads(raw)
env = dict(kv.split("=", 1) for kv in envs.split() if "=" in kv)
cells = [c for c in nb["cells"] if c["cell_type"] == "code" and "_MT_EXPECT = {" in "".join(c["source"])]
patched = {}
if cells:
    c = cells[0]; s = "".join(c["source"])
    m = re.search(r"^_MT_EXPECT = (\{.*\})$", s, re.M); assert m, "no _MT_EXPECT line"
    exp = json.loads(m.group(1)); line = m.group(0)
    for k, v in env.items():
        if k in exp and exp[k] != v:
            old = f'"{k}": "{exp[k]}"'; assert line.count(old) == 1, old
            line = line.replace(old, f'"{k}": "{v}"'); patched[k] = [exp[k], v]; exp[k] = v
    new = s.replace(m.group(0), line); assert new.count(line) == 1
    c["source"] = new.splitlines(keepends=True)
    eff = {k: env.get(k, v) for k, v in json.loads(re.search(r"^_MT_EXPECT = (\{.*\})$", new, re.M).group(1)).items()}
    assert eff == json.loads(re.search(r"^_MT_EXPECT = (\{.*\})$", new, re.M).group(1)), "inconsistent"
for e in extra:
    hits = [(c, "".join(c["source"])) for c in nb["cells"] if e["old"] in "".join(c["source"])]
    assert len(hits) == 1 and hits[0][1].count(e["old"]) == 1, f"patch must match exactly once: {e['old']!r} ({len(hits)} cells)"
    hits[0][0]["source"] = hits[0][1].replace(e["old"], e["new"]).splitlines(keepends=True); patched["text:" + e["old"][:40]] = [e["old"], e["new"]]
if patched:
    open(dst, "w").write(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
else:
    open(dst, "wb").write(raw)
print(json.dumps({"sha_in": hashlib.sha256(raw).hexdigest(), "sha_out": hashlib.sha256(open(dst, "rb").read()).hexdigest(),
                  "patched": patched, "has_check_cell": bool(cells)}))
