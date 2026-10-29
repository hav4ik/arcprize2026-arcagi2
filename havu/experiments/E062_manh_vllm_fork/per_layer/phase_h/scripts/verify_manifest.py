"""Check a local model dir against a sha256 manifest (entries: path, size, sha256): every file's size, and sha256 of
every file (parallel). Prints mismatches and VERIFY_OK/VERIFY_FAIL."""
import concurrent.futures, hashlib, json, os, sys
man, root = json.load(open(sys.argv[1])), sys.argv[2]
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(16 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
bad, missing = [], []
for e in man["entries"]:
    p = os.path.join(root, e["path"])
    if not os.path.exists(p):
        missing.append(e["path"])
    elif os.path.getsize(p) != e["size"]:
        bad.append((e["path"], "size", os.path.getsize(p), e["size"]))
print("sizes checked:", len(man["entries"]), "missing", len(missing), "size mismatches", len(bad), flush=True)
todo = [e for e in man["entries"] if e["path"] not in missing]
with concurrent.futures.ThreadPoolExecutor(12) as ex:
    for e, h in zip(todo, ex.map(lambda e: sha(os.path.join(root, e["path"])), todo)):
        if h != e["sha256"]:
            bad.append((e["path"], "sha256"))
extra = sorted(set(os.path.relpath(os.path.join(d, f), root) for d, _, fs in os.walk(root) for f in fs
                   if not f.startswith(".")) - {e["path"] for e in man["entries"]})
print("sha256 checked:", len(todo), "bad", bad[:10], "missing", missing[:10], "extra files", [x for x in extra if not x.startswith(".cache")][:10])
print("VERIFY_OK" if not bad and not missing else "VERIFY_FAIL")
