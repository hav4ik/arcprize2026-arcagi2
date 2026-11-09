import gzip, io, json, re, sys, tarfile
from collections import Counter
for path in sys.argv[1:]:
    tf = tarfile.open(path)
    c = Counter(); other = []; bad = 0
    for m in tf.getmembers():
        if not m.isfile() or not m.name.endswith(".json.gz"):
            continue
        try:
            tr = json.load(gzip.open(io.BytesIO(tf.extractfile(m).read()), "rt"))
        except Exception:
            bad += 1
            continue
        for e in tr.get("events", []):
            if "fallback" in e:
                o = e["output"]
                if e["accepted"]:
                    cat = "accepted"
                elif "NameError: name 'answer'" in o or "NameError: name 'solve'" in o:
                    cat = "undefined"
                elif "not accepted]" in o and "Traceback" not in o:
                    cat = "format_reject"
                else:
                    cat = "runtime_error"
                c[(e["fallback"], cat)] += 1
                if cat in ("format_reject", "runtime_error", "accepted"):
                    other.append((m.name.split("/")[-1], e["fallback"], cat, o[-200:].replace("\n", " | ")))
    print("==", path.split("E046_multi_test_rollout/")[-1], "unreadable", bad, dict(c))
    for r in other:
        print("   ", r)
