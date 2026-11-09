import gzip, io, json, re, sys, tarfile
from collections import Counter
for path in sys.argv[1:]:
    tf = tarfile.open(path)
    c = Counter(); ex = []
    for m in tf.getmembers():
        if not m.isfile() or not m.name.endswith(".json.gz"):
            continue
        try:
            tr = json.load(gzip.open(io.BytesIO(tf.extractfile(m).read()), "rt"))
        except Exception:
            continue
        rej = [x["content"] for x in tr["messages"] if x.get("role") == "tool" and "[answer not accepted] this task has" in (x.get("content") or "")]
        if rej:
            got = Counter(re.search(r"got (.*?)$", r.split("\n")[0]).group(1)[:30] if re.search(r"got (.*?)$", r.split("\n")[0]) else "?" for r in rej)
            final_ok = any(x.get("role") == "tool" and (x.get("content") or "").startswith("[answer accepted:") for x in tr["messages"])
            c["traces_with_multi_reject"] += 1
            c["ended_accepted" if final_ok else "never_accepted"] += 1
            ex.append((m.name.split("/")[-1], dict(got), final_ok))
    print("==", path.split("E046_multi_test_rollout/")[-1], dict(c))
    for e in ex: print("   ", e)
