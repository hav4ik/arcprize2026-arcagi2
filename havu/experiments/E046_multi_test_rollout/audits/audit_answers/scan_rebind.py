import gzip, io, json, re, sys, tarfile
from collections import Counter
pat = re.compile(r"^\s*(test_inputs?)\s*(=|\+=)(?!=)\s*(.*)$", re.M)
mut = re.compile(r"^\s*(test_inputs?)\s*(\[|\.(append|sort|reverse|insert|pop))", re.M)
tot = Counter(); ex = []
for path in sys.argv[1:]:
    tf = tarfile.open(path)
    for m in tf.getmembers():
        if not m.isfile() or not m.name.endswith(".json.gz"):
            continue
        try:
            tr = json.load(gzip.open(io.BytesIO(tf.extractfile(m).read()), "rt"))
        except Exception:
            continue
        tot["traces"] += 1
        for msg in tr["messages"]:
            for c in msg.get("tool_calls") or []:
                try:
                    code = json.loads(c["function"]["arguments"]).get("code", "")
                except Exception:
                    continue
                for g in pat.finditer(code):
                    tot["rebind " + g.group(1)] += 1
                    ex.append((path.split("runs/")[-1][:20], m.name.split("/")[-1], g.group(0).strip()[:90]))
                for g in mut.finditer(code):
                    tot["mutate " + g.group(1)] += 1
                    ex.append((path.split("runs/")[-1][:20], m.name.split("/")[-1], g.group(0).strip()[:90]))
print(dict(tot))
for e in ex[:40]: print("  ", e)
