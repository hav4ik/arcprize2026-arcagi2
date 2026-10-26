"""Write the records of attempts 2,3 of a results file as attempts 0,1 (for multi_test_compare.py / analysis)."""
import json, sys
src, dst = sys.argv[1], sys.argv[2]
with open(dst, "w") as f:
    for l in open(src):
        r = json.loads(l)
        if r.get("attempt") in (2, 3):
            r["attempt"] -= 2
            f.write(json.dumps(r) + "\n")
