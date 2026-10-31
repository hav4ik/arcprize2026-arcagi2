#!/usr/bin/env python3
"""Publish queue for the dashboard artifact.
  publish_list.py            print dataset id -> file path for every dataset changed since the last publish
  publish_list.py --mark     mark the queued versions as published
  publish_list.py --update   (auto_dash.sh) recompute the queue from the files' sha256
"""
import hashlib, json, os, sys, datetime as dt
T = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
Q = f"{T}/publish_queue.json"
IDS = ["runs", "mem", "tps", "acc", "outages"]
def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()
def load():
    try: return json.load(open(Q))
    except Exception: return {"last_published": {}, "changed": []}
def save(q):
    q["updated_utc"] = dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    json.dump(q, open(Q + ".tmp", "w"), indent=1); os.replace(Q + ".tmp", Q)
q = load()
if "--update" in sys.argv:
    q["changed"] = [{"id": i, "path": f"{T}/{i}.json", "sha256": sha(f"{T}/{i}.json")} for i in IDS
                    if os.path.exists(f"{T}/{i}.json") and sha(f"{T}/{i}.json") != q["last_published"].get(i)]
    save(q)
elif "--mark" in sys.argv:
    for c in q["changed"]: q["last_published"][c["id"]] = c["sha256"]
    q["changed"] = []; q["published_utc"] = dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"); save(q)
    print("marked published")
else:
    for c in q["changed"]: print(f"{c['id']}\t{c['path']}")
    if not q["changed"]: print("(nothing changed since the last publish)")
