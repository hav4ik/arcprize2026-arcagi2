"""Draw the E017 phase-1B reading sample (seeded). Writes sample.jsonl."""
import json, random, collections
from pathlib import Path
OUT = Path(__file__).resolve().parents[1]
rows = [json.loads(l) for l in open(OUT / "attempts_index.jsonl")]
rng = random.Random(2017)
key = lambda r: (r["task_id"], r["pair"])
by_pair = collections.defaultdict(list)
for r in rows:
    by_pair[key(r)].append(r)
sample = []
# (i) wrong submitted answers on pairs another attempt in these runs solved; one per task
cand = [r for r in rows if r["outcome"] == "wrong" and any(x["outcome"] == "correct" for x in by_pair[key(r)])]
tasks = sorted({r["task_id"] for r in cand}); rng.shuffle(tasks)
for t in tasks[:20]:
    pool = [r for r in cand if r["task_id"] == t]
    sample.append(dict(rng.choice(pool), bucket="wrong_solvable"))
# (ii) failed attempts on pairs no config ever solved: up to 2 per pair (prefer wrong over none), 10 total
un = sorted({key(r) for r in rows if not r["pair_solved_any"]})
picked = []
for k in un:
    pool = sorted(by_pair[k], key=lambda r: (r["outcome"] != "wrong", rng.random()))
    picked += pool[:2]
sample += [dict(r, bucket="never_solved") for r in picked[:10]]
# (iii) correct and long (>50k), one per task
long_ = [r for r in rows if r["outcome"] == "correct" and r["tokens"] > 50000]
lt = sorted({r["task_id"] for r in long_} - {s["task_id"] for s in sample}); rng.shuffle(lt)
for t in lt[:10]:
    sample.append(dict(rng.choice([r for r in long_ if r["task_id"] == t]), bucket="correct_long"))
# (iv) correct and short (<20k): prefer tasks already in the sample (contrast), one per task
short = [r for r in rows if r["outcome"] == "correct" and r["tokens"] < 20000]
in_s = [t for t in dict.fromkeys(s["task_id"] for s in sample) if any(r["task_id"] == t for r in short)]
rng.shuffle(in_s)
for t in in_s[:10]:
    sample.append(dict(rng.choice([r for r in short if r["task_id"] == t]), bucket="correct_short"))
with open(OUT / "sample.jsonl", "w") as f:
    for i, s in enumerate(sample):
        s["sid"] = f"S{i:02d}"
        s.pop("turn_tok", None)
        f.write(json.dumps(s) + "\n")
for s in sample:
    print(s["sid"], s["bucket"], s["run"], s["task_id"], s["pair"], s["attempt"], s["outcome"], s["tokens"], s["turns"])
