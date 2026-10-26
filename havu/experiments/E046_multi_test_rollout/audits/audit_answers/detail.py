import json, sys
from collections import defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import reconcile as R
S, CH, SOL = R.S, R.CH, R.SOL
for d in sys.argv[1:]:
    res = R.run(d)
    sub = json.load(open(Path(d) / "submission.json"))
    print("==", res["run"])
    # session-sourced answers
    for (t, p), rs in sorted(S.RECORDS.items()):
        for r in rs:
            if r.get("answer_source") in ("session", "session_solve", "parse", "lenient"):
                print(f"   {r['answer_source']:13s} {t}.{p}#{r['attempt']} n={r.get('multi_test')} correct={r['pred'] == SOL[t][p]} in_sub={SOL[t][p] in (sub[t][p]['attempt_1'], sub[t][p]['attempt_2'])} a1_is_this={sub[t][p]['attempt_1'] == r['pred']} cuts={r.get('cuts')}")
    # multi index correctness
    byq = defaultdict(lambda: [0, 0])
    for (t, p), rs in S.RECORDS.items():
        for r in rs:
            if r.get("multi_test") and S.valid_grid(r.get("pred")):
                byq[p][0] += 1
                byq[p][1] += r["pred"] == SOL[t][p]
    print("   multi preds by pair index [n, correct]:", dict(byq))
    # pairs with >1 record
    multi_rec = 0
    for (t, p), rs in sorted(S.RECORDS.items()):
        if len(rs) > 1:
            multi_rec += 1
            row = sub[t][p]
            print(f"   2 recs {t}.{p}: " + "; ".join(f"a{r['attempt']} tp={r.get('train_pass')} ok={r.get('pred') == SOL[t][p]} valid={S.valid_grid(r.get('pred'))}" for r in rs)
                  + f" | sub a1 ok={row['attempt_1'] == SOL[t][p]} a2 ok={row['attempt_2'] == SOL[t][p]} a2_is_fallback={row['attempt_2'] == CH[t]['test'][p]['input']}")
    print("   pairs with >1 record:", multi_rec)
