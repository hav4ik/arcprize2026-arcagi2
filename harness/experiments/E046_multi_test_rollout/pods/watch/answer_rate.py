"""Answer rate and basic stats per attempts.jsonl. Usage: answer_rate.py <solutions.json> name=path ..."""
import json, sys
sol = json.load(open(sys.argv[1]))
for arg in sys.argv[2:]:
    name, path = arg.split("=", 1)
    recs = [r for r in map(json.loads, open(path)) if "task_id" in r]
    none = sum(r.get("pred") is None for r in recs)
    err = sum(bool(r.get("error")) for r in recs)
    multi = sum(bool(r.get("multi_test")) for r in recs)
    pairs = set(); corr_pairs = set(); n_pair_preds = 0; n_pair_corr = 0
    for r in recs:
        t = r["task_id"]
        if r.get("multi_test"):
            preds = r["pred"] if isinstance(r.get("pred"), list) else [None] * len(sol[t])
            for q, g in enumerate(preds):
                pairs.add((t, q)); n_pair_preds += 1
                if g is not None and g == sol[t][q]:
                    corr_pairs.add((t, q)); n_pair_corr += 1
        else:
            pairs.add((t, r["pair"])); n_pair_preds += 1
            if r.get("pred") is not None and r["pred"] == sol[t][r["pair"]]:
                corr_pairs.add((t, r["pair"])); n_pair_corr += 1
    tp = sum(r.get("train_pass") is True for r in recs)
    out = sum(((r.get("usage") or {}).get("completion") or 0) % 1000000000 for r in recs)  # cut attempts carry +1e9
    stops = {}
    for r in recs:
        stops[r.get("stop_reason")] = stops.get(r.get("stop_reason"), 0) + 1
    tmax = max((r.get("t_end_h") or 0) for r in recs) if recs else 0
    print(f"{name:16s} records {len(recs):3d} (multi {multi}), pred None {none} ({100*none/max(1,len(recs)):.1f}%), errors {err}, "
          f"train_pass {tp}; pair-preds {n_pair_preds}, correct {n_pair_corr}; pairs {len(pairs)}, pairs correct (any) {len(corr_pairs)}; "
          f"out tokens {out/1e6:.2f}M, last t_end {tmax:.2f}h; stops {stops}")
