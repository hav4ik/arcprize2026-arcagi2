"""Per-rank memory line from a run's last sampler record: cap - reserved, cap - peak, reserved, free, retries; nvidia-smi
free; Kaggle-eq RAM. usage: memline.py RUN"""
import json, sys
r = json.loads(open(f"/root/nbruns/{sys.argv[1]}/samp.jsonl").read().strip().splitlines()[-1])
c, v = r.get("alloc_cap") or {}, r.get("vllm_gpu_mem") or {}
print(r["utc"], "; ".join(f"{k}: cap-res {c[k]['cap_minus_reserved_now']}, cap-peak {c[k]['cap_minus_peak']}, reserved {v[k]['reserved']}, "
                          f"free {v[k]['free']}, retries {v[k]['retries']}" for k in sorted(c) if k in v),
      "| smi free GiB", [round(g["free_mib"] / 1024, 2) for g in r.get("gpus", [])], "| Kaggle-eq", r.get("kaggle_eq_gib"))
