"""Score every run on the task-attempts the self-hosted full-model image run finished (ARC Prize weighting)."""
import json, random, os, sys
sys.path.insert(0, "experiments/E066_dsv41_image_pruning/scripts")
from compare_evals import load, done
RUNS = [("full, API, image (E058)", "deepseek-flash_low_e058_pilot30"),
        ("full, self-hosted, image", "dsv41_low_e066_full_ref"),
        ("full, API, text (E046)", "deepseek-flash_low_e046p_note_all"),
        ("full, self-hosted, text (partial)", "dsv41_low_e066_full_ref_txt"),
        ("cand 1 new E128 2-bit, image", "dsv41_low_e066_c1x_img"),
        ("cand 1 new E128 2-bit, text", "dsv41_low_e066_c1x_txt"),
        ("old tool E128 2-bit (B32), text", "dsv41_low_e043_e128t_final_all"),
        ("E72-152 2-bit, image", "dsv41_low_e066_e72_img"),
        ("E72-152 2-bit, text", "dsv41_low_e066_e72_txt")]
D = {lab: load(suf) for lab, suf in RUNS}
ref = D["full, self-hosted, image"]
A = sorted(k for k in ref if done(ref, *k))
tasks2 = sorted({t for t, a in A if (t, 0) in A and (t, 1) in A})
print(f"self-hosted image finished {len(A)} task-attempts; {len(tasks2)} tasks with both attempts")
def frac(d, t, a):
    v = d.get((t, a)) or {}; return sum(bool(x) for x in v.values()) / max(1, len(v))
def p2(d, t):
    pr = sorted(set((d.get((t, 0)) or {})) | set((d.get((t, 1)) or {}))) or [0]
    return sum(bool((d.get((t, 0)) or {}).get(p)) or bool((d.get((t, 1)) or {}).get(p)) for p in pr) / len(pr)
api = D["full, API, image (E058)"]
random.seed(0)
print(f"{'run':36} {'cover':>6} {'pass@1':>7} {'vs API img [95% CI]':>24} {'pass@2':>7} {'vs API img':>11}")
for lab, _ in RUNS:
    d = D[lab]
    if d is None: print(lab, "missing"); continue
    cov = [k for k in A if done(d, *k)]
    x = [frac(d, *k) - frac(api, *k) for k in cov]
    m = 100 * sum(frac(d, *k) for k in cov) / len(cov)
    bs = sorted(100 * sum(random.choice(x) for _ in x) / len(x) for _ in range(2000))
    t2 = [t for t in tasks2 if done(d, t, 0) and done(d, t, 1)]
    m2 = 100 * sum(p2(d, t) for t in t2) / max(1, len(t2)); a2 = 100 * sum(p2(api, t) for t in t2) / max(1, len(t2))
    print(f"{lab:36} {len(cov):6d} {m:6.1f}% {100*sum(x)/len(x):+6.1f} [{bs[50]:+5.1f},{bs[1950]:+5.1f}] {m2:6.1f}% {m2-a2:+6.1f} ({len(t2)}t)")

st_ = D["full, self-hosted, text (partial)"]
B = [k for k in A if done(st_, *k)]
print(f"\non the {len(B)} task-attempts both self-hosted runs finished:")
for lab in ["full, API, image (E058)", "full, self-hosted, image", "full, API, text (E046)", "full, self-hosted, text (partial)",
            "cand 1 new E128 2-bit, text", "old tool E128 2-bit (B32), text"]:
    d = D[lab]; print(f"  {lab:36} {100*sum(frac(d,*k) for k in B)/len(B):5.1f}%")
