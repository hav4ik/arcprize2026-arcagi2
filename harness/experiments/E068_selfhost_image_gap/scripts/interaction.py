import random, sys
sys.path.insert(0, "experiments/E066_dsv41_image_pruning/scripts")
from compare_evals import load, done
L = {"ai": "deepseek-flash_low_e058_pilot30", "si": "dsv41_low_e066_full_ref", "at": "deepseek-flash_low_e046p_note_all",
     "st": "dsv41_low_e066_full_ref_txt", "ci": "dsv41_low_e066_c1x_img", "ct": "dsv41_low_e066_c1x_txt"}
D = {k: load(v) for k, v in L.items()}
def f(d, k):
    v = d.get(k) or {}; return sum(bool(x) for x in v.values()) / max(1, len(v))
def ci(ks, fn, n=4000):
    random.seed(0); x = [fn(k) for k in ks]; m = 100 * sum(x) / len(x)
    b = sorted(100 * sum(random.choice(x) for _ in x) / len(x) for _ in range(n))
    return f"{m:+5.1f} [{b[int(.025*n)]:+5.1f}, {b[int(.975*n)]:+5.1f}] (n={len(ks)})"
B = [k for k in D["si"] if done(D["si"], *k) and done(D["st"], *k)]
print("on attempts both self-hosted runs finished:")
print("  images effect, API        ", ci(B, lambda k: f(D["ai"], k) - f(D["at"], k)))
print("  images effect, self-hosted", ci(B, lambda k: f(D["si"], k) - f(D["st"], k)))
print("  interaction (self - API)  ", ci(B, lambda k: (f(D["si"], k) - f(D["st"], k)) - (f(D["ai"], k) - f(D["at"], k))))
A = [k for k in D["ci"] if done(D["ci"], *k) and done(D["ct"], *k)]
print("all 240 attempts:")
print("  images effect, API        ", ci(A, lambda k: f(D["ai"], k) - f(D["at"], k)))
print("  images effect, cand 1     ", ci(A, lambda k: f(D["ci"], k) - f(D["ct"], k)))
print("  interaction (cand1 - API) ", ci(A, lambda k: (f(D["ci"], k) - f(D["ct"], k)) - (f(D["ai"], k) - f(D["at"], k))))
