#!/usr/bin/env python3
"""E066 tests for class_stats.py on synthetic observer records built from real token streams (CPU only).

1. E043 equivalence: with weights --w-enc 1,1,10,1 --w-dec 0,1,10,0 (images were prompt inputs in E043), the keep
   sets equal E043's reap_aggregate + special_stats --region + select --score contrib --decoder-gen-only
   --protect 0 --weight 10, on E043-style requests (n - 1 inputs).
2. Full-sequence requests (replay_mm.py, "sent": "full") align, score n - 1 positions, and image positions are
   class 3 exactly where the stream has id 129264.
3. A corrupted step is rejected.

  test_class_stats.py --tok DIR --tokenizer tokenizer.json [--n-req 6]
"""
import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
E043 = HERE.parents[1] / "E043_dsv41_real_pruning/scripts"
PY = sys.executable
L, E, K = 40, 384, 6


def fake_obs(obs, tokdir, reqs, full, rng, corrupt=False):
    obs.mkdir(parents=True)
    toks = np.fromfile(tokdir / "tokens.u32", dtype=np.uint32)
    warm = [0, 128000, 2000, 3000]
    stream = [np.asarray(warm, np.int64)]
    with open(obs / "manifest.jsonl", "w") as f:
        f.write(json.dumps({"warmup": warm}) + "\n")
        for r in reqs:
            r = {**r, "dir": str(tokdir)}
            if full:
                r["sent"] = "full"
            f.write(json.dumps(r) + "\n")
            n = r["length"] if full else r["length"] - 1
            stream.append(toks[r["offset"]: r["offset"] + n].astype(np.int64))
    stream = np.concatenate(stream)
    pop = rng.dirichlet(np.full(E, 0.3), size=L)  # skewed expert popularity per layer
    cur, i = 0, 0
    while cur < len(stream):
        T = min(int(rng.integers(500, 9000)), len(stream) - cur)
        ids = np.empty((T, L, K), np.int16)
        for li in range(L):  # K distinct experts per position, by Gumbel top-k on the layer's popularity
            g = np.log(pop[li])[None, :] + rng.gumbel(size=(T, E))
            ids[:, li] = np.argpartition(-g, K, axis=1)[:, :K]
        inp = stream[cur:cur + T].copy()
        if corrupt and i == 1:
            inp[3] += 1
        np.savez(obs / f"step_{i:06d}.npz", input_ids=inp.astype(np.int32), positions=np.arange(T, dtype=np.int32),
                 ids=ids, w=rng.random((T, L, K)).astype(np.float32),
                 amp=rng.gamma(1.0, 1.0, (T, L, K)).astype(np.float32))
        cur += T
        i += 1


def run(*args):
    p = subprocess.run([PY, *map(str, args)], capture_output=True, text=True, cwd=E043)
    return p.returncode, p.stdout + p.stderr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tok", required=True, type=Path)
    ap.add_argument("--n-req", type=int, default=6)
    ap.add_argument("--tokenizer", required=True, help="tokenizer.json (special_stats.py reads its added tokens)")
    a = ap.parse_args()
    a.tok = a.tok.resolve()  # the E043 scripts run from their own directory
    rng = np.random.default_rng(0)
    recs = [json.loads(line) for line in open(a.tok / "index.jsonl")]
    reqs = sorted(recs, key=lambda r: r["length"])[: a.n_req]  # short ones keep the test fast
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        # 1. E043 equivalence
        fake_obs(td / "obs43", a.tok, reqs, full=False, rng=rng)
        code, out = run(E043 / "reap_aggregate.py", "aggregate", "--obs", td / "obs43", "--out", td / "stats.npz")
        assert code == 0, out
        code, out = run(E043 / "special_stats.py", "aggregate", "--region", "--obs", td / "obs43",
                        "--tokenizer", a.tokenizer, "--out", td / "blk.npz")
        assert code == 0, out
        code, out = run(E043 / "special_stats.py", "select", "--stats", td / "stats.npz", "--special", td / "blk.npz",
                        "--score", "contrib", "--decoder-gen-only", "--protect", "0", "--weight", "10",
                        "--experts", "96", "128", "--out-dir", td / "keep43")
        assert code == 0, out
        code, out = run(HERE / "class_stats.py", "aggregate", "--obs", td / "obs43", "--out", td / "cls.npz")
        assert code == 0, out
        code, out = run(HERE / "class_stats.py", "select", "--stats", td / "cls.npz", "--w-enc", "1,1,10,1",
                        "--w-dec", "0,1,10,0", "--avg", "96", "128", "--out-dir", td / "keep66")
        assert code == 0, out
        z43, z66 = np.load(td / "stats.npz"), np.load(td / "cls.npz")
        assert z43["N"].sum() == z66["N"].sum(), "different number of selections"
        for e in (96, 128):
            k43 = np.load(td / f"keep43/keep_contrib_dec_sp0_w10_e{e}.npy")
            k66 = np.load(td / f"keep66/keep_contrib_enc1-1-10-1_dec0-1-10-0_uniform_avg{e}.npy")
            assert (k43 == k66).all(), f"E{e}: keep sets differ in {(k43 != k66).sum()} entries"
        print("1. E043 equivalence: keep sets identical at E96 and E128")

        # 2. full-sequence requests and the image class
        fake_obs(td / "obsfull", a.tok, reqs, full=True, rng=rng)
        code, out = run(HERE / "class_stats.py", "aggregate", "--obs", td / "obsfull", "--out", td / "clsf.npz")
        assert code == 0, out
        pos = np.load(td / "clsf.npz")["pos"]
        toks = np.fromfile(a.tok / "tokens.u32", dtype=np.uint32)
        n_img = sum(int((toks[r["offset"]: r["offset"] + r["length"] - 1] == 129264).sum()) for r in reqs)
        n_scored = sum(r["length"] - 1 for r in reqs)
        assert pos.sum() == n_scored, (pos.sum(), n_scored)
        assert pos[3] == n_img and n_img > 0, (pos[3], n_img)
        print(f"2. full requests: {n_scored} scored positions, {n_img} image positions in class 3, "
              f"classes {dict(zip(['prompt', 'gen', 'block', 'image'], pos.tolist()))}")
        code, out = run(HERE / "class_stats.py", "select", "--stats", td / "clsf.npz", "--avg", "112", "120",
                        "--alloc", "blend", "--floor", "96", "--cap", "160", "--out-dir", td / "keepb")
        assert code == 0, out
        for line in out.strip().splitlines():
            r = json.loads(line)
            k = np.load(td / f"keepb/keep_{r['mask']}.npy")
            assert k.sum() == r["experts_total"] and (k.sum(1) >= 96).all() and (k.sum(1) <= 160).all()
        print("   per-layer allocation: totals and [floor, cap] hold")

        # 4. layer sweep, sens, calibrated step-8 allocation
        code, out = run(HERE / "class_stats.py", "sweep-masks", "--stats", td / "clsf.npz", "--ref", "128", "--cut",
                        "64", "--out-dir", td / "sweep")
        assert code == 0, out
        base = np.load(td / "sweep/keep_base.npy")
        assert (base.sum(1) == 128).all()
        for li in (0, 37):
            k = np.load(td / f"sweep/keep_l{li:02d}.npy")
            assert k[li].sum() == 64 and (k[li] <= base[li]).all(), "cut layer must keep a subset of its 128"
            others = np.delete(np.arange(L), li)
            assert (k[others] == base[others]).all(), "other layers unchanged"
        true_delta = np.full(L, 0.002)
        true_delta[37] = 0.05  # one expensive layer
        with open(td / "sweep_nll.jsonl", "w") as f:
            for tag, d in [("base", 0.0), ("base2", 0.0001)] + [(f"l{li:02d}", true_delta[li]) for li in range(L)]:
                for i in range(4):
                    f.write(json.dumps({"tag": tag, "task_id": f"t{i}", "pair": 0, "attempt": 0, "batch": "b",
                                        "n_gen": 1000, "nll_gen": 1000 * (0.7 + d), "n_blk": 100,
                                        "nll_blk": 100 * (0.1 + d)}) + "\n")
        code, out = run(HERE / "class_stats.py", "sens", "--nll", td / "sweep_nll.jsonl", "--out", td / "sens.json")
        assert code == 0, out
        sens = json.load(open(td / "sens.json"))
        assert abs(sens["delta"][37] - (0.05 - 0.00005)) < 1e-9 and abs(sens["noise_nll_gen"] - 0.0001) < 1e-9
        code, out = run(HERE / "class_stats.py", "select", "--stats", td / "clsf.npz", "--alloc", "calibrated",
                        "--sens", td / "sens.json", "--shape", "blend", "--avg", "112", "120", "--floor", "96",
                        "--cap", "160", "--step", "8", "--out-dir", td / "keepc")
        assert code == 0, out
        code2, out2 = run(HERE / "class_stats.py", "select", "--stats", td / "clsf.npz", "--alloc", "blend",
                          "--avg", "112", "--floor", "96", "--cap", "160", "--step", "8", "--out-dir", td / "keepp")
        assert code2 == 0, out2
        proxy37 = json.loads(out2.strip().splitlines()[0])["per_layer"][37]
        for line in out.strip().splitlines():
            r = json.loads(line)
            k = np.load(td / f"keepc/keep_{r['mask']}.npy")
            cnt = k.sum(1)
            assert k.sum() == r["experts_total"] == round(40 * float(r["mask"].split("avg")[1]))
            assert (cnt >= 96).all() and (cnt <= 160).all() and (cnt % 8 == 0).all()
            assert cnt[37] >= proxy37 and cnt[37] == cnt.max(), (cnt[37], proxy37, cnt.max())
            assert r["predicted_nll_increase_vs_ref"] <= r["predicted_nll_increase_uniform"] + 1e-12
        print(f"4. sweep masks, sens and calibrated step-8 allocation hold; the expensive layer gets {cnt[37]} "
              f"experts (proxy alone: {proxy37})")

        # 3. corrupted stream is rejected
        fake_obs(td / "obsbad", a.tok, reqs, full=True, rng=rng, corrupt=True)
        code, out = run(HERE / "class_stats.py", "aggregate", "--obs", td / "obsbad", "--out", td / "bad.npz")
        assert code != 0 and "differ" in out, out
        print("3. corrupted step rejected:", out.strip().splitlines()[-1])
    print("all class_stats tests passed")


if __name__ == "__main__":
    main()
