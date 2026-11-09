# Per-run decode-step / prefill-cost fit from solver metrics.jsonl windows (notes/2026-10-30_vllm_fork_acceleration_map.md).
# Model per window: 1 = (gen/n)*(a + b*n) + uncached*t_p. Rough regression; run with .venv/bin/python -I.
import json, sys
import numpy as np
runs = {
 'B32 candBf_x2t (x56, 1.56M, old pod)': '/home/vu/Workspace/ARC-AGI-2/experiments/E055_kaggle_replica/runs/candBf_x2t/out/dsv41_logs/metrics.jsonl',
 'B32+R1 b32r1_x2t (x56, 2.49M, pod 2)': '/home/vu/Workspace/ARC-AGI-2/experiments/E062_manh_vllm_fork/l4pod2/runs/b32r1_x2t/out/dsv41_logs/metrics.jsonl',
 'fork b9ad DCP4 NCCL manh_x2t_w80 (x80, 3.06M, pod 1)': '/home/vu/Workspace/ARC-AGI-2/experiments/E062_manh_vllm_fork/l4/runs/manh_x2t_w80/out/dsv41_logs/metrics.jsonl',
 'B32-M 5e1b DCP4 NCCL b32m_x2t (x80, 3.06M, pod 2)': '/home/vu/Workspace/ARC-AGI-2/experiments/E064_manh_notebook_b32_harness/l4e2e/runs/b32m_x2t/out/dsv41_logs/metrics.jsonl',
 'Manh v12 m12_rerun (x64, 1.61M, v11)': '/home/vu/Workspace/ARC-AGI-2/experiments/E062_manh_vllm_fork/per_layer/phase_h/manh_v12/data/dsv41_logs/metrics.jsonl',
 'Manh r1_rerun bec71fb1 (x64, 1.62M, v11)': '/home/vu/Workspace/ARC-AGI-2/experiments/E062_manh_vllm_fork/per_layer/phase_h/runs/r1_rerun_bec71fb1/dsv41_logs/metrics.jsonl',
 'C1 img run3 (x64, 1.45M, v10+2r)': '/home/vu/Workspace/ARC-AGI-2/experiments/E062_manh_vllm_fork/per_layer/pod3_runs/run3_data/dsv41_logs/metrics.jsonl',
}
for name, path in runs.items():
    rows = [json.loads(l) for l in open(path)]
    rows = [r for r in rows if r.get('window_s', 0) >= 200 and r['gen_tok_s'] > 50 and r['srv_running'] >= 8]
    g = np.array([r['gen_tok_s'] for r in rows]); n = np.array([r['srv_running'] for r in rows])
    u = np.array([r['uncached_tok_s'] for r in rows]); h = np.array([r.get('host_tok_s', 0) for r in rows])
    kv = np.array([r['kv'] for r in rows]); pre = np.array([r.get('preemptions', 0) for r in rows])
    per = g / n
    print(f"== {name}: {len(rows)} windows (>=200 s)")
    print(f"   gen tok/s mean {g.mean():.0f}; running mean {n.mean():.1f}; per-stream mean {per.mean():.2f} (p10 {np.percentile(per,10):.2f}, p90 {np.percentile(per,90):.2f})")
    for lo, hi in ((24, 40), (40, 52), (52, 60), (60, 70), (70, 97)):
        m = (n >= lo) & (n < hi)
        if m.sum():
            print(f"   running {lo}-{hi}: n={m.sum():3d} per-stream {per[m].mean():5.2f}  gen {g[m].mean():5.0f}  uncached {u[m].mean():6.0f}  host {h[m].mean():6.0f}  kv {kv[m].mean():.2f}  preempt/win {pre[m].mean():.1f}")
    # fit: 1 = (g/n)*a + g*b + u*tp  (decode step = a + b*n seconds; tp = s per uncached prefill token)
    X = np.stack([g / n, g, u], 1)
    coef, res, rk, sv = np.linalg.lstsq(X, np.ones(len(rows)), rcond=None)
    a, b, tp = coef
    pred = X @ coef
    print(f"   fit: decode step = {a*1e3:.1f} ms + {b*1e3:.2f} ms/stream; prefill {tp*1e3:.3f} ms/token (= {1/tp if tp>0 else float('nan'):.0f} tok/s); R2-ish resid sd {np.std(1-pred):.3f}")
    print(f"   fit step at 56 streams: {(a+b*56)*1e3:.0f} ms (per-stream {1/(a+b*56):.1f} tok/s); at 64: {(a+b*64)*1e3:.0f} ms ({1/(a+b*64):.1f}); at 80: {(a+b*80)*1e3:.0f} ms ({1/(a+b*80):.1f})")
    share = u * tp
    print(f"   fitted prefill time share mean {share.mean():.3f}")
