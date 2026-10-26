#!/usr/bin/env python3
"""E054: go / no-go per cherry-pick from a gpu/run_plan.sh runs directory (stdlib only).

  python3 gpu/report_e054.py /root/e054/runs

#156 (DSV41_INDEXER_SKIP), E053 §5 criteria with the exactness checks moved to saved inputs:
  - kernel/indexer_ab.json: all_ok (OLD self-deterministic; NEW == OLD bitwise on every shape, eager and graph
    replay; identical top-512 sets; kill switch == OLD);
  - decode step (decode_window itl_median_ms, pure decode, every stream active) idx vs old: >= 4% faster at
    B = 36 and B = 48;
  - 0 tracebacks in the idx server log; logprob probe idx vs old inside old's run-to-run noise.
#58678 (DSV41_ENGRAM_WKV_TP), isolated as both vs idx:
  - kernel/engram_wkv_ab.json: wkv and Engram output bit-identical for every T (else REVIEW: see max diffs);
  - startup gate old -> both: weights -0.18..-0.25 GiB/GPU, KV budget +>= 7% of Kaggle's 2.28 GiB, peak activation
    unchanged, same bytes per KV block;
  - decode tok/s (both vs idx) >= -2% at B = 36 and 48; cold prefill tok/s (decode_window's batch prefill) >= -3%;
  - probe both vs old inside the noise.
Combined (both vs old, tool replay C = 36 / 48): gen tok/s, hit rate, preemptions; reported, not gated.
"""

import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load(p):
    try:
        return json.load(open(p))
    except Exception:
        return None


def pct(new, old):
    return round(100 * (new / old - 1), 2) if new and old else None


def main():
    R = Path(sys.argv[1] if len(sys.argv) > 1 else "/root/e054/runs")
    out = {"runs": str(R)}
    ix = load(R / "kernel/indexer_ab.json")
    ew = load(R / "kernel/engram_wkv_ab.json")
    arms = {a: {k: load(R / a / f"{k}.json") for k in ("startup", "dw48", "dw36", "replay36", "replay48")}
            for a in ("old", "idx", "both")}

    def dw(a, b, key):
        d = arms[a][f"dw{b}"]
        return d.get(key) if d else None

    # ---- #156
    s156 = {"kernel_all_ok": bool(ix and ix.get("all_ok")), "kernel_failed": ix.get("failed") if ix else "missing",
            "indexer_us_per_step_mix": ix.get("per_step_mix") if ix else None}
    for b in (36, 48):
        o, n = dw("old", b, "itl_median_ms"), dw("idx", b, "itl_median_ms")
        s156[f"step_ms_old_B{b}"], s156[f"step_ms_idx_B{b}"] = o, n
        s156[f"step_faster_pct_B{b}"] = round(100 * (1 - n / o), 2) if o and n else None
    st = arms["idx"]["startup"] or {}
    s156["tracebacks"] = st.get("tracebacks")
    probe = subprocess.run([sys.executable, str(HERE / "logprob_probe.py"), "--compare", str(R / "old/probe.json"),
                            str(R / "idx/probe.json"), str(R / "both/probe.json")], capture_output=True, text=True)
    lines = [json.loads(x) for x in probe.stdout.splitlines() if x.startswith("{")]
    passes = {Path(x["arm"]).parent.name: x["pass"] for x in lines if "arm" in x}
    s156["probe_pass"] = passes.get("idx")
    s156["go"] = bool(s156["kernel_all_ok"] and all((s156.get(f"step_faster_pct_B{b}") or 0) >= 4 for b in (36, 48))
                      and s156["tracebacks"] == 0 and s156["probe_pass"])
    out["pr156"] = s156

    # ---- #58678
    s586 = {"layer_bit_identical": bool(ew and ew.get("all_bit_identical"))}
    if ew and not ew.get("all_bit_identical"):
        bad = [r for r in ew["rows"] if not r["wkv_bit_identical"] or not r["engram_out_bit_identical"]]
        s586["layer_diffs"] = [{k: r[k] for k in ("layer", "T", "wkv_frac_diff", "wkv_max_abs_diff", "wkv_max_abs",
                                                  "engram_frac_diff", "engram_max_abs_diff")} for r in bad[:20]]
    if ew:
        s586["saved_bytes_per_gpu_measured"] = ew.get("saved_bytes_per_gpu_all_layers")
    g = subprocess.run([sys.executable, str(HERE / "startup_report.py"), "--gate", str(R / "old"), str(R / "both")],
                       capture_output=True, text=True)
    try:
        s586["startup_gate"] = json.loads(g.stdout[g.stdout.index("{\n"):])
    except ValueError:
        s586["startup_gate"] = {"go": False, "error": (g.stdout + g.stderr)[-500:]}
    for b in (36, 48):
        s586[f"decode_tok_s_change_pct_B{b}"] = pct(dw("both", b, "decode_tok_s"), dw("idx", b, "decode_tok_s"))
    s586["prefill_tok_s_change_pct"] = pct(dw("both", 48, "prefill_tok_s"), dw("idx", 48, "prefill_tok_s"))
    s586["probe_pass"] = passes.get("both")
    layer_ok = s586["layer_bit_identical"]
    speed_ok = all((s586.get(f"decode_tok_s_change_pct_B{b}") or -100) >= -2 for b in (36, 48)) and \
        (s586["prefill_tok_s_change_pct"] or -100) >= -3
    s586["go"] = bool(layer_ok and s586["startup_gate"].get("go") and speed_ok and s586["probe_pass"])
    s586["verdict"] = "GO" if s586["go"] else ("REVIEW (layer not bit-identical)" if not layer_ok and
                                               s586["startup_gate"].get("go") and speed_ok else "NO-GO")
    out["pr58678"] = s586

    # ---- combined, tool replay
    comb = {}
    for c in (36, 48):
        o, n = arms["old"][f"replay{c}"], arms["both"][f"replay{c}"]
        if o and n:
            comb[f"C{c}"] = {k: (o.get(k), n.get(k)) for k in ("gen_tok_s", "hit_rate", "preemptions",
                                                               "turns_done_in_window", "window_s", "setup_s")}
            comb[f"C{c}"]["gen_tok_s_change_pct"] = pct(n.get("gen_tok_s"), o.get("gen_tok_s"))
    out["combined_replay"] = comb
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
