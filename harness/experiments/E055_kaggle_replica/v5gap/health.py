import collections, json, sys, contextlib, io, gzip
sys.path.insert(0, '/tmp/claude-1000/-home-vu-Workspace-ARC-AGI-2/81e68821-e1f4-4b05-bad6-9a606f77ad54/scratchpad/v5gap/scripts')
from common import *
def m(x): return sum(x)/len(x) if x else float('nan')
def health(recs, label):
    tx = [r.get('token_exact') or {} for r in recs]
    turns = sum(t.get('turns', 0) for t in tx)
    print(f"{label:22s} att {len(recs):4d} | tool_err/call {sum(r.get('tool_errors') or 0 for r in recs)/max(1,sum(r.get('tool_calls') or 0 for r in recs)):.3f} | rescues {sum(r.get('rescues') or 0 for r in recs)} | tx turns {turns} breaks/turn {sum(t.get('breaks',0) for t in tx)/max(1,turns):.3f} rerenders {sum(t.get('rerenders',0) for t in tx)} resyncs {sum(t.get('resyncs',0) for t in tx)} parse_fb {sum(t.get('parse_fallbacks',0) for t in tx)} | stops {dict(collections.Counter(r.get('stop_reason') for r in recs))}")
health([json.loads(l) for l in open(ROOT/'experiments/E046_multi_test_rollout/pods/runs/v5_w36/out/dsv41_logs/attempts.jsonl')], 'v5_w36')
health([json.loads(l) for l in open(SP/'v5gap/v5rep_attempts.jsonl')], 'v5rep')
for n in ['e128t_w10_u2_gptq_eng64_sub80', 'e128t_pp_sub80_tx', 'e128t_final_all']:
    health([r for k, r in ref_records(n)], n[:22])
for d in ['experiments/E050_effort_ab/runs/eff50/dsv41_logs', 'experiments/E046_multi_test_rollout/pods/runs/mt36_b05_gate60/out/dsv41_logs']:
    health([json.loads(l) for l in open(ROOT/d/'attempts.jsonl') if 'task_id' in l], d.split('/')[3])
# tool timeouts in traces
WC=[0]
def timeouts(tdir):
    n = tt = 0
    for p in Path(tdir).glob('*.json.gz'):
        try:
            t = json.load(gzip.open(p))
        except Exception:
            continue
        for mm in t['messages']:
            if mm['role'] == 'tool':
                n += 1
                c = mm.get('content') or ''
                if 'timed out' in c.lower() or '[timeout]' in c.lower() or 'timeout' in c[:80].lower():
                    tt += 1
                if 'wall-clock' in c.lower() or 'wall clock' in c.lower():
                    WC[0] += 1
    return n, tt
print('v5_w36 tool msgs / timeouts', timeouts(SP/'v5gap/v5w36_traces/traces'))
print('v5rep tool msgs / timeouts', timeouts(SP/'v5gap/v5rep_traces'))
print('wall-clock notes', WC)
