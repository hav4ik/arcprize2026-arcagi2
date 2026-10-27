import collections, json, sys
sys.path.insert(0, '/tmp/claude-1000/-home-vu-Workspace-ARC-AGI-2/81e68821-e1f4-4b05-bad6-9a606f77ad54/scratchpad/v5gap/scripts')
from common import *
from classify import *
which = sys.argv[1]
if which == 'v5_w36':
    keys, sol = kaggle_keys()
    att = ROOT/'experiments/E046_multi_test_rollout/pods/runs/v5_w36/out/dsv41_logs/attempts.jsonl'
    tdir = SP/'v5gap/v5w36_traces/traces'
else:
    keys, sol = x2_keys()
    att = SP/'v5gap/v5rep_attempts.jsonl'
    tdir = SP/'v5gap/v5rep_traces'
inv = {v: k for k, v in keys.items()}
recs = bench_records(att, keys, sol)
refs = collections.defaultdict(list)
for k, r in ref_records('e128t_w10_u2_gptq_eng64_sub80'):
    refs[k].append(bool(r.get('correct')))
out = []
for k, r, ok in recs:
    t, p, a = r['task_id'], r['pair'], r['attempt']
    path = tdir / f'{t}_p{p}_a{a}.json.gz'
    info = trace_info(path) if path.exists() else None
    cls = classify(r, ok, info, None) if info else ('correct' if ok else 'no_trace')
    rr = refs.get(k, [])
    out.append({'task': t, 'pair': p, 'ok': ok, 'cls': cls, 'ref': (sum(rr) / len(rr)) if rr else None,
                'stop': r.get('stop_reason'), 'src': r.get('answer_source'), 'train_pass': r.get('train_pass'),
                'out': r.get('usage', {}).get('completion'), 'turns': r.get('turns'),
                'rej': [o[:300] for c, o in (info['submits'] if info else []) if 'not accepted' in o],
                'rej_code': [c[:200] for c, o in (info['submits'] if info else []) if 'not accepted' in o],
                'n_sub': len(info['submits']) if info else None,
                'shadow': info['shadow'] if info else None, 'fallbacks': info['fallbacks'] if info else None,
                'other': info['other_events'] if info else None, 'tool_errors': r.get('tool_errors'),
                'rescues': r.get('rescues'), 'tx': r.get('token_exact')})
json.dump(out, open(SP/f'v5gap/classified_{which}.json', 'w'), indent=1)
print(collections.Counter(o['cls'] for o in out))
print('any rejection in attempt:', sum(1 for o in out if o['rej']), 'of', len(out))
print('shadow defs:', [(o['task'], o['pair'], o['shadow']) for o in out if o['shadow']])
for o in out:
    if not o['ok']:
        print(o['task'], o['pair'], o['cls'], 'ref', o['ref'], 'stop', o['stop'], 'src', o['src'], 'tp', o['train_pass'],
              'out', o['out'], 'turns', o['turns'], 'nsub', o['n_sub'], 'rej', len(o['rej']), o['fallbacks'][:2] if o['fallbacks'] else '', o['other'][:2] if o['other'] else '')
