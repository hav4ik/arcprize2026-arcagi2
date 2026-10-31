"""One scoring, lifecycle, clock and freshness model for all dashboard sources."""
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[4]
DASH = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT/'experiments/E072_harness_loop/bin'))
import arcscore as A

LABELS = {'q_e073_gold': 'E073 · V12 control', 'q_e073_dev': 'E073 · causal policy screen',
          'q_e073_holdout': 'E073 · selected holdout', 'q_b1_noise_v12_pod2': 'E072 · V12 repeat',
          'q_b2a_autorepair_off_pod2': 'E072 · auto-repair off', 'm12_rerun': 'Manh V12 · gold replica',
          'v12img3_rerun': 'V12 + images · final', 'v12fast2_rerun': 'V12-fast · stopped'}
PODS = {'pod2': '5z3srm3108wcwl', 'pod3': 'ccn5ns0b8emb3o', 'pod5': 'u71eibokv8t34t'}


def read_json(p, default=None):
    try:
        return json.loads(Path(p).read_text())
    except (OSError, ValueError):
        return {} if default is None else default


def read_text(p):
    try:
        return Path(p).read_text(errors='replace')
    except OSError:
        return ''


def iso(t):
    return datetime.fromtimestamp(t, timezone.utc).isoformat().replace('+00:00', 'Z') if t else None


def number(x):
    return float(x) if isinstance(x, (int, float)) and math.isfinite(x) else None


def safe(obj):
    if isinstance(obj, dict):
        return {k: safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [safe(x) for x in obj]
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def task_hash(task):
    return hashlib.sha256(json.dumps(task, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


class Catalog:
    def __init__(self):
        self.tasks = {}; self.solutions = {}; self.workloads = {}
        paths = [ROOT/'data/arc-agi_evaluation_challenges.json',
                 ROOT/'experiments/E055_kaggle_replica/toolkit/workloads/evalx2t_s2026_challenges.json']
        for p in paths:
            ch = read_json(p); sol = read_json(str(p).replace('_challenges.json', '_solutions.json'))
            for t in ch:
                self.tasks[t] = task_hash(ch[t]); self.solutions[t] = sol.get(t)
        for base in [ROOT/'experiments/E072_harness_loop/workloads', ROOT/'experiments/E073_causal_harness/workloads',
                     ROOT/'experiments/E055_kaggle_replica/toolkit/workloads']:
            for p in base.glob('*_challenges.json'):
                self.workloads[p.name.removesuffix('_challenges.json')] = read_json(p)
        self.gold = A.load_jsonl(ROOT/'experiments/E072_harness_loop/data_in/v12_m12_attempts.jsonl')
        self.gold_per = A.task_results(self.gold, {t: s for t, s in self.solutions.items() if s})

    def identify(self, challenges):
        if not challenges or not all(t in self.tasks and self.tasks[t] == task_hash(x) for t, x in challenges.items()):
            return None
        tasks = sorted(challenges)
        return {'version': 'Kaggle evaluation', 'tasks': len(tasks), 'pairs': sum(len(x['test']) for x in challenges.values()),
                'task_ids': tasks, 'cohort': hashlib.sha256('\n'.join(tasks).encode()).hexdigest()[:12],
                'solutions': {t: self.solutions[t] for t in tasks},
                'baseline_pass2': round(100*sum(sum(p['p2'] for p in self.gold_per[t])/len(self.gold_per[t])
                                               for t in tasks)/len(tasks), 2)}


def lifecycle(meta, now, observed_at, connected):
    """Process evidence beats old status strings. Never infer a CUDA OOM from a generic termination."""
    timeline = meta.get('timeline', ''); reason = meta.get('end_reason', '')
    status = meta.get('status') or {}; sup = meta.get('supervisor') or {}
    if not connected or not observed_at or now-observed_at > 180:
        if re.search(r'NOTEBOOK_DONE.*(?:error|failed)', timeline, re.I):
            return 'failed', 'Recorded terminal notebook error; pod not currently observed'
        if 'NOTEBOOK_DONE' in timeline and re.search(r'NOTEBOOK_DONE.*(?:ok|success)', timeline, re.I):
            return 'completed', 'Recorded notebook completion; pod not currently observed'
        if reason:
            return 'ended', reason
        return 'unverified', 'Pod unavailable or observation stale; last values retained'
    if meta.get('runner_alive') or meta.get('solver_alive'):
        if sup.get('state') in ('restarting', 'starting'):
            return ('recovering' if sup.get('start') else 'starting'), 'Server '+sup['state']
        if sup.get('state') in ('stopped', 'failed'):
            return 'blocked', 'Notebook alive; server '+sup['state']
        if status.get('phase') == 'final':
            return 'finishing', 'Solver finalized; notebook still alive'
        if meta.get('solver_alive') and meta.get('server_alive'):
            at = meta.get('telemetry_at')
            if at and now-at > 300:
                return 'telemetry-stale', 'Processes alive; no recent telemetry'
            return 'running', 'Solver and server observed alive'
        return 'starting', 'Notebook alive; solver not started'
    if re.search(r'NOTEBOOK_DONE.*(?:error|failed)', timeline, re.I):
        return 'failed', 'Notebook reports an error'
    if 'NOTEBOOK_DONE' in timeline and re.search(r'NOTEBOOK_DONE.*(?:ok|success)', timeline, re.I):
        return 'completed', 'Notebook finished'
    if reason:
        if 'max Kaggle clock' in reason or 'bound' in reason:
            return 'stopped', reason
        return 'ended', reason
    if re.search(r'\] done\s*$', meta.get('solver_log_tail', ''), re.M):
        return 'completed', 'Solver log reports done; processes exited'
    return 'stopped', 'Notebook and solver are no longer alive; exit reason not recorded'


def score(records, profile, expected=2):
    if not profile:
        return {}, []
    sol = profile['solutions']; final = {}
    for r in records:
        if r.get('task_id') in sol and not r.get('error'):
            final[(r['task_id'], r.get('pair', 0), r['attempt'])] = r
    by_task = {}
    for (t, p, a), r in final.items():
        by_task.setdefault(t, set()).add(a)
    result = A.task_results(records, sol); summary = A.summarize(result)
    summary.update(first_finished=sum(0 in v for v in by_task.values()),
                   both_finished=sum({0, 1} <= v for v in by_task.values()),
                   all_finished=sum(set(range(expected)) <= v for v in by_task.values()),
                   completed_attempts=len(final), expected_attempts=profile['tasks']*expected,
                   tasks_answered=sum(any(p['att'] for p in ps) for ps in result.values()))
    # Rebuild ranked top-2 at each point, including conditional third draws. No best-of-three oracle.
    points = []; buckets = sorted({int(float(r.get('t_end_h') or 0)*6) for r in records})
    for b in buckets:
        end = (b+1)/6
        subset = [r for r in records if float(r.get('t_end_h') or 0) < end]
        s = A.summarize(A.task_results(subset, sol))
        points.append({'h': max(float(r.get('t_end_h') or 0) for r in subset), 'score': s['score_pass2']})
    return summary, points


def sample_from_raw(s):
    minimum = lambda xs: min([v for v in xs if number(v) is not None], default=None)
    if 'ram_gib' in s:
        return s
    caps = list((s.get('alloc_cap') or {}).values()); mem = list((s.get('vllm_gpu_mem') or {}).values())
    return {'t': s.get('t'), 'ram_gib': s.get('kaggle_eq_gib') if s.get('cgroup_gib') else None,
            'gpu_free_gib': minimum([x.get('free') for x in mem]),
            'cap_reserved_gib': minimum([x.get('cap_minus_reserved_now') for x in caps]),
            'cap_peak_gib': minimum([x.get('cap_minus_peak') for x in caps]),
            'ooms': sum(x.get('ooms', 0) for x in mem),
            'allocator_retries': max([x.get('retries', 0) for x in mem], default=0),
            'engine': s.get('engine_last') or {}, 'solver': s.get('solver') or {}}


def normalize(meta, pod, catalog, now, connected=True):
    name = meta['name']; t0 = meta.get('started_at')
    samples = meta.get('samples') or []; last = meta.get('last_sample') or (samples[-1] if samples else {})
    if not t0 and samples:
        t0 = samples[0].get('t')
    profile = catalog.identify(meta.get('challenges') or catalog.workloads.get(meta.get('workload_name'), {}))
    state, reason = lifecycle(meta, now, pod.get('observed_at'), connected)
    expected = int(meta.get('env', {}).get('ARC_ATTEMPTS') or (3 if name == 'q_e073_dev' else 2))
    records = meta.get('attempts') or []
    summary, acc = score(records, profile, expected)
    extra_retry = meta.get('env', {}).get('ARC_EXTRA_IF_NO_TRAIN_PASS') == '1'
    if extra_retry and profile:
        status_counts = meta.get('status') or {}
        planned = sum(status_counts.get(k, 0) or 0 for k in ('done', 'running', 'queued'))
        extra_seen = len({(r['task_id'], r.get('pair', 0), r['attempt']) for r in records if r.get('attempt', 0) >= expected})
        summary['expected_attempts'] = max(profile['tasks']*expected+extra_seen, planned)
    if not records and state not in ('starting', 'running'):
        summary, acc = {}, []
    holdout_locked = name == 'q_e073_holdout' and state not in ('completed', 'ended')
    if holdout_locked:
        for key in ('score_pass2', 'score_pass1', 'attempt0_acc', 'attempt1_acc'):
            summary.pop(key, None)
        acc = []
    metrics = meta.get('metrics') or []; metric = metrics[-1] if metrics else (meta.get('status') or {}).get('server') or {}
    status = meta.get('status') or {}
    memory = [{'h': round((s['t']-t0)/3600, 4), **{k: s.get(k) for k in
               ('ram_gib', 'cap_reserved_gib', 'cap_peak_gib', 'gpu_free_gib')}} for s in samples if s.get('t') and t0]
    # Terminal runs have no "current" throughput. Historical observations remain in the chart.
    active = state in ('running', 'recovering', 'starting', 'finishing', 'telemetry-stale', 'blocked')
    current = metric if active else {}
    elapsed = status.get('elapsed_h')
    if active and t0:
        elapsed = (now-t0)/3600
    elif elapsed is None and records:
        elapsed = max(r.get('t_end_h') or 0 for r in records)
    row = {'id': pod['id']+'/'+name, 'name': LABELS.get(name, name.removeprefix('q_').replace('_', ' ')),
           'run_name': name, 'pod_id': pod['id'], 'pod_name': pod.get('name', pod['id']),
           'state': state, 'reason': reason, 'source': meta.get('source', 'direct SSH'), 'historical': False,
           'observed_at': iso(pod.get('observed_at')), 'telemetry_at': iso(meta.get('telemetry_at')),
           'telemetry_age_s': max(0, now-meta['telemetry_at']) if meta.get('telemetry_at') else None,
           'started_at': iso(t0), 'elapsed_h': elapsed, 'clock': 'hours since notebook start',
           'version': profile['version'] if profile else 'unverified workload',
           'tasks': profile['tasks'] if profile else None, 'pairs': profile['pairs'] if profile else status.get('pairs'),
           'cohort': profile['cohort'] if profile else None, 'workload_name': meta.get('workload_name'),
           'baseline_pass2': profile['baseline_pass2'] if profile else None, 'holdout_locked': holdout_locked,
           'summary': summary, 'expected_draws': expected, 'extra_retry_enabled': extra_retry,
           'scoring': 'two distinct ranked answers; unanswered = 0',
           'gen_tok_s': current.get('gen_tok_s'), 'running': status.get('running') if active else 0,
           'queued': status.get('queued') if active else 0, 'kv': current.get('kv'),
           'prefix_hit': current.get('prefix_hit'), 'host_tok_s': current.get('host_tok_s'),
           'preemptions': metric.get('preemptions'), 'ram_gib': last.get('ram_gib'),
           'cap_reserved_gib': last.get('cap_reserved_gib'), 'cap_peak_gib': last.get('cap_peak_gib'),
           'gpu_free_gib': last.get('gpu_free_gib'), 'ooms': last.get('ooms'),
           'allocator_retries': last.get('allocator_retries'), 'restarts': status.get('restarts'),
           'errors': status.get('errors'), 'log_tail': meta.get('solver_log_tail', '')[-6000:],
           'series': {'score': acc, 'memory': memory, 'throughput': [
               {'h': m.get('elapsed_h'), 'tokens': m.get('gen_tok_s'), 'running': m.get('srv_running'),
                'kv_pct': 100*m['kv'] if number(m.get('kv')) is not None else None} for m in metrics]}}
    if not profile:
        row['score_note'] = 'Score withheld: challenge grids do not match a known Kaggle-version workload'
    return safe(row)


def local_sources(catalog):
    """Discover queue and controller outputs without treating file writes as proof of a live process."""
    result = []
    sources = []
    e73 = ROOT/'experiments/E073_causal_harness/runs'
    for d in e73.iterdir() if e73.exists() else []:
        if d.is_dir() and (d/'nb.timeline').exists():
            sources.append((d, 'ccn5ns0b8emb3o', 'q_e073_'+d.name, 'e073_all' if d.name == 'gold' else 'e073_'+d.name))
    e72 = ROOT/'experiments/E072_harness_loop/runs'
    for d in e72.glob('*'):
        if not d.is_dir() or not (d/'job.env').exists():
            continue
        job = read_text(d/'job.env'); sc = read_json(d/'score.json')
        name = sc.get('run') or 'q_'+d.name
        pod = PODS['pod5'] if 'pod5' in d.name else PODS['pod2'] if 'pod2' in d.name else PODS['pod3']
        m = re.search(r'^WORKLOAD=([^\s]+)', job, re.M)
        sources.append((d, pod, name, m[1] if m else sc.get('workload')))
    for d, pod, name, workload in sources:
        logs = d/'dsv41_logs' if (d/'dsv41_logs').exists() else d
        samples = [sample_from_raw(x) for x in A.load_jsonl(d/'samp.jsonl')]
        status = read_json(logs/'status.json'); timeline = read_text(d/'nb.timeline')
        meta = {'name': name, 'source': 'local synchronized files', 'workload_name': workload,
                'status': status, 'timeline': timeline, 'end_reason': read_text(d/'END_REASON').strip(),
                'samples': samples, 'last_sample': samples[-1] if samples else {},
                'telemetry_at': samples[-1].get('t') if samples else None,
                'started_at': samples[0].get('t') if samples else None,
                'attempts': A.load_jsonl(logs/'attempts.jsonl'), 'metrics': A.load_jsonl(logs/'metrics.jsonl'),
                'solver_log_tail': read_text(logs/'solver.log')[-14000:]}
        result.append((pod, meta))
    phase = ROOT/'experiments/E062_manh_vllm_fork/per_layer/phase_h'
    aliases = {'v12_img': ('pod2', 'v12img3_rerun'), 'v12_fast': ('pod3', 'v12fast2_rerun'),
               'manh_v12_rep2': ('pod5', 'q_rep2_v12_pod5'),
               'v12_offload0': ('pod5', 'q_v12_offload0_pod5'), 'v12_img_ab_r1': ('pod2', 'v12img2_rerun')}
    seen = {(p, m['name']) for p, m in result}
    for folder, (podname, name) in aliases.items():
        pod = PODS[podname]; d = phase/folder/'data'
        if (pod, name) in seen or not d.exists():
            continue
        samp = sorted(d.glob('nbruns/*/samp.jsonl'), key=lambda p: p.stat().st_mtime)
        samples = [sample_from_raw(x) for x in A.load_jsonl(samp[-1])] if samp else []
        result.append((pod, {'name': name, 'source': 'local synchronized files', 'workload_name': 'evalx2t_s2026',
                    'samples': samples, 'last_sample': samples[-1] if samples else {},
                    'telemetry_at': samples[-1].get('t') if samples else None,
                    'started_at': samples[0].get('t') if samples else None,
                    'status': read_json(d/'dsv41_logs/status.json'),
                    'attempts': A.load_jsonl(d/'dsv41_logs/attempts.jsonl'),
                    'metrics': A.load_jsonl(d/'dsv41_logs/metrics.jsonl'),
                    'solver_log_tail': read_text(d/'dsv41_logs/solver.log')[-14000:]}))
    return result


def histories():
    """Preserve legacy curves and commentary as historical observations, never a fabricated live heartbeat."""
    archived = read_json(DASH/'monitor/archive.json').get('directory')
    base = DASH/archived if archived else DASH
    rows = read_json(base/'runs.json', []); acc = read_json(base/'acc.json', [])
    tps = read_json(base/'tps.json', []); mem = read_json(base/'mem.json', [])
    certified = {r['run'] for p in DASH.glob('frozen_*.json') for r in read_json(p).get('runs_row', [])}
    out = []
    for r in rows:
        name = r['run']; recorded = str(r.get('status', ''))
        state = 'completed' if 'finished' in recorded or 'gold standard' in recorded else 'stopped' if 'stopped' in recorded else 'unverified'
        if 'finished early' in recorded and (r.get('ooms') or 0) > 0:
            state = 'failed'
        aps = [x for x in acc if x['run'] == name]; score_value = aps[-1].get('score_so_far') if aps else None
        out.append({'id': 'history/'+hashlib.sha256(name.encode()).hexdigest()[:12], 'name': name,
                    'pod_id': 'historical', 'pod_name': r.get('pod'), 'state': state, 'reason': recorded,
                    'historical': True, 'source': 'preserved legacy snapshot', 'clock': 'legacy recorded elapsed hours',
                    'started_at': r.get('started_utc'), 'observed_at': None, 'telemetry_at': None, 'elapsed_h': r.get('elapsed_h'),
                    'version': 'Kaggle evaluation (recorded)', 'tasks': 240 if name in certified else None,
                    'pairs': 344 if name in certified else None, 'cohort': None,
                    'summary': {'score_pass2': score_value, 'first_finished': r.get('first_done')},
                    'gen_tok_s': None, 'ram_gib': r.get('ram_kaggle_eq_gib'), 'cap_reserved_gib': r.get('cap_reserved_gib'),
                    'cap_peak_gib': r.get('cap_spare_gib'), 'ooms': r.get('ooms'), 'restarts': r.get('restarts'),
                    'errors': r.get('errored'), 'scoring': 'legacy recorded score; preserved',
                    'series': {'score': [{'h': x['t_h'], 'score': x.get('score_so_far')} for x in aps],
                               'throughput': [{'h': x['t_h'], 'tokens': x.get('gen_tok_s'), 'running': x.get('running'),
                                               'kv_pct': x.get('kv_pct')} for x in tps if x['run'] == name],
                               'memory': [{'h': x['t_h'], 'ram_gib': x.get('ram_kaggle_eq_gib'),
                                           'cap_reserved_gib': x.get('cap_minus_reserved_gib'),
                                           'cap_peak_gib': x.get('cap_minus_peak_gib')} for x in mem if x['run'] == name]}})
    return out
