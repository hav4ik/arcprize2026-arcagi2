#!/usr/bin/env python3
"""Capture a local reporting snapshot. Never starts/stops/syncs GPU jobs.

Read already-synced logs, freeze exact scoring inputs, and update analysis tables.
README prose, ETA interpretation and report-timer completion remain caller work.
"""
import hashlib
import importlib.util
import json
import math
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
ANALYSIS = EXP / 'analysis'
TZ = ZoneInfo('America/Los_Angeles')
ARMS = {'E128': 'baseline_txt', '14.02B': 'preview14_txt', '14.58B': 'baseline1458_txt'}
ORDER = ('E128', '14.58B', '14.02B')
LABELS = {'E128': 'E128 (~16B)', '14.58B': 'E074 (~14.58B)', '14.02B': 'E076 (~14.02B)'}
spec = importlib.util.spec_from_file_location(
    'e076_score', ROOT / 'experiments/E066_dsv41_image_pruning/scripts/score_b32_engram.py')
SC = importlib.util.module_from_spec(spec)
spec.loader.exec_module(SC)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def metrics(row):
    values = {}
    for line in row['metrics']:
        name = line.split('{')[0].split()[0]
        try:
            value = float(line.rsplit(' ', 1)[1])
        except ValueError:
            continue
        values[name] = values.get(name, 0) + value
    return values


def read_run(arm, stamp):
    folder = EXP / 'results' / arm
    path = folder / 'arc.jsonl'
    if not path.exists():
        return None
    raw = path.read_bytes()
    frozen = ANALYSIS / f'scheduled_input_{arm}_{stamp}.jsonl'
    frozen.write_bytes(raw)
    rows = [json.loads(s) for s in raw.splitlines()]
    latest = {(r['task_id'], r['attempt'], r['pair']): r for r in rows}
    scores = SC.load_run(frozen)
    both = [t for t in SC.TASKS if all(SC.done(scores, t, a) for a in (0, 1))]
    run = dict(folder=folder, latest=latest, scores=scores, both=both,
               source=str(frozen.relative_to(ROOT)), sha256=sha(raw),
               completed=sum(SC.done(scores, t, a) for t in SC.TASKS for a in (0, 1)))
    cache_raw = (folder / 'cache_metrics.jsonl').read_bytes()
    cache = [json.loads(s) for s in cache_raw.splitlines()]
    last = cache[-1]
    prior = next((r for r in reversed(cache) if r['time'] <= last['time'] - 300), cache[0])
    m, p = metrics(last), metrics(prior)
    q, h = 'vllm:prefix_cache_queries_total', 'vllm:prefix_cache_hits_total'
    log_raw = (folder / 'server.log').read_bytes()
    log = log_raw.decode()
    errors = [line for line in log.splitlines()
              if re.search(r'\bERROR\b|Traceback \(most recent call last\)', line)]
    run['health'] = {
        'snapshot_local': datetime.fromtimestamp(last['time'], TZ).isoformat(),
        'prefix_hit_cumulative_pct': 100 * m[h] / m[q] if m[q] else None,
        'prefix_hit_recent_5min_pct': 100 * (m[h] - p[h]) / (m[q] - p[q]) if m[q] > p[q] else None,
        'kv_usage_pct': 100 * m['vllm:kv_cache_usage_perc'],
        'preemptions': m['vllm:num_preemptions_total'],
        'running': m['vllm:num_requests_running'], 'waiting': m['vllm:num_requests_waiting'],
        'elapsed_minutes': (last['time'] - cache[0]['time']) / 60,
        'server_error_log_lines': len(errors), 'cache_source_sha256': sha(cache_raw),
    }
    year = datetime.fromtimestamp(cache[0]['time'], timezone.utc).year
    samples = []
    for line in log.splitlines():
        match = re.search(r'INFO (\d\d-\d\d \d\d:\d\d:\d\d).*Avg generation throughput: ([\d.]+) tokens/s', line)
        if match:
            t = datetime.strptime(f'{year}-' + match[1], '%Y-%m-%d %H:%M:%S').replace(tzinfo=timezone.utc)
            samples.append((t.timestamp(), float(match[2])))
    run['engine_samples'], run['log_sha256'] = samples, sha(log_raw)
    return run


def stats(run, tasks):
    latest = run['latest']
    logical = [latest[t, a, 0] for t in tasks for a in (0, 1)]
    tokens = [r['usage']['completion'] for r in logical]
    calls = sum(r['tool_calls'] for r in logical)
    errors = sum(r['tool_errors'] for r in logical)
    stops = Counter(r['stop_reason'] for r in logical)
    p1, p2 = SC.score(run['scores'], tasks) if tasks else (None, None)
    result = {
        'tasks': len(tasks), 'attempts': len(logical), 'task_ids': tasks,
        'completed_attempts': run['completed'], 'partial': len(tasks) < 120,
        'pass_at_1': p1, 'pass_at_2': p2,
        'output_tokens_mean': float(np.mean(tokens)) if tokens else None,
        'output_tokens_p50': float(np.quantile(tokens, .5)) if tokens else None,
        'output_tokens_p90': float(np.quantile(tokens, .9)) if tokens else None,
        'tool_calls': calls, 'tool_calls_mean': calls / len(logical) if logical else None,
        'tool_errors': errors, 'tool_error_pct': 100 * errors / calls if calls else None,
        'cap_stops': sum(stops[k] for k in ('output_budget', 'max_turns', 'context')),
        'attempts_any_missing_answer': sum(
            any(not latest[t, a, p].get('pred') for p in range(len(SC.SOL[t])))
            for t in tasks for a in (0, 1)),
        'server_harness_errors': len({(t, a) for (t, a, p), r in latest.items() if 'error' in r}),
        'stop_reasons': dict(stops), 'source_snapshot': run['source'], 'source_sha256': run['sha256'],
    }
    assert sum(stops.values()) == 2 * len(tasks)
    return result


def rate(samples, start_min, end_min):
    origin = samples[0][0] - 10
    lo, hi = origin + start_min * 60, origin + end_min * 60
    previous, total, duration = origin, 0, 0
    for t, value in samples:
        width = max(0, min(t, hi) - max(previous, lo))
        total += width * value
        duration += width
        previous = t
    assert abs(duration - (hi - lo)) < .1
    return total / duration


def tables(snapshot, filename):
    now = datetime.fromisoformat(snapshot['snapshot_local'])
    header = ('| Model | Pass@1 | Pass@2 | Tokens mean | Tokens p50 | Tokens p90 | Tool calls mean (total) | Tool errors |\n'
              '|---|---:|---:|---:|---:|---:|---:|---:|')

    def row(model, r):
        if r.get('state') == 'queued' or not r.get('tasks'):
            status = r.get('state', 'no matched tasks yet')
            return f'| {LABELS[model]} — {status} | — | — | — | — | — | — | — |'
        return (f"| {LABELS[model]} | {r['pass_at_1']:.2f}% | {r['pass_at_2']:.2f}% | "
                f"{r['output_tokens_mean']:,.0f} | {r['output_tokens_p50']:,.0f} | {r['output_tokens_p90']:,.0f} | "
                f"{r['tool_calls_mean']:.2f} ({r['tool_calls']:,}) | {r['tool_error_pct']:.2f}% |")

    text = [f"Snapshot: {now.strftime('%B %d, %Y, %-I:%M %p %Z')}. Kaggle ARC-AGI-2 evaluation.",
            'Pass@1 is task-weighted and averaged across both draws. Token and tool-call means are per logical task attempt; multi-test pair records are counted once.',
            '**Image + tools — finished: 120 tasks × 2 draws per model**',
            header + '\n' + '\n'.join(row(m, snapshot['image'][m]) for m in ORDER)]
    matched = snapshot['text_matched']
    progress = '; '.join(f"{LABELS[m]}: {r['completed_attempts']}/240" if 'completed_attempts' in r
                         else f"{LABELS[m]}: {r.get('state', 'unavailable')}" for m, r in snapshot['text'].items())
    text_title = ('**Text only + tools — finished: 120 tasks × 2 draws per model**'
                  if matched['tasks'] == 120 else '**Text only + tools — matched ongoing comparison**')
    text += [text_title,
             progress + '.',
             f"All rows use the same {matched['tasks']} tasks with both draws complete ({matched['attempts_per_model']} attempts/model). " + matched['caveat'],
             header + '\n' + '\n'.join(row(m, matched['models'][m]) for m in ORDER)]
    for model, r in matched['models'].items():
        if 'tasks' in r:
            text.append(f"{LABELS[model]} matched-subset cap stops: {r['cap_stops']}; attempts missing any answer: {r['attempts_any_missing_answer']}.")
    text.append('Tool errors are failed/rejected calls after auto-repair, divided by all tool calls; infrastructure errors are separate. Token quantiles use linear interpolation.')
    complete = {m: r for m, r in snapshot['text'].items() if r.get('tasks') == 120}
    if complete and matched['tasks'] < 120:
        text += ['**Completed full-set text references — not directly comparable to a partial candidate subset**',
                 header + '\n' + '\n'.join(row(m, r) for m, r in complete.items())]
    tp = snapshot.get('text_throughput', {})
    if tp.get('generation_tok_s'):
        text.append(f"Text generation throughput, matching elapsed minutes {tp['elapsed_window_minutes']}: " +
                    '; '.join(f'{m}: {v:,.2f} tok/s' for m, v in tp['generation_tok_s'].items()) + '. ' + tp['caveat'])
    text.append('Image equal-window minutes5–150: 14.02B 1,083.09 vs E128 1,003.18 / 14.58B 1,052.72 tok/s (+7.97%/+2.89%).')
    text.append(f'[Exact snapshot, task IDs and provenance]({filename}).')
    return '\n\n'.join(text) + '\n'


def main():
    now = datetime.now(TZ)
    stamp = now.strftime('%Y%m%d_%H%M')
    snapshot = json.loads((EXP / 'monitor/latest_scheduled_update.json').read_text())
    snapshot['snapshot_local'] = now.isoformat()
    policy_path = EXP / 'configs/eval_queue_policy.json'
    policy = json.loads(policy_path.read_text()) if policy_path.exists() else {}
    skipped = set(policy.get('skipped_eval_arms', []))
    def unavailable(model):
        return {'state': 'skipped by user' if ARMS[model] in skipped else 'queued'}
    runs = {model: run for model, arm in ARMS.items() if (run := read_run(arm, stamp)) is not None}
    snapshot['text'] = {}
    for model in ORDER:
        if model not in runs:
            snapshot['text'][model] = unavailable(model)
            continue
        run = runs[model]
        r = stats(run, run['both'])
        complete = run['folder'] / 'complete.json'
        if complete.exists():
            r['full_eval_seconds'] = json.loads(complete.read_text())['eval_elapsed_s']
            r['full_eval_gen_tok_s'] = sum(x['usage']['completion'] for (t, a, p), x in run['latest'].items() if p == 0) / r['full_eval_seconds']
        snapshot['text'][model] = r
        snapshot['health']['text_' + model] = run['health']
    matched = sorted(set.intersection(*(set(r['both']) for r in runs.values())))
    snapshot['text_matched'] = {
        'tasks': len(matched), 'attempts_per_model': 2 * len(matched), 'task_ids': matched,
        'models': {m: stats(runs[m], matched) if m in runs else unavailable(m) for m in ORDER},
        'caveat': ('Completed-subset bias from completion order: faster/easier attempts finish first. '
                   'Accuracy and usage use the same tasks and two draws per model. Skipped or queued references have no comparison.'
                   if len(matched) < 120 else 'All120 tasks complete in each available arm; skipped or queued references have no comparison.'),
    }
    available = {m: r for m, r in runs.items() if r['engine_samples']}
    end = 5 * math.floor(min((r['engine_samples'][-1][0] - r['engine_samples'][0][0] + 10) / 60
                            for r in available.values()) / 5)
    rates = {m: rate(r['engine_samples'], 5, end) for m, r in available.items()} if end > 5 else {}
    snapshot['text_throughput'] = {
        'elapsed_window_minutes': [5, end], 'generation_tok_s': rates,
        'gain_14.02B_vs_E128_pct': 100 * (rates['14.02B'] / rates['E128'] - 1) if all(m in rates for m in ('14.02B', 'E128')) else None,
        'gain_14.02B_vs_14.58B_pct': 100 * (rates['14.02B'] / rates['14.58B'] - 1) if all(m in rates for m in ('14.02B', '14.58B')) else None,
        'caveat': 'Matching elapsed windows include unfinished-attempt tokens and evolving batches; not isolated-kernel speed. Skipped or queued references have no throughput comparison.',
        'origin': 'First engine timestamp minus10seconds; duration-weighted overlap of consecutive engine intervals.',
        'server_log_sha256': {m: r['log_sha256'] for m, r in available.items()},
    }
    snapshot['batch_state'] = json.loads((EXP / 'results/batch_status.json').read_text())
    snapshot['all_four_new_arms_complete'] = all((EXP / 'results' / arm / 'complete.json').exists()
                                                for arm in ('preview14_img', *ARMS.values()))
    required = policy.get('required_new_eval_arms', ['preview14_img', *ARMS.values()])
    snapshot['required_new_eval_arms'] = required
    snapshot['skipped_eval_arms'] = sorted(skipped)
    snapshot['all_required_eval_arms_complete'] = all((EXP / 'results' / arm / 'complete.json').exists() for arm in required)
    # Preserve the prior human-reviewed ETA ranges, explicitly marking them for review.
    snapshot['eta_local']['basis'] = f"Prior ranges retained for caller review at {now.strftime('%H:%M %Z')}; candidate text began about16:17PDT, E128 text took158.1minutes. Completion-order bias makes linear ETA unreliable."
    filename = f'scheduled_update_{stamp}.json'
    for path in (ANALYSIS / filename, ANALYSIS / 'summary_tables.json', EXP / 'monitor/latest_scheduled_update.json'):
        path.write_text(json.dumps(snapshot, indent=2) + '\n')
    (ANALYSIS / 'summary_tables.md').write_text(tables(snapshot, filename))
    brief = {k: snapshot[k] for k in ('snapshot_local', 'all_required_eval_arms_complete', 'skipped_eval_arms', 'text_throughput', 'eta_local')}
    brief['text_matched'] = {m: {k: v for k, v in r.items() if k not in ('task_ids', 'source_snapshot', 'source_sha256')}
                             for m, r in snapshot['text_matched']['models'].items()}
    brief['candidate_health'] = snapshot['health'].get('text_14.02B')
    brief['saved'] = str(ANALYSIS / filename)
    print(json.dumps(brief, indent=2))


if __name__ == '__main__':
    main()
