"""Read-only, standard-library probe, sent over SSH stdin. Never reads secrets or changes a workload."""
import json
import math
import os
from pathlib import Path
import re
import subprocess
import time

ROOT = Path('/root')
SAFE = re.compile(r'^[A-Za-z0-9_.-]{1,120}$')
ENV_KEYS = {'ARC_OUT', 'ARC_CHALLENGES', 'ARC_T0', 'ARC_ATTEMPTS', 'ARC_ATTEMPT_FORMATS',
            'ARC_GRID_FORMAT', 'ARC_EXTRA_IF_NO_TRAIN_PASS', 'ARC_MODEL_DIR'}


def finite_json(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: finite_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [finite_json(v) for v in value]
    return value


def text(p, limit=100000):
    try:
        with open(p, 'rb') as f:
            if limit:
                f.seek(0, 2); f.seek(max(0, f.tell()-limit))
            return f.read().decode(errors='replace')
    except OSError:
        return ''


def obj(p):
    try:
        return json.loads(text(p, 0))
    except ValueError:
        return {}


def rows(p):
    out = []
    try:
        for line in open(p, errors='replace'):
            try:
                out.append(json.loads(line))
            except ValueError:
                pass  # a concurrently appended final line is allowed to be incomplete
    except OSError:
        pass
    return out


def mtime(p):
    try:
        return p.stat().st_mtime
    except OSError:
        return None


def command(pid):
    try:
        stat = Path(f'/proc/{int(pid)}/stat').read_text().rsplit(')', 1)[1].split()
        if stat[0] == 'Z':
            return ''
        return Path(f'/proc/{int(pid)}/cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace')
    except (OSError, ValueError):
        return ''


def env(pid):
    try:
        return {k.decode(): v.decode(errors='replace') for chunk in Path(f'/proc/{pid}/environ').read_bytes().split(b'\0')
                if b'=' in chunk for k, v in [chunk.split(b'=', 1)] if k.decode(errors='replace') in ENV_KEYS}
    except OSError:
        return {}


def compact_sample(s):
    caps = list((s.get('alloc_cap') or {}).values())
    mem = list((s.get('vllm_gpu_mem') or {}).values())
    minimum = lambda xs: min([x for x in xs if isinstance(x, (int, float))], default=None)
    ram = s.get('kaggle_eq_gib') if s.get('cgroup_gib') else None
    return {'t': s.get('t'), 'ram_gib': ram,
            'gpu_free_gib': minimum([m.get('free') for m in mem]),
            'cap_reserved_gib': minimum([c.get('cap_minus_reserved_now') for c in caps]),
            'cap_peak_gib': minimum([c.get('cap_minus_peak') for c in caps]),
            'ooms': sum(m.get('ooms', 0) for m in mem),
            'allocator_retries': max([m.get('retries', 0) for m in mem], default=0),
            'engine': s.get('engine_last') or {}, 'solver': s.get('solver') or {}}


def main():
    now = time.time()
    solvers = {}
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():
            continue
        cmd = command(p.name)
        if 'solver_dsv41.py' in cmd and ('python' in cmd):
            e = env(p.name); out = os.path.realpath(e.get('ARC_OUT', ''))
            if out.startswith('/root/kwork_'):
                solvers[Path(out).name.removeprefix('kwork_')] = {'pid': int(p.name), 'env': e}
    result = {'observed_at': now, 'runs': [], 'queue': {}, 'gpus': []}
    dirs = sorted((ROOT/'nbruns').glob('*'), key=lambda p: mtime(p) or 0, reverse=True)[:100]
    for name in solvers:
        if ROOT/'nbruns'/name not in dirs:
            dirs.append(ROOT/'nbruns'/name)
    for folder in dirs:
        name = folder.name
        if not folder.is_dir() or not SAFE.fullmatch(name):
            continue
        try:
            pid = int(text(folder/'run_nb.pid').strip())
        except ValueError:
            pid = 0
        runner = 'run_nb.py' in command(pid)
        solver = solvers.get(name, {})
        work = ROOT/('kwork_'+name); logs = work/'dsv41_logs'
        sup = obj(logs/'supervisor.json')
        status = obj(logs/'status.json')
        timeline = text(folder/'nb.timeline', 8000)
        own_end = text(folder/'END_REASON', 2000).strip()
        samples = rows(folder/'samp.jsonl')
        last = samples[-1] if samples else {}
        job = text(folder/'job.env', 10000)
        wl = re.search(r'^WORKLOAD=([^\s]+)', job, re.M)
        start = mtime(folder/'notebook.sha256')
        if not start and samples:
            start = samples[0].get('t')
        active = bool(runner or solver)
        run = {'name': name, 'runner_alive': runner, 'solver_alive': bool(solver),
               'server_alive': 'vllm' in command(sup.get('pid', 0)),
               'started_at': start, 'supervisor': {k: sup.get(k) for k in ('state', 'start', 'up_s', 'error')},
               'status': status, 'timeline': timeline, 'end_reason': own_end,
               'workload_name': wl[1].strip('"\'') if wl else None,
               'last_sample': compact_sample(last),
               'telemetry_at': last.get('t'), 'status_at': mtime(logs/'status.json'),
               'solver_log_tail': text(logs/'solver.log', 14000), 'env': solver.get('env', {})}
        # Metadata for every notebook prevents old "running" flags from surviving an exit.
        # Payloads for active notebooks only: no expensive transfer of old traces or checkpoints.
        recent_end = not active and (mtime(logs/'status.json') or 0) > now-900
        if active or recent_end:
            keep = {'task_id', 'pair', 'attempt', 'pred', 'multi_test', 'train_pass', 'stop_reason',
                    'error', 't_end_h', 'wall_s', 'turns', 'grid_format', 'usage'}
            run['attempts'] = [{k: v for k, v in r.items() if k in keep} for r in rows(logs/'attempts.jsonl')]
            run['metrics'] = rows(logs/'metrics.jsonl')
            run['samples'] = [compact_sample(s) for s in samples]
            challenge = solver.get('env', {}).get('ARC_CHALLENGES')
            if not challenge:
                candidate = ROOT/'queue/wl'/f'{run["workload_name"]}_challenges.json'
                challenge = str(candidate) if candidate.is_file() else None
            if challenge:
                run['challenges'] = obj(challenge)  # task grids only; never loads a solution key on the pod
            run['attempts_at'] = mtime(logs/'attempts.jsonl')
        result['runs'].append(run)
    for stage in ('pending', 'running', 'done', 'failed'):
        result['queue'][stage] = [p.name for p in sorted((ROOT/'queue'/stage).glob('*.job'))]
    try:
        q = subprocess.run(['nvidia-smi', '--query-gpu=index,utilization.gpu,memory.used,memory.total,power.draw',
                            '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=12, check=True)
        for line in q.stdout.splitlines():
            a = [float(x.strip()) for x in line.split(',')]
            result['gpus'].append(dict(zip(('index', 'utilization_pct', 'used_mib', 'total_mib', 'power_w'), a)))
    except Exception:
        result['gpu_error'] = 'GPU metrics unavailable'
    print(json.dumps(finite_json(result), separators=(',', ':'), allow_nan=False))


if __name__ == '__main__':
    main()
