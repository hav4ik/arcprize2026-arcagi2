"""Autonomous read-only pod collection. An atomic generation replaces the old copied JSON pipeline."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import csv
import hashlib
import ipaddress
import json
from pathlib import Path
import subprocess
import threading
import time
import urllib.request

from model import ROOT, DASH, Catalog, histories, iso, lifecycle, local_sources, normalize, read_json, read_text, safe

FALLBACK = [{'id': '5z3srm3108wcwl', 'name': 'future_tan_falcon', 'host': '213.173.105.27', 'port': 18070},
            {'id': 'ccn5ns0b8emb3o', 'name': 'yawning_salmon_cobra', 'host': '157.157.221.29', 'port': 29668}]


def atomic(p, data):
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name+'.tmp'); tmp.write_text(json.dumps(safe(data), separators=(',', ':'), allow_nan=False))
    tmp.replace(p)


def discover():
    import yaml
    secrets = yaml.safe_load((ROOT/'SECRETS.yaml').read_text())
    query = '{ myself { pods { id name desiredStatus costPerHr machine { gpuDisplayName } runtime { ports { ip isIpPublic privatePort publicPort type } } } } }'
    req = urllib.request.Request('https://api.runpod.io/graphql', data=json.dumps({'query': query}).encode(),
                                 headers={'Authorization': 'Bearer '+secrets['runpod_api_key'], 'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=15) as response:
        data = json.load(response)
    if data.get('errors'):
        raise RuntimeError('Runpod discovery returned an API error')
    pods = []
    for p in data['data']['myself']['pods']:
        # Other GPU work in the account is outside this monitor's ARC scope.
        if 'L4' not in str((p.get('machine') or {}).get('gpuDisplayName')):
            continue
        ports = (p.get('runtime') or {}).get('ports') or []
        ssh = next((x for x in ports if x.get('isIpPublic') and x.get('privatePort') == 22), {})
        pod = {'id': p['id'], 'name': p['name'], 'desired_status': p['desiredStatus'], 'cost_per_hour': p.get('costPerHr'),
               'host': ssh.get('ip'), 'port': ssh.get('publicPort')}
        pods.append(pod)
    return pods


def fetch(pod):
    if not pod.get('host') or not pod.get('port'):
        raise RuntimeError('No public SSH endpoint available')
    ipaddress.ip_address(pod['host']); port = int(pod['port'])
    assert 0 < port < 65536
    cmd = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8', '-o', 'ServerAliveInterval=10',
           '-o', 'ServerAliveCountMax=2', '-o', 'StrictHostKeyChecking=accept-new',
           '-p', str(port), 'root@'+pod['host'], 'python3 -I -']
    r = subprocess.run(cmd, input=(Path(__file__).parent/'probe.py').read_text(), text=True,
                       capture_output=True, timeout=45)
    if r.returncode:
        # Do not expose raw stderr, command environments, request headers or credentials.
        raise RuntimeError('SSH probe failed (exit '+str(r.returncode)+')')
    return json.loads(r.stdout)


def merge_metadata(local, remote):
    out = dict(local or {}); out.update(remote)
    # Empty fields from an inactive remote run do not erase its recorded attempts and curves.
    for key in ('attempts', 'metrics', 'samples', 'challenges'):
        if key not in remote and local and key in local:
            out[key] = local[key]
    if not remote.get('workload_name') and local:
        out['workload_name'] = local.get('workload_name')
    if not remote.get('env') and local:
        out['env'] = local.get('env', {})
    if local:
        for key in ('attempts', 'metrics', 'samples'):
            if len(local.get(key) or []) > len(out.get(key) or []):
                out[key] = local[key]
        if not remote.get('end_reason'):
            out['end_reason'] = local.get('end_reason', '')
        if 'NOTEBOOK_DONE' in local.get('timeline', '') and 'NOTEBOOK_DONE' not in remote.get('timeline', ''):
            out['timeline'] = local['timeline']
    out['source'] = 'direct SSH + synchronized history'
    return out


def redact(value, secrets):
    if isinstance(value, str):
        for secret in secrets:
            value = value.replace(secret, '[redacted]')
        return value
    if isinstance(value, dict):
        return {k: redact(v, secrets) for k, v in value.items()}
    if isinstance(value, list):
        return [redact(v, secrets) for v in value]
    return value


def credential_values(value):
    out = []
    if isinstance(value, dict):
        for k, v in value.items():
            if isinstance(v, str) and len(v) >= 8 and ('key' in k.lower() or 'token' in k.lower()):
                out.append(v)
            out.extend(credential_values(v))
    elif isinstance(value, list):
        for v in value:
            out.extend(credential_values(v))
    return out


class Collector:
    def __init__(self, state, interval=60, remote=True):
        self.state = Path(state); self.interval = interval; self.remote = remote
        self.catalog = Catalog(); self.lock = threading.Lock(); self.request = threading.Event(); self.stop = threading.Event()
        self.snapshot = read_json(self.state/'snapshot.json', {'generated_at': None, 'runs': [], 'pods': [], 'notes': {}, 'errors': []})
        self.refreshing = False; self.last_error = None; self.last_cycle = None
        self.thread = None; self.count = 0

    def start(self):
        self.thread = threading.Thread(target=self.loop, name='arc-collector', daemon=True); self.thread.start()

    def loop(self):
        while not self.stop.is_set():
            self.refreshing = True
            try:
                snapshot = self.collect()
                atomic(self.state/'snapshot.json', snapshot)
                with self.lock:
                    self.snapshot = snapshot
                self.last_error = None; self.last_cycle = time.time()
            except Exception as exc:
                self.last_error = type(exc).__name__+': collection failed; previous snapshot retained'
            finally:
                self.refreshing = False
            self.request.wait(self.interval); self.request.clear()

    def get(self):
        with self.lock:
            return self.snapshot

    def health(self):
        return {'refreshing': self.refreshing, 'last_cycle_at': iso(self.last_cycle), 'error': self.last_error,
                'poll_interval_s': self.interval, 'collector_alive': bool(self.thread and self.thread.is_alive())}

    def collect(self):
        now = time.time(); errors = []; pods = []; observations = {}
        import yaml
        configured = yaml.safe_load((ROOT/'SECRETS.yaml').read_text())
        redactions = credential_values(configured)
        endpoints = ROOT/'ENDPOINTS.yaml'
        if endpoints.exists():
            redactions += credential_values(yaml.safe_load(endpoints.read_text()))
        local = {(p, m['name']): m for p, m in local_sources(self.catalog)}
        if self.remote:
            try:
                pods = discover(); atomic(self.state/'discovery.json', {'pods': pods, 'observed_at': now})
            except Exception:
                errors.append('Runpod discovery unavailable; using last known SSH endpoints')
                pods = read_json(self.state/'discovery.json').get('pods') or FALLBACK
            with ThreadPoolExecutor(max_workers=4) as pool:
                futures = {pool.submit(fetch, p): p for p in pods if p.get('desired_status', 'RUNNING') == 'RUNNING'}
                for future in as_completed(futures):
                    p = futures[future]
                    try:
                        observation = redact(future.result(), redactions)
                        previous = read_json(self.state/'pods'/(p['id']+'.json'))
                        old_runs = {r['name']: r for r in previous.get('runs', [])}
                        observation['runs'] = [merge_metadata(old_runs.get(r['name']), r) for r in observation.get('runs', [])]
                        observations[p['id']] = observation
                        atomic(self.state/'pods'/(p['id']+'.json'), observation)
                        p.update(connected=True, observed_at=observation['observed_at'], error=None)
                    except Exception as exc:
                        p.update(connected=False, error=str(exc)[:160])
                        observation = read_json(self.state/'pods'/(p['id']+'.json'))
                        if observation:
                            observations[p['id']] = observation
                            p['observed_at'] = observation.get('observed_at')
        pod_map = {p['id']: p for p in pods}
        # Keep last known pod payloads in the history even after an instance disappears from Runpod.
        for path in (self.state/'pods').glob('*.json'):
            if path.stem not in observations:
                observations[path.stem] = read_json(path)
        merged = dict(local)
        for pid, observation in observations.items():
            for remote in observation.get('runs', []):
                key = (pid, remote['name'])
                # Inactive setup rehearsals without result data clutter the experiment list.
                if key not in local and not remote.get('runner_alive') and not remote.get('solver_alive'):
                    continue
                merged[key] = merge_metadata(local.get(key), remote)
        runs = []
        for (pid, name), meta in merged.items():
            p = pod_map.get(pid, {'id': pid, 'name': pid, 'connected': False})
            runs.append(normalize(meta, p, self.catalog, now, connected=p.get('connected', False)))
        # A local terminal marker is authoritative about completion even if its old pod has since been removed.
        for r in runs:
            if r['state'] == 'unverified':
                m = merged[(r['pod_id'], r['run_name'])]
                terminal, reason = lifecycle(m, now, now, True)
                if terminal in ('completed', 'ended', 'failed', 'stopped') and (
                        m.get('end_reason') or 'NOTEBOOK_DONE' in m.get('timeline', '') or '] done' in m.get('solver_log_tail', '')):
                    r.update(state=terminal, reason=reason, gen_tok_s=None, running=0)
        historical = histories()
        # Live/local source versions replace outdated legacy copies by label or known aliases.
        replaced = {'V12 + images FINAL', 'v12_fast', 'manh_v12_rep2', 'v12_offload0'}
        runs += [r for r in historical if r['name'] not in replaced]
        priority = {'running': 0, 'recovering': 1, 'blocked': 2, 'telemetry-stale': 3, 'starting': 4, 'finishing': 5,
                    'unverified': 6, 'completed': 7, 'ended': 8, 'stopped': 9, 'failed': 10}
        runs.sort(key=lambda r: (r.get('historical', False), priority.get(r['state'], 20), r['name']))
        for p in pods:
            obs = observations.get(p['id'], {})
            p['observed_at'] = iso(p.get('observed_at'))
            p['gpus'] = obs.get('gpus', [])
            p['queue'] = obs.get('queue', {})
            p.pop('host', None); p.pop('port', None)
        # LLM-curated prose stays in its original files; this service never writes them.
        notes = {'events': read_json(DASH/'events.json', []), 'findings': read_json(DASH/'findings.json', []),
                 'comments': read_json(DASH/'comments.json', []), 'submissions': read_json(DASH/'submissions.json', [])}
        manifest = read_json(DASH/'monitor/archive.json')
        registry = []
        try:
            with open(ROOT/'experiments/experiments.tsv', newline='') as f:
                registry = [r for r in csv.DictReader(f, delimiter='\t') if r['id'] in ('E071', 'E072', 'E073')]
        except OSError:
            pass
        self.count += 1
        return safe(redact({'schema_version': 1, 'generation': self.count, 'generated_at': iso(now), 'generated_epoch': now,
                     'runs': runs, 'pods': pods, 'notes': notes, 'errors': errors, 'registry': registry,
                     'archive': '/'+manifest.get('directory', '')+'/local.html',
                     'poll_interval_s': self.interval, 'baseline': {'name': 'Manh V12', 'pass2': 80.83,
                         'tasks': 240, 'version': 'Kaggle evaluation originals + transformed copies', 'hours': 10.17}}, redactions))
