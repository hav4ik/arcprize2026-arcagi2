#!/usr/bin/env python3
"""Finite controller for yawning_salmon_cobra only. No pod creation, stopping, or Kaggle operations."""
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time
import csv
import fcntl

import analyze

H = Path(__file__).resolve().parent
ROOT = H.parents[1]
HOST = 'root@157.157.221.29'
SSH = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', '-p', '29668', HOST]
START = time.time()
LIMIT = 14 * 3600


def log(s):
    print(time.strftime('%FT%TZ', time.gmtime()), s, flush=True)


def registry(status, result):
    p = ROOT/'experiments/experiments.tsv'
    with open(p, newline='') as f:
        reader = csv.DictReader(f, delimiter='\t'); fields = reader.fieldnames; rows = list(reader)
    for row in rows:
        if row['id'] == 'E073':
            row.update(status=status, result=result, updated=time.strftime('%F', time.gmtime()))
    tmp = p.with_name('experiments.e073.tmp')
    with open(tmp, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter='\t', lineterminator='\n'); writer.writeheader(); writer.writerows(rows)
    tmp.replace(p)


def ssh(cmd, **kw):
    return subprocess.run(SSH + [cmd], check=True, timeout=kw.pop('timeout', 90), **kw)


def sync(label):
    dst = H/'runs'/label; dst.mkdir(parents=True, exist_ok=True)
    # Small live outputs only. Executed notebooks and trace tarballs are copied when a job finishes.
    for folder in [f'/root/nbruns/q_e073_{label}/', f'/root/kwork_q_e073_{label}/dsv41_logs/']:
        subprocess.run(['rsync', '-az', '-e', 'ssh -o BatchMode=yes -o ConnectTimeout=15 -p 29668',
                        '--include=*.json', '--include=*.jsonl', '--include=*.log', '--include=*.timeline',
                        '--include=*.tsv', '--include=*.alerts', '--exclude=traces/', '--exclude=*/', '--exclude=*',
                        HOST+':'+folder, str(dst)+'/'], timeout=120, check=False, stdout=subprocess.DEVNULL)


def wait_job(label):
    last_sync = 0
    while True:
        if time.time() - START > LIMIT:
            # Stop only the run owned by this controller; leave the rented pod itself alone.
            ssh(f'python3 /root/queue/bin/stop_run.py q_e073_{shlex.quote(label)}')
            raise RuntimeError('14-hour research-batch bound reached')
        if time.time() - last_sync > 300:
            sync(label); last_sync = time.time()
        s = ssh(f'cat /root/e073/logs/{shlex.quote(label)}.exit 2>/dev/null || true', capture_output=True, text=True).stdout.strip()
        if s:
            sync(label)
            if s != '0':
                raise RuntimeError(f'{label}: remote job exit {s}')
            # Preserve final executed notebook and compressed tool traces locally before the next job resets /tmp.
            for remote in [f'/root/nbruns/q_e073_{label}/nb.ipynb',
                           f'/root/kwork_q_e073_{label}/dsv41_traces.tar.gz',
                           f'/root/kwork_q_e073_{label}/submission.json']:
                subprocess.run(['scp', '-q', '-P', '29668', HOST+':'+remote, str(H/'runs'/label)+'/'],
                               timeout=180, check=False)
            records = analyze.A.load_jsonl(H/'runs'/label/'attempts.jsonl')
            if not records:
                raise RuntimeError(f'{label}: no attempt records; inspect notebook timeline')
            if any(r.get('error') for r in records):
                raise RuntimeError(f'{label}: error records; investigate before the next arm')
            return records
        time.sleep(30)


def launch(label, workload, extra, max_h):
    assert not ssh(f'test -e /root/e073/logs/{label}.exit && echo exists || true', capture_output=True,
                   text=True).stdout.strip(), 'refusing duplicate stage launch'
    receipt = json.load(open(H/'frozen/receipt.json'))
    job = f'''JOB_LABEL=e073_{label}
NB_PATH=/root/e073/frozen/causal.ipynb
NB_SHA256={receipt['causal']}
WORKLOAD=e073_{workload}
KT_WHEELS_DIR=$WHEELS_SFT
WHEELS_EXPECT_COUNT=294
WHEELS_EXPECT_VLLM=g22df163d2d
MODEL_KIND=none
KT_MODEL_OWNER=chankhavu
KT_MODEL_SLUG=deepseek-v41-flash-1017-arc-tool-img-e128-2bit-causal
KT_MODEL_DIR=/root/models/causal
EXTRA_ENV="{extra}"
MAX_KAGGLE_H={max_h}
'''
    p = H/'jobs'/f'{label}.job'; p.write_text(job)
    subprocess.run(['scp', '-q', '-P', '29668', str(p), HOST+':/root/e073/jobs/'], check=True, timeout=90)
    cmd = f'setsid nohup bash /root/e073/launch_job.sh {shlex.quote(label)} > /root/e073/logs/{label}.launch.log 2>&1 < /dev/null &'
    ssh(cmd, capture_output=True)
    log('launched '+label)


def complete(records, tasks, attempts):
    final = {(r['task_id'], r['attempt']) for r in records if not r.get('error')}
    return len(final) >= tasks * attempts


def main():
    global START
    (H/'runs').mkdir(exist_ok=True)
    START = float(ssh('stat -c %Y /root/nbruns/q_e073_gold/notebook.sha256', capture_output=True, text=True).stdout)
    log('waiting for same-pod V12 control')
    gold = wait_job('gold')
    if not complete(gold, 120, 2):
        raise RuntimeError('V12 control incomplete at its bound; do not replace model yet')
    # Verify HF access locally immediately before the destructive-to-cache model swap.
    import yaml
    from huggingface_hub import HfApi
    secrets = yaml.safe_load(open(ROOT/'SECRETS.yaml'))
    manifest = json.load(open(H/'frozen/model_manifest.json'))
    token = secrets[manifest['token_field']]
    assert HfApi(token=token).model_info(manifest['repo_id'], revision=manifest['revision']).sha == manifest['revision']
    log('staging hash-verified causal model; retaining shared Engram files')
    ssh('python3 -I /root/e073/stage_causal.py', input=token+'\n', text=True, timeout=3600)
    del token, secrets
    launch('dev', 'dev', 'ARC_ATTEMPTS=3 ARC_ATTEMPT_FORMATS=json,json,text+image', 4.5)
    draws = wait_job('dev')
    if not complete(draws, 60, 3):
        raise RuntimeError('development draws incomplete; holdout remains unused')
    result = analyze.screen(gold, draws)
    (H/'runs/selection.json').write_text(json.dumps(result, indent=2)+'\n')
    log('development selection '+json.dumps(result))
    policy = result['selected']
    if policy is None:
        log('no development improvement within the token-cost bound; holdout preserved; batch complete')
        registry('done: no dev win; holdout preserved', 'No causal policy improves dev pass@2 within 1.15x output-token cost; see E073 runs/selection.json')
        return
    formats = 'json,text+image' if policy == 'TI' else 'json,json,text+image'
    extra = f'ARC_ATTEMPTS=2 ARC_ATTEMPT_FORMATS={formats} ARC_EXTRA_IF_NO_TRAIN_PASS={int(policy=="retry")}'
    launch('holdout', 'holdout', extra, 4.0)
    candidate = wait_job('holdout')
    if not complete(candidate, 60, 2):
        raise RuntimeError('holdout incomplete; cannot claim an accuracy improvement')
    result = analyze.validate(gold, candidate, policy)
    (H/'runs/holdout.json').write_text(json.dumps(result, indent=2)+'\n')
    log('holdout result '+json.dumps(result))
    registry('done: holdout measured; deadline replica pending', json.dumps(result))
    log('batch complete; no pod stopped and no Kaggle submission made')


if __name__ == '__main__':
    (H/'runs').mkdir(exist_ok=True)
    lock = open(H/'runs/controller.lock', 'w')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        main()
    except Exception as exc:
        log(f'BLOCKED: {type(exc).__name__}: {exc}')
        (H/'runs/blocked.txt').write_text(f'{type(exc).__name__}: {exc}\n')
        registry('blocked: prerequisite failed', f'{type(exc).__name__}: {exc}')
        raise
