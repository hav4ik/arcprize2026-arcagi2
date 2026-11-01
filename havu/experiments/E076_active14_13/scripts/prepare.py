#!/usr/bin/env python3
"""Freeze disjoint training-family pools and E074's serving/evaluation assets."""
import hashlib
import json
import random
import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[3]
EXP = ROOT / 'experiments/E076_active14_13'
OLD = ROOT / 'experiments/E074_active_params'
KEY = ('batch', 'task_id', 'pair', 'attempt')


def paired_key(r):
    return (r['batch'].replace('e065t_', 'e065_'), r['task_id'], r['pair'], r['attempt'])


def main():
    assert not (EXP / 'data/manifest.json').exists(), 'Already frozen'
    idx = {}
    for mode in ('img', 'txt'):
        idx[mode] = {}
        for p in sorted((ROOT / f'data/e066/tok_{mode}_dev_s').glob('shard*/index.jsonl')):
            for line in p.read_text().splitlines():
                r = json.loads(line)
                if r['length'] <= 60000:
                    r['dir'] = str(p.parent)
                    idx[mode][paired_key(r)] = r
    families = {}
    for key in sorted(set(idx['img']) & set(idx['txt'])):
        parent = idx['img'][key]['parent']
        assert parent == idx['txt'][key]['parent']
        families.setdefault(parent, []).append(key)
    train = json.loads((ROOT / 'data/arc-agi_training_challenges.json').read_text())
    evaluation = json.loads((ROOT / 'data/arc-agi_evaluation_challenges.json').read_text())
    assert set(families) <= set(train) and not set(families) & set(evaluation)
    old = set(json.loads((OLD / 'data/manifest.json').read_text())['families'])
    rng = random.Random(76)
    fresh = sorted(set(families) - old)
    rng.shuffle(fresh)
    audit = fresh[:32]
    short = {p: [k for k in ks if max(idx[m][k]['length'] for m in idx) <= 30000]
             for p, ks in families.items() if p not in audit}
    short_parents = sorted(p for p, ks in short.items() if ks)
    rng.shuffle(short_parents)
    screen = short_parents[:16]
    remaining = sorted(set(families) - set(audit) - set(screen))
    rng.shuffle(remaining)
    select = remaining[:24]
    pools = {'screen': screen, 'select': select, 'audit': audit}
    assert [len(pools[x]) for x in pools] == [16, 24, 32]
    assert len(set(sum(pools.values(), []))) == 72
    chosen = {}
    for pool, parents in pools.items():
        # Alternate random and longer paired traces in select/audit to cover context length.
        chosen[pool] = []
        for i, parent in enumerate(parents):
            options = short[parent] if pool == 'screen' else families[parent]
            if pool != 'screen' and i % 2:
                key = max(options, key=lambda k: (max(idx[m][k]['length'] for m in idx), k))
            else:
                key = rng.choice(options)
            chosen[pool].append(key)
    manifests = {}
    for mode, name in [('img', 'image'), ('txt', 'text')]:
        table = pq.read_table(ROOT / f'data/e066/e065_{name}.parquet')
        positions = {paired_key(r): i for i, r in enumerate(table.select(KEY).to_pylist())}
        for pool, keys in chosen.items():
            dst = EXP / 'data' / pool / mode
            dst.mkdir(parents=True, exist_ok=True)
            offset, mappings = 0, {}
            with (dst/'tokens.u32').open('wb') as ft, (dst/'roles.u8').open('wb') as fr, (dst/'index.jsonl').open('w') as fi:
                for key in keys:
                    r = dict(idx[mode][key])
                    source = Path(r.pop('dir'))
                    if source not in mappings:
                        mappings[source] = (np.memmap(source/'tokens.u32', mode='r', dtype=np.uint32),
                                            np.memmap(source/'roles.u8', mode='r', dtype=np.uint8))
                    tokens, roles = mappings[source]
                    a, b = r['offset'], r['offset'] + r['length']
                    tokens[a:b].tofile(ft)
                    roles[a:b].tofile(fr)
                    r['offset'] = offset
                    offset += r['length']
                    fi.write(json.dumps(r)+'\n')
            pq.write_table(table.take(pa.array([positions[k] for k in keys])), dst.parent/f'{mode}.parquet')
            manifests[f'{pool}_{mode}'] = {'families': pools[pool], 'rollouts': len(keys),
                'tokens': offset, 'max_length': max(idx[mode][k]['length'] for k in keys)}
    # A short generated tool check on ARC training originals; never used for architecture selection.
    diag = {p: train[p] for p in screen[:12]}
    (EXP/'data/training_diagnostic_challenges.json').write_text(json.dumps(diag)+'\n')
    (EXP/'data/manifest.json').write_text(json.dumps({'seed':76, 'pools':manifests,
        'paired_keys':chosen, 'eval_parent_overlap':0, 'audit_e074_family_overlap':0,
        'source':'E065 family-held-out development rollouts; ARC training parents only',
        'diagnostic_training_tasks':list(diag)}, indent=2)+'\n')
    shutil.copytree(OLD/'frozen', EXP/'frozen', ignore=shutil.ignore_patterns('__pycache__'), dirs_exist_ok=True)
    files = {str(p.relative_to(EXP/'frozen')):hashlib.sha256(p.read_bytes()).hexdigest()
             for p in sorted((EXP/'frozen').rglob('*')) if p.is_file()}
    (EXP/'frozen_manifest.json').write_text(json.dumps(files, indent=2)+'\n')
    (EXP/'configs').mkdir(exist_ok=True)
    (EXP/'results').mkdir(exist_ok=True)
    for name in ('uniform6','mean5_profiled'):
        shutil.copyfile(OLD/'configs'/f'{name}.json', EXP/'configs'/f'{name}.json')
    print(json.dumps(manifests, indent=2))


if __name__ == '__main__':
    main()
