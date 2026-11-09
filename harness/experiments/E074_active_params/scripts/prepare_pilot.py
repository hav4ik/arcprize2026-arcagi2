#!/usr/bin/env python3
"""Make a small paired image/text E065 development sample (training parents only)."""
import json
import random
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'experiments/E074_active_params/data'
KEY = ('batch', 'task_id', 'pair', 'attempt')


def paired_key(record):
    key = tuple(record[k] for k in KEY)
    return (key[0].replace('e065t_', 'e065_'),) + key[1:]


def main():
    indices = {}
    for modality in ('img', 'txt'):
        records = {}
        for path in sorted((ROOT / f'data/e066/tok_{modality}_dev_s').glob('shard*/index.jsonl')):
            for line in path.read_text().splitlines():
                r = json.loads(line)
                if r['length'] <= 30000:
                    r['dir'] = str(path.parent)
                    records[paired_key(r)] = r
        indices[modality] = records
    common = sorted(set(indices['img']) & set(indices['txt']))
    assert common, 'No paired rollouts; inspect modality batch names'
    families = {}
    for key in common:
        parent = indices['img'][key]['parent']
        assert parent == indices['txt'][key]['parent']
        families.setdefault(parent, []).append(key)
    train = json.loads((ROOT / 'data/arc-agi_training_challenges.json').read_text())
    evaluation = json.loads((ROOT / 'data/arc-agi_evaluation_challenges.json').read_text())
    assert set(families) <= set(train)
    assert not set(families) & set(evaluation)
    rng = random.Random(74)
    parents = sorted(families)
    rng.shuffle(parents)
    chosen = [rng.choice(families[p]) for p in parents[:32]]
    for modality, name in [('img', 'image'), ('txt', 'text')]:
        dst = OUT / modality
        dst.mkdir(parents=True, exist_ok=True)
        mappings = {}
        offset = 0
        with (dst/'tokens.u32').open('wb') as ft, (dst/'roles.u8').open('wb') as fr, (dst/'index.jsonl').open('w') as fi:
            for key in chosen:
                r = dict(indices[modality][key])
                source = Path(r.pop('dir'))
                if source not in mappings:
                    mappings[source] = (np.memmap(source/'tokens.u32', mode='r', dtype=np.uint32),
                                        np.memmap(source/'roles.u8', mode='r', dtype=np.uint8))
                tokens, roles = mappings[source]
                a, b = r['offset'], r['offset'] + r['length']
                tokens[a:b].tofile(ft); roles[a:b].tofile(fr)
                r['offset'] = offset
                offset += r['length']
                fi.write(json.dumps(r)+'\n')
        table = pq.read_table(ROOT / f'data/e066/e065_{name}.parquet')
        keys = table.select(KEY).to_pylist()
        wanted = set(chosen)
        positions = [i for i, r in enumerate(keys) if paired_key(r) in wanted]
        assert len(positions) == len(chosen)
        pq.write_table(table.take(pa.array(positions)), OUT / f'{modality}.parquet')
    receipt = {'seed':74, 'families':parents[:32], 'rollouts_per_modality':len(chosen),
               'max_tokens_per_rollout':30000, 'eval_parent_overlap':0,
               'source':'E065 family-held-out development split; short paired rollout pilot'}
    (OUT / 'manifest.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
