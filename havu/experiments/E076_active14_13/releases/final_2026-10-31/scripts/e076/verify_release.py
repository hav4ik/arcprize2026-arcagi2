#!/usr/bin/env python3
"""Verify a final E076 bundle and optionally rescore against owner-provided solutions."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('bundle', type=Path)
    ap.add_argument('--solutions', type=Path)
    args = ap.parse_args()
    root = args.bundle.resolve()
    count = 0
    for line in (root / 'FILES.sha256').read_text().splitlines():
        expected, name = line.split('  ', 1)
        path = (root / name).resolve()
        assert path.is_relative_to(root), name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, name
        count += 1
    report = json.loads((root / 'evaluation/completed_tools_report.json').read_text())
    runs = [('E128', 'image', 'E128'), ('A14p58', 'image', '14.58B'),
            ('A14p02', 'image', '14.02B'), ('E128_text', 'text', 'E128'),
            ('A14p02_text', 'text', '14.02B')]
    solutions = None
    if args.solutions:
        meta = json.loads((root / 'metadata.json').read_text())
        expected = meta['evaluation']['data_hashes']['data/arc-agi_evaluation_solutions.json']
        assert hashlib.sha256(args.solutions.read_bytes()).hexdigest() == expected
        solutions = json.loads(args.solutions.read_text())
    for folder, modality, model in runs:
        rows = [json.loads(line) for line in (root / 'evaluation' / folder / 'arc.jsonl').read_text().splitlines()]
        latest = {(r['task_id'], r['attempt'], r['pair']): r for r in rows}
        assert len(latest) == 344 and len({(t, a) for t, a, p in latest}) == 240, folder
        stats = report['image'][model] if modality == 'image' else report['text_matched']['models'][model]
        logical = [r for (t, a, p), r in latest.items() if p == 0]
        assert sum(r['tool_calls'] for r in logical) == stats['tool_calls'], folder
        assert sum(r['tool_errors'] for r in logical) == stats['tool_errors'], folder
        assert abs(sum(r['usage']['completion'] for r in logical) / 240 - stats['output_tokens_mean']) < 1e-8, folder
        if solutions is not None:
            task_p1, task_p2 = [], []
            for task, outputs in sorted(solutions.items()):
                ok = [[latest[task, a, p].get('pred') == output for p, output in enumerate(outputs)] for a in (0, 1)]
                task_p1.append(sum(sum(x) for x in ok) / (2 * len(outputs)))
                task_p2.append(sum(any(ok[a][p] for a in (0, 1)) for p in range(len(outputs))) / len(outputs))
            for metric, values in [('pass_at_1', task_p1), ('pass_at_2', task_p2)]:
                assert abs(100 * sum(values) / len(values) - stats[metric]) < 1e-9, (folder, metric)
    print(json.dumps({'verified_hashed_files': count, 'complete_eval_arms': len(runs),
                      'logical_attempts_per_arm': 240, 'rescored_against_pinned_solutions': solutions is not None}))


if __name__ == '__main__':
    main()
