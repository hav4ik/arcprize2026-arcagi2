#!/usr/bin/env python3
"""Fail closed outside original ARC training and NVARC training-parent rollouts.

This validates provenance/parent boundaries, not semantic grid deduplication.
The user permits both source types; evaluation-derived data stays excluded.
"""
import argparse
import json
from pathlib import Path


def validate_rows(rows, allowed_batches, training_parents, evaluation_parents):
    if not rows:
        raise ValueError('Empty calibration index')
    for r in rows:
        batch, parent, task = r['batch'], r['parent'], r['task_id']
        if batch not in allowed_batches:
            raise ValueError(f'Non-allowlisted rollout batch: {batch}')
        if parent in evaluation_parents or task in evaluation_parents:
            raise ValueError(f'Evaluation-derived rollout: {task}')
        if parent not in training_parents:
            raise ValueError(f'Parent is not in the training registry: {parent}')
        if batch in {'e065_arc2_train', 'e065t_arc2_train'}:
            if task != parent:
                raise ValueError(f'Expected original training task, got: {task}')
        elif not task.startswith(parent + '_'):
            raise ValueError(f'Expected NVARC child task, got: {task}')
    return len(rows)


def validate_calibration_data(exp, pools):
    exp = Path(exp)
    root = exp.parents[1]
    policy = json.loads((exp/'configs/training_data_policy.json').read_text())
    train = set(json.loads((root/'data/arc-agi_training_challenges.json').read_text()))
    evaluation = set(json.loads((root/'data/arc-agi_evaluation_challenges.json').read_text()))
    seen_families = set()
    result = {}
    for pool in pools:
        paired = {}
        for mode in ('img', 'txt'):
            path = exp/'data'/pool/mode/'index.jsonl'
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            result[f'{pool}_{mode}'] = validate_rows(rows, set(policy['allowed_batches']), train, evaluation)
            paired[mode] = {(r['batch'].replace('e065t_', 'e065_'), r['task_id'], r['pair'], r['attempt']) for r in rows}
        if paired['img'] != paired['txt']:
            raise ValueError(f'Image/text rollout mismatch in {pool}')
        families = {r['parent'] for r in rows}
        if families & seen_families:
            raise ValueError(f'Parent-family overlap across pools: {pool}')
        seen_families.update(families)
    return result


def validate_training_diagnostic(exp):
    """Require every original-task diagnostic to match the training registry."""
    exp = Path(exp)
    root = exp.parents[1]
    train = json.loads((root/'data/arc-agi_training_challenges.json').read_text())
    evaluation = set(json.loads((root/'data/arc-agi_evaluation_challenges.json').read_text()))
    tasks = json.loads((exp/'data/training_diagnostic_challenges.json').read_text())
    if not tasks:
        raise ValueError('Empty training diagnostic')
    for task, challenge in tasks.items():
        if task in evaluation or task not in train:
            raise ValueError(f'Diagnostic outside training registry: {task}')
        if challenge != train[task]:
            raise ValueError(f'Diagnostic differs from original training task: {task}')
    return len(tasks)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--experiment', type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument('--pools', nargs='+', choices=('screen', 'select', 'audit'), default=['screen', 'select', 'audit'])
    a = ap.parse_args()
    print(json.dumps(validate_calibration_data(a.experiment, a.pools), indent=2))
