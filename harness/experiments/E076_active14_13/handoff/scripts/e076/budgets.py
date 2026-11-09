#!/usr/bin/env python3
"""Exact routed-expert deltas; total decode counts are anchored to rounded 16B.

This does not treat parameter count as FLOPs or promise throughput gains.
"""
import argparse
import json
from pathlib import Path

EXPERT_PARAMS = 3 * 5120 * 2304


def describe(ks, baseline_b=16.0):
    if len(ks) != 40 or any(type(k) is not int or not 1 <= k <= 6 for k in ks):
        raise ValueError('Expected 40 top-k values in 1..6')
    removed = (240 - sum(ks)) * EXPERT_PARAMS
    return {'top_k': ks, 'expert_slots': sum(ks), 'routed_parameters_removed': removed,
            'estimated_decode_active_b': baseline_b - removed / 1e9,
            'baseline_anchor_b': baseline_b,
            'count_note': 'Approximate total anchored to advertised 16B; routed delta exact; vision encoder excluded'}


def allocate(costs, max_slots):
    """Multiple-choice knapsack over measured per-layer costs (additive proxy).

    Every selected combination must subsequently be measured jointly. No assumption
    that marginal loss curves are monotonic or convex; negative costs are permitted.
    """
    states = {0: (0., [])}
    for choices in costs:
        nxt = {}
        for used, (loss, ks) in states.items():
            for k, cost in choices.items():
                k = int(k)
                if not 1 <= k <= 6:
                    raise ValueError('Unsupported top-k')
                n = used + k
                if n <= max_slots and (n not in nxt or loss + cost < nxt[n][0]):
                    nxt[n] = (loss + cost, ks + [k])
        states = nxt
    if not states:
        raise ValueError('Infeasible expert-slot budget')
    used, (loss, ks) = min(states.items(), key=lambda item: (item[1][0], -item[0]))
    return {'top_k': ks, 'proxy_loss': loss, 'expert_slots': used}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for k in (6, 5, 4, 3):
        (args.out / f'uniform{k}.json').write_text(json.dumps(describe([k]*40), indent=2)+'\n')
    print(json.dumps([describe([k]*40) for k in (6,5,4,3)], indent=2))


if __name__ == '__main__':
    main()
