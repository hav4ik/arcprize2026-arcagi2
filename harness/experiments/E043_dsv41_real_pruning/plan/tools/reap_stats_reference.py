#!/usr/bin/env python3
"""CPU statistics contract, not a model runner or production GPU collector.

Run: python3 tools/reap_stats_reference.py --self-test
Inputs to LayerStats.add are selected original IDs and ALREADY DEFINED
amplitudes (native contribution norm OR gate * unweighted-reference norm).
Calibration observation weights are distinct from router gates.
"""
from __future__ import annotations

import argparse
from array import array
from fractions import Fraction
import json
import math


class LayerStats:
    def __init__(self, experts: int, top_k: int, *, pairs: bool = False):
        if not 1 <= top_k <= experts:
            raise ValueError("Require 1 <= top_k <= experts")
        self.experts, self.top_k, self.pairs = experts, top_k, pairs
        self.count = array("q", [0]) * experts
        self.total = array("d", [0.0]) * experts
        self.denominator = array("d", [0.0]) * experts
        self.square_total = array("d", [0.0]) * experts
        self.maximum = array("d", [0.0]) * experts
        self.positions = 0
        self.position_weight = 0.0
        if pairs:
            n = experts * experts
            self.pair_count = array("q", [0]) * n
            self.pair_total = array("d", [0.0]) * n
            self.pair_denominator = array("d", [0.0]) * n

    def add(self, ids, amplitudes, *, observation_weight=1.0, valid=True):
        if not valid:
            return
        ids, amplitudes = list(ids), list(amplitudes)
        if len(ids) != self.top_k or len(amplitudes) != self.top_k:
            raise ValueError("One complete selected set is required")
        if len(set(ids)) != self.top_k or any(
            type(e) is not int or not 0 <= e < self.experts for e in ids
        ):
            raise ValueError("IDs must be distinct original expert integers")
        u = float(observation_weight)
        amplitudes = [float(a) for a in amplitudes]
        if not math.isfinite(u) or u < 0 or any(
            not math.isfinite(a) or a < 0 for a in amplitudes
        ):
            raise ValueError("Require finite, nonnegative amplitudes and weight")
        self.positions += 1
        self.position_weight += u
        for e, a in zip(ids, amplitudes):
            self.count[e] += 1
            self.total[e] += u * a
            self.denominator[e] += u
            self.square_total[e] += u * a * a
            self.maximum[e] = max(self.maximum[e], a)
        if self.pairs:
            for e, a in zip(ids, amplitudes):
                for f, b in zip(ids, amplitudes):
                    j = e * self.experts + f
                    self.pair_count[j] += 1
                    self.pair_total[j] += u * a * b  # u once, never u squared
                    self.pair_denominator[j] += u

    def merge(self, other):
        if (self.experts, self.top_k, self.pairs) != (
            other.experts, other.top_k, other.pairs
        ):
            raise ValueError("Incompatible statistics shapes")
        # Production runner must ALSO validate model/mask/amplitude/shard manifests.
        names = ["count", "total", "denominator", "square_total"]
        if self.pairs:
            names += ["pair_count", "pair_total", "pair_denominator"]
        for name in names:
            dst, src = getattr(self, name), getattr(other, name)
            for i, value in enumerate(src):
                dst[i] += value
        for i, value in enumerate(other.maximum):
            self.maximum[i] = max(self.maximum[i], value)
        self.positions += other.positions
        self.position_weight += other.position_weight
        return self

    def mean(self, expert):
        d = self.denominator[expert]
        return self.total[expert] / d if d > 0 else 0.0

    def observed(self, expert):
        return self.denominator[expert] > 0

    def pair_mean(self, e, f):
        if not self.pairs:
            raise ValueError("Pair collection was disabled")
        j = e * self.experts + f
        d = self.pair_denominator[j]
        return self.pair_total[j] / d if d > 0 else 0.0


def route_from_unbiased_scores(scores, biases, keep=None, *, top_k=6, scale=1.5):
    """Declared stable scalar routing reference; scores are POST sqrt-softplus.

    Lowest original ID breaks exact ties. This is the intended runtime fixture
    convention, not a claim about generic torch.topk's unspecified tie behavior.
    """
    if len(scores) != len(biases):
        raise ValueError("Score and bias shapes differ")
    if any(not math.isfinite(s) or s < 0 for s in scores):
        raise ValueError("Invalid unbiased score")
    if any(not math.isfinite(b) for b in biases) or not math.isfinite(scale) or scale <= 0:
        raise ValueError("Invalid bias or scale")
    keep = list(range(len(scores))) if keep is None else list(keep)
    if len(set(keep)) != len(keep) or any(
        type(e) is not int or not 0 <= e < len(scores) for e in keep
    ) or not 2 <= top_k <= len(keep):
        raise ValueError("Invalid eligible expert set; normalized routing fixture requires top_k >= 2")
    ids = sorted(keep, key=lambda e: (-(scores[e] + biases[e]), e))[:top_k]
    denominator = math.fsum(scores[e] for e in ids) + 1e-20
    weights = [scale * scores[e] / denominator for e in ids]
    return ids, weights


def self_test():
    passed = []

    def close(a, b):
        assert math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12), (a, b)

    s = LayerStats(3, 1)
    s.add([0], [10])
    for _ in range(10):
        s.add([1], [2])
    assert s.mean(0) > s.mean(1) and s.total[0] < s.total[1]
    assert not s.observed(2)
    passed.append("conditional_mean_is_not_contribution_total")

    a, b = LayerStats(1, 1), LayerStats(1, 1)
    a.add([0], [1])
    b.add([0], [3])
    b.add([0], [5])
    a.merge(b)
    close(a.mean(0), 3)
    assert a.count[0] == 3
    passed.append("merge_sums_and_counts_not_worker_means")

    vectors = [(1.0, 0.0), (-1.0, 0.0)]
    close(sum(math.hypot(*v) for v in vectors) / 2, 1)
    close(math.hypot(*(sum(v[i] for v in vectors) / 2 for i in range(2))), 0)
    passed.append("norm_before_token_averaging")

    close((0.1 * 9 + 0.9 * 1) / 2, 0.9)
    close(((0.1 + 0.9) / 2) * ((9 + 1) / 2), 2.5)
    passed.append("mean_gate_times_norm_not_product_of_means")

    ids, gates = route_from_unbiased_scores([1, 2, 3], [4, 2, 0], top_k=2)
    assert ids == [0, 1]
    close(gates[0], 0.5)
    ids2, gates2 = route_from_unbiased_scores([1, 2, 3], [4, 2, 0], [1, 2], top_k=2)
    assert ids2 == [1, 2]
    close(gates2[1], 0.9)
    close(sum(gates2), 1.5)
    assert gates2[1] > gates[0]
    passed.append("selection_bias_refill_and_native_scale")

    p = LayerStats(3, 2, pairs=True)
    p.add([0, 1], [2, 3])
    p.add([1, 2], [5, 7])
    assert list(p.pair_count) == [1, 1, 0, 1, 2, 1, 0, 1, 1]
    close(p.pair_mean(1, 1), (9 + 25) / 2)
    assert p.pair_mean(1, 1) != p.mean(1) ** 2
    passed.append("true_co_selection_and_diagonal_second_moment")

    w = LayerStats(2, 2, pairs=True)
    w.add([0, 1], [2, 3], observation_weight=2)
    close(w.pair_total[1], 12)
    close(w.pair_denominator[1], 2)
    close(w.pair_mean(0, 1), 6)
    assert w.pair_count[1] == 1
    passed.append("observation_weight_applied_once_to_pair")

    partials = [(1.0, 0.0), (-1.0, 0.0)]
    close(math.hypot(*(sum(v[i] for v in partials) for i in range(2))), 0)
    close(sum(math.hypot(*v) for v in partials), 2)
    passed.append("norm_after_tp_partial_vector_reduction")

    m = LayerStats(2, 1)
    m.add([999], [float("nan")], valid=False)
    m.add([0], [0])
    assert m.positions == 1 and m.mean(0) == m.mean(1) == 0
    assert m.observed(0) and not m.observed(1)
    try:
        m.add([0], [float("nan")])
    except ValueError:
        pass
    else:
        raise AssertionError("Nonfinite amplitude was accepted")
    passed.append("invalid_positions_and_unobserved_distinction")

    # Direct normalized-mixture arithmetic, independent of a paper notation.
    old = Fraction(6, 9)
    new = Fraction(6, 8)
    assert old - new == Fraction(-1, 12)
    pj, q, survivor = Fraction(1, 3), Fraction(1, 4), Fraction(2, 3)
    assert -(pj - q) / (1 - pj) * survivor == old - new
    passed.append("direct_refill_normalization_identity")

    # Original-ID-preserving compaction, including exact equal selection scores.
    scores, biases, keep = [5, 99, 5, 2], [0, 0, 0, 4], [0, 2, 3]
    old_ids, old_weights = route_from_unbiased_scores(scores, biases, keep, top_k=2)
    new_ids, new_weights = route_from_unbiased_scores(
        [scores[e] for e in keep], [biases[e] for e in keep], top_k=2
    )
    assert old_ids == [keep[e] for e in new_ids] == [3, 0]
    for x, y in zip(old_weights, new_weights):
        close(x, y)
    passed.append("mask_and_compact_router_original_id_ties")

    return {"status": "PASS", "scope": "CPU statistics and scalar routing only", "checks": passed}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test(), indent=2))
    else:
        parser.print_help()
