#!/usr/bin/env python3
"""CPU input/target position contract only; no native model or hashing.

Run: python3 tools/ced_phase_masks_reference.py --self-test
One episode has P prompt tokens and R observed completion targets.
The minimal consumed inputs have length P+R-1.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json


@dataclass(frozen=True)
class EpisodePositions:
    prompt: int
    targets: int
    warm_window: int = 128

    def __post_init__(self):
        if any(type(v) is not int or v < 1 for v in
               (self.prompt, self.targets, self.warm_window)):
            raise ValueError("P, R and warmup window must be positive integers")

    @property
    def inputs(self):
        return self.prompt + self.targets - 1

    def ranges(self):
        p, n = self.prompt, self.inputs
        return {
            "encoder_prompt": range(0, p),
            "encoder_continuation": range(p, n),
            "decoder_generation": range(p - 1, n),
            "decoder_warmup": range(max(0, p - self.warm_window), p - 1),
            "engram_inputs": range(0, n),
        }

    def counts(self):
        return {name: len(span) for name, span in self.ranges().items()}


def observation_key(episode_id, input_position, layer_id, score_definition):
    """Run manifest supplies model/Engram/schedule identity in production.

    Physical call IDs deliberately do not belong to semantic observations.
    """
    return episode_id, input_position, layer_id, score_definition


def self_test():
    passed = []
    e = EpisodePositions(3, 4)
    spans = e.ranges()
    assert list(spans["decoder_generation"]) == [2, 3, 4, 5]
    assert [i + 1 for i in spans["decoder_generation"]] == [3, 4, 5, 6]
    assert 2 in spans["encoder_prompt"] and 2 in spans["decoder_generation"]
    assert 6 not in spans["engram_inputs"]
    passed.append("target_shift_last_prompt_and_unconsumed_final_token")

    for p, r in ((1, 1), (1, 4), (4, 1), (128, 3), (129, 1), (300, 20)):
        e = EpisodePositions(p, r)
        c = e.counts()
        assert c["encoder_prompt"] == p
        assert c["encoder_continuation"] == r - 1
        assert c["decoder_generation"] == r
        assert c["decoder_warmup"] == min(p, 128) - 1
        assert c["engram_inputs"] == p + r - 1
        assert c["encoder_prompt"] + c["encoder_continuation"] == e.inputs
        s = e.ranges()
        assert not (set(s["decoder_generation"]) & set(s["decoder_warmup"]))
    passed.append("single_token_and_long_prompt_boundary_counts")

    e = EpisodePositions(300, 20)
    spans = e.ranges()
    generated = set(spans["decoder_generation"])
    physical_calls = [range(172, 305), range(299, e.inputs)]
    seen = set()
    for call in physical_calls:
        for pos in call:
            if pos in generated:
                seen.add(observation_key("episode", pos, 20, "native_amp"))
    assert len(seen) == e.targets
    passed.append("overlap_replay_does_not_duplicate_semantic_positions")

    validity = [True] * e.inputs
    validity[0] = False  # Fixture for a native-ineligible input position.
    semantic_positions = [p for p in spans["engram_inputs"] if validity[p]]
    lookup_events = sum(3 * 8 for layer in (1, 14) for p in semantic_positions)
    assert lookup_events == 48 * (e.inputs - 1)
    assert 6 * len(generated) == 6 * e.targets
    passed.append("eligibility_and_per_module_lookup_count")

    for args in ((0, 1), (1, 0), (True, 1), (1, 1, 0), (1.0, 1)):
        try:
            EpisodePositions(*args)
        except ValueError:
            pass
        else:
            raise AssertionError(args)
    passed.append("invalid_episode_parameters_rejected")
    print(json.dumps({"status": "PASS", "scope": "CPU position masks only", "checks": passed}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if not args.self_test:
        parser.error("Use --self-test; this is not a model runner")
    self_test()
