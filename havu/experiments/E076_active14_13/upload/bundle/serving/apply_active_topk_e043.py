#!/usr/bin/env python3
"""Static per-layer top-k for the E043 vision11 fork, selected before engine init.

E074_TOPK_FILE contains {"top_k": [40 integers in 1..6]}. Shape changes require
a fresh engine (including new CUDA graphs); no runtime mutation is supported.
"""
import ast
import hashlib
import json
from pathlib import Path
import sys


def patch(vllm):
    paths = {'router': vllm / 'models/deepseek_v4_1/nvidia/router.py',
             'model': vllm / 'models/deepseek_v4_1/nvidia/model.py'}
    original = {k: p.read_text() for k, p in paths.items()}
    if any('# E074:' in text for text in original.values()):
        if all('# E074:' in text for text in original.values()):
            print('E074 already installed')
            return
        raise RuntimeError('Partial E074 installation')
    anchor = '\n\ndef can_use_deepseek_v41_topk('
    assert original['router'].count(anchor) == 1
    router = original['router'].replace(anchor,
        '\n# E074: existing native kernel supports TOP_K <= 8; only E128 enabled here.\n'
        '_SUPPORTED_ROUTING_SHAPES |= frozenset((128, k) for k in range(1, 7))\n' + anchor)
    anchor = '        n_activated_experts = config.num_experts_per_tok\n'
    assert original['model'].count(anchor) == 1
    model = original['model'].replace(anchor, anchor + '''        # E074: freeze the layer schedule before constructing FusedMoE/workspaces.
        import json as _e074_json
        import os as _e074_os
        _e074_path = _e074_os.environ.get("E074_TOPK_FILE")
        if _e074_path and extract_layer_index(prefix) < config.num_hidden_layers:
            with open(_e074_path) as _e074_f:
                _e074_ks = _e074_json.load(_e074_f)["top_k"]
            if (len(_e074_ks) != config.num_hidden_layers or
                    any(type(k) is not int or not 1 <= k <= 6 for k in _e074_ks) or
                    n_routed_experts != 128):
                raise ValueError("E074 requires E128 and one integer top-k in 1..6 per layer")
            n_activated_experts = _e074_ks[extract_layer_index(prefix)]
            logger.info("E074 layer=%d top_k=%d", extract_layer_index(prefix), n_activated_experts)
''')
    updated = {'router': router, 'model': model}
    for k, text in updated.items():
        ast.parse(text)
    receipt = {}
    for k, path in paths.items():
        path.with_suffix('.py.e074-original').write_text(original[k])
        path.write_text(updated[k])
        receipt[k] = {'before': hashlib.sha256(original[k].encode()).hexdigest(),
                      'after': hashlib.sha256(updated[k].encode()).hexdigest()}
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    patch(Path(sys.argv[1]))
