#!/usr/bin/env python3
"""Cross-check native E128 top-k against independent Torch selection, incl. images."""
import torch
from vllm.models.deepseek_v4_1.nvidia.router import deepseek_v41_topk
from vllm.models.deepseek_v4_1.common.mm_preprocess import IMAGE_SENTINEL_BASE_ID

torch.manual_seed(7401)
checks = 0
for rows in (1, 17, 129):
    for tied in (False, True):
        logits = torch.randn(rows, 128, device='cuda', dtype=torch.float32) * 5
        bias = torch.randn(128, device='cuda', dtype=torch.float32) * .1
        vl = torch.randn_like(bias) * .2
        if tied:
            logits.zero_(); bias.zero_(); vl.zero_()
        tokens = torch.full((rows,), 42, device='cuda', dtype=torch.int64)
        tokens[::2] = IMAGE_SENTINEL_BASE_ID
        scores = torch.nn.functional.softplus(logits).sqrt()
        correction = bias.expand(rows, -1).clone()
        correction[::2] = vl
        expected_order = torch.argsort(scores + correction, descending=True, stable=True)
        for k in (1, 2, 3, 4, 5, 6):
            weights, ids = deepseek_v41_topk(logits, bias, torch.int32, 1.5,
                topk=k, correction_bias_vl=vl, input_tokens=tokens)
            expected_ids = expected_order[:, :k]
            expected_weights = scores.gather(1, expected_ids)
            expected_weights *= 1.5 / expected_weights.sum(1, keepdim=True)
            torch.testing.assert_close(ids.long(), expected_ids, atol=0, rtol=0)
            torch.testing.assert_close(weights, expected_weights, atol=2e-6, rtol=2e-6)
            checks += 1
print(f'PASS: {checks} E128 top-k checks, text/image positions, ties, k=1..6')
