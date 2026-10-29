"""Triton dsv4_topk vs the CUDA topk_softplus_sqrt op at E=128 (both exist there), random fp32 logits + bias, image rows."""
import torch
import vllm._custom_ops as ops
from vllm.model_executor.layers.fused_moe.router.dsv4_topk import dsv4_topk

torch.manual_seed(0)
SENT = 1000
for n in (1, 80, 2048):
    for vl in (False, True):
        g = torch.randn(n, 128, device="cuda") * 2
        bias = torch.randn(128, device="cuda") * 0.1
        bvl = torch.randn(128, device="cuda") * 0.5 if vl else None
        ids_in = torch.randint(0, SENT, (n,), device="cuda", dtype=torch.int32)
        if vl:
            ids_in[::3] = SENT + (torch.arange(n, device="cuda")[::3] % 5).to(torch.int32)
        w_t, i_t = dsv4_topk(g, bias, torch.int32, 1.5, input_ids=ids_in if vl else None, bias_vl=bvl,
                             image_sentinel_lo=SENT if vl else 0)
        w_c = torch.empty(n, 6, device="cuda"); i_c = torch.empty(n, 6, device="cuda", dtype=torch.int32)
        tei = torch.empty(n, 6, device="cuda", dtype=torch.int32)
        ops.topk_hash_softplus_sqrt(w_c, i_c, tei, g, True, 1.5, bias, ids_in if vl else None, None,
                                    bias_vl=bvl, image_sentinel_lo=SENT if vl else 0)
        # compare as sets per row (the CUDA op may order the 6 slots differently)
        si_t, o_t = i_t.sort(1); si_c, o_c = i_c.sort(1)
        same = bool(torch.equal(si_t, si_c))
        dw = float((w_t.gather(1, o_t) - w_c.gather(1, o_c)).abs().max())
        print(f"E128 rows={n} image_rows={vl}: ids_equal={same} (ordered {bool(torch.equal(i_t, i_c))}) max|dw|={dw:.2e}")
