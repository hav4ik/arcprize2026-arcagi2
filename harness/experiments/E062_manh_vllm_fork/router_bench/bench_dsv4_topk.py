"""Standalone timing + correctness of the fork's Triton dsv4_topk router kernel (copied from
vllm/model_executor/layers/fused_moe/router/dsv4_topk.py, minus the warmup decorator and PDL) at per-layer expert
counts, decode (80 rows) and prefill-chunk (2048 rows) sizes, with and without image rows. Times are GPU time per
call, measured inside a CUDA graph (as vLLM replays decode steps) and eagerly."""
import json
import sys

import torch
import triton
import triton.language as tl


@triton.jit
def _dsv4_topk_kernel(gating_output_ptr, correction_bias_ptr, is_padding_ptr, topk_weights_ptr, topk_ids_ptr,
                      routed_scaling_factor, input_ids_ptr, bias_vl_ptr, image_sentinel_lo,
                      NUM_EXPERTS: tl.constexpr, BLOCK_N: tl.constexpr, HAS_PADDING: tl.constexpr,
                      HAS_VL: tl.constexpr):
    row = tl.program_id(0)
    expert_offsets = tl.arange(0, BLOCK_N)
    expert_mask = expert_offsets < NUM_EXPERTS
    bias = tl.load(correction_bias_ptr + expert_offsets, mask=expert_mask, other=0.0).to(tl.float32)
    if HAS_VL:
        token_id = tl.load(input_ids_ptr + row).to(tl.int64)
        bias_vl = tl.load(bias_vl_ptr + expert_offsets, mask=expert_mask, other=0.0).to(tl.float32)
        is_image = (token_id >= image_sentinel_lo) & (token_id < image_sentinel_lo + 5)
        bias = tl.where(is_image, bias_vl, bias)
    logits = tl.load(gating_output_ptr + row * NUM_EXPERTS + expert_offsets, mask=expert_mask, other=0.0).to(tl.float32)
    weights = tl.sqrt(tl.where(logits > 20.0, logits, tl.log(1.0 + tl.exp(logits))))
    current = tl.where(expert_mask, weights + bias, -float("inf"))
    current = tl.where(current == current, current, -1e30)
    topk_offsets = tl.arange(0, 8)
    selected_weights = tl.zeros([8], dtype=tl.float32)
    selected_ids = tl.zeros([8], dtype=tl.int32)
    for slot in tl.static_range(6):
        max_value = tl.max(current, axis=0)
        candidate = tl.where(current == max_value, expert_offsets, NUM_EXPERTS)
        expert_id = tl.min(candidate, axis=0).to(tl.int32)
        selected_weight = tl.sum(tl.where(expert_offsets == expert_id, weights, 0.0), axis=0)
        is_slot = topk_offsets == slot
        selected_weights = tl.where(is_slot, selected_weight, selected_weights)
        selected_ids = tl.where(is_slot, expert_id, selected_ids)
        current = tl.where(expert_offsets == expert_id, -float("inf"), current)
    weight_sum = tl.sum(selected_weights, axis=0)
    selected_weights *= routed_scaling_factor / tl.where(weight_sum > 0.0, weight_sum, 1.0)
    output_mask = topk_offsets < 6
    output_offsets = row * 6 + topk_offsets
    if HAS_PADDING:
        is_padding = tl.load(is_padding_ptr + row)
        selected_weights = tl.where(is_padding, 0.0, selected_weights)
        selected_ids = tl.where(is_padding, -1, selected_ids)
    tl.store(topk_weights_ptr + output_offsets, selected_weights, mask=output_mask)
    tl.store(topk_ids_ptr + output_offsets, selected_ids, mask=output_mask)


SENT = 129264  # image placeholder id; the fork's sentinel block is 5 consecutive ids


def launch(g, bias, pad, w, ids, inp, bvl):
    n, e = g.shape
    _dsv4_topk_kernel[(n,)](g, bias, pad, w, ids, 1.5, inp, bvl, SENT, NUM_EXPERTS=e,
                            BLOCK_N=triton.next_power_of_2(e), HAS_PADDING=True, HAS_VL=bvl is not None, num_warps=1)


def reference(g, bias, pad, inp, bvl):
    s = torch.sqrt(torch.where(g > 20, g, torch.log1p(torch.exp(g))))
    b = bias.expand_as(g)
    if bvl is not None:
        img = (inp >= SENT) & (inp < SENT + 5)
        b = torch.where(img[:, None], bvl.expand_as(g), b)
    sel = s + b
    # top-6 with ties to the lowest id (as the kernel): stable sort on -sel
    order = torch.sort(-sel, dim=1, stable=True).indices[:, :6]
    wsel = torch.gather(s, 1, order)
    wsel = wsel * 1.5 / wsel.sum(1, keepdim=True).clamp_min(1e-30)
    order = torch.where(pad[:, None], -1, order)
    wsel = torch.where(pad[:, None], 0.0, wsel)
    return wsel, order.to(torch.int32)


def time_graph(fn, iters=200):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(40):  # 40 layers per step, as in the model
            fn()
    graph.replay()
    torch.cuda.synchronize()
    t0, t1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    t0.record()
    for _ in range(iters):
        graph.replay()
    t1.record()
    torch.cuda.synchronize()
    return t0.elapsed_time(t1) * 1000 / iters  # us per 40 calls (one model step)


def time_eager(fn, iters=400):
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    t0, t1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    t0.record()
    for _ in range(iters):
        fn()
    t1.record()
    torch.cuda.synchronize()
    return t0.elapsed_time(t1) * 1000 / iters  # us per call


def main():
    free, total = torch.cuda.mem_get_info()
    assert free > 400 * 2 ** 20, f"only {free / 2 ** 20:.0f} MiB free on the GPU; not running"
    torch.manual_seed(0)
    out = []
    for e in [int(x) for x in sys.argv[1].split(",")]:
        for n in (80, 2048):
            for vl in (False, True):
                g = torch.randn(n, e, device="cuda", dtype=torch.float32) * 2
                bias = torch.randn(e, device="cuda") * 0.1
                bvl = (torch.randn(e, device="cuda") * 0.5) if vl else None
                inp = torch.randint(0, 129000, (n,), device="cuda", dtype=torch.int32)
                if vl:
                    inp[::3] = SENT  # a third of the rows are image tokens
                pad = torch.zeros(n, dtype=torch.bool, device="cuda")
                pad[-3:] = True
                w = torch.empty(n, 6, device="cuda")
                ids = torch.empty(n, 6, device="cuda", dtype=torch.int32)
                fn = lambda: launch(g, bias, pad, w, ids, inp, bvl)  # noqa: E731
                fn()
                torch.cuda.synchronize()
                rw, rid = reference(g, bias, pad, inp, bvl)
                ok_ids = bool(torch.equal(ids, rid))
                max_dw = float((w - rw).abs().max())
                step_us = time_graph(fn)
                eager_us = time_eager(fn)
                out.append({"E": e, "rows": n, "image_rows": vl, "ids_equal_ref": ok_ids, "max_abs_dweight": max_dw,
                            "graph_us_per_step_40_layers": round(step_us, 1), "eager_us_per_call": round(eager_us, 2)})
                print(json.dumps(out[-1]), flush=True)
                del g, bias, bvl, inp, pad, w, ids
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
