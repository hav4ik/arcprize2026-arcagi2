# E060: per-layer routed-expert selection counts for planning a Flash-Next prune (drop-in module for vLLM 0.32.0,
# installed as vllm/e060_expert_counts.py in /root/e060/svenv only). Enabled by env E060_EXPERT_COUNTS=<dir>.
# Reuses vLLM's RoutedExpertsCapturer (the routing-replay hook on each MoE router: logical expert ids, written into a
# fixed device buffer, so it works inside CUDA graphs). After every model step, TP rank 0 adds the step's real rows
# (input_batch.query_start_loc_np[n_reqs], so CUDA-graph padding rows are excluded) to a [layers, experts] int64
# counter on the GPU; every E060_EXPERT_COUNTS_EVERY seconds (default 300) it saves the cumulative counter as
# <dir>/counts_<unix time>.npy plus one JSON line in <dir>/index.jsonl ({t, file, tokens, steps}).
# Per-arm counts = the difference of the snapshots at the arm's start and end. Tokens include prefill and decode
# (and, with speculative decoding, the target's verify rows).
import json
import os
import time

import numpy as np
import torch

from vllm.distributed.parallel_state import get_tp_group
from vllm.model_executor.layers.fused_moe.routed_experts_capturer import (
    RoutedExpertsCapturer,
    bind_routed_experts_capturer,
)


class ExpertCounter:
    def __init__(self, model, vllm_config, path):
        self.capturer = RoutedExpertsCapturer(
            max_num_batched_tokens=vllm_config.scheduler_config.max_num_batched_tokens, vllm_config=vllm_config)
        bind_routed_experts_capturer(model, self.capturer)
        self.L, self.K = self.capturer.shape_per_token
        self.E = vllm_config.model_config.get_num_experts()
        dev = self.capturer.device_buffer.device
        self.counts = torch.zeros(self.L * self.E, dtype=torch.int64, device=dev)
        self.offsets = (torch.arange(self.L, device=dev, dtype=torch.int64) * self.E).view(1, self.L, 1)
        self.rank0 = get_tp_group().is_first_rank
        self.path, self.tokens, self.steps = path, 0, 0
        self.every = float(os.environ.get("E060_EXPERT_COUNTS_EVERY", "300"))
        self.last = time.time()
        if self.rank0:
            os.makedirs(path, exist_ok=True)

    def step(self, input_batch):
        if not self.rank0:
            return
        n = int(input_batch.query_start_loc_np[len(input_batch.req_ids)])
        if n <= 0:
            return
        ids = self.capturer.device_buffer[:n].to(torch.int64) + self.offsets  # [n, L, K] -> flat layer*E + expert
        self.counts.index_add_(0, ids.reshape(-1), torch.ones(ids.numel(), dtype=torch.int64, device=ids.device))
        self.tokens += n
        self.steps += 1
        if time.time() - self.last >= self.every:
            self.dump()

    def dump(self):
        t = int(time.time())
        name = f"counts_{t}.npy"
        np.save(os.path.join(self.path, name), self.counts.view(self.L, self.E).cpu().numpy())
        with open(os.path.join(self.path, "index.jsonl"), "a") as f:
            f.write(json.dumps({"t": t, "file": name, "tokens": self.tokens, "steps": self.steps}) + "\n")
        self.last = time.time()
