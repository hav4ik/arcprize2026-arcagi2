"""Minimal example: load a 2-bit DeepSeek-V4.1 E128 checkpoint and generate greedily.

Works for both chankhavu's base checkpoint and any SFT milestone export, e.g.
  /cns/lu-d/home/arazdai/kaggle/models/dsv41_e128_2bit                                     (base)
  /cns/lu-d/home/arazdai/kaggle/sft/checkpoints_dsv41_e128/v12_fixed_oom_moe_only_full_sft/wu_1/export_2bit/epoch_0p25

Checkpoint layout (identical for base and exports):
  model-0000{1,2}-of-00002.safetensors   FP8 block-32 backbone (attention, norms, router, shared experts, embed, head)
  gptq/layer_00..39.safetensors          2-bit Humming routed experts: per expert `e{idx}.{w1,w2,w3}.{q,s,z}`
  model-engram-*.safetensors             compact Engram tables (only in the base dir -> pass it as `engram_model_dir`)
  config.json, tokenizer.json, EXPORT_COMPLETED.json (exports only; written after all shards are flushed)

The routed experts are de-quantized to bf16 and sharded across the 8 ranks (expert parallelism:
16 of 128 experts per layer per GPU, ~45 GB/GPU), so this must run as 8 lock-stepped processes
on an 8xB200 host (one process per GPU, e.g. via `torchrun --nproc_per_node=8`). All ranks run the
same forward; rank 0 samples the next token and broadcasts it.

This is exactly what `eval_dsv41_b32_arc2.py` does before handing the model to Havu's harness.
"""

from __future__ import annotations

import datetime
import json
import os

import tokenizers
import torch
import torch.distributed as dist

from google3.experimental.users.arazdai.kaggle.sft import dsv41_sft_model
from google3.pyglib import gfile

CHECKPOINT_2BIT_DIR = (
    "/cns/lu-d/home/arazdai/kaggle/sft/checkpoints_dsv41_e128/"
    "v12_fixed_oom_moe_only_full_sft/wu_1/export_2bit/epoch_0p25"
)
BASE_MODEL_DIR = "/cns/lu-d/home/arazdai/kaggle/models/dsv41_e128_2bit"  # Engram shards + tokenizer live here


def load_model(ckpt_dir: str, device: torch.device, rank: int, world_size: int) -> dsv41_sft_model.Dsv41SftModel:
  """Builds the model skeleton for this rank and fills it from the 2-bit checkpoint."""
  cfg_path = os.path.join(ckpt_dir, "config.json")
  with gfile.Open(cfg_path, "r") as f:
    cfg = dsv41_sft_model.Dsv41E128Config.from_json_dict(json.load(f))
  cfg.routed_expert_lora_r = 0  # inference: LoRA is already merged into the 2-bit experts
  cfg.enable_qat = False  # plain bf16 de-quantized forward (set True for Havu-exact FP8/FP4 activation quant)
  cfg.qat_trainable_expert_scales = False
  cfg.load_compact_engram = True  # layers 1 & 14 Engram tables, sharded across ranks
  cfg.max_seq_len = 163840

  with torch.device(device):
    model = dsv41_sft_model.Dsv41SftModel(
        cfg=cfg,
        ep_rank=rank,
        ep_world_size=world_size,
        gradient_checkpointing=False,
        ep_replicated_input=True,  # every rank receives the same tokens (SPMD decode)
    )
  stats = dsv41_sft_model.load_dequantized_dsv41_e128(
      model_dir=ckpt_dir,
      model=model,
      device=device,
      engram_model_dir=BASE_MODEL_DIR,  # exports don't carry the 69 GB Engram shards; read them from the base dir
  )
  # Sanity: a correct load reports 330 FP8 tensors, 594 dense tensors, 640 unpacked 2-bit experts (per rank), 6 engram slices.
  if rank == 0:
    print(f"loaded {ckpt_dir}: {stats}", flush=True)
  assert stats["fp8_dequantized_tensors"] == 330, stats
  model.eval()
  return model


@torch.no_grad()
def greedy_generate(model, tok, prompt: str, device: torch.device, max_new_tokens: int = 256) -> str:
  """Prefill + incremental decode with the model's KV cache (`forward_hidden_cached`)."""
  ids = torch.tensor([tok.encode(prompt, add_special_tokens=False).ids], dtype=torch.long, device=device)
  kv_cache: dict = {}  # fresh cache for a new sequence; the model appends to it in place
  hidden = model.forward_hidden_cached(ids, kv_cache)  # [1, T, dim] -> only the last position is needed
  out: list[int] = []
  eos = tok.token_to_id("<｜end▁of▁sentence｜>")
  for _ in range(max_new_tokens):
    logits = model.head(hidden[:, -1:, :]).float()  # [1, 1, vocab]
    nxt = logits.argmax(dim=-1)  # greedy; sample here for temperature > 0
    if dist.is_initialized():
      dist.broadcast(nxt, src=0)  # keep all 8 EP ranks in lock-step on the same token
    tid = int(nxt.item())
    if tid == eos:
      break
    out.append(tid)
    hidden = model.forward_hidden_cached(nxt, kv_cache)  # 1-token decode step
  return tok.decode(out, skip_special_tokens=True)


def main() -> None:
  rank = int(os.environ.get("RANK", "0"))
  world_size = int(os.environ.get("WORLD_SIZE", "1"))
  device = torch.device(f"cuda:{rank}")
  torch.cuda.set_device(device)
  if world_size > 1 and not dist.is_initialized():
    dist.init_process_group(
        backend="nccl", rank=rank, world_size=world_size, device_id=device,
        timeout=datetime.timedelta(minutes=90),
    )

  model = load_model(CHECKPOINT_2BIT_DIR, device, rank, world_size)

  with gfile.Open(os.path.join(BASE_MODEL_DIR, "tokenizer.json"), "rb") as f:
    tok = tokenizers.Tokenizer.from_buffer(f.read())

  text = greedy_generate(model, tok, "The pattern in the training examples is", device)
  if rank == 0:
    print(text, flush=True)
  if dist.is_initialized():
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
  main()
