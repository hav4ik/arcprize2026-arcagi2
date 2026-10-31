"""Load and evaluate a 2-bit DeepSeek-V4.1 E128 SFT / QAT Image checkpoint.

Supports all three model-loading and evaluation workflows for the latest Round-2
Multimodal QAT Image SFT (`round2_v14_16xb200_fast`) and Round-1 Text SFT (`v12`):

  1. Load a standalone exported 2-bit checkpoint directly (`load_model`):
     - Best QAT Image SFT (0.80 ep):
       `/cns/lu-d/home/arazdai/kaggle/sft/checkpoints_dsv41_e128_qat_iter2/round2_v14_16xb200_fast/wu_1/export_2bit/epoch_0p80`
     - Final QAT Image SFT (1.00 ep, HuggingFace `chankhavu/ds-qat-image-v02-ep1-2bit`):
       `/cns/lu-d/home/arazdai/kaggle/sft/checkpoints_dsv41_e128_qat_iter2/round2_v14_16xb200_fast/wu_1/export_2bit/epoch_1p00`
     - Base Image 2-bit checkpoint:
       `/cns/lu-d/home/arazdai/kaggle/models/dsv41_img_e128_2bit_causal`

  2. Load the base 2-bit image model + overlay a resumable QAT adapter/scale delta
     checkpoint (`load_model_from_resumable_qat_delta`), and optionally export it
     to standalone 2-bit format (`--export_2bit_out_dir`):
     - Best QAT Image SFT resumable delta (0.80 ep, HuggingFace `chankhavu/ds-qat-image-v02-ep08`):
       `/cns/lu-d/home/arazdai/kaggle/sft/checkpoints_dsv41_e128_qat_iter2/round2_v14_16xb200_fast/wu_1/step-00154-epoch_0p80`

  3. Greedy generation for both text-only prompts (`greedy_generate`) and E058
     multimodal `text+image` prompts with `<｜deepseek_image｜>` placeholders
     (`greedy_generate_with_images`). For full 120-task Candidate B32 ARC-AGI-2
     evaluation (`--grid_format=text+image --image_cell=0`), use
     `eval_dsv41_b32_arc2.py` or `launch_dsv41_b32_eval_xm.py`.

Checkpoint layout (exported 2-bit):
  model-0000{1..3}-of-00003.safetensors  FP8 block-32 backbone + BF16 vision tower
  gptq/layer_00..39.safetensors          2-bit Humming routed experts: `e{idx}.{w1,w2,w3}.{q,s,z}`
  model-engram-*.safetensors             Compact Engram tables (layers 1 & 14; read from `engram_model_dir`)
  config.json, EXPORT_COMPLETED.json
"""

from __future__ import annotations

from collections.abc import Sequence
import datetime
import json
import os

from absl import app
from absl import flags
import tokenizers
import torch
import torch.distributed as dist

from google3.experimental.users.arazdai.kaggle.sft import dsv41_image_tokens
from google3.experimental.users.arazdai.kaggle.sft import dsv41_sft_model
from google3.experimental.users.arazdai.kaggle.sft import train_dsv41_sft
from google3.pyglib import gfile

CHECKPOINT_2BIT_DIR = (
    "/cns/lu-d/home/arazdai/kaggle/sft/checkpoints_dsv41_e128_qat_iter2/"
    "round2_v14_16xb200_fast/wu_1/export_2bit/epoch_0p80"
)
RESUMABLE_QAT_DELTA_DIR = (
    "/cns/lu-d/home/arazdai/kaggle/sft/checkpoints_dsv41_e128_qat_iter2/"
    "round2_v14_16xb200_fast/wu_1/step-00154-epoch_0p80"
)
BASE_IMAGE_MODEL_DIR = (
    "/cns/lu-d/home/arazdai/kaggle/models/dsv41_img_e128_2bit_causal"
)
BASE_MODEL_DIR = (
    "/cns/lu-d/home/arazdai/kaggle/models/dsv41_e128_2bit"
)  # Compact Engram shards + tokenizer live here

_CHECKPOINT_2BIT_DIR = flags.DEFINE_string(
    "checkpoint_2bit_dir",
    CHECKPOINT_2BIT_DIR,
    "Path to an exported 2-bit checkpoint directory (e.g. export_2bit/epoch_0p80 "
    "or chankhavu/ds-qat-image-v02-ep1-2bit).",
    allow_override=True,
)
_RESUMABLE_CKPT_DIR = flags.DEFINE_string(
    "resumable_ckpt_dir",
    "",
    "Optional path to a resumable QAT delta checkpoint (backbone_trainable.pt + "
    "expert_lora_rank_00..07.pt, e.g. step-00154-epoch_0p80 or "
    "chankhavu/ds-qat-image-v02-ep08). When set, loads --base_image_model_dir "
    "and overlays the QAT delta.",
    allow_override=True,
)
_BASE_IMAGE_MODEL_DIR = flags.DEFINE_string(
    "base_image_model_dir",
    BASE_IMAGE_MODEL_DIR,
    "Base 2-bit image model directory used when --resumable_ckpt_dir is set.",
    allow_override=True,
)
_ENGRAM_MODEL_DIR = flags.DEFINE_string(
    "engram_model_dir",
    BASE_MODEL_DIR,
    "Directory containing model-engram-*.safetensors and tokenizer.json.",
    allow_override=True,
)
_EXPORT_2BIT_OUT_DIR = flags.DEFINE_string(
    "export_2bit_out_dir",
    "",
    "Optional output directory to export a standalone 2-bit checkpoint after "
    "loading --resumable_ckpt_dir.",
    allow_override=True,
)
_ENABLE_QAT_EVAL = flags.DEFINE_bool(
    "enable_qat_eval",
    True,
    "Enable Havu-exact FP8/FP4 dynamic activation & KV quantization in forward pass.",
    allow_override=True,
)
_PROMPT = flags.DEFINE_string(
    "prompt",
    "The pattern in the training examples is",
    "Prompt for greedy generation demo.",
    allow_override=True,
)
_MAX_NEW_TOKENS = flags.DEFINE_integer(
    "max_new_tokens",
    256,
    "Maximum new tokens to decode greedily.",
    allow_override=True,
)


def load_model(
    ckpt_dir: str,
    device: torch.device,
    rank: int,
    world_size: int,
    engram_model_dir: str = BASE_MODEL_DIR,
    enable_qat: bool = False,
) -> dsv41_sft_model.Dsv41SftModel:
  """Builds the model skeleton for this rank and fills it from a 2-bit export."""
  cfg_path = os.path.join(ckpt_dir, "config.json")
  with gfile.Open(cfg_path, "r") as f:
    cfg = dsv41_sft_model.Dsv41E128Config.from_json_dict(json.load(f))
  cfg.routed_expert_lora_r = 0  # LoRA + QAT scales already merged into 2-bit experts
  cfg.enable_qat = enable_qat
  cfg.qat_trainable_expert_scales = False
  cfg.load_compact_engram = True
  cfg.max_seq_len = 163840

  with torch.device(device):
    model = dsv41_sft_model.Dsv41SftModel(
        cfg=cfg,
        ep_rank=rank,
        ep_world_size=world_size,
        gradient_checkpointing=False,
        ep_replicated_input=True,
    )
  stats = dsv41_sft_model.load_dequantized_dsv41_e128(
      model_dir=ckpt_dir,
      model=model,
      device=device,
      engram_model_dir=engram_model_dir,
  )
  if rank == 0:
    print(f"loaded {ckpt_dir}: {stats}", flush=True)
  assert stats["fp8_dequantized_tensors"] == 330, stats
  model.eval()
  return model


def load_model_from_resumable_qat_delta(
    base_model_dir: str,
    resumable_ckpt_dir: str,
    device: torch.device,
    rank: int,
    world_size: int,
    engram_model_dir: str = BASE_MODEL_DIR,
    export_2bit_out_dir: str = "",
) -> dsv41_sft_model.Dsv41SftModel:
  """Loads the base 2-bit image model and overlays a resumable QAT delta checkpoint.

  Restores `backbone_trainable.pt` (shared experts + vision aligner) and
  `expert_lora_rank_{ep_rank:02d}.pt` (routed-expert LoRA r=32 + trainable 2-bit
  group scales `qat_scale`, e.g. from `chankhavu/ds-qat-image-v02-ep08`).
  Optionally merges and exports a standalone 2-bit checkpoint when
  `export_2bit_out_dir` is provided.
  """
  cfg_path = os.path.join(base_model_dir, "config.json")
  with gfile.Open(cfg_path, "r") as f:
    cfg = dsv41_sft_model.Dsv41E128Config.from_json_dict(json.load(f))
  cfg.routed_expert_lora_r = 32
  cfg.routed_expert_lora_alpha = 64.0
  cfg.enable_qat = True
  cfg.qat_trainable_expert_scales = True
  cfg.load_compact_engram = True
  cfg.max_seq_len = 163840

  with torch.device(device):
    model = dsv41_sft_model.Dsv41SftModel(
        cfg=cfg,
        ep_rank=rank,
        ep_world_size=world_size,
        gradient_checkpointing=False,
        ep_replicated_input=True,
    )
  base_stats = dsv41_sft_model.load_dequantized_dsv41_e128(
      model_dir=base_model_dir,
      model=model,
      device=device,
      engram_model_dir=engram_model_dir,
  )
  train_dsv41_sft.configure_dsv41_trainable_parameters(
      model=model,
      train_moe_only=True,
      train_vision="aligner" if cfg.vision_enabled else "none",
  )
  resume_info = train_dsv41_sft.load_dsv41_resumable_checkpoint(
      ckpt_dir=resumable_ckpt_dir,
      model=model,
      rank=rank,
      world_size=world_size,
  )
  if rank == 0:
    print(
        f"loaded base {base_model_dir} ({base_stats}) + QAT delta "
        f"{resumable_ckpt_dir}: {resume_info}",
        flush=True,
    )
  if export_2bit_out_dir:
    export_summary = train_dsv41_sft.export_dsv41_e128_2bit_checkpoint(
        model=model,
        base_model_dir=base_model_dir,
        export_dir=export_2bit_out_dir,
        rank=rank,
        world_size=world_size,
    )
    if rank == 0:
      print(
          f"exported standalone 2-bit checkpoint to {export_2bit_out_dir}: "
          f"{export_summary}",
          flush=True,
      )
  model.eval()
  return model


@torch.no_grad()
def greedy_generate(
    model: dsv41_sft_model.Dsv41SftModel,
    tok: tokenizers.Tokenizer,
    prompt: str,
    device: torch.device,
    max_new_tokens: int = 256,
) -> str:
  """Prefill + incremental decode with the model's KV cache (`forward_hidden_cached`)."""
  ids = torch.tensor(
      [tok.encode(prompt, add_special_tokens=False).ids],
      dtype=torch.long,
      device=device,
  )
  kv_cache: dict[str, object] = {}
  hidden = model.forward_hidden_cached(ids, kv_cache)
  out: list[int] = []
  eos = tok.token_to_id("<｜end▁of▁sentence｜>")
  for _ in range(max_new_tokens):
    logits = model.head(hidden[:, -1:, :]).float()
    nxt = logits.argmax(dim=-1)
    if dist.is_initialized():
      dist.broadcast(nxt, src=0)
    tid = int(nxt.item())
    if tid == eos:
      break
    out.append(tid)
    hidden = model.forward_hidden_cached(nxt, kv_cache)
  return tok.decode(out, skip_special_tokens=True)


@torch.no_grad()
def greedy_generate_with_images(
    model: dsv41_sft_model.Dsv41SftModel,
    tok: tokenizers.Tokenizer,
    prompt_with_placeholders: str,
    image_png_blobs: Sequence[bytes],
    device: torch.device,
    max_new_tokens: int = 256,
) -> str:
  """Multimodal E058 `text+image` prefill + incremental KV-cached greedy decode.

  Each `<｜deepseek_image｜>` token in `prompt_with_placeholders` is expanded into
  its exact `[IMAGE_START] + ([IMAGE]*W + [IMAGE_NEW_LINE])*H + [IMAGE_END]` span
  and routed through the ViT + 3x3 aligner + `ffn.gate.bias_vl` expert router bias.
  """
  raw_ids = tok.encode(prompt_with_placeholders, add_special_tokens=False).ids
  exp_ids_np, tok_types_np, _, _ = dsv41_image_tokens.expand_image_placeholders(
      raw_ids, image_png_blobs
  )
  ids = torch.as_tensor(exp_ids_np, dtype=torch.long, device=device).unsqueeze(0)
  token_types = torch.as_tensor(
      tok_types_np, dtype=torch.int8, device=device
  ).unsqueeze(0)
  images: list[dsv41_sft_model.ImageInput] = []
  for blob in image_png_blobs:
    patches, plan = dsv41_image_tokens.preprocess_image(blob)
    images.append(
        dsv41_sft_model.ImageInput(
            patches=torch.from_numpy(patches).to(
                device=device, dtype=torch.bfloat16
            ),
            grid_hw=(plan.n_vit_h, plan.n_vit_w),
            token_plan=(plan.n_llm_h, plan.n_llm_w),
        )
    )
  kv_cache: dict[str, object] = {}
  hidden = model.forward_hidden_cached(
      ids, kv_cache, token_types=token_types, images=images
  )
  out: list[int] = []
  eos = tok.token_to_id("<｜end▁of▁sentence｜>")
  for _ in range(max_new_tokens):
    logits = model.head(hidden[:, -1:, :]).float()
    nxt = logits.argmax(dim=-1)
    if dist.is_initialized():
      dist.broadcast(nxt, src=0)
    tid = int(nxt.item())
    if tid == eos:
      break
    out.append(tid)
    hidden = model.forward_hidden_cached(nxt, kv_cache)
  return tok.decode(out, skip_special_tokens=True)


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError(f"Unexpected positional arguments: {argv[1:]}")
  rank = int(os.environ.get("RANK", "0"))
  world_size = int(os.environ.get("WORLD_SIZE", "1"))
  device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
  if device.type == "cuda":
    torch.cuda.set_device(device)
  if world_size > 1 and not dist.is_initialized():
    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size,
        device_id=device,
        timeout=datetime.timedelta(minutes=90),
    )

  if _RESUMABLE_CKPT_DIR.value:
    model = load_model_from_resumable_qat_delta(
        base_model_dir=_BASE_IMAGE_MODEL_DIR.value,
        resumable_ckpt_dir=_RESUMABLE_CKPT_DIR.value,
        device=device,
        rank=rank,
        world_size=world_size,
        engram_model_dir=_ENGRAM_MODEL_DIR.value,
        export_2bit_out_dir=_EXPORT_2BIT_OUT_DIR.value,
    )
  else:
    model = load_model(
        ckpt_dir=_CHECKPOINT_2BIT_DIR.value,
        device=device,
        rank=rank,
        world_size=world_size,
        engram_model_dir=_ENGRAM_MODEL_DIR.value,
        enable_qat=_ENABLE_QAT_EVAL.value,
    )

  tok_path = os.path.join(_ENGRAM_MODEL_DIR.value, "tokenizer.json")
  with gfile.Open(tok_path, "rb") as f:
    tok = tokenizers.Tokenizer.from_buffer(f.read())

  text = greedy_generate(
      model, tok, _PROMPT.value, device, max_new_tokens=_MAX_NEW_TOKENS.value
  )
  if rank == 0:
    print(text, flush=True)
  if dist.is_initialized():
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
  app.run(main)
