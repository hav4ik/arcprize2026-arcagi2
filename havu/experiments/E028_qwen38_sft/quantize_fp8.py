"""E028: FP8 post-training quantization of the merged Qwen3.8-27B SFT model with NVIDIA ModelOpt.

Recipe (mirrors nvidia/Qwen3.8-27B-NVFP4's hf_quant_config, but FP8 everywhere it quantizes):
- static per-tensor FP8 (E4M3) weights + activations (mtq.FP8_DEFAULT_CFG) on every language-model linear:
  self_attn q/k/v/o, linear_attn in_proj_qkv / in_proj_z / out_proj, mlp gate/up/down;
- kept in BF16: lm_head, the small linear-attention gate projections (in_proj_a / in_proj_b, also BF16 in
  NVIDIA's checkpoint), the vision tower, MTP layers;
- KV cache not quantized in the checkpoint (choose fp8 KV at serve time with --kv-cache-dtype).
Calibration: in-distribution windows of the E028 training sequences (calib.npy, int32 [n, seq_len]).
Export: modelopt.torch.export.export_hf_checkpoint (unified HF checkpoint, served by SGLang/vLLM as modelopt FP8).

  python quantize_fp8.py --model /workspace/models/Qwen3.8-27B-e028-sft --calib calib.npy \
      --out /workspace/models/Qwen3.8-27B-e028-sft-FP8
"""
import argparse
import copy
import json
import time

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--calib", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=0, help="use only the first n calibration windows")
    args = ap.parse_args()

    import modelopt
    import modelopt.torch.quantization as mtq
    from modelopt.torch.export import export_hf_checkpoint
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    t0 = time.time()
    model = AutoModelForImageTextToText.from_pretrained(args.model, dtype=torch.bfloat16, device_map="auto")
    model.eval()
    print(f"loaded in {time.time() - t0:.0f}s; modelopt {modelopt.__version__}", flush=True)

    cfg = copy.deepcopy(mtq.FP8_DEFAULT_CFG)
    # ModelOpt >= 0.46: quant_cfg is an ordered list of rules (later rules win). The default FP8 recipe already
    # disables lm_head, linear_attn.in_proj_a/b and conv1d, mtp.*, and the vision modules; these are belt-and-braces.
    for pat in ("*lm_head*", "*visual*", "*vision*", "*in_proj_a*", "*in_proj_b*", "*mtp*"):
        if isinstance(cfg["quant_cfg"], list):
            cfg["quant_cfg"].append({"quantizer_name": pat, "enable": False})
        else:
            cfg["quant_cfg"][pat] = {"enable": False}

    calib = np.load(args.calib)
    if args.n:
        calib = calib[:args.n]
    dev = next(model.parameters()).device

    def forward_loop(m):
        with torch.no_grad():
            for i, row in enumerate(calib):
                m(input_ids=torch.as_tensor(row, dtype=torch.long, device=dev)[None])
                if i % 16 == 0:
                    print(f"calib {i}/{len(calib)} {time.time() - t0:.0f}s", flush=True)

    t1 = time.time()
    model = mtq.quantize(model, cfg, forward_loop)
    print(f"quantized in {time.time() - t1:.0f}s", flush=True)
    mtq.print_quant_summary(model)

    export_hf_checkpoint(model, export_dir=args.out)
    AutoTokenizer.from_pretrained(args.model).save_pretrained(args.out)
    for f in ("chat_template.jinja", "generation_config.json", "preprocessor_config.json", "video_preprocessor_config.json"):
        try:
            import shutil
            shutil.copy(f"{args.model}/{f}", f"{args.out}/{f}")
        except FileNotFoundError:
            pass
    json.dump({"source": args.model, "calib": args.calib, "calib_windows": int(len(calib)),
               "calib_seq_len": int(calib.shape[1]), "modelopt": modelopt.__version__,
               "disabled": ["lm_head", "visual", "in_proj_a", "in_proj_b", "mtp"], "algo": "FP8_DEFAULT_CFG"},
              open(f"{args.out}/e028_quant_info.json", "w"), indent=1)
    print(f"exported to {args.out} in {time.time() - t0:.0f}s total", flush=True)


if __name__ == "__main__":
    main()
