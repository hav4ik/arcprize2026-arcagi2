"""H100 GPU integration test for Google3 vLLM CUDA shim with Qwen3.5 (GatedDeltaNet)."""

from __future__ import annotations

import json
import os
import tempfile

from absl.testing import absltest
import torch

from google3.experimental.users.arazdai.kaggle import vllm_cuda_shim
from google3.experimental.users.arazdai.kaggle import vllm_fla_triton_ops


class VllmH100GpuTest(absltest.TestCase):

  @classmethod
  def setUpClass(cls) -> None:
    super().setUpClass()
    vllm_cuda_shim.install_google3_vllm_cuda_shim()

  def test_01_cuda_available_and_fla_triton_equivalence(self) -> None:
    """Verifies H100 GPU is active and FLA chunk prefill matches recurrent decode."""
    self.assertTrue(torch.cuda.is_available(), "CUDA GPU must be available")
    device = torch.device("cuda:0")
    print(f"[Test 1] Running on GPU: {torch.cuda.get_device_name(0)}")

    torch.manual_seed(42)
    T = 80
    H = 4
    HV = 12
    K = 128
    V = 128
    dtype = torch.bfloat16

    conv_dim = 2 * H * K + HV * V
    conv_output = torch.randn(T, conv_dim, device=device, dtype=dtype) * 0.1
    a = torch.randn(T, HV, device=device, dtype=dtype) * 0.1
    b = torch.randn(T, HV, device=device, dtype=dtype) * 0.1
    A_log = torch.randn(HV, device=device, dtype=torch.float32) * 0.1
    dt_bias = torch.ones(HV, device=device, dtype=torch.float32)

    q, k, v, g, beta = vllm_fla_triton_ops.fused_post_conv_prep(
        conv_output=conv_output,
        a=a,
        b=b,
        A_log=A_log,
        dt_bias=dt_bias,
        num_k_heads=H,
        head_k_dim=K,
        head_v_dim=V,
        apply_l2norm=True,
        output_g_exp=False,
    )
    cu_seqlens = torch.tensor([0, 50, 80], device=device, dtype=torch.int32)
    init_state_chunk = (
        torch.randn(2, HV, V, K, device=device, dtype=torch.float32) * 0.05
    )

    o_chunk, final_state_chunk = vllm_fla_triton_ops.chunk_gated_delta_rule(
        q=q.unsqueeze(0),
        k=k.unsqueeze(0),
        v=v.unsqueeze(0),
        g=g.unsqueeze(0),
        beta=beta.unsqueeze(0),
        initial_state=init_state_chunk.clone(),
        output_final_state=True,
        cu_seqlens=cu_seqlens,
        use_qk_l2norm_in_kernel=False,
    )

    # Compare against step-by-step recurrent kernel
    q_raw, k_raw, v_raw = torch.split(
        conv_output, [H * K, H * K, HV * V], dim=-1
    )
    q_raw = q_raw.reshape(1, T, H, K).contiguous()
    k_raw = k_raw.reshape(1, T, H, K).contiguous()
    v_raw = v_raw.reshape(1, T, HV, V).contiguous()
    init_state_rec = init_state_chunk.clone()
    ssm_indices = torch.tensor([0, 1], device=device, dtype=torch.int32)

    o_rec, final_state_rec = (
        vllm_fla_triton_ops.fused_sigmoid_gating_delta_rule_update(
            A_log=A_log,
            a=a,
            b=b,
            dt_bias=dt_bias,
            q=q_raw,
            k=k_raw,
            v=v_raw,
            initial_state=init_state_rec,
            inplace_final_state=True,
            cu_seqlens=cu_seqlens,
            ssm_state_indices=ssm_indices,
            use_qk_l2norm_in_kernel=True,
        )
    )

    max_diff_o = (o_chunk.float() - o_rec.float()).abs().max().item()
    max_diff_s = (
        (final_state_chunk.float() - final_state_rec.float()).abs().max().item()
    )
    print(f"[Test 1] max_diff_o={max_diff_o:.6f}, max_diff_state={max_diff_s:.6f}")
    self.assertLess(max_diff_o, 2e-2)
    self.assertLess(max_diff_s, 2e-2)

  def test_02_vllm_qwen3_5_end_to_end_gpu_generation(self) -> None:
    """Verifies vllm.LLM initializes Qwen3_5ForConditionalGeneration and generates on H100."""
    import vllm
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt

    with tempfile.TemporaryDirectory() as tmpdir:
      config_dict = {
          "architectures": ["Qwen3_5ForConditionalGeneration"],
          "model_type": "qwen3_5",
          "tie_word_embeddings": False,
          "image_token_id": 151655,
          "video_token_id": 151656,
          "vision_start_token_id": 151652,
          "vision_end_token_id": 151653,
          "vision_config": {
              "model_type": "qwen3_5",
              "depth": 2,
              "hidden_size": 128,
              "intermediate_size": 256,
              "num_heads": 4,
              "in_channels": 3,
              "patch_size": 16,
              "spatial_merge_size": 2,
              "temporal_patch_size": 2,
              "out_hidden_size": 512,
              "num_position_embeddings": 64,
          },
          "text_config": {
              "model_type": "qwen3_5_text",
              "vocab_size": 152064,
              "hidden_size": 512,
              "intermediate_size": 1024,
              "num_hidden_layers": 4,
              "num_attention_heads": 24,
              "num_key_value_heads": 4,
              "head_dim": 256,
              "hidden_act": "silu",
              "max_position_embeddings": 4096,
              "rms_norm_eps": 1e-6,
              "tie_word_embeddings": False,
              "linear_conv_kernel_dim": 4,
              "linear_key_head_dim": 128,
              "linear_value_head_dim": 128,
              "linear_num_key_heads": 16,
              "linear_num_value_heads": 48,
              "layer_types": [
                  "linear_attention",
                  "linear_attention",
                  "linear_attention",
                  "full_attention",
              ],
              "rope_parameters": {
                  "rope_type": "default",
                  "rope_theta": 1000000.0,
                  "partial_rotary_factor": 0.25,
                  "mrope_interleaved": True,
                  "mrope_section": [11, 11, 10],
              },
          },
      }
      with open(os.path.join(tmpdir, "config.json"), "w") as f:
        json.dump(config_dict, f)

      llm = vllm.LLM(
          model=tmpdir,
          load_format="dummy",
          skip_tokenizer_init=True,
          language_model_only=True,
          trust_remote_code=True,
          dtype="bfloat16",
          tensor_parallel_size=1,
          max_model_len=512,
          gpu_memory_utilization=0.3,
          enforce_eager=True,
          attention_backend="FLASH_ATTN",
          compilation_config={"custom_ops": ["none"]},
          additional_config={"gdn_prefill_backend": "triton"},
      )

      sampling_params = SamplingParams(
          temperature=0.0,
          max_tokens=12,
          ignore_eos=True,
      )
      prompts = [
          TokensPrompt(prompt_token_ids=[10, 20, 30, 40, 50, 60, 70, 80]),
          TokensPrompt(prompt_token_ids=[11, 22, 33, 44, 55]),
      ]
      outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
      self.assertLen(outputs, 2)
      for idx, out in enumerate(outputs):
        token_ids = list(out.outputs[0].token_ids)
        print(
            f"[Test 2] Prompt {idx} generated {len(token_ids)} token_ids:"
            f" {token_ids}"
        )
        self.assertLen(token_ids, 12)


if __name__ == "__main__":
  absltest.main()

