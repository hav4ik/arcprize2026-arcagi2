"""Unit tests for Qwen3.8-27B ARC-AGI-2 SFT data pipeline, masking, and loss computation."""

from __future__ import annotations

import json
import os
import tempfile
from typing import Sequence

import numpy as np
import torch
import transformers

from google3.experimental.users.arazdai.kaggle.sft import data_pipeline
from google3.experimental.users.arazdai.kaggle.sft import export_and_quantize
from google3.experimental.users.arazdai.kaggle.sft import fast_deltanet
from google3.experimental.users.arazdai.kaggle.sft import train_full_sft
from google3.experimental.users.arazdai.kaggle.sft import train_sft
from google3.testing.pybase import googletest


class FakeQwenTokenizer:
    """Deterministic byte-level fake tokenizer with exact special tokens for unit testing."""

    IM_START_ID = 248044
    IM_END_ID = 248045
    ASST_ID = 77091
    USER_ID = 872
    SYS_ID = 8948
    NEWLINE_ID = 198

    SPECIAL_MAP = {
        "<|im_start|>": IM_START_ID,
        "<|im_end|>": IM_END_ID,
        "assistant": ASST_ID,
        "user": USER_ID,
        "system": SYS_ID,
    }

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        out: list[int] = []
        i = 0
        n = len(text)
        while i < n:
            matched = False
            for tok_str in ("<|im_start|>", "<|im_end|>"):
                if text.startswith(tok_str, i):
                    out.append(self.SPECIAL_MAP[tok_str])
                    i += len(tok_str)
                    # Check if immediately followed by role + \n
                    for role_str in ("assistant", "user", "system"):
                        if text.startswith(role_str + "\n", i):
                            out.append(self.SPECIAL_MAP[role_str])
                            out.append(self.NEWLINE_ID)
                            i += len(role_str) + 1
                            break
                    matched = True
                    break
            if matched:
                continue
            ch = text[i]
            if ch == "\n":
                out.append(self.NEWLINE_ID)
            else:
                out.append(1000 + ord(ch))
            i += 1
        return out


class SftPipelineTest(googletest.TestCase):

    def test_parent_puzzle_id_and_90_10_split_zero_leakage(self):
        records = []
        # Create 20 parent puzzles, each with 3 synthetic / pair derivatives
        for p_idx in range(20):
            parent = f"p{p_idx:04x}"
            src = "arc2_train" if p_idx < 10 else "nvarc_training"
            for v_idx in range(3):
                records.append(
                    {
                        "task_id": f"{parent}_aug{v_idx}" if src == "nvarc_training" else parent,
                        "parent": parent,
                        "pair": v_idx,
                        "source": src,
                    }
                )

        train_recs, val_recs, stats = data_pipeline.split_by_parent_puzzle_id(
            records, val_fraction=0.10, seed=42
        )
        self.assertEqual(stats["parent_overlap_count"], 0)
        self.assertEqual(stats["total_unique_parents"], 20)
        self.assertEqual(stats["val_unique_parents"], 2)
        self.assertEqual(stats["train_unique_parents"], 18)
        self.assertEqual(len(train_recs) + len(val_recs), 60)

        train_parents = {r["parent_id"] for r in train_recs}
        val_parents = {r["parent_id"] for r in val_recs}
        self.assertTrue(train_parents.isdisjoint(val_parents))

    def test_contamination_filter_catches_known_derivatives_and_eval_ids(self):
        eval_index = {"eval_task_ids": {"89565ca0", "12345678"}}
        is_contam, reason = data_pipeline.is_task_contaminated("2753e76c_abc123", "2753e76c", eval_index)
        self.assertTrue(is_contam)
        self.assertIn("2753e76c", reason)

        is_contam2, _ = data_pipeline.is_task_contaminated("81c0276b", None, eval_index)
        self.assertTrue(is_contam2)

        is_contam3, reason3 = data_pipeline.is_task_contaminated("12345678_syn", "12345678", eval_index)
        self.assertTrue(is_contam3)
        self.assertIn("direct_eval_task_id", reason3)

        is_contam4, _ = data_pipeline.is_task_contaminated("00576224_syn", "00576224", eval_index)
        self.assertFalse(is_contam4)

    def test_canonical_grid_hash_d8_and_color_invariance(self):
        grid = [[0, 1, 1], [2, 0, 1]]
        # Rotate 90 degrees and swap colors (0->5, 1->8, 2->3)
        rot90 = np.rot90(np.asarray(grid), 1)
        color_map = {0: 5, 1: 8, 2: 3}
        transformed = [[color_map[int(v)] for v in row] for row in rot90]

        h1 = data_pipeline.canonical_grid_hash(grid, invariant_d8_color=True)
        h2 = data_pipeline.canonical_grid_hash(transformed, invariant_d8_color=True)
        self.assertEqual(h1, h2)

        # Exact hash should differ
        e1 = data_pipeline.canonical_grid_hash(grid, invariant_d8_color=False)
        e2 = data_pipeline.canonical_grid_hash(transformed, invariant_d8_color=False)
        self.assertNotEqual(e1, e2)

    def test_segment_rendering_and_assistant_only_loss_mask(self):
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "Solve puzzle X."},
            {
                "role": "assistant",
                "reasoning_content": "Let's inspect the shapes.",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "python_exec",
                            "arguments": json.dumps({"code": "print(len(train))"}),
                        },
                    }
                ],
            },
            {"role": "tool", "content": "3"},
            {"role": "user", "content": "[Turn 1/40 used]"},
            {
                "role": "assistant",
                "reasoning_content": "Verified on all 3 train pairs. Submitting.",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_2",
                        "type": "function",
                        "function": {
                            "name": "submit_answer",
                            "arguments": json.dumps({"code": "answer = [[1, 2], [3, 4]]"}),
                        },
                    }
                ],
            },
        ]

        segments = data_pipeline.build_segments(messages, max_turns=40)
        tok = FakeQwenTokenizer()
        input_ids, loss_mask = data_pipeline.tokenize_and_verify_segments(
            segments=segments,
            tokenizer=tok,
            im_start_id=tok.IM_START_ID,
            im_end_id=tok.IM_END_ID,
            asst_role_id=tok.ASST_ID,
            newline_id=tok.NEWLINE_ID,
        )
        self.assertEqual(input_ids.shape, loss_mask.shape)
        self.assertGreater(int(loss_mask.sum()), 0)
        self.assertLess(int(loss_mask.sum()), len(input_ids))

        # Tampering with 1 masked position must trigger AssertionError in verify_token_loss_mask
        bad_mask = loss_mask.copy()
        bad_mask[0] = 1
        with self.assertRaises(AssertionError):
            data_pipeline.verify_token_loss_mask(
                input_ids=input_ids,
                loss_mask=bad_mask,
                im_start_id=tok.IM_START_ID,
                im_end_id=tok.IM_END_ID,
                asst_role_id=tok.ASST_ID,
                newline_id=tok.NEWLINE_ID,
            )

    def test_tiny_qwen35_chunked_causal_lm_loss_and_backward(self):
        text_cfg = transformers.Qwen3_5TextConfig(
            vocab_size=256,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            linear_conv_kernel_dim=4,
            linear_key_head_dim=16,
            linear_value_head_dim=16,
            linear_num_key_heads=2,
            linear_num_value_heads=2,
        )
        cfg = transformers.Qwen3_5Config(text_config=text_cfg)
        model = transformers.Qwen3_5ForConditionalGeneration(cfg)
        lang_model = model.model.language_model
        lm_head = model.lm_head

        input_ids = torch.randint(0, 256, (1, 32), dtype=torch.long)
        # Mask first 12 tokens (system/user), supervise last 20 tokens (assistant)
        labels = input_ids.clone()
        labels[:, :12] = -100

        out = lang_model(input_ids=input_ids, use_cache=False)
        loss_sum, n_valid = train_sft.compute_chunked_causal_lm_loss(
            hidden_states=out.last_hidden_state,
            lm_head=lm_head,
            labels=labels,
            chunk_size=8,
        )
        self.assertEqual(n_valid, 20)
        mean_loss = loss_sum / float(n_valid)

        # Compare against standard unchunked PyTorch cross_entropy
        full_logits = lm_head(out.last_hidden_state[:, :-1, :]).float()
        ref_loss = torch.nn.functional.cross_entropy(
            full_logits.reshape(-1, 256),
            labels[:, 1:].reshape(-1),
            ignore_index=-100,
            reduction="mean",
        )
        self.assertAlmostEqual(float(mean_loss.item()), float(ref_loss.item()), places=5)

        mean_loss.backward()
        self.assertIsNotNone(lm_head.weight.grad)

    def test_fast_deltanet_forward_and_backward_parity(self):
        from transformers.models.qwen3_5 import modeling_qwen3_5

        torch.manual_seed(42)
        b, seq_len, h, k_dim, v_dim = 1, 160, 2, 16, 16
        q_ref = torch.randn(b, seq_len, h, k_dim, dtype=torch.float32, requires_grad=True)
        k_ref = torch.randn(b, seq_len, h, k_dim, dtype=torch.float32, requires_grad=True)
        v_ref = torch.randn(b, seq_len, h, v_dim, dtype=torch.float32, requires_grad=True)
        g_ref = (-torch.rand(b, seq_len, h, dtype=torch.float32) * 0.2).detach().requires_grad_(True)
        beta_ref = torch.sigmoid(torch.randn(b, seq_len, h, dtype=torch.float32)).detach().requires_grad_(True)

        q_fast = q_ref.detach().clone().requires_grad_(True)
        k_fast = k_ref.detach().clone().requires_grad_(True)
        v_fast = v_ref.detach().clone().requires_grad_(True)
        g_fast = g_ref.detach().clone().requires_grad_(True)
        beta_fast = beta_ref.detach().clone().requires_grad_(True)

        orig_fn = modeling_qwen3_5.torch_chunk_gated_delta_rule
        out_ref, _ = orig_fn(
            q_ref, k_ref, v_ref, g=g_ref, beta=beta_ref, chunk_size=64, use_qk_l2norm_in_kernel=True
        )
        out_fast, _ = fast_deltanet.fast_triton_chunk_gated_delta_rule(
            q_fast, k_fast, v_fast, g=g_fast, beta=beta_fast, chunk_size=64, use_qk_l2norm_in_kernel=True
        )

        self.assertTrue(torch.allclose(out_ref, out_fast, atol=1e-5, rtol=1e-4))

        grad_out = torch.randn_like(out_ref)
        (out_ref * grad_out).sum().backward()
        (out_fast * grad_out).sum().backward()

        self.assertTrue(torch.allclose(q_ref.grad, q_fast.grad, atol=1e-5, rtol=1e-4))
        self.assertTrue(torch.allclose(k_ref.grad, k_fast.grad, atol=1e-5, rtol=1e-4))
        self.assertTrue(torch.allclose(v_ref.grad, v_fast.grad, atol=1e-5, rtol=1e-4))
        self.assertTrue(torch.allclose(g_ref.grad, g_fast.grad, atol=1e-5, rtol=1e-4))
        self.assertTrue(torch.allclose(beta_ref.grad, beta_fast.grad, atol=1e-5, rtol=1e-4))

    def test_validate_deltanet_backward_all_params_and_fallback(self):
        report = fast_deltanet.validate_deltanet_backward_all_params(
            device="cpu",
            dtype=torch.float32,
            cos_sim_threshold=0.9999,
            rel_l2_threshold=1e-3,
        )
        expected_params = {
            "in_proj_qkv.weight",
            "in_proj_z.weight",
            "in_proj_b.weight",
            "in_proj_a.weight",
            "conv1d.weight",
            "dt_bias",
            "A_log",
            "norm.weight",
            "out_proj.weight",
        }
        self.assertEqual(set(report.keys()), expected_params)
        for p_name, p_info in report.items():
            self.assertTrue(p_info["has_grad"], f"{p_name} missing grad")
            self.assertTrue(p_info["finite"], f"{p_name} grad not finite")
            self.assertTrue(p_info["passed"], f"{p_name} parity check failed: {p_info}")

    def test_full_sft_causal_lm_excludes_visual_and_updates_all_text_groups(self):
        text_cfg = transformers.Qwen3_5TextConfig(
            vocab_size=256,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            linear_conv_kernel_dim=4,
            linear_key_head_dim=16,
            linear_value_head_dim=16,
            linear_num_key_heads=2,
            linear_num_value_heads=2,
        )
        cfg = transformers.Qwen3_5Config(text_config=text_cfg)
        full_model = transformers.Qwen3_5ForConditionalGeneration(cfg)
        self.assertTrue(hasattr(full_model.model, "visual"))

        sft_model = train_full_sft.FullSFTCausalLM(full_model)
        self.assertFalse(hasattr(full_model.model, "visual"))
        sft_model = train_full_sft.apply_fsdp2_full_sharding(
            sft_model, device=torch.device("cpu"), world_size=1
        )

        for name, p in sft_model.named_parameters():
            self.assertTrue(p.requires_grad)
            self.assertEqual(p.dtype, torch.float32)
            self.assertNotIn("visual", name)

        groups = train_full_sft.categorize_parameter_groups(sft_model)
        for g_name, items in groups.items():
            self.assertGreater(len(items), 0, f"Expected non-empty group {g_name}")

        initial_snap = train_full_sft.snapshot_group_local_clones(groups)
        optimizer = torch.optim.AdamW(
            sft_model.parameters(), lr=1e-3, betas=(0.9, 0.95), weight_decay=0.01
        )

        input_ids = torch.randint(0, 256, (1, 24), dtype=torch.long)
        labels = input_ids.clone()
        labels[:, :8] = -100

        loss_sum, n_valid = sft_model(input_ids=input_ids, labels=labels, chunk_size=16)
        self.assertEqual(n_valid, 16)
        (loss_sum / float(n_valid)).backward()
        grad_norm = train_full_sft.clip_fsdp2_grad_norm_(
            list(sft_model.parameters()), max_norm=1.0, device=torch.device("cpu")
        )
        self.assertTrue(np.isfinite(grad_norm))
        self.assertGreater(grad_norm, 0.0)
        optimizer.step()

        update_norms = train_full_sft.compute_group_update_norms(
            groups, initial_snap, device=torch.device("cpu")
        )
        for g_name, norm_val in update_norms.items():
            self.assertGreater(norm_val, 0.0, f"Expected group {g_name} to update")

    def test_partial_batch_schedule_and_distributed_checkpoint_save_reload_export(self):
        # 15 trajectories across 4 ranks => 3 full batches of 4 + 1 partial batch of 3 (with 1 None on rank 3)
        order = np.arange(15, dtype=np.int64)
        batches = train_full_sft.build_epoch_batches(
            order=order,
            effective_batch=4,
            world_size=4,
            micro_batch_size=1,
            grad_accum_steps=1,
        )
        self.assertEqual(len(batches), 4)
        self.assertTrue(all(x is not None for x in batches[0]))
        self.assertTrue(all(x is not None for x in batches[1]))
        self.assertTrue(all(x is not None for x in batches[2]))
        self.assertEqual(sum(1 for x in batches[3] if x is not None), 3)
        self.assertIsNone(batches[3][3])

        # Verify all 15 unique trajectory indices appear exactly once
        seen = [x for b in batches for x in b if x is not None]
        self.assertEqual(len(seen), 15)
        self.assertEqual(set(seen), set(range(15)))

        # Verify distributed checkpoint save -> incomplete ignore -> reload -> export_and_quantize reconstruction
        text_cfg = transformers.Qwen3_5TextConfig(
            vocab_size=128,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            head_dim=16,
            linear_conv_kernel_dim=4,
            linear_key_head_dim=16,
            linear_value_head_dim=16,
            linear_num_key_heads=2,
            linear_num_value_heads=2,
        )
        cfg = transformers.Qwen3_5Config(text_config=text_cfg)
        full_model = transformers.Qwen3_5ForConditionalGeneration(cfg)
        sft_model = train_full_sft.FullSFTCausalLM(full_model)
        sft_model = train_full_sft.apply_fsdp2_full_sharding(
            sft_model, device=torch.device("cpu"), world_size=1
        )

        optimizer = torch.optim.AdamW(sft_model.parameters(), lr=5e-6)
        rng = np.random.default_rng(20260405)

        input_ids = torch.randint(0, 128, (1, 16), dtype=torch.long)
        labels = input_ids.clone()
        loss_sum, n_valid = sft_model(input_ids=input_ids, labels=labels, chunk_size=16)
        (loss_sum / float(n_valid)).backward()
        optimizer.step()

        with tempfile.TemporaryDirectory(prefix="full_sft_ckpt_test_") as tmp_dir:
            ckpt_root = os.path.join(tmp_dir, "checkpoints")
            step2_dir = train_full_sft.save_distributed_checkpoint(
                model=sft_model,
                optimizer=optimizer,
                ckpt_root_dir=ckpt_root,
                ckpt_name="step-00002",
                rank=0,
                world_size=1,
                step=2,
                epoch=0,
                batch_idx_in_epoch=2,
                order=order,
                scheduler_meta={"total_steps": 4, "warmup_steps": 1, "min_lr_ratio": 0.1},
                rng=rng,
                num_parts=2,
                keep_last_n=1,
                milestone_every_steps=0,
                save_optimizer_state=False,
                save_weights_dtype="bfloat16",
            )
            self.assertTrue(os.path.exists(os.path.join(step2_dir, "COMPLETED.json")))
            part0 = torch.load(
                os.path.join(step2_dir, "rank_00_part_00.pt"),
                map_location="cpu",
                weights_only=False,
            )
            self.assertEqual(part0["optimizer"], {})

            # Create an incomplete step-00003 directory without COMPLETED.json
            incomplete_dir = os.path.join(ckpt_root, "step-00003")
            os.makedirs(incomplete_dir, exist_ok=True)
            with open(os.path.join(incomplete_dir, "rank_00_part_00.pt"), "wb") as f:
                f.write(b"partial_corrupt_shard")

            found = train_full_sft.find_latest_completed_checkpoint(
                ckpt_root_dir=ckpt_root, expected_world_size=1
            )
            self.assertIsNotNone(found)
            latest_dir, latest_meta = found
            self.assertEqual(latest_dir, step2_dir)

            # Mutate model parameters and reload from step2_dir
            saved_weight = sft_model.lm_head.weight.detach().to(torch.bfloat16).to(torch.float32).clone()
            with torch.no_grad():
                sft_model.lm_head.weight.zero_()
            resumed_state = train_full_sft.load_distributed_checkpoint(
                model=sft_model,
                optimizer=optimizer,
                ckpt_dir=latest_dir,
                meta=latest_meta,
                rank=0,
                device=torch.device("cpu"),
                rng=rng,
            )
            self.assertEqual(resumed_state["step"], 2)
            self.assertEqual(resumed_state["batch_idx_in_epoch"], 2)
            self.assertTrue(torch.allclose(sft_model.lm_head.weight, saved_weight))

            # Verify reconstruct_full_sft_fp32_weights maps keys to HF format
            reconstructed = export_and_quantize.reconstruct_full_sft_fp32_weights(step2_dir)
            self.assertIn("lm_head.weight", reconstructed)
            self.assertIn("model.language_model.embed_tokens.weight", reconstructed)
            self.assertTrue(torch.allclose(reconstructed["lm_head.weight"], saved_weight))

            # Verify keep_last_n=1, milestone_every_steps=0 prunes step-00002 when step-00004 completes
            os.remove(os.path.join(incomplete_dir, "rank_00_part_00.pt"))
            os.rmdir(incomplete_dir)
            step4_dir = train_full_sft.save_distributed_checkpoint(
                model=sft_model,
                optimizer=optimizer,
                ckpt_root_dir=ckpt_root,
                ckpt_name="step-00004",
                rank=0,
                world_size=1,
                step=4,
                epoch=0,
                batch_idx_in_epoch=4,
                order=order,
                scheduler_meta={"total_steps": 4, "warmup_steps": 1, "min_lr_ratio": 0.1},
                rng=rng,
                num_parts=2,
                keep_last_n=1,
                milestone_every_steps=0,
                save_optimizer_state=False,
                save_weights_dtype="bfloat16",
            )
            self.assertFalse(os.path.exists(step2_dir))
            self.assertTrue(os.path.exists(step4_dir))

    def test_lora_and_optimizer_dtypes_and_fp32_merge_quantize(self):
        import peft

        text_cfg = transformers.Qwen3_5TextConfig(
            vocab_size=256,
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=32,
            linear_conv_kernel_dim=4,
            linear_key_head_dim=32,
            linear_value_head_dim=32,
            linear_num_key_heads=2,
            linear_num_value_heads=2,
        )
        cfg = transformers.Qwen3_5Config(text_config=text_cfg)
        base_model = transformers.Qwen3_5ForConditionalGeneration(cfg).to(torch.bfloat16)
        lora_cfg = peft.LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "in_proj_qkv",
                "in_proj_z",
                "in_proj_b",
                "in_proj_a",
                "out_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            bias="none",
            task_type=peft.TaskType.CAUSAL_LM,
        )
        model = peft.get_peft_model(base_model, lora_cfg)
        trainable = [p for p in model.parameters() if p.requires_grad]
        frozen = [p for p in model.parameters() if not p.requires_grad]

        # 1. Verify frozen base weights are bfloat16 and trainable LoRA weights are float32
        self.assertTrue(all(p.dtype == torch.bfloat16 for p in frozen))
        self.assertTrue(all(p.dtype == torch.float32 for p in trainable))

        optimizer = torch.optim.AdamW(trainable, lr=1e-4)
        input_ids = torch.randint(0, 256, (1, 16), dtype=torch.long)
        out = model.base_model.model.model.language_model(
            input_ids=input_ids, use_cache=False
        )
        loss = out.last_hidden_state.float().pow(2).mean()
        loss.backward()
        optimizer.step()

        # 2. Verify optimizer states (exp_avg, exp_avg_sq) and parameters after step are float32
        for p in trainable:
            self.assertEqual(p.dtype, torch.float32)
            st = optimizer.state[p]
            self.assertEqual(st["exp_avg"].dtype, torch.float32)
            self.assertEqual(st["exp_avg_sq"].dtype, torch.float32)

        # 3. Verify extract_lora_pairs and merge_lora_weight_fp32 preserve sub-BF16 updates
        w_base = torch.ones((128, 128), dtype=torch.bfloat16)
        # Delta of +0.002 is smaller than BF16 ULP at 1.0 (2^-7 = 0.0078125), so 1.0_bf16 + 0.002_bf16 == 1.0_bf16!
        lora_a = torch.ones((8, 128), dtype=torch.float32) * 0.01
        lora_b = torch.ones((128, 8), dtype=torch.float32) * 0.0125
        # scaling = 2.0 => scaling * (B @ A) = 2.0 * (8 * 0.01 * 0.0125) = +0.002
        w_merged_fp32 = export_and_quantize.merge_lora_weight_fp32(
            w_base, lora_a, lora_b, scaling=2.0
        )
        self.assertEqual(w_merged_fp32.dtype, torch.float32)
        self.assertAlmostEqual(float(w_merged_fp32[0, 0].item()), 1.002, places=6)

        # Whereas naive BF16 merge rounds 1.002 back to 1.000!
        w_naive_bf16 = w_base + (2.0 * (lora_b @ lora_a)).to(torch.bfloat16)
        self.assertEqual(float(w_naive_bf16[0, 0].item()), 1.0)

        # And quantizing from FP32 preserves the updated scale in weight_scale_inv!
        _, scale_fp32 = export_and_quantize.quantize_weight_block128(w_merged_fp32, block_size=128)
        _, scale_naive = export_and_quantize.quantize_weight_block128(w_naive_bf16, block_size=128)
        self.assertGreater(float(scale_fp32.item()), float(scale_naive.item()))

    def test_top10k_v8_rollout_row_e031_schema_and_qwen_sft_tokenization(self):
        import pyarrow as pa
        import pyarrow.parquet as pq

        messages = [
            {
                "role": "system",
                "content": "Reasoning Effort: Minimal. Keep your thinking short and terse.",
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "run_python",
                            "description": "Execute a Python cell in your persistent IPython session.",
                            "parameters": {
                                "type": "object",
                                "properties": {"code": {"type": "string"}},
                                "required": ["code"],
                            },
                        },
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "submit_answer",
                            "description": "End this attempt with your test output.",
                            "parameters": {
                                "type": "object",
                                "properties": {"code": {"type": "string"}},
                                "required": ["code"],
                            },
                        },
                    },
                ],
            },
            {"role": "user", "content": "Solve ARC puzzle 00576224_syn0."},
            {
                "role": "assistant",
                "content": "",
                "reasoning_content": "Let's test a 3x3 tiling rule on all train pairs.",
                "tool_calls": [
                    {
                        "id": "call_1_0",
                        "type": "function",
                        "function": {
                            "name": "run_python",
                            "arguments": json.dumps(
                                {"code": "def solve(g):\n    return g\nprint(all(solve(p['input'])==p['output'] for p in train))"}
                            ),
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_1_0",
                "content": "[completed]\nTrue\n[budget: 1k output tokens used; aim to finish within 40k (hard limit 80k)]",
            },
            {
                "role": "assistant",
                "content": "",
                "reasoning_content": "All training pairs match. Submitting solve(test_input).",
                "tool_calls": [
                    {
                        "id": "call_2_0",
                        "type": "function",
                        "function": {
                            "name": "submit_answer",
                            "arguments": json.dumps({"code": "solve(test_input)"}),
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_2_0",
                "content": "[answer accepted: 2x2 grid]",
            },
        ]
        events = [
            {
                "turn": 1,
                "output_tokens": 42,
                "thinking_tokens": 18,
                "tool_use_tokens": 24,
                "answer_tokens": 0,
                "finish_reason": "stop",
                "num_tool_calls": 1,
                "is_canonical": True,
            },
            {
                "turn": 2,
                "output_tokens": 28,
                "thinking_tokens": 14,
                "tool_use_tokens": 14,
                "answer_tokens": 0,
                "finish_reason": "stop",
                "num_tool_calls": 1,
                "is_canonical": True,
            },
        ]

        verified_row = {
            "task_id": "00576224_syn0",
            "batch": "top10k_v8_pass1",
            "source": "nvarc",
            "parent": "00576224",
            "pair": 0,
            "attempt": 0,
            "correct": True,
            "train_pass": True,
            "train_frac": 1.0,
            "answer_source": "submit",
            "stop_reason": "submitted",
            "error": None,
            "pred": "[[1,2],[3,4]]",
            "turns": 2,
            "tool_calls": 2,
            "prompt_tokens": 1200,
            "cache_hit_tokens": 0,
            "completion_tokens": 70,
            "seconds": 3.5,
            "est_cost_usd": 0.0,
            "model": "DeepSeek-V4.1-Flash-1017 (Full-Precision BF16->FP32)",
            "provider": "ds41_full_precision",
            "messages": json.dumps(messages),
            "events": json.dumps(events),
            "test_exact": True,
            "level3_pass": True,
            "level3_frac": 1.0,
            "num_level3_pairs": 25,
            "all_3_levels_verified": True,
            "verified_clean_sft": True,
        }
        overfit_row = dict(verified_row)
        overfit_row["task_id"] = "00576224_syn1"
        overfit_row["correct"] = False  # Failed Level-3 held-out generator pairs
        overfit_row["level3_pass"] = False
        overfit_row["all_3_levels_verified"] = False
        overfit_row["verified_clean_sft"] = False

        with tempfile.TemporaryDirectory(prefix="top10k_v8_test_") as tmp_dir:
            jsonl_path = os.path.join(tmp_dir, "rollouts_shard_000of064.jsonl")
            parquet_path = os.path.join(tmp_dir, "rollouts_shard_000of064.parquet")
            with open(jsonl_path, "w") as jf:
                jf.write(json.dumps(verified_row) + "\n")
                jf.write(json.dumps(overfit_row) + "\n")
            pq.write_table(pa.Table.from_pylist([verified_row, overfit_row]), parquet_path)

            eval_index = {"eval_task_ids": {"89565ca0"}}
            for path in (jsonl_path, parquet_path):
                stats, recs = data_pipeline.audit_rollout_file(path, eval_index)
                self.assertEqual(stats["rows"], 2)
                self.assertEqual(stats["train_pass_count"], 2)
                self.assertEqual(stats["correct_count"], 1)
                self.assertEqual(stats["verified_clean_count"], 1)
                self.assertEqual(len(recs), 1)
                self.assertEqual(recs[0]["task_id"], "00576224_syn0")
                self.assertEqual(recs[0]["parent_id"], "00576224")

                segments = data_pipeline.build_segments(recs[0]["messages"], max_turns=40)
                tok = FakeQwenTokenizer()
                input_ids, loss_mask = data_pipeline.tokenize_and_verify_segments(
                    segments=segments,
                    tokenizer=tok,
                    im_start_id=tok.IM_START_ID,
                    im_end_id=tok.IM_END_ID,
                    asst_role_id=tok.ASST_ID,
                    newline_id=tok.NEWLINE_ID,
                )
                self.assertGreater(int(loss_mask.sum()), 0)


if __name__ == "__main__":
    googletest.main()
