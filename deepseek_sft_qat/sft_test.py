"""Unit tests for Qwen3.8-27B ARC-AGI-2 SFT data pipeline, masking, and loss computation."""

from __future__ import annotations

import io
import json
import os
import pickle
import shutil
import tempfile
from typing import Sequence

import numpy as np
from PIL import Image
import torch
import transformers

from google3.experimental.users.arazdai.kaggle.sft import data_pipeline
from google3.experimental.users.arazdai.kaggle.sft import dsv41_image_tokens
from google3.experimental.users.arazdai.kaggle.sft import dsv41_sft_model
from google3.experimental.users.arazdai.kaggle.sft import evaluate_difficulty_argon
from google3.experimental.users.arazdai.kaggle.sft import export_and_quantize
from google3.experimental.users.arazdai.kaggle.sft import fast_deltanet
from google3.experimental.users.arazdai.kaggle.sft import train_dsv41_sft
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

    def test_compute_example_loss_weight_and_load_initial_lora_adapter(self):
        import peft

        # 1. Verify gamma == 0 returns uniform 1.0
        self.assertEqual(
            train_sft.compute_example_loss_weight(1.2, 0.6, gamma=0.0), 1.0
        )
        # 2. Verify higher loss gets higher weight and clamping works
        w_low = train_sft.compute_example_loss_weight(
            0.40, 0.80, gamma=1.5, min_weight=0.25, max_weight=3.0
        )
        w_mid = train_sft.compute_example_loss_weight(
            0.80, 0.80, gamma=1.5, min_weight=0.25, max_weight=3.0
        )
        w_high = train_sft.compute_example_loss_weight(
            1.20, 0.80, gamma=1.5, min_weight=0.25, max_weight=3.0
        )
        self.assertLess(w_low, w_mid)
        self.assertAlmostEqual(w_mid, 1.0, places=5)
        self.assertGreater(w_high, w_mid)
        self.assertAlmostEqual(w_high, 1.5 ** 1.5, places=5)

        w_clamped_min = train_sft.compute_example_loss_weight(
            0.05, 0.80, gamma=1.5, min_weight=0.25, max_weight=3.0
        )
        w_clamped_max = train_sft.compute_example_loss_weight(
            5.0, 0.80, gamma=1.5, min_weight=0.25, max_weight=3.0
        )
        self.assertEqual(w_clamped_min, 0.25)
        self.assertEqual(w_clamped_max, 3.0)

        # 3. Verify load_initial_lora_adapter restores saved PEFT weights
        text_cfg = transformers.Qwen3_5TextConfig(
            vocab_size=128,
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
        lora_cfg = peft.LoraConfig(
            r=4,
            lora_alpha=8,
            target_modules=["in_proj_qkv", "out_proj"],
            bias="none",
            task_type="CAUSAL_LM",
        )
        m1 = peft.get_peft_model(
            transformers.Qwen3_5ForConditionalGeneration(cfg), lora_cfg
        )
        with torch.no_grad():
            for n, p in m1.named_parameters():
                if "lora_" in n:
                    p.fill_(0.125)

        with tempfile.TemporaryDirectory(prefix="init_lora_test_") as tmp_dir:
            m1.save_pretrained(tmp_dir)
            m2 = peft.get_peft_model(
                transformers.Qwen3_5ForConditionalGeneration(cfg), lora_cfg
            )
            train_sft.load_initial_lora_adapter(m2, tmp_dir)
            for (n1, p1), (n2, p2) in zip(
                m1.named_parameters(), m2.named_parameters()
            ):
                if "lora_" in n1:
                    self.assertEqual(n1, n2)
                    self.assertTrue(torch.allclose(p1, p2))

    def test_argon_difficulty_prompt_parsing_and_filtering(self):
        candidate = {
            "puzzle_id": "nvarc_syn_001",
            "family_id": "nvarc_syn",
            "source_dataset": "NVARC-Synthetic-103k",
            "empirical_difficulty": {
                "bucket_3level": "0/2",
                "pass_rate_3level": 0.0,
                "has_direct_empirical": True,
            },
            "puzzle": {
                "train": [
                    {"input": [[0, 1], [2, 0]], "output": [[1, 0], [0, 2]]}
                ],
                "test": [{"input": [[0, 3], [4, 0]], "output": [[3, 0], [0, 4]]}],
            },
        }
        rollout_row = {
            "task_id": "nvarc_syn_001",
            "turns": 6,
            "tool_calls": 5,
            "prompt_tokens": 9000,
            "completion_tokens": 5000,
            "thinking_tokens": 3500,
            "train_pass": True,
            "correct": True,
            "all_3_levels_verified": True,
            "submitted_solve_code": "def solve(g):\n    return [row[::-1] for row in g]\n",
        }
        prompt = evaluate_difficulty_argon.build_argon_difficulty_prompt(
            candidate, rollout_row
        )
        self.assertIn("Qwen3.8-27B", prompt)
        self.assertIn("nvarc_syn_001", prompt)
        self.assertIn("def solve(g):", prompt)

        raw_json = """```json
        {
          "qwen38_difficulty_score": 8,
          "algorithmic_complexity": "hard_multi_step",
          "qwen38_failure_modes_targeted": ["multi_step_composition", "marker_relative_indexing"],
          "transformation_rule_summary": "Marker-anchored reflection with parity shift.",
          "solution_code_quality": 5,
          "sft_recommend_include": true,
          "sft_priority_tier": "tier1_high_value_hard_for_qwen",
          "qwen38_sft_rationale": "Targets multi-step marker-relative coordinate reasoning."
        }
        ```"""
        parsed = evaluate_difficulty_argon.parse_argon_json_response(raw_json)
        self.assertEqual(parsed["qwen38_difficulty_score"], 8)
        self.assertTrue(parsed["sft_recommend_include"])
        score = evaluate_difficulty_argon.compute_composite_sft_value_score(
            parsed, rollout_row=rollout_row, candidate=candidate
        )
        self.assertGreaterEqual(score, 90.0)

        with tempfile.TemporaryDirectory(prefix="argon_filter_test_") as tmp_dir:
            ratings_path = os.path.join(tmp_dir, "argon_ratings.jsonl")
            with open(ratings_path, "w") as f:
                f.write(
                    json.dumps(
                        {
                            "puzzle_id": "nvarc_syn_001",
                            "family_id": "nvarc_syn",
                            **parsed,
                            "sft_value_score": score,
                        }
                    )
                    + "\n"
                )
                f.write(
                    json.dumps(
                        {
                            "puzzle_id": "trivial_1d_001",
                            "family_id": "trivial_1d",
                            "qwen38_difficulty_score": 2,
                            "sft_recommend_include": False,
                            "sft_priority_tier": "tier3_skip_trivial_or_redundant",
                            "sft_value_score": 12.0,
                        }
                    )
                    + "\n"
                )
            ratings_map = data_pipeline.load_argon_ratings(ratings_path)
            records = [
                {"task_id": "nvarc_syn_001", "parent": "nvarc_syn"},
                {"task_id": "trivial_1d_001", "parent": "trivial_1d"},
                {"task_id": "official_arc_001", "parent": "official_arc_001"},
            ]
            kept, fstats = data_pipeline.filter_records_by_argon_difficulty(
                records,
                argon_ratings=ratings_map,
                min_qwen38_difficulty=6,
                max_qwen38_difficulty=9,
                only_argon_recommended=True,
                keep_unrated_official_arc=True,
            )
            self.assertEqual(len(kept), 2)
            self.assertEqual(fstats["kept_hard_for_qwen38"], 1)
            self.assertEqual(fstats["dropped_trivial_or_low_diff"], 1)
            self.assertEqual(fstats["kept_unrated_official"], 1)

    def test_dsv41_dequant_2bit_quant_and_differentiable_sft_forward_backward(self):
        torch.manual_seed(1234)
        # 1. Verify FP8 block32 quantize -> dequantize round-trip
        w_orig = torch.randn((32, 64), dtype=torch.float32) * 0.25
        w_fp8, s_fp8 = dsv41_sft_model.quantize_fp8_block32_torch(w_orig, block_size=32)
        self.assertEqual(w_fp8.dtype, torch.float8_e4m3fn)
        self.assertEqual(s_fp8.dtype, torch.float8_e8m0fnu)
        self.assertEqual(tuple(s_fp8.shape), (32, 2))
        w_deq = dsv41_sft_model.dequantize_fp8_block32_torch(
            w_fp8, s_fp8, out_dtype=torch.float32
        )
        rel_err_fp8 = float(
            (w_deq - w_orig).norm() / w_orig.norm().clamp(min=1e-6)
        )
        self.assertLess(rel_err_fp8, 0.15)

        # 2. Verify 2-bit Humming uint2 quantize -> dequantize round-trip & shapes
        # Construct an exact 4-level grid per group so 2-bit quantization is near-exact
        n_out, k_in, g_size = 32, 128, 64
        q_levels = torch.randint(0, 4, (n_out, k_in // g_size, g_size), dtype=torch.float32)
        # Ensure both 0 and 3 appear in each group so scale recovery is exact
        q_levels[:, :, 0] = 0.0
        q_levels[:, :, 1] = 3.0
        scales_true = torch.full((n_out, k_in // g_size, 1), 0.125, dtype=torch.float32)
        zp_true = torch.full((n_out, k_in // g_size, 1), 1.0, dtype=torch.float32)
        w_4level = ((q_levels - zp_true) * scales_true).reshape(n_out, k_in)

        q_p, s_p, z_p = dsv41_sft_model.quantize_uint2_humming_expert_tensor(
            w_4level, group=g_size
        )
        self.assertEqual(tuple(q_p.shape), (n_out, k_in // 16))
        self.assertEqual(tuple(s_p.shape), (n_out, k_in // g_size))
        self.assertEqual(tuple(z_p.shape), (n_out // 16, k_in // g_size))
        self.assertEqual(q_p.dtype, torch.int32)
        self.assertEqual(s_p.dtype, torch.bfloat16)
        self.assertEqual(z_p.dtype, torch.int32)

        w_2bit_deq = dsv41_sft_model.dequant_gptq_expert_tensor(
            q_p, s_p, z_p, group=g_size, out_dtype=torch.float32
        )
        self.assertEqual(tuple(w_2bit_deq.shape), (n_out, k_in))
        self.assertTrue(torch.allclose(w_2bit_deq, w_4level, atol=2e-3, rtol=2e-2))

        # 3. Verify differentiable Dsv41SftModel forward + backward with token-level loss_weights
        cfg = dsv41_sft_model.Dsv41E128Config.tiny_test_config()
        model = dsv41_sft_model.Dsv41SftModel(
            cfg=cfg,
            ep_rank=0,
            ep_world_size=1,
            gradient_checkpointing=True,
        )
        # Initialize LoRA B with non-zero weights so w1/w2/w3 base weights also receive gradients
        with torch.no_grad():
            for n, p in model.named_parameters():
                if "lora_B" in n or "experts." in n:
                    p.normal_(0.0, 0.02)

        seq_len = 24
        tokens = torch.randint(0, cfg.vocab_size, (1, seq_len), dtype=torch.long)
        labels = tokens.clone()
        labels[:, :6] = -100  # mask prompt prefix
        loss_weights = torch.ones((1, seq_len), dtype=torch.float32)
        loss_weights[:, :6] = 0.0
        loss_weights[:, 10:14] = 10.0  # WIC-alnum tokens
        loss_weights[:, 18:] = 5.0     # final submit turn tokens

        hidden = model.forward_hidden(tokens)
        loss_sum, n_valid = train_sft.compute_chunked_causal_lm_loss(
            hidden_states=hidden,
            lm_head=model.head,
            labels=labels,
            chunk_size=16,
            loss_weights=loss_weights,
        )
        self.assertEqual(n_valid, seq_len - 6)
        mean_loss = loss_sum / float(n_valid)
        self.assertTrue(torch.isfinite(mean_loss))
        mean_loss.backward()

        # Verify gradients flowed into backbone, attention, shared experts, and routed expert LoRA
        self.assertIsNotNone(model.head.weight.grad)
        self.assertIsNotNone(model.layers[0].attn.wq_a.weight.grad)
        self.assertIsNotNone(model.layers[0].ffn.shared_experts.w1.weight.grad)
        lora_grads = [
            p.grad
            for n, p in model.named_parameters()
            if "lora_B" in n and p.grad is not None
        ]
        self.assertGreater(len(lora_grads), 0)

        # 4. Verify milestone step computation (0.25, 0.50, 0.75, 1.00 epoch) and optimizer parameter groups
        milestones = train_dsv41_sft.compute_milestone_steps(
            steps_per_epoch=114,
            total_steps=114,
            fractions_csv="0.25,0.50,0.75,1.00",
        )
        self.assertEqual(
            milestones,
            {
                28: "epoch_0p25",
                57: "epoch_0p50",
                86: "epoch_0p75",
                114: "epoch_1p00",
            },
        )
        groups, p_stats = train_dsv41_sft.build_dsv41_optimizer_param_groups(
            model, weight_decay=0.01, lr_mults={"lora": 5.0, "scale": 2.5}
        )
        # Groups are split by LR tier x decay: every group carries `tier` + `lr_mult`.
        self.assertGreaterEqual(len(groups), 2)
        tiers = {g["tier"] for g in groups}
        self.assertIn("base", tiers)
        for g in groups:
            self.assertIn(g["tier"], ("base", "lora", "scale", "vision", "aligner"))
            expected_mult = {"lora": 5.0, "scale": 2.5}.get(g["tier"], 1.0)
            self.assertEqual(g["lr_mult"], expected_mult)
            if g["tier"] == "scale":
                self.assertEqual(g["weight_decay"], 0.0)  # qat_scale never decays
        self.assertGreater(p_stats["decay_tensors"], 0)
        self.assertGreater(p_stats["no_decay_tensors"], 0)
        n_all = sum(len(g["params"]) for g in groups)
        self.assertEqual(n_all, sum(1 for p in model.parameters() if p.requires_grad))

        # 5. Verify build_length_bucketed_step_batches on a synthetic PackedTrajectoryDataset
        with tempfile.TemporaryDirectory(prefix="dsv41_bucket_test_") as tmp_dir:
            npz_path = os.path.join(tmp_dir, "synth_tokens.npz")
            lengths = np.array([12, 24, 18, 30, 16], dtype=np.int32)
            offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
            total_tok = int(offsets[-1])
            np.savez(
                npz_path,
                input_ids=np.ones(total_tok, dtype=np.int32),
                loss_mask=np.ones(total_tok, dtype=np.uint8),
                loss_weights=np.ones(total_tok, dtype=np.float32),
                offsets=offsets,
                lengths=lengths,
            )
            ds = train_sft.PackedTrajectoryDataset(npz_path, max_seq_len=64)
            step_batches = train_dsv41_sft.build_length_bucketed_step_batches(
                dataset=ds,
                seed=42,
                epoch=0,
                effective_batch=4,
                world_size=2,
                grad_accum=2,
                mega_batch_multiplier=2,
            )
            self.assertEqual(len(step_batches), 2)
            self.assertEqual(len(step_batches[0]), 4)
            self.assertEqual(len(step_batches[1]), 4)
            real_seen = [idx for b in step_batches for idx in b if idx is not None]
            self.assertCountEqual(real_seen, [0, 1, 2, 3, 4])

    def test_dsv41_moe_only_freezing_and_loss_decrease(self):
        torch.manual_seed(2026)
        cfg = dsv41_sft_model.Dsv41E128Config.tiny_test_config()
        cfg.routed_expert_lora_r = 8
        cfg.routed_expert_lora_alpha = 16.0
        cfg.trainable_routed_experts_dense = False
        model = dsv41_sft_model.Dsv41SftModel(
            cfg=cfg,
            ep_rank=0,
            ep_world_size=1,
            gradient_checkpointing=True,
        )
        # Initialize expert base weights so routed expert forward/backward is non-trivial
        with torch.no_grad():
            for n, p in model.named_parameters():
                if "experts." in n and "lora_" not in n:
                    p.normal_(0.0, 0.02)

        freeze_stats = train_dsv41_sft.configure_dsv41_trainable_parameters(
            model, train_moe_only=True
        )
        self.assertGreater(freeze_stats["trainable_tensors"], 0)
        self.assertGreater(freeze_stats["frozen_tensors"], 0)

        # Verify attention, ngrams (engram), router gate, embed, head, mHC, and norms are all frozen
        for name, p in model.named_parameters():
            if any(
                k in name
                for k in (
                    "embed.",
                    "head.",
                    ".attn.",
                    ".engram.",
                    ".ffn.gate.",
                    ".hc_",
                    "norm",
                )
            ):
                self.assertFalse(
                    p.requires_grad, f"Expected {name} to be frozen when train_moe_only=True"
                )
            elif ".ffn.shared_experts." in name or (
                ".ffn.experts." in name and "lora_" in name
            ):
                self.assertTrue(
                    p.requires_grad, f"Expected MoE parameter {name} to be trainable"
                )

        param_groups, p_stats = train_dsv41_sft.build_dsv41_optimizer_param_groups(
            model, weight_decay=0.01
        )
        self.assertGreater(p_stats["decay_tensors"], 0)
        optimizer = torch.optim.AdamW(param_groups, lr=5e-3)

        tokens = torch.randint(0, cfg.vocab_size, (1, 20), dtype=torch.long)
        labels = tokens.clone()
        labels[:, :4] = -100
        loss_weights = torch.ones((1, 20), dtype=torch.float32)
        loss_weights[:, 8:12] = 10.0

        losses: list[float] = []
        for _ in range(4):
            optimizer.zero_grad(set_to_none=True)
            hidden = model.forward_hidden(tokens)
            loss_sum, n_valid = train_sft.compute_chunked_causal_lm_loss(
                hidden_states=hidden,
                lm_head=model.head,
                labels=labels,
                chunk_size=16,
                loss_weights=loss_weights,
            )
            loss = loss_sum / float(n_valid)
            losses.append(float(loss.detach().item()))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], max_norm=1.0
            )
            optimizer.step()

        # Verify training loss strictly decreases across the 4 MoE-only steps
        self.assertLess(losses[-1], losses[0])

        # Verify resumable checkpoint saving succeeds
        with tempfile.TemporaryDirectory(prefix="dsv41_moe_ckpt_") as tmp_dir:
            ckpt_dir = train_dsv41_sft.save_dsv41_resumable_checkpoint(
                output_dir=tmp_dir,
                step=4,
                epoch=0,
                step_in_epoch=3,
                model=model,
                optimizer=optimizer,
                ema_loss=losses[-1],
                best_val_loss=losses[-1],
                metrics_history=[{"step": 4, "train_loss": losses[-1]}],
                rank=0,
                world_size=1,
                milestone_tag="epoch_0p25",
            )
            self.assertTrue(os.path.exists(os.path.join(ckpt_dir, "COMPLETED.json")))
            self.assertTrue(
                os.path.exists(os.path.join(ckpt_dir, "backbone_trainable.pt"))
            )
            self.assertTrue(
                os.path.exists(os.path.join(ckpt_dir, "expert_lora_rank_00.pt"))
            )

            # Resume round-trip (auto_resume after a Borg preemption/restart):
            # (a) discovery picks the highest step that has COMPLETED.json + all weight files; a
            #     marker-less directory (preempted mid-save: at v12 step 57 all files existed but the
            #     backbone was truncated) and a directory with missing files are both skipped;
            self.assertEqual(
                train_dsv41_sft.parse_checkpoint_step(ckpt_dir), (4, "epoch_0p25")
            )
            self.assertIsNone(train_dsv41_sft.parse_checkpoint_step("export_2bit"))
            self.assertTrue(train_dsv41_sft.model_has_expert_checkpoint_state(model))
            with open(os.path.join(tmp_dir, "metrics.jsonl"), "w") as f:
                for s in (0, 4, 6, 8):
                    f.write(json.dumps({"step": s, "train_loss": 1.0 / (s + 1)}) + "\n")
            # Take two more optimizer steps, save as step 6 without the marker.
            for _ in range(2):
                optimizer.zero_grad(set_to_none=True)
                hidden = model.forward_hidden(tokens)
                loss_sum, n_valid = train_sft.compute_chunked_causal_lm_loss(
                    hidden_states=hidden,
                    lm_head=model.head,
                    labels=labels,
                    chunk_size=16,
                    loss_weights=loss_weights,
                )
                (loss_sum / float(n_valid)).backward()
                optimizer.step()
            ckpt6 = train_dsv41_sft.save_dsv41_resumable_checkpoint(
                output_dir=tmp_dir,
                step=6,
                epoch=0,
                step_in_epoch=5,
                model=model,
                optimizer=optimizer,
                ema_loss=0.5,
                best_val_loss=0.4,
                metrics_history=[{"step": s, "train_loss": 1.0 / (s + 1)} for s in (0, 4, 6, 8)],
                rank=0,
                world_size=1,
            )
            self.assertEqual(
                train_dsv41_sft.find_latest_resumable_checkpoint(
                    tmp_dir, world_size=1, has_expert_state=True
                ),
                ckpt6,
            )
            os.remove(os.path.join(ckpt6, "COMPLETED.json"))
            partial_dir = os.path.join(tmp_dir, "step-00008")
            os.makedirs(partial_dir)
            with open(os.path.join(partial_dir, "backbone_trainable.pt"), "wb") as f:
                f.write(b"")  # expert file missing -> incomplete, must be skipped
            with open(os.path.join(partial_dir, "COMPLETED.json"), "w") as f:
                f.write("{}")
            latest = train_dsv41_sft.find_latest_resumable_checkpoint(
                tmp_dir, world_size=1, has_expert_state=True
            )
            self.assertEqual(latest, ckpt_dir)  # step 4: marker + all files; 6 (no marker) and 8 (partial) skipped
            self.assertIsNone(
                train_dsv41_sft.find_latest_resumable_checkpoint(
                    os.path.join(tmp_dir, "does_not_exist"), 1, True
                )
            )

            # (b) loading into a fresh model restores every trainable tensor exactly and
            #     leaves frozen tensors untouched;
            torch.manual_seed(7)
            fresh = dsv41_sft_model.Dsv41SftModel(
                cfg=cfg, ep_rank=0, ep_world_size=1, gradient_checkpointing=True
            )
            train_dsv41_sft.configure_dsv41_trainable_parameters(
                fresh, train_moe_only=True
            )
            frozen_before = {
                n: p.detach().clone()
                for n, p in fresh.named_parameters()
                if not p.requires_grad
            }
            info = train_dsv41_sft.load_dsv41_resumable_checkpoint(
                ckpt_dir=ckpt6, model=fresh, rank=0, world_size=1
            )
            self.assertEqual(info["step"], 6)
            self.assertEqual(info["milestone_tag"], "")
            self.assertNotIn("ema_loss", info)  # marker was removed
            src_params = dict(model.named_parameters())
            n_checked = 0
            for n, p in fresh.named_parameters():
                if p.requires_grad:
                    self.assertTrue(torch.equal(p.detach(), src_params[n].detach()), n)
                    n_checked += 1
                else:
                    self.assertTrue(torch.equal(p.detach(), frozen_before[n]), n)
            self.assertEqual(n_checked, info["loaded_tensors"])
            self.assertGreater(n_checked, 0)
            info4 = train_dsv41_sft.load_dsv41_resumable_checkpoint(
                ckpt_dir=ckpt_dir, model=fresh, rank=0, world_size=1
            )
            self.assertEqual(info4["step"], 4)
            self.assertAlmostEqual(info4["ema_loss"], losses[-1], places=6)

            # (b2) also works when directory basename is a HuggingFace repo name
            # (e.g. `ds-qat-image-v02-ep08`) rather than `step-XXXXX`, reading step from COMPLETED.json.
            hf_style_dir = os.path.join(tmp_dir, "ds-qat-image-v02-ep08")
            shutil.copytree(ckpt_dir, hf_style_dir)
            info_hf = train_dsv41_sft.load_dsv41_resumable_checkpoint(
                ckpt_dir=hf_style_dir, model=fresh, rank=0, world_size=1
            )
            self.assertEqual(info_hf["step"], 4)
            self.assertEqual(info_hf["milestone_tag"], "epoch_0p25")

            from google3.experimental.users.arazdai.kaggle.sft import load_dsv41_2bit_example
            self.assertIn("round2_v14_16xb200_fast", load_dsv41_2bit_example.CHECKPOINT_2BIT_DIR)

            # (c) metrics history is reloaded up to the resume step;
            self.assertEqual(
                [r["step"] for r in train_dsv41_sft.load_metrics_history(tmp_dir, 6)],
                [0, 4, 6],
            )
            # (d) a partial checkpoint is rejected rather than silently resumed.
            with self.assertRaises(Exception):
                train_dsv41_sft.load_dsv41_resumable_checkpoint(
                    ckpt_dir=partial_dir, model=fresh, rank=0, world_size=1
                )

    def test_eval_dsv41_b32_arc2_run_havu_b32_shard_without_vllm(self):
        from google3.experimental.users.arazdai.kaggle.arc_agi_2.pilot_2 import havu_b32_harness
        from google3.experimental.users.arazdai.kaggle.arc_agi_2.pilot_2 import prompts_and_selector
        from google3.experimental.users.arazdai.kaggle.sft import eval_dsv41_b32_arc2

        eval_dsv41_b32_arc2._install_vllm_sampling_params_shim()
        import vllm  # type: ignore

        self.assertEqual(
            vllm.sampling_params.RequestOutputKind.CUMULATIVE, "CUMULATIVE"
        )
        sp = vllm.SamplingParams(temperature=1.0, top_p=1.0, top_k=-1, max_tokens=64)
        self.assertEqual(sp.max_tokens, 64)

        cfg = dsv41_sft_model.Dsv41E128Config.tiny_test_config()
        model = dsv41_sft_model.Dsv41SftModel(
            cfg=cfg,
            ep_rank=0,
            ep_world_size=1,
            gradient_checkpointing=False,
            ep_replicated_input=True,
        ).eval()

        class _DummyTok:
            eos_token_id = 1

            def encode(self, s: str, add_special_tokens: bool = False) -> list[int]:
                del add_special_tokens
                if s == havu_b32_harness.DS41_EOS_TOKEN:
                    return [1]
                return [min(cfg.vocab_size - 1, max(2, ord(c) % cfg.vocab_size)) for c in s[:16]] or [2]

            def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:
                del skip_special_tokens
                if ids:
                    return '</think>\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name="submit_answer">\n<｜DSML｜ parameter name="code" string="true">[[0]]</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n</｜DSML｜ calls><｜end▁of▁sentence｜>'
                return "thinking..."

        engine = eval_dsv41_b32_arc2.Dsv41DistributedStepEngine(
            model=model,
            tokenizer=_DummyTok(),  # type: ignore[arg-type]
            device=torch.device("cpu"),
            max_model_len=2048,
        )
        with tempfile.TemporaryDirectory() as tmp:
            ds_dir = os.path.join(tmp, "eval_data")
            out_dir = os.path.join(tmp, "eval_out")
            os.makedirs(ds_dir, exist_ok=True)
            with open(os.path.join(ds_dir, "00000001.json"), "w") as f:
                json.dump(
                    {
                        "train": [{"input": [[0]], "output": [[0]]}],
                        "test": [{"input": [[0]], "output": [[0]]}],
                    },
                    f,
                )
            base_cfg = prompts_and_selector.PILOT2_HYPOTHESES["havu_b32"]
            eval_cfg = prompts_and_selector.Pilot2HypothesisConfig(
                **{
                    **base_cfg.__dict__,
                    "exp_id": "dsv41_e128_b32_epoch_0p25",
                    "title": "DeepSeek-V4.1 E128 B32 Test",
                    "num_attempts": 1,
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "top_k": -1,
                }
            )
            recs = havu_b32_harness.run_havu_b32_shard(
                llm=engine,
                cfg=eval_cfg,
                dataset_dir=ds_dir,
                out_dir=out_dir,
                shard_index=0,
                num_shards=1,
                max_tasks=1,
                model_path="/tmp/dsv41_e128",
            )
            self.assertEqual(len(recs), 1)
            self.assertTrue(recs[0]["pass1"])

    def test_dsv41_qat_qad_nemotron_and_kv_cached_parity(self):
        torch.manual_seed(20261029)
        # 1. Verify 2D [out//32, in//32] FP8 block-32 quantize/dequantize & STE
        w_orig = torch.randn((64, 96), dtype=torch.float32) * 0.2
        w_fp8_2d, s_fp8_2d = dsv41_sft_model.quantize_fp8_block32_torch(
            w_orig, block_size=32, scale_2d=True
        )
        self.assertEqual(tuple(s_fp8_2d.shape), (2, 3))
        w_deq_2d = dsv41_sft_model.dequantize_fp8_block32_torch(
            w_fp8_2d, s_fp8_2d, out_dtype=torch.float32
        )
        self.assertEqual(tuple(w_deq_2d.shape), (64, 96))

        # 2. Verify locked-z_orig 2-bit Humming zero-degradation round-trip & train-to-export parity
        n_out, k_in, g_size = 32, 128, 64
        q_init = torch.randint(0, 4, (n_out, k_in // g_size, g_size), dtype=torch.uint8)
        z_init = torch.randint(0, 4, (n_out, k_in // g_size), dtype=torch.uint8)
        s_init = (torch.rand((n_out, k_in // g_size), dtype=torch.bfloat16) * 0.08 + 0.02)
        w_init = (
            (q_init.to(torch.float32) - z_init.to(torch.float32).unsqueeze(-1))
            * s_init.to(torch.float32).unsqueeze(-1)
        ).reshape(n_out, k_in).to(torch.bfloat16)

        q_packed_orig = dsv41_sft_model.pack_uint2_torch(
            q_init.reshape(n_out, k_in)
        )
        z_packed_orig = (
            dsv41_sft_model.pack_uint2_torch(z_init.T.contiguous()).T.contiguous()
        )

        lora_lin = dsv41_sft_model.LoRALinear(
            in_features=k_in,
            out_features=n_out,
            lora_r=8,
            lora_alpha=16.0,
            group=g_size,
            enable_qat=True,
            trainable_scale=True,
            lock_gptq_codes=False,
        )
        lora_lin.set_gptq_state(w_init, s_init, z_packed_orig)

        # At delta W == 0, export_humming_2bit must be 100% bit-for-bit identical to Havu's GPTQ checkpoint
        q_exp0, s_exp0, z_exp0 = lora_lin.export_humming_2bit()
        self.assertTrue(torch.equal(q_exp0, q_packed_orig))
        self.assertTrue(torch.equal(s_exp0, s_init))
        self.assertTrue(torch.equal(z_exp0, z_packed_orig))
        _, _, _, st0 = lora_lin.export_humming_2bit_with_stats()
        self.assertEqual(st0["codes_changed"], 0)
        self.assertEqual(st0["scales_changed"], 0)
        self.assertEqual(st0["codes_total"], n_out * k_in)
        self.assertEqual(st0["scales_total"], n_out * (k_in // g_size))

        # After non-zero LoRA + qat_scale updates, forward QAT weight must match exported 2-bit dequant exactly
        with torch.no_grad():
            lora_lin.lora_B.normal_(0.0, 0.05)
            lora_lin.qat_scale.mul_(1.07)
        q_exp1, s_exp1, z_exp1 = lora_lin.export_humming_2bit()
        self.assertTrue(torch.equal(z_exp1, z_packed_orig))  # z_orig remains locked!
        _, _, _, st1 = lora_lin.export_humming_2bit_with_stats()
        self.assertGreater(st1["scales_changed"], 0)  # +7% on s moves every BF16 scale
        self.assertGreater(st1["codes_changed"], 0)  # LoRA delta of this size crosses s/2 somewhere
        self.assertEqual(st1["scales_changed"], int((s_exp1 != s_init).sum()))
        w_deq_exp1 = dsv41_sft_model.dequant_gptq_expert_tensor(
            q_exp1, s_exp1, z_exp1, group=g_size, out_dtype=torch.bfloat16
        )
        x_probe = torch.randn((2, k_in), dtype=torch.bfloat16)
        x_probe_q = dsv41_sft_model.act_quant_fp8_ste(x_probe, block_size=32)
        y_qat = lora_lin(x_probe)
        y_exp = torch.nn.functional.linear(x_probe_q, w_deq_exp1)
        self.assertTrue(torch.allclose(y_qat, y_exp, atol=1e-3, rtol=1e-3))

        # 2b. Non-QAT (plain bf16 + LoRA SFT) must NOT materialize the int8 `d_orig` cache (it cost
        # ~22.6 GB/rank and OOM'd the v11b SFT run), yet must export bit-identically to the QAT path.
        self.assertGreater(lora_lin.d_orig.numel(), 0)
        lora_plain = dsv41_sft_model.LoRALinear(
            in_features=k_in,
            out_features=n_out,
            lora_r=8,
            lora_alpha=16.0,
            group=g_size,
            enable_qat=False,
        )
        lora_plain.set_gptq_state(w_init, s_init, z_packed_orig)
        self.assertTrue(lora_plain.has_gptq_state)
        self.assertEqual(lora_plain.d_orig.numel(), 0)
        q_plain0, s_plain0, z_plain0 = lora_plain.export_humming_2bit()
        self.assertEqual(lora_plain.d_orig.numel(), 0)  # export must not populate the cache either
        self.assertTrue(torch.equal(q_plain0, q_packed_orig))
        self.assertTrue(torch.equal(s_plain0, s_init))
        self.assertTrue(torch.equal(z_plain0, z_packed_orig))
        with torch.no_grad():
            lora_plain.lora_A.copy_(lora_lin.lora_A.to(lora_plain.lora_A.dtype))
            lora_plain.lora_B.copy_(lora_lin.lora_B.to(lora_plain.lora_B.dtype))
        lora_ref = dsv41_sft_model.LoRALinear(
            in_features=k_in,
            out_features=n_out,
            lora_r=8,
            lora_alpha=16.0,
            group=g_size,
            enable_qat=True,
            trainable_scale=False,
        )
        lora_ref.set_gptq_state(w_init, s_init, z_packed_orig)
        with torch.no_grad():
            lora_ref.lora_A.copy_(lora_plain.lora_A.float())
            lora_ref.lora_B.copy_(lora_plain.lora_B.float())
        self.assertGreater(lora_ref.d_orig.numel(), 0)
        q_plain1, s_plain1, z_plain1 = lora_plain.export_humming_2bit()
        q_ref1, s_ref1, z_ref1 = lora_ref.export_humming_2bit()
        self.assertTrue(torch.equal(q_plain1, q_ref1))
        self.assertTrue(torch.equal(s_plain1, s_ref1))
        self.assertTrue(torch.equal(z_plain1, z_ref1))

        # 3. Verify Nemotron QAT/QAD forward + backward with frozen teacher reference and MoE-only training
        cfg = dsv41_sft_model.Dsv41E128Config.tiny_test_config()
        cfg.enable_qat = True
        cfg.qat_trainable_expert_scales = True
        cfg.qat_lock_gptq_codes = False
        model = dsv41_sft_model.Dsv41SftModel(
            cfg=cfg,
            ep_rank=0,
            ep_world_size=1,
            gradient_checkpointing=True,
        )
        # Populate synthetic GPTQ states on all routed experts and non-zero LoRA B
        with torch.no_grad():
            for layer in model.layers:
                for exp in layer.ffn.experts:
                    for proj in (exp.w1, exp.w2, exp.w3):
                        proj.weight.normal_(0.0, 0.05)
                        q_p, s_p, z_p = dsv41_sft_model.quantize_uint2_humming_expert_tensor(
                            proj.weight, group=proj.group
                        )
                        w_deq = dsv41_sft_model.dequant_gptq_expert_tensor(
                            q_p, s_p, z_p, group=proj.group, out_dtype=torch.bfloat16
                        )
                        proj.set_gptq_state(w_deq, s_p, z_p)
                        proj.lora_B.normal_(0.0, 0.02)
        model.snapshot_teacher_weights()

        freeze_stats = train_dsv41_sft.configure_dsv41_trainable_parameters(
            model, train_moe_only=True
        )
        self.assertGreater(freeze_stats["qat_scale_params"], 0)
        self.assertEqual(
            model.layers[0].ffn.shared_experts.w1.weight.dtype, torch.float32
        )

        param_groups, p_stats = train_dsv41_sft.build_dsv41_optimizer_param_groups(
            model, weight_decay=0.01
        )
        self.assertGreater(p_stats["no_decay_tensors"], 0)
        optimizer = torch.optim.AdamW(param_groups, lr=5e-3)

        tokens = torch.randint(0, cfg.vocab_size, (1, 16), dtype=torch.long)
        labels = tokens.clone()
        labels[:, :4] = -100
        loss_weights = torch.ones((1, 16), dtype=torch.float32)

        with torch.no_grad():
            model.set_teacher_mode(True)
            teacher_hidden = model.forward_hidden(tokens).detach()
            model.set_teacher_mode(False)

        student_hidden = model.forward_hidden(tokens)
        total_loss_sum, n_valid, kl_sum = (
            train_dsv41_sft.compute_chunked_qat_qad_loss(
                student_hidden=student_hidden,
                teacher_hidden=teacher_hidden,
                lm_head=model.head,
                labels=labels,
                chunk_size=8,
                loss_weights=loss_weights,
                qat_kl_weight=0.2,
                qat_kl_temperature=1.0,
            )
        )
        self.assertEqual(n_valid, 12)
        self.assertGreater(float(total_loss_sum.item()), 0.0)
        self.assertGreaterEqual(float(kl_sum), 0.0)
        (total_loss_sum / float(n_valid)).backward()

        # Attention & Engram must have None grad; MoE shared_experts, lora_A/B, and qat_scale must have finite grad
        self.assertIsNone(model.layers[0].attn.wq_a.weight.grad)
        self.assertIsNotNone(model.layers[0].ffn.shared_experts.w1.weight.grad)
        exp0 = model.layers[0].ffn.experts[0]
        self.assertIsNotNone(exp0.w1.lora_A.grad)
        self.assertIsNotNone(exp0.w1.lora_B.grad)
        self.assertIsNotNone(exp0.w1.qat_scale.grad)
        optimizer.step()

        # 4. Verify incremental KV-cached forward_hidden_cached parity with uncached forward_hidden
        model.eval()
        model.set_teacher_mode(True)
        with torch.no_grad():
            seq_tokens = torch.randint(0, cfg.vocab_size, (1, 12), dtype=torch.long)
            full_h = model.forward_hidden(seq_tokens)
            kv_cache = model.create_kv_cache()
            # Prefill first 8 tokens, then decode 4 tokens one-by-one
            h_prefill = model.forward_hidden_cached(seq_tokens[:, :8], kv_cache)
            h_steps = [h_prefill]
            for pos in range(8, 12):
                h_step = model.forward_hidden_cached(
                    seq_tokens[:, pos : pos + 1], kv_cache
                )
                h_steps.append(h_step)
            cached_h = torch.cat(h_steps, dim=1)
            self.assertEqual(kv_cache["seqlen"], 12)
            self.assertTrue(
                torch.allclose(full_h, cached_h, atol=2e-2, rtol=2e-2)
            )

            # 4b. Verify batched multi-stream decode (`begin/step/end_batched_decode`) matches
            # independent single-stream decode: partial window (6 < window_size=8), mixed
            # ratio-2 compressor phases (even vs odd start), candidate-source layer, and
            # continued single-stream decoding on the written-back caches afterwards.
            seq_a = torch.randint(0, cfg.vocab_size, (1, 16), dtype=torch.long)
            seq_b = torch.randint(0, cfg.vocab_size, (1, 20), dtype=torch.long)
            pre_a, pre_b, n_batched, n_tail = 6, 11, 5, 2
            cache_a_single = model.create_kv_cache()
            cache_b_single = model.create_kv_cache()
            _ = model.forward_hidden_cached(seq_a[:, :pre_a], cache_a_single)
            _ = model.forward_hidden_cached(seq_b[:, :pre_b], cache_b_single)

            cache_a_batch = model.create_kv_cache()
            cache_b_batch = model.create_kv_cache()
            _ = model.forward_hidden_cached(seq_a[:, :pre_a], cache_a_batch)
            _ = model.forward_hidden_cached(seq_b[:, :pre_b], cache_b_batch)

            session = model.begin_batched_decode(
                [cache_a_batch, cache_b_batch], max_steps=n_batched
            )
            for step_i in range(n_batched):
                tok_a = seq_a[:, pre_a + step_i : pre_a + step_i + 1]
                tok_b = seq_b[:, pre_b + step_i : pre_b + step_i + 1]
                ha_s = model.forward_hidden_cached(tok_a, cache_a_single)
                hb_s = model.forward_hidden_cached(tok_b, cache_b_single)
                h_ab = model.step_batched_decode(
                    torch.cat([tok_a, tok_b], dim=0), session
                )
                self.assertTrue(
                    torch.allclose(ha_s, h_ab[0:1], atol=1e-2, rtol=1e-2),
                    f"step {step_i} stream a: {(ha_s - h_ab[0:1]).abs().max().item()}",
                )
                self.assertTrue(
                    torch.allclose(hb_s, h_ab[1:2], atol=1e-2, rtol=1e-2),
                    f"step {step_i} stream b: {(hb_s - h_ab[1:2]).abs().max().item()}",
                )
            model.end_batched_decode(session)
            self.assertEqual(cache_a_batch["seqlen"], pre_a + n_batched)
            self.assertEqual(cache_b_batch["seqlen"], pre_b + n_batched)

            # Written-back caches must be usable by the single-stream path afterwards
            for step_i in range(n_tail):
                pa = pre_a + n_batched + step_i
                pb = pre_b + n_batched + step_i
                tok_a = seq_a[:, pa : pa + 1]
                tok_b = seq_b[:, pb : pb + 1]
                ha_s = model.forward_hidden_cached(tok_a, cache_a_single)
                hb_s = model.forward_hidden_cached(tok_b, cache_b_single)
                ha_c = model.forward_hidden_cached(tok_a, cache_a_batch)
                hb_c = model.forward_hidden_cached(tok_b, cache_b_batch)
                self.assertTrue(torch.allclose(ha_s, ha_c, atol=1e-2, rtol=1e-2))
                self.assertTrue(torch.allclose(hb_s, hb_c, atol=1e-2, rtol=1e-2))

            # One-shot convenience wrapper (begin + 1 step + end) must agree too
            pa = pre_a + n_batched + n_tail
            pb = pre_b + n_batched + n_tail
            tok_a = seq_a[:, pa : pa + 1]
            tok_b = seq_b[:, pb : pb + 1]
            ha_s = model.forward_hidden_cached(tok_a, cache_a_single)
            hb_s = model.forward_hidden_cached(tok_b, cache_b_single)
            h_ab = model.forward_hidden_cached_batch(
                torch.cat([tok_a, tok_b], dim=0), [cache_a_batch, cache_b_batch]
            )
            self.assertTrue(torch.allclose(ha_s, h_ab[0:1], atol=1e-2, rtol=1e-2))
            self.assertTrue(torch.allclose(hb_s, h_ab[1:2], atol=1e-2, rtol=1e-2))

        # 5. Verify _load_compact_engram_from_shards with CNS single-integer `rows_{r0}` keys and shard 00009 row_map
        with tempfile.TemporaryDirectory() as eng_tmp:
            from safetensors.torch import save_file

            eng_blk = next(b for b in model.layers if b.engram is not None)
            eng_lid = eng_blk.layer_id
            eng_mod = eng_blk.engram
            assert eng_mod is not None
            t_rows = eng_mod.table_rows
            h_dim = eng_mod.head_dim
            p_vocab = eng_mod.part_vocab_idx_end - eng_mod.vocab_start_idx if hasattr(eng_mod, "part_vocab_idx_end") else eng_mod.part_vocab_size

            # Shard 1: scale, Shard 2: rows_0, Shard 9: row_map
            save_file(
                {
                    f"layers.{eng_lid}.engram.embed.scale": torch.full(
                        (t_rows, h_dim // 32), 127, dtype=torch.uint8
                    ),
                },
                os.path.join(eng_tmp, "model-engram-00001-of-00016.safetensors"),
            )
            save_file(
                {
                    f"layers.{eng_lid}.engram.embed.weight.rows_0": torch.ones(
                        (t_rows, h_dim), dtype=torch.float32
                    ).to(torch.float8_e4m3fn),
                },
                os.path.join(eng_tmp, "model-engram-00002-of-00016.safetensors"),
            )
            save_file(
                {
                    f"layers.{eng_lid}.engram.embed_tokens.row_map": torch.zeros(
                        (p_vocab,), dtype=torch.int32
                    ),
                },
                os.path.join(eng_tmp, "model-engram-00009-of-00016.safetensors"),
            )
            loaded_slices = dsv41_sft_model._load_compact_engram_from_shards(
                eng_tmp, model, torch.device("cpu"), False, 0, 1
            )
            self.assertEqual(loaded_slices, 3)
            self.assertTrue(eng_mod.has_compact_table)

            # 6. With the compact Engram table active, batched decode (batched n-gram hashing
            #    fast path + per-stream `engram_tail` bookkeeping) must still match single-stream.
            model.eval()
            with torch.no_grad():
                seq_c = torch.randint(0, cfg.vocab_size, (1, 12), dtype=torch.long)
                seq_d = torch.randint(0, cfg.vocab_size, (1, 15), dtype=torch.long)
                pre_c, pre_d, n_steps = 5, 9, 4
                cc_s, cd_s = model.create_kv_cache(), model.create_kv_cache()
                cc_b, cd_b = model.create_kv_cache(), model.create_kv_cache()
                for cache in (cc_s, cc_b):
                    _ = model.forward_hidden_cached(seq_c[:, :pre_c], cache)
                for cache in (cd_s, cd_b):
                    _ = model.forward_hidden_cached(seq_d[:, :pre_d], cache)
                self.assertIsNotNone(cc_b.get("engram_tail"))
                sess = model.begin_batched_decode([cc_b, cd_b], max_steps=n_steps)
                for step_i in range(n_steps):
                    tok_c = seq_c[:, pre_c + step_i : pre_c + step_i + 1]
                    tok_d = seq_d[:, pre_d + step_i : pre_d + step_i + 1]
                    hc_s = model.forward_hidden_cached(tok_c, cc_s)
                    hd_s = model.forward_hidden_cached(tok_d, cd_s)
                    h_cd = model.step_batched_decode(torch.cat([tok_c, tok_d], dim=0), sess)
                    self.assertTrue(
                        torch.allclose(hc_s, h_cd[0:1], atol=1e-2, rtol=1e-2),
                        f"engram step {step_i} stream c: {(hc_s - h_cd[0:1]).abs().max().item()}",
                    )
                    self.assertTrue(
                        torch.allclose(hd_s, h_cd[1:2], atol=1e-2, rtol=1e-2),
                        f"engram step {step_i} stream d: {(hd_s - h_cd[1:2]).abs().max().item()}",
                    )
                model.end_batched_decode(sess)
                self.assertTrue(torch.equal(cc_s["engram_tail"], cc_b["engram_tail"]))
                self.assertTrue(torch.equal(cd_s["engram_tail"], cd_b["engram_tail"]))

            # 7. Mid-session rollback: stream `e` keeps only k of the S session steps (as after an
            #    EOS / stop string inside a decode chunk) and must then be identical to a stream
            #    that only ever saw those k tokens; stream `f` keeps all S steps.
            with torch.no_grad():
                seq_e = torch.randint(0, cfg.vocab_size, (1, 20), dtype=torch.long)
                seq_f = torch.randint(0, cfg.vocab_size, (1, 22), dtype=torch.long)
                pre_e, pre_f, n_sess = 6, 9, 7
                for k_keep in (0, 1, 3, 4):
                    ce_b, cf_b = model.create_kv_cache(), model.create_kv_cache()
                    _ = model.forward_hidden_cached(seq_e[:, :pre_e], ce_b)
                    _ = model.forward_hidden_cached(seq_f[:, :pre_f], cf_b)
                    sess = model.begin_batched_decode([ce_b, cf_b], max_steps=n_sess)
                    for step_i in range(n_sess):
                        tok_e = seq_e[:, pre_e + step_i : pre_e + step_i + 1]
                        tok_f = seq_f[:, pre_f + step_i : pre_f + step_i + 1]
                        _ = model.step_batched_decode(torch.cat([tok_e, tok_f], dim=0), sess)
                    model.end_batched_decode(sess, valid_steps=[k_keep, n_sess])
                    self.assertEqual(ce_b["seqlen"], pre_e + k_keep)
                    self.assertEqual(cf_b["seqlen"], pre_f + n_sess)

                    # Reference: single-stream cache that saw exactly pre_e + k_keep tokens
                    ce_ref = model.create_kv_cache()
                    _ = model.forward_hidden_cached(seq_e[:, : pre_e + k_keep], ce_ref)
                    # Continue both for 3 more tokens single-stream and compare hidden states
                    for j in range(3):
                        pos = pre_e + k_keep + j
                        tok = seq_e[:, pos : pos + 1]
                        h_ref = model.forward_hidden_cached(tok, ce_ref)
                        h_rb = model.forward_hidden_cached(tok, ce_b)
                        self.assertTrue(
                            torch.allclose(h_ref, h_rb, atol=1e-2, rtol=1e-2),
                            f"rollback k={k_keep} continue step {j}: {(h_ref - h_rb).abs().max().item()}",
                        )
                    self.assertTrue(torch.equal(ce_ref["engram_tail"], ce_b["engram_tail"]))
                    # Stream f (no rollback) must match a plain single-stream run too
                    cf_ref = model.create_kv_cache()
                    _ = model.forward_hidden_cached(seq_f[:, : pre_f + n_sess], cf_ref)
                    pos = pre_f + n_sess
                    tok = seq_f[:, pos : pos + 1]
                    self.assertTrue(
                        torch.allclose(
                            model.forward_hidden_cached(tok, cf_ref),
                            model.forward_hidden_cached(tok, cf_b),
                            atol=1e-2,
                            rtol=1e-2,
                        )
                    )

    def test_dsv41_stacked_eval_moe_parity_in_inference_config(self):
        """Regression: stacked zero-sync MoE eval path must match the per-expert loop.

        Reproduces the production Candidate B32 evaluation configuration
        (`enable_qat=False`, `routed_expert_lora_r=0`, non-dense bf16 experts,
        `model.eval()` + `torch.no_grad()`), where `merged_weight()` aliases the live
        nn.Parameter.  A previous version released the parameters before
        `torch.stack`, producing an empty `[E, 0]` `w2` stack and crashing with
        `size mismatch ... vec (0)` on the first prefill.
        """
        torch.manual_seed(7)
        cfg = dsv41_sft_model.Dsv41E128Config.tiny_test_config()
        cfg.enable_qat = False
        cfg.routed_expert_lora_r = 0
        cfg.trainable_routed_experts_dense = False
        cfg.qat_trainable_expert_scales = False
        model = dsv41_sft_model.Dsv41SftModel(
            cfg=cfg, ep_rank=0, ep_world_size=1, gradient_checkpointing=False
        )
        with torch.no_grad():
            for n, p in model.named_parameters():
                if "experts." in n:
                    p.normal_(0.0, 0.05)
        model.eval()

        moe0 = model.layers[0].ffn
        exp0 = moe0.experts[moe0.experts_start_idx]
        # Production aliasing precondition: merged_weight() returns the live Parameter.
        self.assertIs(exp0.w2.merged_weight(), exp0.w2.weight)
        self.assertEqual(exp0.w2.weight.dtype, torch.bfloat16)
        self.assertFalse(exp0.trainable_dense)

        long_tokens = torch.randint(0, cfg.vocab_size, (1, 80), dtype=torch.long)
        short_tokens = long_tokens[:, :16]

        # Reference outputs via the per-expert loop (grad enabled => stacked path disabled,
        # and the per-expert parameters are guaranteed to still be intact).
        with torch.enable_grad():
            self.assertFalse(moe0._can_use_stacked_eval_path())
            ref_long = model.forward_hidden(long_tokens).detach()
            ref_short = model.forward_hidden(short_tokens).detach()
            self.assertIsNone(moe0._stacked_w13)
            self.assertGreater(exp0.w2.weight.numel(), 0)

        n_local = moe0.n_local_experts
        with torch.no_grad():
            self.assertTrue(moe0._can_use_stacked_eval_path())
            # Prefill path (l_tok > 64)
            out_long = model.forward_hidden(long_tokens)
            self.assertIsNotNone(moe0._stacked_w13)
            self.assertIsNotNone(moe0._stacked_w2)
            self.assertEqual(
                tuple(moe0._stacked_w13.shape),
                (n_local, 2 * cfg.moe_inter_dim, cfg.dim),
            )
            self.assertEqual(
                tuple(moe0._stacked_w2.shape), (n_local, cfg.dim, cfg.moe_inter_dim)
            )
            # Per-expert parameters must have been released (memory-neutral stacking)...
            for proj in (exp0.w1, exp0.w2, exp0.w3):
                self.assertEqual(proj.weight.numel(), 0)
            # ...but the stacked copies must hold the real (non-zero) weights.
            self.assertGreater(float(moe0._stacked_w2.float().abs().sum()), 0.0)
            self.assertTrue(torch.isfinite(out_long).all())
            self.assertTrue(
                torch.allclose(ref_long, out_long, atol=2e-2, rtol=2e-2),
                f"prefill max abs diff = {(ref_long - out_long).abs().max().item()}",
            )

            # Decode fast path (l_tok <= 64), evaluated after the parameters were released
            out_short = model.forward_hidden(short_tokens)
            self.assertTrue(torch.isfinite(out_short).all())
            self.assertTrue(
                torch.allclose(ref_short, out_short, atol=2e-2, rtol=2e-2),
                f"decode max abs diff = {(ref_short - out_short).abs().max().item()}",
            )

            # KV-cached single-stream + batched multi-stream decode also run on the stacked path
            cache_a = model.create_kv_cache()
            cache_b = model.create_kv_cache()
            _ = model.forward_hidden_cached(long_tokens[:, :70], cache_a)
            _ = model.forward_hidden_cached(long_tokens[:, :12], cache_b)
            h_ab = model.forward_hidden_cached_batch(
                torch.cat([long_tokens[:, 70:71], long_tokens[:, 12:13]], dim=0),
                [cache_a, cache_b],
            )
            self.assertEqual(tuple(h_ab.shape), (2, 1, ref_long.shape[-1]))
            self.assertTrue(torch.isfinite(h_ab).all())
            self.assertEqual(cache_a["seqlen"], 71)
            self.assertEqual(cache_b["seqlen"], 13)

        # Trainer-style configs (LoRA / trainable QAT scales) must keep the per-expert path:
        # no stacked copies are built and nothing is released during validation forwards.
        cfg_tr = dsv41_sft_model.Dsv41E128Config.tiny_test_config()
        cfg_tr.enable_qat = True
        cfg_tr.routed_expert_lora_r = 8
        cfg_tr.trainable_routed_experts_dense = False
        cfg_tr.qat_trainable_expert_scales = True
        model_tr = dsv41_sft_model.Dsv41SftModel(
            cfg=cfg_tr, ep_rank=0, ep_world_size=1, gradient_checkpointing=False
        )
        model_tr.eval()
        moe_tr = model_tr.layers[0].ffn
        with torch.no_grad():
            self.assertFalse(moe_tr._can_use_stacked_eval_path())
            _ = model_tr.forward_hidden(short_tokens)
            self.assertIsNone(moe_tr._stacked_w13)
            self.assertIsNone(moe_tr._shared_prequant_cache)
            exp_tr = moe_tr.experts[moe_tr.experts_start_idx]
            self.assertGreater(exp_tr.w2.weight.numel(), 0)

    def test_dsv41_two_step_sft_export_reload_and_long_moe_ckpt_parity(self):
        """Verifies >=512-token MoE checkpointing parity and 2-step SFT -> 2-bit export -> reload -> eval."""
        import safetensors
        from safetensors.torch import save_file

        torch.manual_seed(20261030)
        cfg = dsv41_sft_model.Dsv41E128Config.tiny_test_config()
        cfg.enable_qat = False
        cfg.routed_expert_lora_r = 8
        cfg.routed_expert_lora_alpha = 16.0
        cfg.trainable_routed_experts_dense = False

        # 1. Build a synthetic Havu-format 2-bit GPTQ + cross-shard FP8 block-32 base checkpoint
        with tempfile.TemporaryDirectory(prefix="dsv41_e2e_pipe_") as tmp:
            base_dir = os.path.join(tmp, "base_2bit")
            export_dir = os.path.join(tmp, "export_2bit_step2")
            os.makedirs(os.path.join(base_dir, "gptq"), exist_ok=True)
            with open(os.path.join(base_dir, "config.json"), "w") as f:
                json.dump({"model_type": "deepseek_v4_1"}, f)

            ref_init = dsv41_sft_model.Dsv41SftModel(
                cfg=cfg, ep_rank=0, ep_world_size=1, gradient_checkpointing=True
            )
            with torch.no_grad():
                for _, p in ref_init.named_parameters():
                    p.normal_(0.0, 0.04)

            # Split backbone across shard 1 & shard 2 with `layers.0.ffn.shared_experts.w1.weight`
            # in shard 1 and its `.scale` in shard 2 (reproducing the production boundary split).
            shard1: dict[str, torch.Tensor] = {}
            shard2: dict[str, torch.Tensor] = {}
            for name, p in ref_init.named_parameters():
                if ".experts." in name or "lora_" in name:
                    continue
                t = p.detach().float()
                if (
                    name.endswith(".weight")
                    and t.dim() == 2
                    and t.size(0) % 32 == 0
                    and t.size(1) % 32 == 0
                    and not name.startswith("embed.")
                    and not name.startswith("head.")
                    and ".gate." not in name
                ):
                    use_2d = ".shared_experts." in name
                    q_fp8, s_e8 = dsv41_sft_model.quantize_fp8_block32_torch(
                        t, block_size=32, scale_2d=use_2d
                    )
                    s_key = name[: -len(".weight")] + ".scale"
                    shard1[name] = q_fp8
                    if name == "layers.0.ffn.shared_experts.w1.weight":
                        shard2[s_key] = s_e8
                    else:
                        shard1[s_key] = s_e8
                else:
                    shard2[name] = p.detach().clone()
            for bname, b in ref_init.named_buffers():
                if bname.endswith(".bias_vl"):
                    shard2[bname] = b.detach().clone()

            save_file(shard1, os.path.join(base_dir, "model-00001-of-00002.safetensors"))
            save_file(shard2, os.path.join(base_dir, "model-00002-of-00002.safetensors"))

            for lid, blk in enumerate(ref_init.layers):
                layer_gptq: dict[str, torch.Tensor] = {}
                for eid, exp in enumerate(blk.ffn.experts):
                    assert exp is not None
                    for pname in ("w1", "w2", "w3"):
                        proj = getattr(exp, pname)
                        q_p, s_p, z_p = (
                            dsv41_sft_model.quantize_uint2_humming_expert_tensor(
                                proj.weight.detach(), group=proj.group
                            )
                        )
                        layer_gptq[f"e{eid}.{pname}.q"] = q_p
                        layer_gptq[f"e{eid}.{pname}.s"] = s_p
                        layer_gptq[f"e{eid}.{pname}.z"] = z_p
                save_file(
                    layer_gptq,
                    os.path.join(base_dir, f"gptq/layer_{lid:02d}.safetensors"),
                )

            # 2. Load base checkpoint into trainable SFT model and verify >=512-token MoE ckpt path
            sft_model = dsv41_sft_model.Dsv41SftModel(
                cfg=cfg, ep_rank=0, ep_world_size=1, gradient_checkpointing=True
            )
            stats_base = dsv41_sft_model.load_dequantized_dsv41_e128(
                model_dir=base_dir, model=sft_model, device=torch.device("cpu")
            )
            self.assertGreater(stats_base["fp8_dequantized_tensors"], 0)
            self.assertEqual(
                stats_base["gptq_2bit_experts_unpacked"],
                cfg.n_layers * cfg.n_routed_experts,
            )

            train_dsv41_sft.configure_dsv41_trainable_parameters(
                sft_model, train_moe_only=True
            )
            groups, _ = train_dsv41_sft.build_dsv41_optimizer_param_groups(
                sft_model, weight_decay=0.01
            )
            opt = torch.optim.AdamW(groups, lr=1e-2)

            # Run 2 SFT steps on a 544-token sequence so `x_flat.size(0) >= 512` triggers
            # the checkpointed shared_experts + routed-expert path.
            long_seq = torch.randint(0, cfg.vocab_size, (1, 544), dtype=torch.long)
            long_labels = long_seq.clone()
            long_labels[:, :16] = -100
            step_losses: list[float] = []
            for _ in range(2):
                opt.zero_grad(set_to_none=True)
                hid = sft_model.forward_hidden(long_seq)
                l_sum, n_val = train_sft.compute_chunked_causal_lm_loss(
                    hidden_states=hid,
                    lm_head=sft_model.head,
                    labels=long_labels,
                    chunk_size=128,
                )
                loss = l_sum / float(n_val)
                step_losses.append(float(loss.detach().item()))
                loss.backward()
                opt.step()
            self.assertLess(step_losses[-1], step_losses[0])

            # 3. Export 2-bit Humming + FP8 block-32 checkpoint after 2 steps
            exp_stats = dsv41_sft_model.export_dsv41_e128_2bit_checkpoint(
                model=sft_model,
                base_model_dir=base_dir,
                export_cns_dir=export_dir,
                rank=0,
                world_size=1,
            )
            self.assertGreater(exp_stats["exported_files_count"], 0)

            # Verify `layers.0.ffn.shared_experts.w1.scale` was re-homed next to `.weight` in shard 1
            # and updated to match the trained `.weight`
            with safetensors.safe_open(
                os.path.join(export_dir, "model-00001-of-00002.safetensors"),
                framework="pt",
            ) as f1:
                w1_q_exp = f1.get_tensor("layers.0.ffn.shared_experts.w1.weight")
                s1_exp = f1.get_tensor("layers.0.ffn.shared_experts.w1.scale")
                w1_deq_exp = dsv41_sft_model.dequantize_fp8_block32_torch(
                    w1_q_exp, s1_exp, out_dtype=torch.float32
                )
                w1_trained = sft_model.layers[0].ffn.shared_experts.w1.weight.detach().float()
                rel_err = float(
                    (w1_deq_exp - w1_trained).norm() / (w1_trained.norm() + 1e-12)
                )
                self.assertLess(rel_err, 0.08)

            # 4. Reload exported checkpoint into a pure-inference model (routed_expert_lora_r=0)
            #    and verify stacked-eval forward + KV-cached decode work cleanly.
            cfg_eval = dsv41_sft_model.Dsv41E128Config.tiny_test_config()
            cfg_eval.enable_qat = False
            cfg_eval.routed_expert_lora_r = 0
            cfg_eval.trainable_routed_experts_dense = False
            eval_model = dsv41_sft_model.Dsv41SftModel(
                cfg=cfg_eval,
                ep_rank=0,
                ep_world_size=1,
                gradient_checkpointing=False,
                ep_replicated_input=True,
            ).eval()
            stats_eval = dsv41_sft_model.load_dequantized_dsv41_e128(
                model_dir=export_dir,
                model=eval_model,
                device=torch.device("cpu"),
            )
            self.assertEqual(
                stats_eval["fp8_dequantized_tensors"],
                stats_base["fp8_dequantized_tensors"],
            )
            self.assertEqual(
                stats_eval["gptq_2bit_experts_unpacked"],
                cfg.n_layers * cfg.n_routed_experts,
            )
            with torch.no_grad():
                kv = eval_model.create_kv_cache()
                h_pre = eval_model.forward_hidden_cached(long_seq[:, :12], kv)
                h_dec = eval_model.forward_hidden_cached(long_seq[:, 12:13], kv)
                logits_dec = eval_model.head(h_dec)
                self.assertTrue(torch.isfinite(h_pre).all())
                self.assertTrue(torch.isfinite(logits_dec).all())

    def test_dsv41_round2_child_flags_and_vl_dataset_contract(self):
        # 1. Every round-2 flag must be forwarded to the 8 per-GPU worker subprocesses (round-2 bug:
        #    workers silently ran with train_vision=none and scale/LoRA LR == base LR).
        flags_obj = train_dsv41_sft.FLAGS
        flags_obj.mark_as_parsed()
        saved = {
            n: flags_obj[n].value
            for n in ("train_vision", "scale_learning_rate", "lora_learning_rate", "engram_model_dir")
        }
        try:
            flags_obj.train_vision = "aligner+last_k"
            flags_obj.scale_learning_rate = 5e-5
            flags_obj.lora_learning_rate = 1e-4
            flags_obj.engram_model_dir = "/cns/x/engram"
            child_args = train_dsv41_sft._serialize_dsv41_flags_for_children()  # pylint: disable=protected-access
        finally:
            for n, v in saved.items():
                setattr(flags_obj, n, v)
        self.assertIn("--train_vision=aligner+last_k", child_args)
        self.assertIn("--scale_learning_rate=5e-05", child_args)
        self.assertIn("--lora_learning_rate=0.0001", child_args)
        self.assertIn("--engram_model_dir=/cns/x/engram", child_args)
        for name in (
            "vision_trainable_blocks",
            "vision_learning_rate",
            "aligner_learning_rate",
            "disable_vision",
            "allow_non_qat_export",
            "global_grad_clip",
        ):
            self.assertTrue(any(a.startswith(f"--{name}=") for a in child_args), name)

        # 2. VL dataset contract: `token_types` in the npz + `{split}_images.pkl` sidecar.
        img = Image.new("RGB", (84, 42), (255, 0, 0))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        png = buf.getvalue()
        base_ids = np.array([5, 6, 7, dsv41_image_tokens.IMAGE_TOKEN_ID, 8, 9], dtype=np.int32)
        ids, types, (mask,), plans = dsv41_image_tokens.expand_image_placeholders(
            base_ids, [png], per_token_arrays=[np.array([0, 0, 0, 0, 1, 1], dtype=np.uint8)]
        )
        self.assertEqual(len(plans), 1)
        n_img_slots = int((types == dsv41_image_tokens.IMAGE).sum())
        self.assertGreater(n_img_slots, 0)
        text_ids = np.array([11, 12, 13, 14], dtype=np.int32)
        with tempfile.TemporaryDirectory(prefix="dsv41_vl_ds_") as tmp_dir:
            npz_path = os.path.join(tmp_dir, "train_tokens.npz")
            lengths = np.array([len(ids), len(text_ids)], dtype=np.int32)
            offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
            np.savez(
                npz_path,
                input_ids=np.concatenate([ids, text_ids]).astype(np.int32),
                loss_mask=np.concatenate([mask, np.ones(4, dtype=np.uint8)]).astype(np.uint8),
                loss_weights=np.concatenate([mask, np.ones(4, dtype=np.uint8)]).astype(np.uint8),
                offsets=offsets,
                lengths=lengths,
                token_types=np.concatenate(
                    [types, np.full(4, dsv41_image_tokens.TEXT, dtype=np.int8)]
                ).astype(np.int8),
            )
            with open(os.path.join(tmp_dir, "train_images.pkl"), "wb") as f:
                pickle.dump([[png], []], f)
            ds = train_dsv41_sft.VLPackedTrajectoryDataset(npz_path, max_seq_len=4096)
            self.assertEqual(len(ds), 2)
            self.assertEqual(ds.n_with_images, 1)
            s0 = ds[0]
            s1 = ds[1]
            self.assertEqual(int(s0["token_types"].numel()), len(ids))
            self.assertEqual(int((s0["token_types"] == dsv41_image_tokens.IMAGE).sum()), n_img_slots)
            self.assertEqual(len(s0["images"]), 1)
            self.assertEqual(s1["images"], [])
            self.assertTrue(bool((s1["token_types"] < 0).all()))
            # Image spans are never supervised.
            self.assertEqual(int(s0["loss_mask"][s0["token_types"] >= 0].sum()) if "loss_mask" in s0 else 0, 0)
            tt, imgs = train_dsv41_sft.prepare_vl_inputs(s0, torch.device("cpu"))
            self.assertIsNotNone(tt)
            self.assertEqual(tuple(tt.shape), (1, len(ids)))
            self.assertEqual(len(imgs[0]), 1)
            self.assertEqual(tuple(imgs[0][0].patches.shape[1:]), (3, 14, 14))
            tt1, imgs1 = train_dsv41_sft.prepare_vl_inputs(s1, torch.device("cpu"))
            self.assertIsNone(tt1)
            self.assertIsNone(imgs1)

            # Text-only legacy npz (no token_types key, no sidecar) loads exactly as before.
            legacy_path = os.path.join(tmp_dir, "legacy_tokens.npz")
            np.savez(
                legacy_path,
                input_ids=text_ids,
                loss_mask=np.ones(4, dtype=np.uint8),
                loss_weights=np.ones(4, dtype=np.uint8),
                offsets=np.array([0, 4], dtype=np.int64),
                lengths=np.array([4], dtype=np.int32),
            )
            ds_legacy = train_dsv41_sft.VLPackedTrajectoryDataset(legacy_path, max_seq_len=4096)
            self.assertEqual(ds_legacy.n_with_images, 0)
            self.assertNotIn("token_types", ds_legacy[0])
            self.assertNotIn("images", ds_legacy[0])

    def test_e065_audit_split_and_gemini38_difficulty_ranking(self):
        from google3.experimental.users.arazdai.kaggle.sft import audit_and_split_e065_rollouts as split_mod
        from google3.experimental.users.arazdai.kaggle.sft import rank_e065_difficulty_gemini38_flash as rank_mod

        gt_tasks = {
            "t_pass": {
                "task_id": "t_pass",
                "parent": "t_pass",
                "batch": "arc2_train",
                "source": "kaggle_arc2_training",
                "train": [{"input": [[0, 1]], "output": [[1, 0]]}],
                "test": [{"input": [[1, 1]], "output": [[0, 0]]}],
            },
            "t_fail": {
                "task_id": "t_fail",
                "parent": "t_fail",
                "batch": "arc2_train",
                "source": "kaggle_arc2_training",
                "train": [{"input": [[0, 2]], "output": [[2, 0]]}],
                "test": [
                    {"input": [[2, 2]], "output": [[0, 0]]},
                    {"input": [[0, 0]], "output": [[2, 2]]},
                ],
            },
        }
        msgs_sample = json.dumps(
            [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "Solve."},
                {
                    "role": "assistant",
                    "reasoning_content": "Let's test a rule.",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "c1",
                            "type": "function",
                            "function": {
                                "name": "submit_answer",
                                "arguments": json.dumps({"code": "solve(test_input)"}),
                            },
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "c1", "content": "Submitted"},
            ]
        )
        rows = [
            {
                "task_id": "t_pass",
                "pair": 0,
                "attempt": 0,
                "source": "arc2_train",
                "parent": "t_pass",
                "correct": True,
                "pred": json.dumps([[0, 0]]),
                "answer_source": "submit",
                "stop_reason": "submitted",
                "turns": 2,
                "tool_calls": 2,
                "prompt_tokens": 500,
                "completion_tokens": 1200,
                "reasoning_tokens": 900,
                "elapsed_s": 5.0,
                "messages": msgs_sample,
                "events": "[]",
                "error": None,
                "train_pass": True,
                "train_frac": 1.0,
                "train_trace": "expr",
            },
            {
                "task_id": "t_fail",
                "pair": 0,
                "attempt": 0,
                "source": "arc2_train",
                "parent": "t_fail",
                "correct": True,
                "pred": json.dumps([[0, 0]]),
                "answer_source": "submit",
                "stop_reason": "submitted",
                "turns": 8,
                "tool_calls": 8,
                "prompt_tokens": 900,
                "completion_tokens": 18000,
                "reasoning_tokens": 15000,
                "elapsed_s": 40.0,
                "messages": msgs_sample,
                "events": "[]",
                "error": None,
                "train_pass": True,
                "train_frac": 1.0,
                "train_trace": "expr",
            },
            {
                "task_id": "t_fail",
                "pair": 1,
                "attempt": 0,
                "source": "arc2_train",
                "parent": "t_fail",
                "correct": False,
                "pred": json.dumps([[9, 9]]),
                "answer_source": "submit",
                "stop_reason": "submitted",
                "turns": 0,
                "tool_calls": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "reasoning_tokens": 0,
                "elapsed_s": 0.0,
                "messages": msgs_sample,
                "events": "[]",
                "error": None,
                "train_pass": True,
                "train_frac": 1.0,
                "train_trace": "expr",
            },
        ]
        import pyarrow as pa
        import pyarrow.parquet as pq

        with tempfile.TemporaryDirectory(prefix="e065_split_") as tmp_dir:
            pq_file = os.path.join(tmp_dir, "e065_images_arc2_train.parquet")
            pq.write_table(pa.Table.from_pylist(rows), pq_file)
            res = split_mod.audit_and_split_modality(
                rollouts_dir=tmp_dir,
                file_names=("e065_images_arc2_train.parquet",),
                gt_tasks=gt_tasks,
                eval_index={"eval_task_ids": set()},
            )
            self.assertEqual(res["total_discrepancies"], 0)
            st = res["per_file_stats"]["e065_images_arc2_train.parquet"]
            self.assertEqual(st["test_passing_rollouts"], 1)
            self.assertEqual(st["non_test_passing_rollouts"], 1)
            self.assertEqual(st["fail_demo_overfit_train_pass"], 1)
            self.assertEqual(len(res["non_test_passing_records"]), 1)

            # Verify Gemini 3.8 Flash prompt builder and composite hardness scoring
            fail_rec = res["non_test_passing_records"][0]
            pass_rec = res["rollout_status_by_task"]["t_pass"]
            prompt_str = rank_mod.build_difficulty_prompt(fail_rec)
            self.assertIn("t_fail", prompt_str)
            self.assertIn("demo_overfit_train_pass_test_fail", prompt_str)

            hard_score_fail = rank_mod.compute_composite_hardness_score(
                fail_rec, {"difficulty_score": 8.5}
            )
            hard_score_pass = rank_mod.compute_composite_hardness_score(
                pass_rec, {"difficulty_score": 3.0}
            )
            self.assertGreater(hard_score_fail, hard_score_pass)
            self.assertEqual(
                data_pipeline.discover_e065_extra_parquet_shards(tmp_dir),
                [pq_file],
            )


if __name__ == "__main__":
    googletest.main()



