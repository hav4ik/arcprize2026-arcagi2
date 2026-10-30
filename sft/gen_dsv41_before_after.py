"""Before/after SFT sanity check for the 2-bit DeepSeek-V4.1 E128 checkpoint on one held-out val example.

For each of (a) the original `dsv41_e128_2bit` base checkpoint and (b) an exported SFT 2-bit milestone
checkpoint (`export_2bit/epoch_1p00`), this script:
  1. Loads the checkpoint into `Dsv41SftModel` on 8x B200 (expert-parallel, same path as `eval_dsv41_b32_arc2`).
  2. Computes teacher-forced NLL / top-1 accuracy on one `val_tokens.npz` trajectory (supervised tokens only,
     and all tokens) through the KV-cached prefill path used for generation.
  3. Greedy-decodes the first assistant turn from the trajectory's prompt prefix (`...<｜Assistant｜><think>`).
  4. Greedy-decodes a few plain-English sanity prompts (DSML format) to check basic fluency.
Results (prompt, ground-truth turn, generations, NLLs) are printed and written as JSON to `--gen_out_dir`.

`--gen_mode=inspect` runs on CPU without any model and only decodes / prints the selected val example.
"""

from __future__ import annotations

import datetime
import json
import os
import subprocess
import sys
import time
from typing import Any

from absl import app
from absl import flags
import numpy as np
import torch
import torch.distributed as dist

from google3.experimental.users.arazdai.kaggle.arc_agi_2.pilot_2 import havu_b32_harness
from google3.experimental.users.arazdai.kaggle.sft import dsv41_sft_model
from google3.experimental.users.arazdai.kaggle.sft import eval_dsv41_b32_arc2 as eval_lib
from google3.experimental.users.arazdai.kaggle.sft import export_and_quantize
from google3.experimental.users.arazdai.kaggle.sft import train_sft
from google3.pyglib import gfile
from google3.pyglib.contrib.g3_multiprocessing import g3_multiprocessing

FLAGS = flags.FLAGS

flags.DEFINE_enum(
    "gen_mode",
    "generate",
    ["inspect", "generate"],
    "`inspect` only decodes the selected val example on CPU; `generate` runs both checkpoints on GPUs.",
)
flags.DEFINE_string(
    "gen_base_model_dir",
    "/cns/lu-d/home/arazdai/kaggle/models/dsv41_e128_2bit",
    "Original (pre-SFT) 2-bit DeepSeek-V4.1 E128 checkpoint directory (also provides Engram shards + tokenizer).",
)
flags.DEFINE_string(
    "gen_sft_dir",
    "/cns/lu-d/home/arazdai/kaggle/sft/checkpoints_dsv41_e128/xid_305548873_dsv41_e128_moe_only_full_sft_8xb200_v4/wu_1/export_2bit/epoch_1p00",
    "Exported SFT 2-bit checkpoint directory to compare against the base checkpoint.",
)
flags.DEFINE_string(
    "gen_val_npz",
    "/cns/lu-d/home/arazdai/kaggle/sft/data/dsv41_e128_short_clean/val_tokens.npz",
    "Held-out validation trajectories (same file the SFT trainer evaluated on).",
)
flags.DEFINE_integer(
    "gen_val_index",
    -1,
    "Index into the val dataset (after overlong filtering). -1 = shortest trajectory.",
)
flags.DEFINE_integer("gen_max_new_tokens", 640, "Greedy tokens to generate for the ARC val example.")
flags.DEFINE_integer("gen_sanity_max_new_tokens", 160, "Greedy tokens to generate per sanity prompt.")
flags.DEFINE_string(
    "gen_out_dir",
    "/cns/lu-d/home/arazdai/kaggle/sft/evals/dsv41_e128_before_after_sanity",
    "CNS directory for the JSON report.",
)
flags.DEFINE_integer("gen_num_gpus", 8, "Number of GPUs (expert-parallel world size).")
flags.DEFINE_bool("gen_enable_qat", False, "Enable Havu FP8/FP4 fake-quant activations in the forward pass.")
flags.DEFINE_bool("gen_load_compact_engram", True, "Load the compact Engram shards (layers 1 & 14).")
flags.DEFINE_bool("gen_skip_base", False, "Skip the base checkpoint (only run the SFT checkpoint).")
flags.DEFINE_bool("gen_skip_sft", False, "Skip the SFT checkpoint (only run the base checkpoint).")

_BOS = havu_b32_harness.DS41_BOS_TOKEN
_USER = havu_b32_harness.DS41_USER_SP_TOKEN
_ASSISTANT = havu_b32_harness.DS41_ASSISTANT_SP_TOKEN

SANITY_PROMPTS: list[tuple[str, str]] = [
    (
        "plain_qa_nothink",
        f"{_BOS}{_USER}In two or three sentences, explain what the ARC-AGI benchmark tests.{_ASSISTANT}</think>",
    ),
    (
        "arith_think",
        f"{_BOS}{_USER}What is 17 * 23? Think briefly, then give the answer.{_ASSISTANT}<think>",
    ),
    (
        "python_nothink",
        f"{_BOS}{_USER}Write a Python function that returns the transpose of a 2D list.{_ASSISTANT}</think>",
    ),
]


def select_val_example(val_npz: str, val_index: int) -> dict[str, Any]:
    """Loads `val_npz` and returns the chosen trajectory split into prompt / first assistant turn / rest."""
    ds = train_sft.PackedTrajectoryDataset(val_npz, max_seq_len=1 << 20)
    if val_index < 0:
        lengths = [int(ds.lengths[r]) for r in ds.valid_indices]
        val_index = int(np.argmin(lengths))
    sample = ds[val_index]
    ids = sample["input_ids"].tolist()
    labels = sample["labels"].tolist()
    sup = [l != -100 for l in labels]
    if not any(sup):
        raise ValueError(f"Val example {val_index} has no supervised tokens.")
    first_sup = sup.index(True)
    turn_end = len(ids)
    for j in range(first_sup, len(ids)):
        if not sup[j]:
            turn_end = j
            break
    return {
        "val_index": val_index,
        "real_index": int(ds.valid_indices[val_index]),
        "n_tokens": len(ids),
        "n_supervised": int(sum(sup)),
        "first_supervised_pos": first_sup,
        "first_turn_end_pos": turn_end,
        "input_ids": ids,
        "labels": labels,
        "prompt_ids": ids[:first_sup],
        "first_turn_ids": ids[first_sup:turn_end],
    }


def _prefill_collect_hidden(
    engine: eval_lib.Dsv41DistributedStepEngine,
    slot_id: int,
    token_ids: list[int],
    device: torch.device,
    prefill_chunk: int = 4096,
) -> torch.Tensor:
    """Mirrors `Dsv41DistributedStepEngine._run_distributed_cached_forward` but keeps all `[1, S, dim]` hidden states."""
    is_dist = dist.is_initialized() and dist.get_world_size() > 1
    engine._cache_slots[slot_id] = {}  # pylint: disable=protected-access
    kv_cache = engine._cache_slots[slot_id]  # pylint: disable=protected-access
    hiddens: list[torch.Tensor] = []
    for start in range(0, len(token_ids), prefill_chunk):
        sub_ids = token_ids[start : start + prefill_chunk]
        op = 2 if start == 0 else 1
        cmd = torch.tensor([op, slot_id, len(sub_ids)], dtype=torch.int64, device=device)
        if is_dist:
            dist.broadcast(cmd, src=0)
        tok_t = torch.tensor([sub_ids], dtype=torch.long, device=device)
        if is_dist:
            dist.broadcast(tok_t, src=0)
        hiddens.append(engine.model.forward_hidden_cached(tok_t, kv_cache))
    return torch.cat(hiddens, dim=1)


@torch.no_grad()
def teacher_forced_metrics(
    model: dsv41_sft_model.Dsv41SftModel,
    hidden: torch.Tensor,
    input_ids: list[int],
    labels: list[int],
    chunk: int = 1024,
) -> dict[str, float]:
    """Per-token NLL and top-1 accuracy (supervised-only and all tokens) from `[1, S, dim]` hidden states."""
    device = hidden.device
    h = hidden[0, :-1].to(model.head.weight.dtype)
    tgt_all = torch.tensor(input_ids[1:], dtype=torch.long, device=device)
    tgt_sup = torch.tensor(labels[1:], dtype=torch.long, device=device)
    sup_mask = tgt_sup != -100
    nll_all_sum = 0.0
    nll_sup_sum = 0.0
    top1_all = 0
    top1_sup = 0
    ent_sum = 0.0
    for s in range(0, h.size(0), chunk):
        logits = model.head(h[s : s + chunk]).float()
        logp = torch.log_softmax(logits, dim=-1)
        t_all = tgt_all[s : s + chunk]
        m_sup = sup_mask[s : s + chunk]
        tok_nll = -logp.gather(-1, t_all[:, None])[:, 0]
        pred = logits.argmax(dim=-1)
        nll_all_sum += float(tok_nll.sum().item())
        nll_sup_sum += float(tok_nll[m_sup].sum().item())
        top1_all += int((pred == t_all).sum().item())
        top1_sup += int((pred[m_sup] == t_all[m_sup]).sum().item())
        ent_sum += float((-(logp.exp() * logp).sum(-1)).sum().item())
    n_all = int(tgt_all.numel())
    n_sup = int(sup_mask.sum().item())
    return {
        "n_all_tokens": n_all,
        "n_supervised_tokens": n_sup,
        "nll_all": nll_all_sum / max(1, n_all),
        "nll_supervised": nll_sup_sum / max(1, n_sup),
        "ppl_supervised": float(np.exp(min(nll_sup_sum / max(1, n_sup), 30.0))),
        "top1_acc_all": top1_all / max(1, n_all),
        "top1_acc_supervised": top1_sup / max(1, n_sup),
        "mean_predictive_entropy_nats": ent_sum / max(1, n_all),
        "ln_vocab": float(np.log(model.cfg.vocab_size)),
    }


def greedy_generate(
    engine: eval_lib.Dsv41DistributedStepEngine,
    prompt_ids: list[int],
    max_new_tokens: int,
) -> dict[str, Any]:
    """Greedy (temperature=0) decode via the same step engine used by the Candidate B32 evaluator."""
    sp = eval_lib._CompatSamplingParams(temperature=0.0, max_tokens=max_new_tokens, stop=[])  # pylint: disable=protected-access
    [rid] = engine.enqueue([{"prompt_token_ids": list(prompt_ids)}], [sp])
    t0 = time.time()
    final = None
    while engine.has_unfinished_requests():
        for ro in engine.step():
            if ro.request_id == rid:
                final = ro
    assert final is not None
    out = final.outputs[0]
    dt = time.time() - t0
    return {
        "text": out.text,
        "n_tokens": len(out.token_ids),
        "finish_reason": out.finish_reason,
        "seconds": round(dt, 1),
        "tok_per_sec": round(len(out.token_ids) / max(dt, 1e-6), 2),
        "unique_token_frac": round(len(set(out.token_ids)) / max(1, len(out.token_ids)), 3),
    }


def _rank0_evaluate_checkpoint(
    tag: str,
    model: dsv41_sft_model.Dsv41SftModel,
    tok: eval_lib._FastTokenizerAdapter,  # pylint: disable=protected-access
    device: torch.device,
    example: dict[str, Any],
) -> dict[str, Any]:
    """Runs NLL + generations for the currently loaded weights; terminates the follower loop at the end."""
    engine = eval_lib.Dsv41DistributedStepEngine(
        model=model, tokenizer=tok, device=device, max_model_len=model.cfg.max_seq_len
    )
    res: dict[str, Any] = {"tag": tag}
    t0 = time.time()
    hidden = _prefill_collect_hidden(engine, 999, example["input_ids"], device)
    res["teacher_forced"] = teacher_forced_metrics(model, hidden, example["input_ids"], example["labels"])
    res["teacher_forced"]["seconds"] = round(time.time() - t0, 1)
    del hidden
    engine._evict_slot(999)  # pylint: disable=protected-access
    print(f"[{tag}] teacher-forced: {json.dumps(res['teacher_forced'])}", flush=True)

    res["arc_first_turn_generation"] = greedy_generate(
        engine, example["prompt_ids"], FLAGS.gen_max_new_tokens
    )
    print(
        f"[{tag}] ARC first-turn greedy generation ({res['arc_first_turn_generation']['n_tokens']} tok, "
        f"{res['arc_first_turn_generation']['tok_per_sec']} tok/s):\n"
        f"{res['arc_first_turn_generation']['text'][:3000]}\n----",
        flush=True,
    )
    res["sanity"] = {}
    for name, text in SANITY_PROMPTS:
        ids = tok.encode(text, add_special_tokens=False)
        g = greedy_generate(engine, ids, FLAGS.gen_sanity_max_new_tokens)
        g["prompt"] = text
        res["sanity"][name] = g
        print(f"[{tag}] sanity/{name}:\n{text}\n>>> {g['text'][:1500]}\n----", flush=True)

    if dist.is_initialized() and dist.get_world_size() > 1:
        dist.broadcast(torch.zeros(3, dtype=torch.int64, device=device), src=0)
    return res


def _worker_main(rank: int, world_size: int) -> None:
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if world_size > 1 and not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "::1")
        os.environ.setdefault("MASTER_PORT", "29531")
        os.environ.setdefault("NCCL_SOCKET_FAMILY", "AF_INET6")
        store_path = os.environ.get("TORCH_DIST_INIT_FILE", "/tmp/gen_dsv41_c10d_filestore")
        dist.init_process_group(
            backend="nccl",
            init_method=f"file://{store_path}",
            rank=rank,
            world_size=world_size,
            device_id=device,
            timeout=datetime.timedelta(minutes=90),
        )

    try:
        example = select_val_example(FLAGS.gen_val_npz, FLAGS.gen_val_index)
        checkpoints: list[tuple[str, str]] = []
        if not FLAGS.gen_skip_base:
            checkpoints.append(("base_dsv41_e128_2bit", FLAGS.gen_base_model_dir))
        if not FLAGS.gen_skip_sft:
            checkpoints.append(("sft_" + os.path.basename(FLAGS.gen_sft_dir.rstrip("/")), FLAGS.gen_sft_dir))
        if len(checkpoints) != 1:
            # The pure-inference stacked expert path frees the per-expert parameters after the first
            # forward, so a second checkpoint cannot be loaded in place; run one checkpoint per process.
            raise app.UsageError(
                "Run exactly one checkpoint per process: pass --gen_skip_sft=true or --gen_skip_base=true."
            )

        cfg_path = os.path.join(FLAGS.gen_base_model_dir, "config.json")
        if gfile.Exists(cfg_path):
            with gfile.Open(cfg_path, "r") as f:
                cfg = dsv41_sft_model.Dsv41E128Config.from_json_dict(json.load(f))
        else:
            cfg = dsv41_sft_model.Dsv41E128Config()
        cfg.routed_expert_lora_r = 0
        cfg.enable_qat = FLAGS.gen_enable_qat
        cfg.qat_trainable_expert_scales = False
        cfg.load_compact_engram = FLAGS.gen_load_compact_engram
        cfg.max_seq_len = max(cfg.max_seq_len, 65536)

        with torch.device(device):
            model = dsv41_sft_model.Dsv41SftModel(
                cfg=cfg,
                ep_rank=rank,
                ep_world_size=world_size,
                gradient_checkpointing=False,
                ep_replicated_input=True,
            )
        model.eval()

        tok = None
        results: dict[str, Any] = {}
        if rank == 0:
            tok = eval_lib._FastTokenizerAdapter(os.path.join(FLAGS.gen_base_model_dir, "tokenizer.json"))  # pylint: disable=protected-access
            results = {
                "val_npz": FLAGS.gen_val_npz,
                "val_index": example["val_index"],
                "real_index": example["real_index"],
                "n_tokens": example["n_tokens"],
                "n_supervised": example["n_supervised"],
                "first_supervised_pos": example["first_supervised_pos"],
                "first_turn_end_pos": example["first_turn_end_pos"],
                "prompt_text": tok.decode(example["prompt_ids"]),
                "ground_truth_first_turn_text": tok.decode(example["first_turn_ids"]),
                "enable_qat": FLAGS.gen_enable_qat,
                "load_compact_engram": FLAGS.gen_load_compact_engram,
                "checkpoints": {},
            }
            print(
                f"[gen] val example idx={example['val_index']} n_tokens={example['n_tokens']} "
                f"prompt_len={len(example['prompt_ids'])} first_turn_len={len(example['first_turn_ids'])}",
                flush=True,
            )

        for tag, ckpt_dir in checkpoints:
            t_load = time.time()
            stats = dsv41_sft_model.load_dequantized_dsv41_e128(
                model_dir=ckpt_dir,
                model=model,
                device=device,
                engram_model_dir=FLAGS.gen_base_model_dir,
            )
            model.eval()
            if world_size > 1:
                dist.barrier()
            if rank == 0:
                print(f"[gen] Loaded {tag} from {ckpt_dir} in {time.time() - t_load:.1f}s: {stats}", flush=True)
                assert tok is not None
                res = _rank0_evaluate_checkpoint(tag, model, tok, device, example)
                res["checkpoint_dir"] = ckpt_dir
                res["load_stats"] = stats
                results["checkpoints"][tag] = res
                export_and_quantize.ensure_cns_quota_dir(FLAGS.gen_out_dir)
                out_path = os.path.join(FLAGS.gen_out_dir, f"report_{tag}.json")
                with gfile.Open(out_path, "w") as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
                print(f"[gen] Wrote report to {out_path}", flush=True)
            else:
                eval_lib._follower_worker_loop(model, device)  # pylint: disable=protected-access
            if world_size > 1:
                dist.barrier()
            torch.cuda.empty_cache()

        if rank == 0:
            print("[gen] ===== SUMMARY =====", flush=True)
            for tag, res in results["checkpoints"].items():
                tf = res["teacher_forced"]
                print(
                    f"[gen] {tag}: nll_sup={tf['nll_supervised']:.3f} (ppl {tf['ppl_supervised']:.1f}) "
                    f"nll_all={tf['nll_all']:.3f} top1_sup={tf['top1_acc_supervised']:.3f} "
                    f"ln(V)={tf['ln_vocab']:.2f}",
                    flush=True,
                )
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _serialize_flags_for_children() -> list[str]:
    names = [
        "gen_mode",
        "gen_base_model_dir",
        "gen_sft_dir",
        "gen_val_npz",
        "gen_val_index",
        "gen_max_new_tokens",
        "gen_sanity_max_new_tokens",
        "gen_out_dir",
        "gen_num_gpus",
        "gen_enable_qat",
        "gen_load_compact_engram",
        "gen_skip_base",
        "gen_skip_sft",
    ]
    args = [
        "--undefok=cfs_d_cid_enable,cfs_d_resource_user_override,cfs_default_traffic_class,cfs_max_write_bytes_inflight",
        "--cfs_d_cid_enable=true",
        "--cfs_d_resource_user_override=deepmind-resources-colossus",
        "--cfs_default_traffic_class=AF1",
        "--cfs_max_write_bytes_inflight=33554432",
    ]
    for name in names:
        val = FLAGS[name].value
        if isinstance(val, bool):
            args.append(f"--{name}={'true' if val else 'false'}")
        else:
            args.append(f"--{name}={val}")
    return args


def main(argv: list[str]) -> None:
    if len(argv) > 1:
        raise app.UsageError(f"Unknown command-line arguments: {argv[1:]}")

    if FLAGS.gen_mode == "inspect":
        example = select_val_example(FLAGS.gen_val_npz, FLAGS.gen_val_index)
        tok = eval_lib._FastTokenizerAdapter(os.path.join(FLAGS.gen_base_model_dir, "tokenizer.json"))  # pylint: disable=protected-access
        print(json.dumps({k: v for k, v in example.items() if not k.endswith("_ids") and k != "labels"}, indent=2))
        print("=== PROMPT ===")
        print(tok.decode(example["prompt_ids"]))
        print("=== GROUND-TRUTH FIRST ASSISTANT TURN ===")
        print(tok.decode(example["first_turn_ids"]))
        print("=== REMAINDER OF TRAJECTORY (first 1500 chars) ===")
        print(tok.decode(example["input_ids"][example["first_turn_end_pos"] :])[:1500])
        return

    env_rank = os.environ.get("RANK")
    env_world = os.environ.get("WORLD_SIZE")
    if env_rank is not None and env_world is not None:
        _worker_main(rank=int(env_rank), world_size=int(env_world))
        return

    world_size = FLAGS.gen_num_gpus if torch.cuda.is_available() else 1
    if world_size <= 1:
        _worker_main(rank=0, world_size=1)
        return

    exe_path = sys.argv[0]
    child_args = _serialize_flags_for_children()
    store_file = f"/tmp/gen_dsv41_c10d_filestore_{os.getpid()}_{int(time.time())}"
    print(f"Launching {world_size} worker subprocesses via {exe_path} (init_method=file://{store_file})...", flush=True)
    procs: list[subprocess.Popen[bytes]] = []
    for r in range(world_size):
        env = os.environ.copy()
        env["RANK"] = str(r)
        env["LOCAL_RANK"] = str(r)
        env["WORLD_SIZE"] = str(world_size)
        env["MASTER_ADDR"] = "::1"
        env["MASTER_PORT"] = os.environ.get("MASTER_PORT", "29531")
        env["NCCL_SOCKET_FAMILY"] = "AF_INET6"
        env["TORCH_DIST_INIT_FILE"] = store_file
        procs.append(subprocess.Popen([exe_path, *child_args], env=env))
    exit_codes: list[int] = []
    for r, p in enumerate(procs):
        rc = p.wait()
        exit_codes.append(rc)
        if rc != 0:
            print(f"Worker rank {r} exited with code {rc}; terminating remaining workers.", flush=True)
            for other in procs:
                if other.poll() is None:
                    other.terminate()
            break
    if any(rc != 0 for rc in exit_codes):
        raise RuntimeError(f"Worker subprocesses failed with exit codes: {exit_codes}")


if __name__ == "__main__":
    g3_multiprocessing.handle_main(main)
