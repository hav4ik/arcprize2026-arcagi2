"""E028: LoRA SFT of Qwen3.8-27B on Tinker, on the E025 DeepSeek-Flash rollouts built by build_dataset.py.

Own training loop (the cookbook's supervised loop only supports Adam): AdamW or Dimuon
(https://tinker-docs.thinkingmachines.ai/tinker/optimizers/), linear/cosine/constant LR decay to 0,
one-step pipelining, checkpoints (state + sampler) into <run>/checkpoints.jsonl, resume from the last one.

Data: data/sft/e028/train.npz (tokens int32, weights uint8 0/1 aligned with tokens, offsets), pre-rendered
and mask-checked by build_dataset.py. Only assistant tokens have weight 1; prompts and tool outputs are 0.

Batching ("packing" without concatenation): each epoch shuffles the rollouts and fills batches up to a
token budget (--batch-tokens, total tokens incl. masked ones), so steps are similar in size; every rollout
stays its own sequence (no cross-rollout attention).

Loss: Tinker's cross_entropy is a plain sum of weight * NLL over tokens and datums. The 0/1 weights are
divided by the number of trained tokens in the batch, so each step's loss is the per-token mean NLL over
the batch: every trained token counts the same, and masked tokens are in neither the sum nor the count.
The logged train_nll is that value, measured before the step on that batch (in epoch 1 every batch is
unseen, so it is a held-out loss).

  python -I train.py --dry-run
  # LR pilot = the first 20 steps of the real run (same data, same 252-step schedule), then stop:
  python -I train.py --name full2500_dimuon_1e-3 --lr 1e-3 --stop-after 20
  # continue the winner from its step-20 checkpoint to the end (same --name and args, no --stop-after):
  python -I train.py --name full2500_dimuon_1e-3 --lr 1e-3 --max-cost 300
"""
import argparse
import asyncio
import hashlib
import json
import logging
import os
import random
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path("/home/vu/Workspace/ARC-AGI-2")
DATA = ROOT / "data/sft/e028"
RENDERER = "qwen3_8_xhigh_reasoning"
MODELS = {  # Tinker id -> (max sequence tokens, train $/M); models_and_pricing page, 2026-10-16
    "Qwen/Qwen3.8-27B": (65536, 4.103),
    "Qwen/Qwen3.8-27B:peft:262144": (262144, 7.46),
}
RESUME_KEYS = ("model", "optimizer", "lr", "lr_schedule", "rank", "batch_tokens", "epochs", "limit", "max_steps",
               "seed", "beta1", "grad_clip")
log = logging.getLogger("e028")


# ---------------------------------------------------------------- data

def load_npz(path):
    z = np.load(path)
    return z["tokens"], z["weights"], z["offsets"]


def data_fingerprint(path):
    """Identifies the exact dataset (order, lengths, masks) so a resume can't silently switch data."""
    _, weights, offsets = load_npz(path)
    h = hashlib.sha1(offsets.tobytes())
    h.update(np.add.reduceat(weights.astype(np.int64), offsets[:-1]).tobytes())
    return f"{len(offsets) - 1}:{h.hexdigest()[:16]}"


def to_datum(tokens, weights, scale):
    import tinker
    import torch
    from tinker_cookbook.supervised.common import datum_from_model_input_weights
    mi = tinker.ModelInput.from_ints(tokens.tolist())
    w = torch.tensor(weights, dtype=torch.float32) * scale
    # max_length=None: sequences were length-checked at build time, never truncated here.
    return datum_from_model_input_weights(mi, w, max_length=None, reduction="none")


class Batches:
    """Token-budget batches. The stored order is shuffled, so a --limit prefix is a random sample.
    Epoch 0 keeps the stored order (identical across runs, so pilots see the same batches); epoch e > 0
    reshuffles with seed (seed + e). Batches are filled greedily in that order up to batch_tokens; a rollout
    that would overflow the current batch starts the next one (so a batch is within one rollout of the
    budget). The last, short batch of an epoch is kept if it holds at least half the budget."""

    def __init__(self, path, batch_tokens, limit, max_len, seed, epochs):
        self.tokens, self.weights, self.offsets = load_npz(path)
        n = len(self.offsets) - 1
        self.idx = list(range(min(n, limit) if limit else n))
        self.lens = np.diff(self.offsets)
        assert self.lens[self.idx].max() <= max_len, f"a sequence exceeds model context {max_len}"
        assert batch_tokens >= max_len, "--batch-tokens must hold the longest possible rollout"
        self.trained = np.add.reduceat(self.weights.astype(np.int64), self.offsets[:-1])
        self.batch_tokens, self.seed = batch_tokens, seed
        self.plan = []  # list of (epoch, [rollout indices])
        for e in range(epochs):
            self.plan += [(e, b) for b in self._pack(self._order(e))]
        self.steps_per_epoch = sum(1 for e, _ in self.plan if e == 0)

    def _order(self, epoch):
        o = list(self.idx)
        if epoch:
            random.Random(self.seed + epoch).shuffle(o)
        return o

    def _pack(self, order):
        batches, cur, size = [], [], 0
        for i in order:
            if cur and size + self.lens[i] > self.batch_tokens:
                batches.append(cur)
                cur, size = [], 0
            cur.append(i)
            size += int(self.lens[i])
        if cur and (size >= self.batch_tokens // 2 or not batches):
            batches.append(cur)
        return batches

    def rows(self, step):
        return self.plan[step][1]

    def batch(self, step):
        rows = self.rows(step)
        scale = 1.0 / max(int(self.trained[rows].sum()), 1)  # per-token mean over the batch's trained tokens
        out = []
        for i in rows:
            a, b = self.offsets[i], self.offsets[i + 1]
            out.append(to_datum(self.tokens[a:b], self.weights[a:b], scale))
        return out

    def tokens_in(self, step):
        return int(self.lens[self.rows(step)].sum())


# ---------------------------------------------------------------- schedule / optimizer

def lr_at(args, step, total):
    if args.lr_schedule == "constant":
        return args.lr
    frac = step / total
    if args.lr_schedule == "linear":
        return args.lr * (1 - frac)
    return args.lr * 0.5 * (1 + np.cos(np.pi * frac))


def optimizer_config(args):
    from tinker import types
    return types.DimuonOptimizerConfig() if args.optimizer == "dimuon" else types.AdamOptimizerConfig()


def optim_params(args, lr):
    from tinker import types
    if args.optimizer == "dimuon":  # docs: tune only the LR; keep beta1=0.9, no clipping
        return types.DimuonParams(learning_rate=lr, beta1=args.beta1, grad_clip_norm=args.grad_clip)
    return types.AdamParams(learning_rate=lr, beta1=args.beta1, beta2=0.95, eps=1e-8, weight_decay=0.0,
                            grad_clip_norm=args.grad_clip)


# ---------------------------------------------------------------- cost / setup

def schedule_steps(args, data):
    """Length of the LR schedule (the whole run), independent of --stop-after."""
    return min(len(data.plan), args.max_steps) if args.max_steps else len(data.plan)


def estimate(args, data, train_price, start, end):
    """Cost of the steps this launch runs: [start, end). Batch stats are over the whole schedule."""
    steps = schedule_steps(args, data)
    tokens = sum(data.tokens_in(s) for s in range(start, end))
    sizes = np.array([data.tokens_in(s) for s in range(steps)])
    nroll = np.array([len(data.rows(s)) for s in range(steps)])
    used = {i for s in range(steps) for i in data.rows(s)}
    return {
        "model": args.model, "optimizer": args.optimizer, "lr": args.lr, "lr_schedule": args.lr_schedule,
        "lora_rank": args.rank, "rollouts": len(data.idx), "rollouts_trained": len(used),
        "batch_tokens": args.batch_tokens, "steps_per_epoch": data.steps_per_epoch, "epochs": args.epochs,
        "total_steps": steps, "this_launch_steps": [start, end], "tokens_sent": tokens,
        "batch_tokens_min_mean_max": [int(sizes.min()), int(sizes.mean()), int(sizes.max())],
        "rollouts_per_batch_min_mean_max": [int(nroll.min()), round(float(nroll.mean()), 1), int(nroll.max())],
        "est_cost_usd": round(tokens / 1e6 * train_price, 2),
        "note": "every datum token billed at the train price (unverified); excludes storage",
    }


def tinker_key():
    if os.environ.get("TINKER_API_KEY"):
        return
    for line in open(ROOT / "SECRETS.yaml"):
        if line.startswith("tinker_api_key:"):
            os.environ["TINKER_API_KEY"] = line.split(":", 1)[1].strip().strip("\"' ")
            return
    sys.exit("no tinker_api_key in SECRETS.yaml")


def batch_nll(fwd_result, data):
    """Per-token mean NLL over the trained tokens of a batch (weights > 0), plus per-rollout (nll_sum, n)."""
    num = den = 0.0
    per = []
    for out, d in zip(fwd_result.loss_fn_outputs, data):
        lp = np.asarray(out["logprobs"].tolist(), dtype=np.float64)
        w = np.asarray(d.loss_fn_inputs["weights"].tolist()) > 0
        assert len(lp) == len(w), f"logprobs {len(lp)} vs weights {len(w)}"
        per.append((float(-lp[w].sum()), int(w.sum())))
        num -= lp[w].sum()
        den += w.sum()
    return num / max(den, 1), int(den), per


# ---------------------------------------------------------------- training loop

async def train(args, data, run_dir, total, end):
    import tinker
    from tinker_cookbook import checkpoint_utils

    service = tinker.ServiceClient()
    meta = {"renderer_name": RENDERER, "recipe": "e028_qwen38_sft", "optimizer": args.optimizer}
    rec = checkpoint_utils.get_last_checkpoint(str(run_dir), required_key="state_path")
    if rec is not None and rec.final:
        log.info(f"{run_dir.name} already finished ({rec.state_path}); nothing to do")
        return
    if rec is not None:
        start = int(rec.batch)
        client = await service.create_training_client_from_state_with_optimizer_async(rec.state_path,
                                                                                       user_metadata=meta)
        log.info(f"resumed from {rec.state_path} after {start} steps")
    else:
        start = 0
        client = await service.create_lora_training_client_async(
            base_model=args.model, rank=args.rank, seed=args.seed, user_metadata=meta,
            optimizer=optimizer_config(args))
    metrics_f = open(run_dir / "metrics.jsonl", "a")
    rollout_f = open(run_dir / "rollout_nll.jsonl", "a")  # per rollout: npz/manifest index, NLL before its step

    async def save(done, final=False, tries=1):
        """Checkpoint after `done` optimizer steps (state for resume + sampler weights for inference).
        The final checkpoint and a --stop-after stop are kept indefinitely; periodic ones expire."""
        name = "final" if final else f"step{done:05d}"
        keep = final or done == end
        for attempt in range(tries):
            try:
                paths = await checkpoint_utils.save_checkpoint_async(
                    client, name, str(run_dir), loop_state={"batch": done, "step": done, "final": final},
                    kind="both", ttl_seconds=None if keep else args.ttl_days * 86400)
                log.info(f"checkpoint {name}: {paths}")
                return
            except Exception as exc:
                if attempt == tries - 1:
                    raise
                log.warning(f"save {name} failed ({type(exc).__name__}: {exc}); retrying in 60s")
                await asyncio.sleep(60)

    async def finish(p):
        step, batch, fwd_f, opt_f, lr, t0 = p
        fwd = await fwd_f.result_async()
        opt = await opt_f.result_async()
        nll, n_tok, per = batch_nll(fwd, batch)
        for i, (nll_sum, n) in zip(data.rows(step), per):
            rollout_f.write(json.dumps({"step": step, "idx": int(i), "nll": nll_sum / max(n, 1), "trained": n}) + "\n")
        rollout_f.flush()
        m = {"step": step, "epoch": data.plan[step][0], "lr": lr, "train_nll": nll, "rollouts": len(batch),
             "trained_tokens": n_tok, "tokens": sum(d.model_input.length + 1 for d in batch),
             "seconds": round(time.time() - t0, 1)}
        for k, v in (getattr(fwd, "metrics", None) or {}).items():
            m[f"fwd/{k}"] = v
        for k, v in (getattr(opt, "metrics", None) or {}).items():
            m[f"opt/{k}"] = v
        metrics_f.write(json.dumps(m) + "\n")
        metrics_f.flush()
        log.info(f"step {step + 1}/{total} lr {lr:.3g} nll {nll:.4f} trained {n_tok} ({m['seconds']}s)")
        if not np.isfinite(nll):
            raise RuntimeError(f"non-finite loss at step {step}")

    pending = None  # (step, batch, fwd_future, opt_future, lr, t0) of the step queued last
    if start >= end:
        log.info(f"{run_dir.name} already has {start} steps (stop at {end}); nothing to do")
        return
    for step in range(start, end):
        t0 = time.time()
        batch = data.batch(step)
        lr = lr_at(args, step, total)
        fwd_f = await client.forward_backward_async(batch, loss_fn="cross_entropy")
        opt_f = await client.optim_step_async(optim_params(args, lr))
        if pending is not None:  # one-step lookahead: step s is queued before s-1's results are read
            await finish(pending)
        pending = (step, batch, fwd_f, opt_f, lr, t0)
        done = step + 1
        if args.save_every and done % args.save_every == 0 and done < end:
            await finish(pending)  # drain, so the checkpoint holds exactly `done` steps
            pending = None
            await save(done)
    if pending is not None:
        await finish(pending)
    await save(end, final=end == total, tries=5)  # a --stop-after stop is an ordinary (resumable) checkpoint
    metrics_f.close()
    rollout_f.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="dryrun")
    ap.add_argument("--model", default="Qwen/Qwen3.8-27B", choices=list(MODELS))
    ap.add_argument("--optimizer", default="dimuon", choices=["dimuon", "adamw"])
    ap.add_argument("--lr", type=float, default=1e-3, help="peak LR (Dimuon needs ~10-100x the AdamW LR)")
    ap.add_argument("--lr-schedule", default="linear", choices=["linear", "cosine", "constant"])
    ap.add_argument("--beta1", type=float, default=0.9)
    ap.add_argument("--grad-clip", type=float, default=0.0, help="0 = off (docs: keep off for Dimuon)")
    ap.add_argument("--rank", type=int, default=128)
    ap.add_argument("--batch-tokens", type=int, default=300_000,
                    help="token budget per optimizer step (total tokens incl. masked; >= model context)")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0, help="use only the first N (random) train rollouts")
    ap.add_argument("--max-steps", type=int, default=0, help="cap the schedule length (the run ends there)")
    ap.add_argument("--stop-after", type=int, default=0,
                    help="stop after N steps of the full schedule, saving a resumable checkpoint (LR pilots)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-every", type=int, default=0, help="0 = about 4 checkpoints per run (+ final)")
    ap.add_argument("--ttl-days", type=int, default=7, help="retention of intermediate checkpoints")
    ap.add_argument("--max-cost", type=float, default=50.0, help="refuse to launch above this estimate (USD)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--yes", action="store_true", help="skip the interactive confirmation")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    max_len, train_price = MODELS[args.model]
    data = Batches(DATA / "train.npz", args.batch_tokens, args.limit, max_len, args.seed, args.epochs)
    if not data.plan:
        sys.exit(f"0 steps: {len(data.idx)} rollouts, batch budget {args.batch_tokens} tokens")
    from tinker_cookbook import checkpoint_utils
    run_dir = Path(__file__).parent / "runs" / args.name
    cfg_file = run_dir / "e028_config.json"
    total = schedule_steps(args, data)
    end = min(total, args.stop_after) if args.stop_after else total
    rec = checkpoint_utils.get_last_checkpoint(str(run_dir), required_key="state_path") if run_dir.exists() else None
    start = int(rec.batch) if rec is not None else 0
    est = estimate(args, data, train_price, start, end)
    if not args.save_every:
        args.save_every = max(1, total // 4)
    print(json.dumps(est, indent=1))
    if args.dry_run:
        return
    if start >= end:
        sys.exit(f"{run_dir} already has {start} steps (stop at {end}); nothing to do")
    if est["est_cost_usd"] > args.max_cost:
        sys.exit(f"estimated ${est['est_cost_usd']} > --max-cost {args.max_cost}; raise it explicitly")

    fingerprint = data_fingerprint(DATA / "train.npz")
    if cfg_file.exists():
        cfg = json.load(open(cfg_file))
        old = cfg["args"]
        diff = {k: (old.get(k), vars(args)[k]) for k in RESUME_KEYS if old.get(k) != vars(args)[k]}
        if cfg.get("data_fingerprint") != fingerprint:
            diff["data_fingerprint"] = (cfg.get("data_fingerprint"), fingerprint)
        if diff:
            sys.exit(f"{run_dir} holds a run with different args/data {diff}; use a new --name")
    if not args.yes and input(f"Launch '{args.name}' for ~${est['est_cost_usd']}? [y/N] ").strip().lower() != "y":
        sys.exit("aborted")
    run_dir.mkdir(parents=True, exist_ok=True)
    if not cfg_file.exists():
        json.dump({**est, "args": vars(args), "data": str(DATA), "data_fingerprint": fingerprint},
                  open(cfg_file, "w"), indent=1)
    with open(run_dir / "launches.jsonl", "a") as f:
        f.write(json.dumps({"time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), **est,
                            "argv": sys.argv[1:]}) + "\n")
    fh = logging.FileHandler(run_dir / "train.log")
    fh.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    logging.getLogger().addHandler(fh)
    tinker_key()
    asyncio.run(train(args, data, run_dir, total, end))


if __name__ == "__main__":
    main()
