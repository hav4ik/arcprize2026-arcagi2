"""Single-device, restart-safe gradient accumulation schedules.

Trainer sees one cheap descriptor per optimizer update (outer GA=1). Only
training_step materializes that update's microbatches, one at a time, and the
ordinary Trainer training_step performs each backward with /actual_GA. This
avoids depending on private fixed-GA epoch-loop arithmetic or prefetching ARC
corruptions from the next update before saving its collator RNG state.
"""

from __future__ import annotations

import argparse
from bisect import bisect_right
from dataclasses import dataclass
import json

import torch
from torch.utils.data import DataLoader


GA_SCHEDULE_FORMAT = "arc-ga-schedule-v1"
GA_SAMPLER = "cpu-seeded-randperm-v1"


def parse_integer_list(value: str) -> list[int]:
    try:
        result = json.loads(value)
    except (ValueError, TypeError) as error:
        raise argparse.ArgumentTypeError("expected a JSON integer list, e.g. '[24,32]'") from error
    if not isinstance(result, list) or not result or any(type(n) is not int for n in result):
        raise argparse.ArgumentTypeError("expected a nonempty JSON integer list")
    return result


@dataclass(frozen=True)
class GASchedule:
    values: tuple[int, ...]
    milestones: tuple[int, ...]

    def __post_init__(self):
        if not self.values or len(self.values) != len(self.milestones):
            raise ValueError("GA schedule and milestones must have the same nonzero length")
        if any(type(value) is not int or value <= 0 for value in self.values):
            raise ValueError("GA schedule values must be positive integers")
        if any(type(step) is not int or step < 0 for step in self.milestones):
            raise ValueError("GA milestones must be nonnegative integers")
        if self.milestones[0] != 0 or any(a >= b for a, b in zip(self.milestones, self.milestones[1:])):
            raise ValueError("GA milestones must start at 0 and strictly increase")

    def stage(self, completed_steps: int) -> int:
        if type(completed_steps) is not int or completed_steps < 0:
            raise ValueError("completed steps must be a nonnegative integer")
        return bisect_right(self.milestones, completed_steps) - 1

    def at(self, completed_steps: int) -> int:
        """GA for the next update, after completed_steps optimizer updates."""
        return self.values[self.stage(completed_steps)]

    def consumed_microbatches(self, completed_steps: int) -> int:
        self.stage(completed_steps)  # Validate before integrating the schedule.
        total = 0
        for i, (start, ga) in enumerate(zip(self.milestones, self.values)):
            stop = min(completed_steps, self.milestones[i + 1]) if i + 1 < len(self.values) else completed_steps
            total += max(0, stop - start) * ga
        return total

    def contract(self) -> dict:
        return {"format": GA_SCHEDULE_FORMAT, "values": list(self.values),
                "milestones": list(self.milestones), "sampler": GA_SAMPLER}

    def cursor(self, completed_steps: int, micro_batch_size: int) -> dict:
        consumed = self.consumed_microbatches(completed_steps)
        return {"completed_steps": completed_steps, "consumed_microbatches": consumed,
                "consumed_packs": consumed * micro_batch_size,
                "next_stage": self.stage(completed_steps), "next_ga": self.at(completed_steps)}


def schedule_from_options(options) -> GASchedule | None:
    values = getattr(options, "gradient_accumulation_schedule", None)
    milestones = getattr(options, "gradient_accumulation_milestones", None)
    if values is None and milestones is None:
        return None
    if values is None or milestones is None:
        raise ValueError("--gradient-accumulation-schedule and --gradient-accumulation-milestones must be supplied together")
    return GASchedule(tuple(values), tuple(milestones))


def schedule_from_contract(contract: dict) -> GASchedule | None:
    payload = contract.get("gradient_accumulation_schedule")
    if payload is None:
        return None
    schedule = GASchedule(tuple(payload["values"]), tuple(payload["milestones"]))
    if payload != schedule.contract():
        raise ValueError("unsupported GA schedule contract or sampler")
    return schedule


def required_training_packs(options) -> int:
    schedule = schedule_from_options(options)
    microbatches = (schedule.consumed_microbatches(options.max_steps) if schedule
                    else options.gradient_accumulation_steps * options.max_steps)
    return microbatches * options.per_device_train_batch_size


def _update_descriptor(features):
    if len(features) != 1:
        raise RuntimeError("scheduled GA requires exactly one descriptor per outer batch")
    return {"arc_update_index": features[0]}


def make_scheduled_trainer_class(base_class, *, schedule: GASchedule, micro_batch_size: int, seed: int):
    """Keep optimizer/LR/clipping/save/eval in Trainer; own only accumulation.

    Intentionally single-process and num_workers=0. The seeded permutation is
    independent of model/collator RNG and update grouping. Epoch is Trainer's
    schedule progress; arc_data_fraction reports the actual dataset fraction.
    Full resume from a *fixed*-GA checkpoint is rejected by the recipe contract.
    """

    class ScheduledGATrainer(base_class):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            if self.args.gradient_accumulation_steps != 1 or self.args.per_device_train_batch_size != 1:
                raise ValueError("scheduled GA requires outer Trainer batch_size=1 and GA=1")
            if (self.accelerator.num_processes != 1 or self.args.n_gpu > 1
                    or getattr(self, "is_deepspeed_enabled", False)
                    or getattr(self, "is_fsdp_enabled", False)):
                raise ValueError("GA schedules currently support one device without DDP/FSDP/DeepSpeed")
            if self.args.dataloader_num_workers or self.args.ignore_data_skip:
                raise ValueError("scheduled GA requires workers=0 and ignore_data_skip=False")
            if self.accelerator.gradient_accumulation_steps != 1:
                raise ValueError("scheduled GA requires Accelerator outer GA=1 (no double loss division)")
            if self.args.max_steps <= schedule.milestones[-1]:
                raise ValueError("all GA milestones must be smaller than max_steps")
            required = schedule.consumed_microbatches(self.args.max_steps) * micro_batch_size
            if self.train_dataset is None or len(self.train_dataset) < required:
                raise ValueError(f"GA schedule requires at least {required} packed samples (one pass, no repeats)")
            self.model_accepts_loss_kwargs = False
            if self.compute_loss_func is not None:
                raise ValueError("scheduled GA does not support a custom compute_loss_func normalizer")
            self.arc_ga_permutation = None
            self.arc_ga_last_stage = None
            self.arc_ga_last_completed_update = None

        def get_train_dataloader(self):
            # No underlying row or corruption is touched by descriptor prefetch
            # or skip_first_batches during full-state resume.
            generator = torch.Generator(device="cpu").manual_seed(seed)
            self.arc_ga_permutation = torch.randperm(len(self.train_dataset), generator=generator)
            loader = DataLoader(
                range(self.args.max_steps), batch_size=1, shuffle=False,
                collate_fn=_update_descriptor, num_workers=0, pin_memory=False,
                generator=torch.Generator(device="cpu").manual_seed(seed),
            )
            return self.accelerator.prepare_data_loader(loader, device_placement=False)

        def training_step(self, model, inputs, num_items_in_batch=None):
            completed = int(self.state.global_step)
            if inputs.get("arc_update_index") != completed:
                raise RuntimeError(f"GA data cursor mismatch: descriptor={inputs}, completed={completed}")
            if self.arc_ga_permutation is None:
                raise RuntimeError("GA sampler has not been initialized")
            ga = schedule.at(completed)
            stage = schedule.stage(completed)
            if stage != self.arc_ga_last_stage:
                print(f"[DiffusionGemma] GA schedule: update={completed + 1}, GA={ga}, "
                      f"micro_batch={micro_batch_size}, consumed_packs="
                      f"{schedule.consumed_microbatches(completed) * micro_batch_size:,}", flush=True)
                self.arc_ga_last_stage = stage
            offset = schedule.consumed_microbatches(completed) * micro_batch_size
            total = torch.zeros((), device=self.args.device)
            outer_ga = self.current_gradient_accumulation_steps
            self.current_gradient_accumulation_steps = ga
            try:
                for micro in range(ga):
                    start = offset + micro * micro_batch_size
                    indices = self.arc_ga_permutation[start:start + micro_batch_size].tolist()
                    batch = self.data_collator([self.train_dataset[index] for index in indices])
                    self.accelerator.gradient_state._set_sync_gradients(micro == ga - 1)
                    # Preserve the original per-microbatch diagnostics, loss,
                    # checkpointing, fused gradients and backward implementation.
                    total += super().training_step(model, batch, num_items_in_batch=None)
                    del batch
                    if micro != ga - 1:
                        self.control = self.callback_handler.on_substep_end(self.args, self.state, self.control)
            finally:
                self.current_gradient_accumulation_steps = outer_ga
                self.accelerator.gradient_state._set_sync_gradients(True)
            self.arc_ga_last_completed_update = completed + 1
            return total

        def arc_ga_checkpoint_state(self):
            completed = int(self.state.global_step)
            if self.arc_ga_last_completed_update != completed:
                raise RuntimeError("cannot checkpoint an incomplete GA update")
            return schedule.cursor(completed, micro_batch_size)

        def log(self, logs, *args, **kwargs):
            if "loss" in logs and self.state.global_step:
                completed = int(self.state.global_step)
                cursor = schedule.cursor(completed, micro_batch_size)
                logs = {**logs, "arc_gradient_accumulation_steps": schedule.at(completed - 1),
                        "arc_consumed_packs": cursor["consumed_packs"],
                        "arc_data_fraction": cursor["consumed_packs"] / len(self.train_dataset)}
            return super().log(logs, *args, **kwargs)

    return ScheduledGATrainer
