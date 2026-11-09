"""Shared task-time optimizer integration for SFT and GRPO trainers."""

from collections import defaultdict
import sys

from muon_optimizers import create_muon_composite_optimizer
from training_args_compat import create_optimizer_with_optional_model


BNB_MUON_OPTIMIZERS = frozenset({"muon32bit", "muon8bit"})


def configure_task_optimizer(
    trainer,
    *,
    optimizer_name,
    flash_adamw_quantize=False,
    flash_adamw_master_weight_bits=32,
    normuon_adjust_lr="rms_norm",
    normuon_momentum=0.95,
    normuon_beta2=0.95,
    normuon_megabatch_size=8,
    normuon_use_triton=False,
    muon_adjust_lr="rms_norm",
    muon_momentum=0.95,
    muon_nesterov=True,
    muon_ns_chunk_size=1,
    muon_min_8bit_size=4096,
    muon_scalar_optimizer="adamw_torch",
    log_prefix="[Trainer]",
):
    """Configure a trainer mixin before its lazy optimizer creation."""
    trainer._arc_optimizer_name = optimizer_name
    trainer._arc_flash_adamw_quantize = bool(flash_adamw_quantize)
    trainer._arc_flash_adamw_master_weight_bits = int(
        flash_adamw_master_weight_bits
    )
    trainer._arc_normuon_adjust_lr = normuon_adjust_lr
    trainer._arc_normuon_momentum = float(normuon_momentum)
    trainer._arc_normuon_beta2 = float(normuon_beta2)
    trainer._arc_normuon_megabatch_size = int(normuon_megabatch_size)
    trainer._arc_normuon_use_triton = bool(normuon_use_triton)
    trainer._arc_muon_adjust_lr = muon_adjust_lr
    trainer._arc_muon_momentum = float(muon_momentum)
    trainer._arc_muon_nesterov = bool(muon_nesterov)
    trainer._arc_muon_ns_chunk_size = int(muon_ns_chunk_size)
    trainer._arc_muon_min_8bit_size = int(muon_min_8bit_size)
    trainer._arc_muon_scalar_optimizer = muon_scalar_optimizer
    trainer._arc_optimizer_log_prefix = log_prefix
    return trainer


def make_task_normuon_parameter_groups(model, max_megabatch_size=8):
    """Keep embedding/head tensors on AdamW and orthogonalize LoRA matrices."""
    if max_megabatch_size < 0:
        raise ValueError("NorMuon megabatch size cannot be negative")

    special_parameter_ids = set()
    for getter_name in ("get_input_embeddings", "get_output_embeddings"):
        getter = getattr(model, getter_name, None)
        module = getter() if callable(getter) else None
        if module is not None:
            special_parameter_ids.update(
                id(parameter)
                for parameter in module.parameters(recurse=False)
            )
    for _, module in model.named_modules():
        if type(module).__name__.lower().endswith("embedding"):
            special_parameter_ids.update(
                id(parameter)
                for parameter in module.parameters(recurse=False)
            )

    matrix_parameters = []
    adamw_parameters_by_dtype = defaultdict(list)
    stats = {
        "matrix_parameters": 0,
        "matrix_numel": 0,
        "adamw_parameters": 0,
        "adamw_numel": 0,
        "special_numel": 0,
        "special_parameter_names": [],
    }
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        normalized_name = name.lower()
        is_special = id(parameter) in special_parameter_ids or any(
            marker in normalized_name
            for marker in ("embed_tokens", "embedding", "lm_head")
        )
        if parameter.ndim == 2 and not is_special:
            matrix_parameters.append(parameter)
            stats["matrix_parameters"] += 1
            stats["matrix_numel"] += parameter.numel()
        else:
            adamw_parameters_by_dtype[parameter.dtype].append(parameter)
            stats["adamw_parameters"] += 1
            stats["adamw_numel"] += parameter.numel()
            if is_special:
                stats["special_numel"] += parameter.numel()
                stats["special_parameter_names"].append(name)

    if not matrix_parameters:
        raise RuntimeError(
            "NorMuon found no eligible trainable matrix parameters; use an "
            "AdamW optimizer for this adapter configuration"
        )

    if max_megabatch_size:
        matrices_by_shape = defaultdict(list)
        for parameter in matrix_parameters:
            matrices_by_shape[(tuple(parameter.shape), parameter.dtype)].append(
                parameter
            )
        matrix_groups = [
            parameters[start : start + max_megabatch_size]
            for parameters in matrices_by_shape.values()
            for start in range(0, len(parameters), max_megabatch_size)
        ]
    else:
        matrix_groups = [matrix_parameters]

    parameter_groups = [
        {
            "params": parameters,
            "algorithm": "normuon",
            "weight_decay": 0.0,
        }
        for parameters in matrix_groups
    ]
    for dtype, parameters in sorted(
        adamw_parameters_by_dtype.items(),
        key=lambda item: str(item[0]),
    ):
        parameter_groups.append(
            {
                "params": parameters,
                "algorithm": "adamw",
                "weight_decay": 0.0,
                "parameter_dtype": str(dtype),
            }
        )

    stats["adamw_dtypes"] = sorted(
        str(dtype) for dtype in adamw_parameters_by_dtype
    )
    stats["matrix_optimizer_groups"] = len(matrix_groups)
    stats["max_megabatch_size"] = max_megabatch_size or None
    return parameter_groups, stats


class TaskOptimizerTrainerMixin:
    """Add opt-in FlashAdamW, NorMuon, and bnb Muon to a Trainer."""

    def create_optimizer(self, model=None):
        target_model = self.model if model is None else model
        optimizer_name = getattr(
            self,
            "_arc_optimizer_name",
            self.args.optim,
        )
        if optimizer_name not in {
            "flash_adamw",
            "normuon",
            *BNB_MUON_OPTIMIZERS,
        }:
            return create_optimizer_with_optional_model(super().create_optimizer, model)
        if self.optimizer is not None:
            return self.optimizer

        if optimizer_name in BNB_MUON_OPTIMIZERS:
            parameter_groups, stats = make_task_normuon_parameter_groups(
                target_model,
                # bnb Muon does its own same-shape chunking. Keeping one
                # annotated matrix group avoids changing optimizer math.
                max_megabatch_size=0,
            )
            adjust_lr = getattr(self, "_arc_muon_adjust_lr", "rms_norm")
            momentum_bits = 8 if optimizer_name == "muon8bit" else 32
            self.optimizer = create_muon_composite_optimizer(
                parameter_groups,
                momentum_bits=momentum_bits,
                lr=self.args.learning_rate,
                momentum=float(getattr(self, "_arc_muon_momentum", 0.95)),
                betas=(self.args.adam_beta1, self.args.adam_beta2),
                eps=self.args.adam_epsilon,
                weight_decay=self.args.weight_decay,
                nesterov=bool(getattr(self, "_arc_muon_nesterov", True)),
                adjust_lr=None if adjust_lr == "none" else adjust_lr,
                ns_chunk_size=int(
                    getattr(self, "_arc_muon_ns_chunk_size", 1)
                ),
                min_8bit_size=int(
                    getattr(self, "_arc_muon_min_8bit_size", 4096)
                ),
                scalar_optimizer=getattr(
                    self,
                    "_arc_muon_scalar_optimizer",
                    "adamw_torch",
                ),
                log_prefix=getattr(
                    self,
                    "_arc_optimizer_log_prefix",
                    "[Trainer]",
                ),
            )
            print(
                f"{getattr(self, '_arc_optimizer_log_prefix', '[Trainer]')} "
                "Muon parameter split: "
                f"matrix_parameters={stats['matrix_parameters']}/"
                f"{stats['matrix_numel']:,} values, "
                f"AdamW_parameters={stats['adamw_parameters']}/"
                f"{stats['adamw_numel']:,} values, "
                f"special_embedding_head={stats['special_numel']:,} values",
                file=sys.__stdout__,
                flush=True,
            )
            return self.optimizer

        if optimizer_name == "normuon":
            try:
                import dion
                from dion import NorMuon
            except ImportError as error:
                raise RuntimeError(
                    "--optimizer normuon requires DION. Install the dion "
                    "package or add the cloned DION repository root to "
                    "PYTHONPATH."
                ) from error

            max_megabatch_size = int(
                getattr(self, "_arc_normuon_megabatch_size", 8)
            )
            parameter_groups, stats = make_task_normuon_parameter_groups(
                target_model,
                max_megabatch_size=max_megabatch_size,
            )
            adjust_lr = getattr(self, "_arc_normuon_adjust_lr", "rms_norm")
            self.optimizer = NorMuon(
                parameter_groups,
                distributed_mesh=None,
                lr=self.args.learning_rate,
                mu=float(getattr(self, "_arc_normuon_momentum", 0.95)),
                muon_beta2=float(
                    getattr(self, "_arc_normuon_beta2", 0.95)
                ),
                betas=(self.args.adam_beta1, self.args.adam_beta2),
                weight_decay=0.0,
                epsilon=self.args.adam_epsilon,
                adjust_lr=None if adjust_lr == "none" else adjust_lr,
                use_triton=bool(
                    getattr(self, "_arc_normuon_use_triton", False)
                ),
                use_polar_express=True,
            )
            print(
                f"{getattr(self, '_arc_optimizer_log_prefix', '[Trainer]')} "
                "NorMuon initialized: "
                f"dion={getattr(dion, '__version__', 'unknown')}, "
                f"matrix_parameters={stats['matrix_parameters']}/"
                f"{stats['matrix_numel']:,} values, "
                f"AdamW_parameters={stats['adamw_parameters']}/"
                f"{stats['adamw_numel']:,} values, "
                f"special_embedding_head={stats['special_numel']:,} values, "
                f"matrix_groups={stats['matrix_optimizer_groups']}, "
                f"max_megabatch_size={stats['max_megabatch_size']}, "
                f"adjust_lr={adjust_lr}, "
                "use_triton="
                f"{getattr(self, '_arc_normuon_use_triton', False)}",
                file=sys.__stdout__,
                flush=True,
            )
            return self.optimizer

        try:
            import flashoptim
            from flashoptim import FlashAdamW
        except ImportError as error:
            raise RuntimeError(
                "--optimizer flash_adamw requires FlashOptim. Install it "
                "or add the cloned FlashOptim repository root to PYTHONPATH."
            ) from error

        trainable_parameters = [
            parameter
            for parameter in target_model.parameters()
            if parameter.requires_grad
        ]
        trainable_numel = sum(
            parameter.numel() for parameter in trainable_parameters
        )
        quantize = bool(
            getattr(self, "_arc_flash_adamw_quantize", False)
        )
        master_weight_bits = int(
            getattr(
                self,
                "_arc_flash_adamw_master_weight_bits",
                32,
            )
        )
        self.optimizer = FlashAdamW(
            trainable_parameters,
            lr=self.args.learning_rate,
            betas=(self.args.adam_beta1, self.args.adam_beta2),
            eps=self.args.adam_epsilon,
            weight_decay=self.args.weight_decay,
            quantize=quantize,
            master_weight_bits=master_weight_bits,
            fused=True,
        )
        print(
            f"{getattr(self, '_arc_optimizer_log_prefix', '[Trainer]')} "
            "FlashAdamW initialized: "
            f"flashoptim={getattr(flashoptim, '__version__', 'unknown')}, "
            f"parameters={len(trainable_parameters)}, "
            f"values={trainable_numel:,}, "
            f"quantized_moments={quantize}, "
            f"master_weight_bits={master_weight_bits}",
            file=sys.__stdout__,
            flush=True,
        )
        return self.optimizer
