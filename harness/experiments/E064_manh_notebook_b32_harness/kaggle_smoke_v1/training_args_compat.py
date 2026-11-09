"""Small, import-safe compatibility helpers shared by TTT and pre-SFT."""

import inspect
import math


def create_optimizer_with_optional_model(create_optimizer, model=None):
    """Forward Transformers 5.x's prepared model without breaking 4.x parents.

    Inspect the callable rather than catching TypeError from optimizer creation:
    an optimizer's own TypeError must propagate instead of retrying initialization.
    """
    if model is not None:
        try:
            parameter = inspect.signature(create_optimizer).parameters.get("model")
        except (TypeError, ValueError):
            parameter = None
        if parameter is not None:
            if parameter.kind == inspect.Parameter.POSITIONAL_ONLY:
                return create_optimizer(model)
            return create_optimizer(model=model)
    return create_optimizer()


def warmup_dataclass_arguments(arguments_class, warmup_ratio):
    """Preserve ratio semantics across old and new HF training arguments.

    Unsloth wrappers accept **kwargs, so inspect the underlying dataclass
    fields, not the public call signature. New Transformers uses fractional
    warmup_steps where older versions expose warmup_ratio. Pass only one:
    silently dropping warmup_ratio otherwise changes the first LR to full LR.
    """
    ratio = float(warmup_ratio)
    if not math.isfinite(ratio) or not 0.0 <= ratio < 1.0:
        raise ValueError("warmup_ratio must be finite and in [0, 1)")
    fields = getattr(arguments_class, "__dataclass_fields__", {})
    if "warmup_ratio" in fields:
        return {"warmup_ratio": ratio}
    if "warmup_steps" in fields:
        return {"warmup_steps": ratio}
    raise RuntimeError(
        "Installed training arguments expose neither warmup_ratio nor warmup_steps"
    )
