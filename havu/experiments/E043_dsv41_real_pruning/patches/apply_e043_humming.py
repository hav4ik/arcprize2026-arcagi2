#!/usr/bin/env python3
"""E043 low-bit round: serve DeepSeek-V4.1's routed experts through Humming with the real 1017 weights.

Usage: apply_e043_humming.py <site-packages>/vllm      (idempotent, marker "E043H:")

Env at serve time (unset = model unchanged):
  DSV41_EXPERT_HUMMING     Humming weight schema JSON for the routed experts, e.g.
                           {"quant_method": "gptq", "bits": 2, "group_size": 64, "sym": false, "desc_act": false}
                           (E042's hook: the experts get HummingMoEMethod instead of DeepseekV41Mxfp4MoEMethod).
  DSV41_EXPERT_HUMMING_W2  optional JSON merged into that schema for the down projection only (e.g. {"bits": 3}):
                           2-bit gate/up with 3-bit down, as in Project Maya's Maya-S.
  E043_GPTQ_DIR            optional directory of precomputed GPTQ tensors (see e043_humming.py); without it Humming
                           rounds the decoded weights itself (RTN).
The checkpoint's MXFP4 experts are decoded exactly to BF16 at load (e043_humming.dequant_expert_weights). Router,
attention, shared expert, Engram and every other tensor are loaded and executed as released.
"""
import shutil
import sys
from pathlib import Path

MARK = "E043H:"


def edit(path: Path, pairs):
    text = path.read_text()
    if MARK in text:
        print(f"already patched: {path}")
        return
    for old, new in pairs:
        if text.count(old) != 1:
            raise SystemExit(f"{path}: anchor found {text.count(old)}x: {old[:120]!r}")
        text = text.replace(old, new)
    path.write_text(text)
    print(f"patched: {path}")


def main():
    vllm = Path(sys.argv[1])
    shutil.copy(Path(__file__).with_name("e043_humming.py"), vllm / "e043_humming.py")
    print(f"installed {vllm / 'e043_humming.py'}")

    edit(vllm / "models/deepseek_v4_1/quant_config.py", [(
        "            return DeepseekV41Mxfp4MoEMethod(layer.moe_config)\n",
        "            # E043H: DSV41_EXPERT_HUMMING=<humming weight schema JSON> serves the routed experts through\n"
        "            # Humming (E042's hook); DSV41_EXPERT_HUMMING_W2 overrides the schema for the down projection.\n"
        "            import json as _json\n"
        "            import os as _os\n"
        "            _hcfg = _os.environ.get(\"DSV41_EXPERT_HUMMING\")\n"
        "            if _hcfg:\n"
        "                from vllm.model_executor.layers.quantization.humming import HummingConfig\n"
        "                _m = HummingConfig(full_config=_json.loads(_hcfg)).get_quant_method(layer, prefix)\n"
        "                _w2 = _os.environ.get(\"DSV41_EXPERT_HUMMING_W2\")\n"
        "                if _w2:\n"
        "                    _c2 = {**_json.loads(_hcfg), **_json.loads(_w2)}\n"
        "                    _m.weight_schema_w2 = HummingConfig(full_config=_c2).get_quant_method(\n"
        "                        layer, prefix).weight_schema\n"
        "                return _m\n"
        "            return DeepseekV41Mxfp4MoEMethod(layer.moe_config)\n",
    )])

    edit(vllm / "models/deepseek_v4_1/nvidia/model.py", [(
        "        stacked_params_mapping = [\n            # (param_name, shard_name, shard_id)\n"
        "            (\"gate_up_proj\", \"w1\", 0),\n",
        "        # E043H: real-weight low-bit experts through Humming: decode each MXFP4 expert matrix to exact BF16.\n"
        "        import os as _e043_os\n"
        "        if _e043_os.environ.get(\"DSV41_EXPERT_HUMMING\"):\n"
        "            from vllm import e043_humming\n"
        "            weights = e043_humming.dequant_expert_weights(weights)\n"
        "        stacked_params_mapping = [\n            # (param_name, shard_name, shard_id)\n"
        "            (\"gate_up_proj\", \"w1\", 0),\n",
    )])

    edit(vllm / "model_executor/layers/quantization/humming.py", [
        (
            "            if is_unquantized:\n"
            "                assert isinstance(self.weight_schema, _hm.HummingWeightSchema)\n"
            "                f16_dtype = _hm.DataType.from_torch_dtype(layer.param_dtype)\n"
            "                has_global_scale = \"TENSOR\" in str(self.weight_schema.weight_scale_type)\n"
            "                tensor_list = _hm.quantize_weight(\n"
            "                    weight=loaded_weight,\n"
            "                    dtype=self.weight_schema.b_dtype,\n"
            "                    scale_dtype=self.weight_schema.bs_dtype or f16_dtype,\n"
            "                    group_size=self.weight_schema.weight_scale_group_size,\n"
            "                    has_zero_point=self.weight_schema.has_zero_point,\n"
            "                    has_global_scale=has_global_scale,\n"
            "                    is_fp_zero_point=self.weight_schema.is_fp_zero_point,\n"
            "                    pack=True,\n"
            "                )\n",
            "            if is_unquantized:\n"
            "                # E043H: per-sublayer schema (down projection override) and precomputed GPTQ tensors.\n"
            "                _ws = (getattr(self, \"weight_schema_w2\", self.weight_schema)\n"
            "                       if shard_id == \"w2\" else self.weight_schema)\n"
            "                assert isinstance(_ws, _hm.HummingWeightSchema)\n"
            "                f16_dtype = _hm.DataType.from_torch_dtype(layer.param_dtype)\n"
            "                has_global_scale = \"TENSOR\" in str(_ws.weight_scale_type)\n"
            "                from vllm import e043_humming as _e043h\n"
            "                _pre = _e043h.precomputed(layer.layer_name, expert_id, shard_id)\n"
            "                tensor_list = _pre if _pre is not None else _hm.quantize_weight(\n"
            "                    weight=loaded_weight,\n"
            "                    dtype=_ws.b_dtype,\n"
            "                    scale_dtype=_ws.bs_dtype or f16_dtype,\n"
            "                    group_size=_ws.weight_scale_group_size,\n"
            "                    has_zero_point=_ws.has_zero_point,\n"
            "                    has_global_scale=has_global_scale,\n"
            "                    is_fp_zero_point=_ws.is_fp_zero_point,\n"
            "                    pack=True,\n"
            "                )\n",
        ),
        (
            "            \"w2\": {\n                \"shape_n\": hidden_size,\n"
            "                \"shape_k\": intermediate_size_per_partition,\n"
            "                \"tensors_attrs\": self.weight_schema.get_padded_tensors_attrs(\n",
            "            \"w2\": {\n                \"shape_n\": hidden_size,\n"
            "                \"shape_k\": intermediate_size_per_partition,\n"
            "                \"tensors_attrs\": getattr(self, \"weight_schema_w2\", self.weight_schema)"
            ".get_padded_tensors_attrs(\n",
        ),
        (
            "            sublayer_configs=layer.sublayer_configs,\n"
            "            weight_schema=self.weight_schema,\n",
            "            sublayer_configs=layer.sublayer_configs,\n"
            "            weight_schema=({\"w13\": self.weight_schema, \"w2\": self.weight_schema_w2}\n"
            "                           if hasattr(self, \"weight_schema_w2\") else self.weight_schema),\n",
        ),
    ])

    edit(vllm / "model_executor/layers/quantization/utils/humming_utils.py", [(
        "                shape_k=configs[\"shape_k\"],\n"
        "                weight_schema=weight_schema,\n",
        "                shape_k=configs[\"shape_k\"],\n"
        "                # E043H: per-sublayer weight schemas ({\"w13\": ..., \"w2\": ...}).\n"
        "                weight_schema=(weight_schema[sublayer_name] if isinstance(weight_schema, dict)\n"
        "                               else weight_schema),\n",
    )])


    # humming 0.1.12 has no WeightScaleType.GROUP_TENSOR (the second-level scale moved to weight_scale_2_type);
    # the fork's quant-key mapping for the native "humming" schema still references it.
    p = vllm / "model_executor/layers/quantization/utils/humming_utils.py"
    text = p.read_text()
    old = "    if schema.weight_scale_type == WeightScaleType.GROUP_TENSOR:\n"
    new = ("    # E043H2: humming 0.1.12 compatibility (no GROUP_TENSOR)\n"
           "    if str(getattr(schema, \"weight_scale_2_type\", None) or \"\").lower().endswith(\"tensor\"):\n")
    if "E043H2:" in text:
        print(f"already patched (H2): {p}")
    else:
        assert text.count(old) == 1, text.count(old)
        p.write_text(text.replace(old, new))
        print(f"patched (H2): {p}")

    # humming 0.1.12's quantize_weight() takes weight_scale_2_type instead of has_global_scale.
    p = vllm / "model_executor/layers/quantization/humming.py"
    text = p.read_text()
    old = "                    has_global_scale=has_global_scale,\n"
    new = ("                    # E043H3: humming 0.1.12 API\n"
           "                    weight_scale_2_type=(\"tensor\" if has_global_scale else None),\n")
    if "E043H3:" in text:
        print(f"already patched (H3): {p}")
    else:
        assert text.count(old) >= 1, text.count(old)  # MoE (E043H block) and dense linear method
        p.write_text(text.replace(old, new))
        print(f"patched (H3, {text.count(old)}x): {p}")

    # humming 0.1.12 gives zero_point no scale_type, so the MoE weight loader rejects it ("quant method must be one
    # of ..."). Zero points are per-group like the scales (packed along N): load them with the group-scale path.
    text = p.read_text()
    old = ("    if \"scale_type\" in extra_attrs:\n"
           "        extra_attrs[\"quant_method\"] = extra_attrs[\"scale_type\"]\n")
    new = old + ("    elif name == \"zero_point\":  # E043H4: humming 0.1.12 zero points are group-wise\n"
                 "        extra_attrs[\"quant_method\"] = \"group\"\n")
    if "E043H4:" in text:
        print(f"already patched (H4): {p}")
    else:
        assert text.count(old) == 1, text.count(old)
        p.write_text(text.replace(old, new))
        print(f"patched (H4): {p}")

    # Router shapes for physically pruned low-bit exports (E96 / E104 / E112 / E128 / E160 with top-6).
    p = vllm / "models/deepseek_v4_1/nvidia/router.py"
    lines = p.read_text().splitlines(keepends=True)
    i = next(i for i, l in enumerate(lines) if l.startswith("_SUPPORTED_ROUTING_SHAPES = frozenset({"))
    add = [x for x in ("(96, 6)", "(104, 6)", "(112, 6)", "(128, 6)", "(160, 6)") if x not in lines[i]]
    if add:
        lines[i] = lines[i].rstrip()[:-2] + ", " + ", ".join(add) + "})\n"
        p.write_text("".join(lines))
        print(f"router shapes added (H5): {', '.join(add)}")


if __name__ == "__main__":
    main()
