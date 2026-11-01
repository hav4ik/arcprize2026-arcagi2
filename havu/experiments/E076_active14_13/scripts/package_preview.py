#!/usr/bin/env python3
"""Build a small overlay for the exact E076 seed14 checkpoint and its evidence."""
import hashlib
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[3]
EXP=ROOT/'experiments/E076_active14_13'
OLD=ROOT/'experiments/E074_active_params'
UPLOAD=EXP/'upload'
BUNDLE=UPLOAD/'bundle'
REPO='chankhavu/DeepSeek-V4.1-Flash-1017-arc-img-E128-A14B-preview'


def put(name,value):
    p=BUNDLE/name;p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(value,indent=2)+'\n')


def main():
    BUNDLE.mkdir(exist_ok=True)
    active=json.loads((EXP/'configs/seed14.json').read_text())
    alternative=json.loads((OLD/'results/candidate_arc/schedule.json').read_text())
    assert len(active['top_k'])==40 and sum(active['top_k'])==184
    assert sum(alternative['top_k'])==200
    cfg=json.loads((UPLOAD/'causal_config.json').read_text())
    eng_cfg=json.loads((UPLOAD/'engram_config.json').read_text())
    for k in set(cfg['text_config'])|set(eng_cfg['text_config']):
        if k.startswith('engram_'):assert cfg['text_config'].get(k)==eng_cfg['text_config'].get(k),k
    cfg['text_config']['num_experts_per_tok_per_layer']=active['top_k']
    cfg['e076_active_pruning']={'candidate':'seed14','status':'preview; full ARC evaluation pending',
        'active_experts_file':'active_experts.json','mean_top_k':4.6,
        'estimated_decode_active_b':active['estimated_decode_active_b'],
        'requires_per_layer_topk_patch':True,
        'warning':'Unpatched servers can ignore the per-layer field and execute the six-expert baseline. The included E043 patch reads E074_TOPK_FILE.',
        'weight_updates':False,'calibration_family':'E065 paired image/tool and text/tool traces'}
    put('config.json',cfg);put('active_experts.json',active)
    put('schedules/active_experts.json',active)
    put('schedules/e074_mean5_14p58B.json',alternative)
    put('schedules/baseline_k6.json',json.loads((OLD/'results/baseline_arc/schedule.json').read_text()))
    index=json.loads((UPLOAD/'causal_model.safetensors.index.json').read_text())
    eng_index=json.loads((UPLOAD/'engram_model.safetensors.index.json').read_text())
    weight_map={k:v for k,v in index['weight_map'].items() if not v.startswith('model-engram-')}
    weight_map.update({k:v for k,v in eng_index['weight_map'].items() if v.startswith('model-engram-')})
    index['weight_map']=weight_map;put('model.safetensors.index.json',index)
    report=json.loads((EXP/'results/report.json').read_text())
    nll=report['training_nll']['seed14_select_select']
    evidence={'candidate':'seed14','status':'preliminary; E076 broader search still running',
        'calibration_data':'E065 image/tool and paired text/tool traces; ARC training parents only',
        'selection_family_count':24,'relative_nll_increase':nll,
        'short_diagnostics':{k:v for k,v in report['training_diagnostics'].items()
            if k.startswith(('baseline_calibration','seed14_select','uniform4_select'))},
        'diagnostic_warning':'12 ARC training originals x1 with a 24k base output budget; not validation-set accuracy',
        'full_arc_evaluation':'not yet run for seed14; do not attribute E074 results to this schedule',
        'comparison_13p17B':report['training_nll']['uniform4_select_select']}
    put('evaluation/seed14_preview.json',evidence)
    put('evaluation/e074_mean5_kaggle120.json',json.loads((OLD/'results/arc_report.json').read_text()))
    sources={n:json.loads((UPLOAD/(n+'.json')).read_text()) for n in ('source_causal','source_engram')}
    files={f['path']:{'source_repo':sources['source_causal']['repo'],
           'source_revision':sources['source_causal']['revision'],**f}
           for f in sources['source_causal']['files'] if f['path'] not in ('README.md','config.json','model.safetensors.index.json')}
    for f in sources['source_engram']['files']:
        if f['path'].startswith('model-engram-') and f['path'].endswith('.safetensors'):
            files[f['path']]={'source_repo':sources['source_engram']['repo'],
                'source_revision':sources['source_engram']['revision'],**f}
    assert set(weight_map.values())<=set(files)
    assert len([f for f in files if f.startswith('gptq/') and f.endswith('.safetensors')])==40
    manifest={'destination_repo':REPO,'private_required':True,
        'causal_source':{'repo':sources['source_causal']['repo'],'revision':sources['source_causal']['revision']},
        'engram_source':{'repo':sources['source_engram']['repo'],'revision':sources['source_engram']['revision']},
        'unchanged_weight_files':{k:v for k,v in files.items() if k.endswith('.safetensors')},
        'stored_weight_bytes':sum(v['size'] for k,v in files.items() if k.endswith('.safetensors')),
        'weight_modification':'none; active sparsity is a static routing schedule',
        'default_schedule':'active_experts.json','default_active_slots':184,
        'alternative_schedule':'schedules/e074_mean5_14p58B.json','alternative_active_slots':200}
    put('candidate_manifest.json',manifest)
    (BUNDLE/'serving').mkdir(exist_ok=True)
    shutil.copyfile(OLD/'scripts/apply_topk.py',BUNDLE/'serving/apply_active_topk_e043.py')
    # Preserve exact implementation evidence for the tested experimental stack.
    shutil.copyfile(OLD/'scripts/test_router_gpu.py',BUNDLE/'serving/test_active_router_e043.py')
    (BUNDLE/'README.md').write_text(CARD)
    (BUNDLE/'serving/ACTIVE_EXPERTS.md').write_text(SERVING)
    overlays={str(p.relative_to(BUNDLE)):{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
              for p in BUNDLE.rglob('*') if p.is_file()}
    (UPLOAD/'upload_plan.json').write_text(json.dumps({'repo':REPO,'private':True,'copy':files,'overlay':overlays},indent=2)+'\n')
    print(json.dumps({'repo':REPO,'copied_files':len(files),'overlay_files':len(overlays),
                      'stored_weight_bytes':manifest['stored_weight_bytes']}))


CARD='''---
license: mit
library_name: vllm
tags:
- deepseek
- mixture-of-experts
- pruning
- gptq
- experimental
- arc-agi
pipeline_tag: image-text-to-text
base_model: deepseek-ai/DeepSeek-V4.1-Flash-1017
---

# DeepSeek V4.1 Flash 1017 — causal image E128, ~14.02B active preview

**Private experimental preview, E076 `seed14`, 2026-10-31. Full ARC evaluation of
this schedule is pending.** This checkpoint keeps 128 stored routed experts in
each of 40 layers and activates 4 in 28 layers and 6 in 12 layers. The shared
expert remains active. Mean routed top-k is 4.6.

The exact routed-parameter reduction is 1,981,808,640 per decoded token. Relative
to the rounded advertised 16B baseline, the estimated decode-active size is
14.018B. This excludes vision-encoder work and is not a total stored-parameter,
download-size, VRAM or proportional-speedup claim.

## Contents and provenance

The quantized weights are unchanged from the causal image E128 checkpoint at
`chankhavu/DeepSeek-V4.1-Flash-1017-arc-tool-img-E128-2bit-causal`, revision
`209c810a747c043c2aca474843de8e357f920ef6`. This repository includes the compatible
three-file compact Engram table from the tool E128 checkpoint at revision
`9bce98c5ecda0a9226edc6eab72a689089db1e37`, exactly as used in E074/E076. The
combined weight index is included. File origins and hashes are recorded in
`candidate_manifest.json`.

This is a routing change, with no additional weight training or requantization.
The default schedule is `active_experts.json`. Calibration uses E065 image/tool
traces and their paired E065 text/tool traces, with ARC training parents only.
No evaluation task labels are used to select this schedule.

## Evidence so far

On 24 paired E065 training families, generated-token NLL increases by 1.057%
for images and 1.213% for text, averaging 1.135% relative to k6. Tool-block NLL
increases by 3.746% and 4.105%, respectively. The initial uniform-k4 (~13.17B)
candidate has a larger average generated-token NLL increase of 3.153%.

Short generated diagnostics on **12 Kaggle ARC training originals, pass@1,
one draw, 24k base output budget**: seed14 100% image / 100% text; k6 baseline
100% image / 83.3% text; uniform-k4 83.3% image / 91.7% text. These are small
training-only diagnostics, not evidence of improved or preserved evaluation
accuracy. The broader E076 architecture search and full 120-task validation are
still running. See `evaluation/seed14_preview.json` for the exact measurements.

## Alternative with completed evaluation

`schedules/e074_mean5_14p58B.json` activates an average of five routed experts,
approximately 14.584B active parameters. It uses these same stored weights.
On **120 Kaggle ARC-AGI-2 evaluation tasks, two draws per task, images + tools**,
task-weighted pass@1/pass@2 were 70.83%/81.81%, versus 72.43%/83.47% for k6.
The pass@1 difference is -1.60 percentage points, paired-task 95% interval
[-7.01,+3.54]. Noninferiority at a two-point margin is inconclusive.
These E074 results belong to the alternative schedule, **not the default seed14**.

## Serving requirement

**A per-layer active-top-k patch is required.** An unpatched server can ignore
`text_config.num_experts_per_tok_per_layer` and run the six-expert baseline.
The custom metadata alone does not activate this architecture.

`serving/apply_active_topk_e043.py` is the patch tested on our E043 vision11
experimental fork. Set `E074_TOPK_FILE` to this model's `active_experts.json`
before starting that server. Full details are in `serving/ACTIVE_EXPERTS.md`.
The retained `inference/` reference implementation has not been adapted to this
schedule; do not treat its default generation script as an active-14B runner.

**Manh's fork `g22df163d2d` still needs the small router/model port described in
the serving notes. This upload does not claim drop-in compatibility with it.**

The checkpoint retains the causal image path, native routing scores and biases,
shared experts, vision tower, attention and Engram. Use prefix caching for tool
traffic and a context limit matching the server. Restart the engine when changing
schedules so workspace shapes, kernel specializations and CUDA graphs agree.
'''

SERVING='''# Serving the static active-expert schedules

## Tested experimental stack

The included `apply_active_topk_e043.py` targets the E043 vision11 fork with files
`vllm/models/deepseek_v4_1/nvidia/model.py` and `router.py`. It extends the E128
router allowlist and reads one integer top-k per layer from `E074_TOPK_FILE`
before constructing FusedMoE. It is idempotent and refuses unsupported layouts.
The patch and 36 router reference cases passed on the experimental stack;
seed14's real model ran image/text NLL and generated tool diagnostics there.

After installing that stack and the existing compact-Engram/Humming support:

```bash
python "$MODEL_DIR/serving/apply_active_topk_e043.py" "$VLLM_PACKAGE_DIR"
export E074_TOPK_FILE="$MODEL_DIR/active_experts.json"
export E043_GPTQ_DIR="$MODEL_DIR/gptq"
export DSV41_EXPERT_HUMMING='{"quant_method":"humming","weight_dtype":"uint2","weight_scale_dtype":"bfloat16","group_size":128,"has_zero_point":true}'
export DSV41_EXPERT_HUMMING_W2='{"group_size":64}'
```

Use the normal E043 TP4 serving command with prefix caching, compact Engram CPU
offload, native FP8 KV, the DeepSeek V4.1 tokenizer/tool parsers, and
`--hf-overrides '{"is_mm_prefix_lm":false}' --disable-chunked-mm-input`.
The E076 calibration server uses max-model-len 65536; full ARC uses 163840.
Check startup logs for all 40 `E074 layer=... top_k=...` lines and match them to
the selected JSON. The active-top-k file is the runtime source on this stack.

For the evaluated mean-five alternative, change E074_TOPK_FILE to
`$MODEL_DIR/schedules/e074_mean5_14p58B.json` and restart the engine. Its active
schedule overrides the preview metadata on this experimental stack; report its
separate name and evidence. Baseline k6 is in `schedules/baseline_k6.json`.

## Manh's current fork: port required, not yet GPU validated

Read-only audit of `g22df163d2d`, Humming 0.1.18:

1. `models/deepseek_v41/nvidia/model.py`: read and validate the per-layer list,
   then pass the selected layer's value as `n_activated_experts` to the base MoE.
2. `model_executor/layers/fused_moe/router/dsv4_topk.py`: replace fixed-six
   eligibility, loop bound, output stride/mask, allocation and warmup shapes with
   the requested compile-time TOP_K (3..6 for E076).
3. `fused_topk_bias_router.py`: forward topk to the router call.

Humming already consumes topk_ids/topk_weights through the modular interface.
The GB300-only opt-in latency_specialized path has additional fixed-six kernels;
it is outside the L4/RTX PRO 6000 serving path and must not be enabled here.

Before claiming compatibility, run text/image router reference cases including
ties and padding, an unchanged-k6 check, real mixed-k Humming prefill/decode with
CUDA graphs, and long-context prefix-cached tool traffic. A passing model load
alone is insufficient. Full Kaggle submission qualification remains separate.
'''


if __name__=='__main__':main()
