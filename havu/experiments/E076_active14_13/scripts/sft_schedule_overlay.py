#!/usr/bin/env python3
"""Create a routing-config overlay for compatible SFT weights; never edit weights."""
import argparse
import hashlib
import json
from pathlib import Path


def overlay(config, schedule):
    c=json.loads(json.dumps(config))
    t=c.get('text_config',c)
    expected={'num_hidden_layers':40,'hidden_size':5120,'moe_intermediate_size':2304,
              'n_routed_experts':128,'n_shared_experts':1,'num_experts_per_tok':6}
    for key,value in expected.items():
        if t.get(key)!=value:
            raise ValueError(f'Incompatible {key}: expected {value}, got {t.get(key)}')
    for key in ('n_routed_experts_per_layer','num_experts_per_layer'):
        if key in t and t[key]!=[128]*40:
            raise ValueError(f'Non-E128 stored expert layout: {key}')
    ks=schedule['top_k']
    if len(ks)!=40 or any(type(k) is not int or not 1<=k<=6 for k in ks):
        raise ValueError('Expected forty integer top-k values in 1..6')
    if schedule.get('expert_slots',sum(ks))!=sum(ks):
        raise ValueError('Schedule slot total disagrees with top_k')
    t['num_experts_per_tok_per_layer']=ks
    # Never carry provenance from an older, unrelated preview configuration.
    c.pop('active_expert_pruning',None)
    c.pop('e076_active_pruning',None)
    return c


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-config',type=Path,required=True)
    p.add_argument('--schedule',type=Path,required=True)
    p.add_argument('--source-id',required=True,help='SFT checkpoint identity, not the original base model')
    p.add_argument('--source-revision',required=True,help='Immutable checkpoint revision or local manifest hash')
    p.add_argument('--output-dir',type=Path,required=True,help='New overlay directory; must not already exist')
    a=p.parse_args()
    source=a.source_config.read_bytes();raw_schedule=a.schedule.read_bytes()
    schedule=json.loads(raw_schedule);config=overlay(json.loads(source),schedule)
    a.output_dir.mkdir(parents=True,exist_ok=False)
    (a.output_dir/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    (a.output_dir/'active_experts.json').write_bytes(raw_schedule)
    metadata={'source_id':a.source_id,'source_revision':a.source_revision,
              'source_config_sha256':hashlib.sha256(source).hexdigest(),
              'schedule_sha256':hashlib.sha256(raw_schedule).hexdigest(),
              'expert_slots':sum(schedule['top_k']),
              'estimated_active_b_rounded16_anchor':16-(240-sum(schedule['top_k']))*35389440/1e9,
              'weight_files_included':False,'source_weights_modified':False,
              'requires_original_sft_weights_and_tokenizer':True,
              'runtime_patch_required':True,'sft_accuracy_revalidated':False,
              'runtime_note':'On tested E043 stack set E074_TOPK_FILE to active_experts.json and restart engine. Config field alone is insufficient.'}
    (a.output_dir/'overlay_metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps(metadata,indent=2))


if __name__=='__main__':main()
