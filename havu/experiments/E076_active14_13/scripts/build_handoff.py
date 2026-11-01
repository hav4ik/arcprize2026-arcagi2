#!/usr/bin/env python3
"""Snapshot explicit non-secret E074/E076 artifacts for the SFT pruning handoff."""
import collections
import datetime
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
from zoneinfo import ZoneInfo

ROOT=Path(__file__).resolve().parents[3]
EXP=ROOT/'experiments/E076_active14_13'
OLD=ROOT/'experiments/E074_active_params'
OUT=EXP/'handoff'


def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    OUT.mkdir(exist_ok=True)
    origins={}
    def copy(src,dst):
        src=Path(src);dest=OUT/dst
        assert src.is_file(),src
        assert src.name not in ('SECRETS.yaml','ENDPOINTS.yaml')
        dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dest)
        origins[dst]=str(src.relative_to(ROOT))
    for src,dst in [(EXP/'configs/seed14.json','schedules/active_experts.json'),
                    (OLD/'results/candidate_arc/schedule.json','schedules/e074_mean5_14p58B.json'),
                    (EXP/'configs/uniform6.json','schedules/baseline_k6.json'),
                    (EXP/'data/manifest.json','calibration/e076_manifest.json'),
                    (EXP/'data/source_provenance.json','calibration/source_provenance.json'),
                    (OLD/'data/manifest.json','calibration/e074_manifest.json'),
                    (OLD/'results/sensitivity.json','calibration/e074_sensitivity.json'),
                    (OLD/'results/selection.json','calibration/e074_selection.json'),
                    (EXP/'results/sensitivity.json','calibration/e076_partial_sensitivity.json'),
                    (EXP/'configs/tools_first_manifest.json','provenance/tools_first_manifest.json'),
                    (EXP/'configs/dual_baseline_manifest.json','provenance/dual_baseline_manifest.json'),
                    (EXP/'configs/exploratory12_constraints.json','provenance/future12_constraints.json'),
                    (EXP/'results/tools_first_selection.json','provenance/pre_eval_selection.json'),
                    (EXP/'results/reproduction_environment.json','serving/environment.json'),
                    (EXP/'analysis/manh_g22df_audit/audit.json','serving/manh_audit.json'),
                    (EXP/'frozen_manifest.json','provenance/frozen_harness_manifest.json'),
                    (EXP/'upload/receipt.json','provenance/hf_upload_receipt.json')]:copy(src,dst)
    if (EXP/'upload/schedule_copy_receipt.json').exists():
        copy(EXP/'upload/schedule_copy_receipt.json','provenance/hf_schedule_copy_receipt.json')
    for tag in ('baseline_screen','uniform6_verify','uniform5_verify','mean5_profiled_verify'):
        for mode in ('img','txt'):
            copy(OLD/'results'/tag/(mode+'.jsonl'),'calibration/e074_measurements/'+tag+'/'+mode+'.jsonl')
    for p in (EXP/'data').rglob('*'):
        if p.is_file():copy(p,'calibration/prepared/'+str(p.relative_to(EXP/'data')))
    for p in (EXP/'frozen').rglob('*'):
        if p.is_file() and '__pycache__' not in p.parts:copy(p,'frozen/'+str(p.relative_to(EXP/'frozen')))
    for name in ('candidate_manifest.json','config.json','model.safetensors.index.json'):
        copy(EXP/'upload/bundle'/name,'provenance/'+name)
    for p in (EXP/'upload/bundle/serving').iterdir():
        if p.is_file():copy(p,'serving/'+p.name)
    for group,path,names in [
        ('e074',OLD/'scripts',('apply_topk.py','test_router_gpu.py','budgets.py','batch.py','prepare_pilot.py','score_arc.py')),
        ('e076',EXP/'scripts',('arm.py','batch.py','prepare.py','search.py','budgets.py','reference.py','report.py','diagnostic.py','process_utils.py','sft_schedule_overlay.py','test_search.py','build_handoff.py')),
        ('e066',ROOT/'experiments/E066_dsv41_image_pruning/scripts',('b32_local_eval.py','nll_eval_mm.py','mm_common.py','score_b32_engram.py'))]:
        for name in names:copy(path/name,'scripts/'+group+'/'+name)
    for tag in ('baseline_calibration','seed14_select','uniform4_select'):
        for p in (EXP/'results'/tag).glob('*.jsonl'):
            if p.name!='cache_metrics.jsonl':copy(p,'calibration/measurements/'+tag+'/'+p.name)
    for p in (EXP/'results').glob('layer*/screen_*.jsonl'):
        copy(p,'calibration/measurements/'+p.parent.name+'/'+p.name)
    runs={}
    all_usage={}
    for name,folder in [('E128',OLD/'results/baseline_arc'),('A14p58',OLD/'results/candidate_arc'),('A14p02',EXP/'results/preview14_img')]:
        target='evaluation/'+name
        for f in ('arc.jsonl','complete.json','command.json','eval_command.json','schedule.json','cache_metrics.jsonl'):
            copy(folder/f,target+'/'+f)
        if (folder/'reference_check.json').exists():copy(folder/'reference_check.json',target+'/reference_check.json')
        runs[name]={'path':target,'source':str(folder.relative_to(ROOT))}
        records={}
        for line in (folder/'arc.jsonl').read_text().splitlines():
            r=json.loads(line)
            if r['pair']==0:records[r['task_id'],r['attempt']]=r
        rows=list(records.values());assert len(rows)==240
        total=lambda f:sum(f(r) for r in rows)
        calls=total(lambda r:r['tool_calls']);errors=total(lambda r:r['tool_errors'])
        stops=collections.Counter(r['stop_reason'] for r in rows)
        all_usage[name]={'task_attempts':240,'output_tokens':total(lambda r:r['usage']['completion']),
            'mean_output_tokens':total(lambda r:r['usage']['completion'])/240,
            'mean_prompt_tokens_including_cached_replays':total(lambda r:r['usage']['prompt'])/240,
            'tool_calls':calls,'mean_tool_calls':calls/240,'tool_errors_after_repair':errors,
            'tool_error_rate_pct':100*errors/calls,'server_harness_errors':total(lambda r:bool(r.get('error'))),
            'format_rescues':total(lambda r:r.get('rescues',0)),
            'auto_repairs':{k:total(lambda r:r.get('auto_repairs',{}).get(k,0)) for k in ('proposed','applied','retry_ok')},
            'attempts_with_auto_repair':total(lambda r:r.get('auto_repairs',{}).get('applied',0)>0),
            'missing_pair0_answer_attempts':total(lambda r:r.get('pred') is None),
            'stop_reasons':dict(stops),'train_pass_counts':dict(collections.Counter(str(r.get('train_pass')) for r in rows))}
    (OUT/'evaluation/full_usage.json').write_text(json.dumps(all_usage,indent=2)+'\n')
    copy(EXP/'analysis/completed_image_tools_report.json','evaluation/full_image_report.json')
    report=json.loads((EXP/'results/report.json').read_text())
    (OUT/'calibration/nll_summary.json').write_text(json.dumps(report['training_nll'],indent=2)+'\n')
    schedule=json.loads((EXP/'configs/seed14.json').read_text())
    ranked=json.loads((OLD/'results/sensitivity.json').read_text())
    reconstructed=[6]*40
    for row in sorted(ranked,key=lambda r:(r['delta'],r['layer']))[:28]:reconstructed[row['layer']]=4
    assert reconstructed==schedule['top_k']
    file_hashes={f: digest(ROOT/f) for f in ('data/arc-agi_evaluation_challenges.json','data/arc-agi_evaluation_solutions.json','data/arc-agi_training_challenges.json','data/arc-agi_training_solutions.json')}
    copy(ROOT/'data/arc-agi_evaluation_challenges.json','evaluation/kaggle_challenges.json')
    metadata={'schema_version':1,'created_local':datetime.datetime.now(ZoneInfo('America/Los_Angeles')).isoformat(),
        'experiment':'E076 active routing seed14','candidate':schedule,
        'schedule_semantics':'Static per transformer layer; token-dependent expert identities still selected by native router; weights unchanged; all128 stored experts retained',
        'layer_index_base':0,'layers_kept_at6':[i for i,k in enumerate(schedule['top_k']) if k==6],
        'architecture':{'layers':40,'hidden_size':5120,'moe_intermediate_size':2304,'stored_routed_experts_per_layer':128,'shared_experts_per_layer':1,'original_routed_top_k':6,'routed_scaling_factor':1.5},
        'selection':{'method':'E074 one-layer6->4 costs, sorted by (delta,layer); lowest28 layers reduced to4',
            'ranking_screen_rollouts_per_modality':4,'ranking_is_e076_complete_search':False,
            'e076_select_rollouts_per_modality':24,'seed14_audit_completed':False,
            'completed_e076_screens':{'k4':40,'k5':6,'k3':0},'arc_results_used_to_select_schedule':False},
        'evaluation':{'version':'Kaggle ARC-AGI-2 evaluation','tasks':120,'test_pairs':172,'draws_per_task':2,'logical_attempts':240,'pair_records':344,'runs':runs,'raw_solutions_bundled':False,'data_hashes':file_hashes,
            'primary':'task-weighted pass@1 averaged over both draws','secondary':'task-weighted per-test-input pass@2',
            'bootstrap':{'unit':'task','replicates':5000,'seed':0,'sorted_quantile_indices':[125,4875]},
            'image_status':'complete','text_status':'running, results not established in this snapshot','controls':'E074 image references reused; matching44 frozen harness assets and command parity; not new simultaneous replicas'},
        'sft_transfer':{'architecture_required':'compatible40-layer E128 with same expert width; confirm topology and tokenizer',
            'copy_only_schedule_not_base_weights':True,'accuracy_transfer_proven':False,
            'references':'same SFT checkpoint at k6 and at E074 mean5; retain historical base comparisons separately',
            'priority':'text+tools first, then images+tools','sensitive_layers':'keep6; do not sacrifice to reach approximate parameter budget'},
        'hash_note':'FILES.sha256 authenticates bundle bytes; FILE_INDEX.json maps copied files to original workspace paths; these are integrity hashes, not cryptographic signatures'}
    (OUT/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    index={str(p.relative_to(OUT)):{'sha256':digest(p),'bytes':p.stat().st_size,'source':origins.get(str(p.relative_to(OUT)),'generated for handoff')} for p in sorted(OUT.rglob('*')) if p.is_file() and p.name not in ('FILES.sha256','FILE_INDEX.json')}
    (OUT/'FILE_INDEX.json').write_text(json.dumps(index,indent=2)+'\n')
    listed=[p for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='FILES.sha256']
    (OUT/'FILES.sha256').write_text(''.join(digest(p)+'  '+str(p.relative_to(OUT))+'\n' for p in listed))
    print(json.dumps({'files':len(listed),'bytes':sum(p.stat().st_size for p in listed),'path':str(OUT),'full_usage':all_usage},indent=2))


if __name__=='__main__':main()
