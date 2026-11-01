#!/usr/bin/env python3
"""Publish the user-requested private preview with pinned server-side weight copies."""
import hashlib
import json
from pathlib import Path
import time

import yaml
from huggingface_hub import HfApi, CommitOperationAdd, CommitOperationCopy, hf_hub_download

ROOT=Path(__file__).resolve().parents[3]
EXP=ROOT/'experiments/E076_active14_13'
UPLOAD=EXP/'upload'
BUNDLE=UPLOAD/'bundle'


def main():
    token=yaml.safe_load((ROOT/'SECRETS.yaml').read_text())['huggingface_full_access_api_key']
    api=HfApi(token=token)
    plan=json.loads((UPLOAD/'upload_plan.json').read_text())
    repo=plan['repo']
    assert plan['private'] is True and api.whoami()['name']==repo.split('/')[0]
    if api.repo_exists(repo):
        info=api.model_info(repo)
        assert info.private, 'Refusing to overwrite a public repository'
        names={f.rfilename for f in info.siblings}
        if names- {'.gitattributes'}:
            # Resume only an upload of this exact candidate, never an unrelated model.
            path=hf_hub_download(repo,'candidate_manifest.json',token=token)
            assert json.loads(Path(path).read_text())==json.loads((BUNDLE/'candidate_manifest.json').read_text())
    else:
        api.create_repo(repo,repo_type='model',private=True,exist_ok=False)
    info=api.model_info(repo)
    assert info.private
    operations=[]
    for path,row in sorted(plan['copy'].items()):
        operations.append(CommitOperationCopy(src_path_in_repo=path,path_in_repo=path,
            src_revision=row['source_revision'],src_repo_id=row['source_repo'],src_repo_type='model'))
    for path,row in sorted(plan['overlay'].items()):
        local=BUNDLE/path
        assert hashlib.sha256(local.read_bytes()).hexdigest()==row['sha256'],path
        operations.append(CommitOperationAdd(path_in_repo=path,path_or_fileobj=local))
    print(f'Publishing {len(operations)} files to private {repo}',flush=True)
    start=time.monotonic()
    commit=api.create_commit(repo,repo_type='model',operations=operations,parent_commit=info.sha,
        commit_message='E076 seed14 active-expert preview with complete weights, Engram, schedules and evidence',
        commit_description='Private upload requested by the owner. No weight updates: pinned causal E128 plus a 184-slot schedule. Full seed14 ARC evaluation pending. Includes the separately evaluated E074 mean-five alternative and explicit serving-patch requirements.',
        num_threads=4)
    print('Committed',commit.oid,flush=True)
    info=api.model_info(repo,revision=commit.oid,files_metadata=True)
    remote={f.rfilename:f for f in info.siblings}
    assert info.private, 'Repository privacy changed'
    expected=set(plan['copy'])|set(plan['overlay'])
    assert expected<=set(remote), sorted(expected-set(remote))
    for path,row in plan['copy'].items():
        got=remote[path]
        assert got.size==row['size'],path+' size'
        if row['lfs']:
            assert got.lfs and got.lfs.sha256==row['lfs']['sha256'],path+' LFS hash'
        else:
            assert got.blob_id==row['blob_id'],path+' git blob hash'
    for path,row in plan['overlay'].items():
        got=hf_hub_download(repo,path,revision=commit.oid,token=token)
        assert hashlib.sha256(Path(got).read_bytes()).hexdigest()==row['sha256'],path+' overlay hash'
    index=json.loads((BUNDLE/'model.safetensors.index.json').read_text())
    assert set(index['weight_map'].values())<=set(remote)
    receipt={'repo':repo,'url':f'https://huggingface.co/{repo}','revision':commit.oid,
        'private':info.private,'verified_files':len(expected),
        'verified_weight_files':sum(p.endswith('.safetensors') for p in expected),
        'stored_weight_bytes':sum(remote[p].size for p in expected if p.endswith('.safetensors')),
        'verification':'Source LFS SHA256 or git blob equality for every copied file; SHA256 download check for every overlay; complete weight index',
        'elapsed_s':time.monotonic()-start,'candidate':'E076 seed14 ~14.018B preview',
        'full_arc_evaluated':False,'serving':'Per-layer top-k patch required; Manh g22df port pending'}
    (UPLOAD/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2),flush=True)


if __name__=='__main__':main()
