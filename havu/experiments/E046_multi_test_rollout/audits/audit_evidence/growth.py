import json,gzip,glob,os,statistics,collections
W='/home/vu/Workspace/ARC-AGI-2/'
d=json.load(open(W+'experiments/E043_dsv41_real_pruning/results/tool_shapes_e128t_w10_u2_gptq_eng64_sub80.json'))
G={os.path.basename(f)[:-5]:json.load(open(f)) for f in glob.glob(W+'data/arc-agi-2-github/data/evaluation/*.json')}
npairs={t:len(g['test']) for t,g in G.items()}
def growth(turns):
    if len(turns)<2: return None
    p0=turns[0]['prompt']; last=turns[-1]['prompt']+turns[-1]['completion']
    comp=sum(t['completion'] for t in turns)
    return p0,last,comp
pp=collections.defaultdict(list); fp_pp={}
for a in d['attempts']:
    g=growth(a['turns'])
    if g: pp['multi' if npairs[a['task_id']]>1 else 'single'].append(g)
    if a['turns']: fp_pp[(a['task_id'],a['pair'])]=a['turns'][0]['prompt']
def summ(lab,v):
    r=[(l-p)/c for p,l,c in v if c>0]
    tot=sum(l-p for p,l,c in v)/sum(c for p,l,c in v)
    print(f'{lab:28s} n={len(v)} growth (ctx-ctx0)/gen: median {statistics.median(r):.2f} pooled {tot:.2f}; final ctx mean {statistics.mean(l for p,l,c in v)/1e3:.1f}k p90 {sorted(l for p,l,c in v)[int(.9*len(v))]/1e3:.1f}k')
summ('E043 per-pair, single tasks',pp['single']); summ('E043 per-pair, multi tasks',pp['multi'])
# multi from traces
for run in ['final_all','mt3_sub80_tx_b05']:
    R=W+f'evals/results/tools_persistent_grid_lean2sub_dsv41_low_e043_e128t_{run}/'
    v=collections.defaultdict(list); fpm={}
    for f in glob.glob(R+'*.messages.json.gz'):
        t=os.path.basename(f).split('_')[0]
        try: ev=json.load(gzip.open(f))['events']
        except Exception: continue
        us=[e['usage'] if isinstance(e['usage'],dict) else eval(e['usage']) for e in ev if 'usage' in e]
        turns=[{'prompt':u['prompt_tokens'],'completion':u['completion_tokens']} for u in us]
        g=growth(turns)
        if g: v['multi' if npairs[t]>1 else 'single'].append(g)
        if turns and npairs[t]>1: fpm.setdefault(t,[]).append(turns[0]['prompt'])
    for k in v: summ(f'{run} {k}',v[k])
    rat=[min(fpm[t])/sum(fp_pp.get((t,q),0) for q in range(npairs[t])) for t in fpm if all((t,q) in fp_pp for q in range(npairs[t]))]
    print(f'  {run}: multi first prompt / sum of per-pair first prompts: median {statistics.median(rat):.2f} mean {statistics.mean(rat):.2f} (sim uses 0.60)')
