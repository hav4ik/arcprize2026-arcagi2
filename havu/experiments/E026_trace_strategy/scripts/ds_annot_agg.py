import json,glob,collections,statistics as st
rows=[json.loads(l) for f in sorted(glob.glob(__import__('os').path.join(__import__('os').path.dirname(__file__),'../ds_annotations/*.jsonl'))) for l in open(f) if l.strip().endswith('}')]
rows=[r for r in rows if r['cond']!='COMPARE' and r['task']!='4c3d4a41']
by=collections.defaultdict(list)
for r in rows: by[r['cond']].append(r)
keys=['perception_hand','perception_code','hypothesis_prose','code_solve_debug','verification','test_application','post_answer_doubt','other']
for c in ['NT','S80','V1']:
    R=by[c]; tot=sum(r['tokens'] for r in R)
    tw={k:sum(r['tokens']*r['token_share_pct'].get(k,0) for r in R)/tot for k in keys}
    print(c,len(R),'acc',sum(r['correct'] for r in R),'tok',tot//len(R),'tokw',{k:round(v) for k,v in tw.items()})
    print('   perc_tok mean',st.mean(r['perception_tokens_est'] for r in R),'median',st.median(r['perception_tokens_est'] for r in R),'hand',collections.Counter(r['hand_transcription'] for r in R))
    idea=[r['idea_pct'] for r in R if isinstance(r.get('idea_pct'),(int,float))]
    print('   idea_pct median',st.median(idea), 'final_rule median',st.median([r['final_rule_pct'] for r in R if isinstance(r.get('final_rule_pct'),(int,float))]))
    ss=collections.defaultdict(collections.Counter)
    for r in R:
        for k,v in r['self_stats'].items(): ss[k][v]+=1
    for k,v in ss.items(): print('     ',k,dict(v))
for r in rows:
    if r['cond']=='V1': print('V1',r['task'],'used',r.get('summary_used_items'))
