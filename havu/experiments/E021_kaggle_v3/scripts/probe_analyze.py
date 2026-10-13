"""Analyse a downloaded load-probe run dir. Usage: probe_analyze.py kaggle/runs/probe-v2 > out.md"""
import json, re, sys, statistics as st, os
from collections import defaultdict
d = sys.argv[1]
logf = [f for f in os.listdir(d) if f.endswith('.log')][0]
lines = []
for l in open(os.path.join(d, logf)):
    l = l.strip().lstrip(',')
    try: lines.append(json.loads(l)['data'].rstrip('\n'))
    except Exception: pass
txt = '\n'.join(lines)
S = json.load(open(os.path.join(d, 'summary_local.json')))
ph = S['phases_s']
def clock(s):
    m = re.match(r'\[\s*[\d.]+h (\d\d):(\d\d):(\d\d)\]', s); return int(m[1])*3600+int(m[2])*60+int(m[3])
ready = [clock(s) for s in lines if 'server ready in' in s][0]
force = [s for s in lines if 'local time box' in s][0]
print('## ', d)
print('GPUs:', ph['gpus'], '| server_ready(s since T0):', ph['server_ready'], '| first_token:', ph['first_token'])
print(force); print([s for s in lines if 'concurrency' in s and 'max_seqs' in s][0])
prog = [s for s in lines if ' progress:' in s]
rows = []
for s in prog:
    t = (clock(s)-ready)/60
    g = lambda p: re.search(p, s)
    rows.append(dict(t=t, done=int(g(r'(\d+) attempts done')[1]), infl=int(g(r'in flight (\d+)')[1]), tok=int(g(r'tokens (\d+)')[1]),
        win=float(g(r'window (\d+) tok/s')[1]), est=float(g(r'rate est (\d+)')[1]), avg=float(g(r'avg (\d+) tok/s')[1]),
        hit=(float(g(r'cache hit ([\d.]+)%')[1]) if g(r'cache hit ([\d.]+)%') else None), pt=int(g(r'of (\d+) prompt')[1]),
        run=int(g(r'running (\d+)')[1]), wait=int(g(r'waiting (\d+)')[1]), kv=float(g(r'KV usage ([\d.]+)%')[1]), pre=int(g(r'total (\d+)\)')[1])))
print('| min after ready | done | in flight | tokens | win tok/s | avg tok/s | rate est | KV % | cache hit % | prompt tok/2min | running | waiting | preempt |')
print('|--|--|--|--|--|--|--|--|--|--|--|--|--|')
for i, r in enumerate(rows):
    if i % 5 == 4 or i == len(rows)-1:
        # 10-min window means for window tok/s and hit
        w = rows[max(0, i-4):i+1]
        print(f"| {r['t']:.0f} | {r['done']} | {r['infl']} | {r['tok']} | {st.mean(x['win'] for x in w):.0f} | {r['avg']:.0f} | {r['est']:.0f} | {max(x['kv'] for x in w):.0f} | {st.mean(x['hit'] for x in w if x['hit'] is not None):.0f} | {st.mean(x['pt'] for x in w):.0f} | {r['run']} | {r['wait']} | {r['pre']} |")
print('peak KV (progress):', max(r['kv'] for r in rows), 'summary kv_usage_max_pct:', S.get('kv_usage_max_pct'), 'max waiting:', max(r['wait'] for r in rows), 'final preemptions:', rows[-1]['pre'])
# steady state: from 15 to 60 min after ready, and last-third
def mean(rs, k): return st.mean(r[k] for r in rs)
for a, b in ((0, 20), (20, 60), (60, 130)):
    rs = [r for r in rows if a <= r['t'] < b]
    if rs: print(f"window {a}-{b} min: mean tok/s {mean(rs,'win'):.0f}, KV {mean(rs,'kv'):.0f}%, hit {st.mean(r['hit'] for r in rs if r['hit'] is not None):.0f}%")
# vllm engine log
vl = open(os.path.join(d, 'arc_logs/vllm_server.log')).read().splitlines()
eng = []
for l in vl:
    m = re.search(r'Avg prompt throughput: ([\d.]+) tokens/s, Avg generation throughput: ([\d.]+) tokens/s, Running: (\d+) reqs, Waiting: (\d+) reqs, GPU KV cache usage: ([\d.]+)%, Prefix cache hit rate: ([\d.]+)%', l)
    if m: eng.append(tuple(map(float, m.groups())))
act = [e for e in eng if e[1] > 0]
print('engine log: samples', len(eng), 'mean gen tok/s (active samples)', f"{st.mean(e[1] for e in act):.0f}", 'mean prompt tok/s', f"{st.mean(e[0] for e in act):.0f}", 'max KV', max(e[4] for e in eng), 'max waiting', max(e[3] for e in eng))
n = len(act)
for k in range(4):
    seg = act[k*n//4:(k+1)*n//4]
    print(f"  engine quarter {k+1}: gen {st.mean(e[1] for e in seg):.0f} tok/s, prompt {st.mean(e[0] for e in seg):.0f} tok/s, running {st.mean(e[2] for e in seg):.0f}, KV {st.mean(e[4] for e in seg):.0f}%, engine cum. hit {seg[-1][5]:.1f}%")
print('warnings:', [l[:160] for l in vl if re.search(r'preempt|Traceback|OOM|out of memory|CUDA error|restart', l, re.I)][:5], '| solver:', [s[:160] for s in lines if re.search(r'Traceback|OOM|restart|WARNING', s)][:5])
# budgets
A = [json.loads(l) for l in open(os.path.join(d, 'arc_logs/attempts_local.jsonl'))]
done_lines = [s for s in lines if re.search(r'\] done ', s)]
bud = [(clock(s)-ready)/60 for s in done_lines], [int(re.search(r'budget=(\d+)', s)[1]) for s in done_lines]
b = sorted(bud[1])
bc = sorted(x['budget'] for x in A)
print('capped budgets actually run (min(TOOLS_CAP, assigned)): min %d median %d max %d' % (bc[0], st.median(bc), bc[-1]))
print('scheduler-assigned budgets (n=%d): min %d median %d max %d' % (len(b), b[0], st.median(b), b[-1]))
print('budget counts:', {k: b.count(k) for k in sorted(set(b))} if len(set(b)) < 12 else 'many')
print('budget of attempts in order of finish:', bud[1])
# which attempts started late? start = finish - elapsed
starts = sorted(((clock(s) - int(re.search(r' (\d+)s', s)[1]) - ready)/60, int(re.search(r'budget=(\d+)', s)[1]), s.split('done ')[1].split(' ')[0]) for s in done_lines)
print('attempt start (min after ready), budget:', [(round(a), b_) for a, b_, _ in starts])
# results
pp = S['per_pair']; sl = S['slots']
c1 = sum(1 for x in sl if x['correct1']); print('pair-weighted pass@1: %d/%d = %.1f%%' % (c1, len(sl), 100*c1/len(sl)))
bt = defaultdict(list)
for x in sl: bt[x['task']].append(x['correct1'])
tw = sum(sum(v)/len(v) for v in bt.values()); print('task-weighted pass@1: %.2f/%d = %.1f%%' % (tw, len(bt), 100*tw/len(bt)), '| summary score:', S['score'])
print('answered', S['answered'], 'errors', S['errors'], 'attempts', S['attempts'], 'tokens', S['tokens'], 'gen_minutes', S['gen_minutes'])
print('answer_sources', S['answer_sources'])
tk = [x['tokens'] for x in A]; print('tokens/attempt mean %.0f median %.0f max %d' % (st.mean(tk), st.median(tk), max(tk)))
print('forced:', sum(1 for x in A if x.get('forced')), 'force reasons:', {k: sum(1 for x in A if x.get('force_reason') == k) for k in set(x.get('force_reason') for x in A)})
print('stop reasons:', {k: sum(1 for x in A if x.get('stop_reason') == k) for k in set(x.get('stop_reason') for x in A)})
print('never attempted pairs:', len(sl) - len(A), '| attempts_by_index', S['v3'].get('attempts_by_index'), '| gated', S['v3'].get('gated'))
cor_f = sum(1 for x in A if x.get('forced') and x.get('pred') is not None and any(p['correct'] and p['task']==x['task_id'] and p['pair']==x['pair'] for p in pp))
by_src = defaultdict(lambda: [0,0])
for p in pp:
    by_src[p['source']][0] += 1; by_src[p['source']][1] += bool(p['correct'])
print('per-pair source -> [n, correct]:', dict(by_src))
print('correct among forced:', cor_f, '| correct among submitted:', sum(1 for p in pp if p['correct'] and p['source']=='submit'))
el = [x['elapsed_s'] for x in A]; print('elapsed per attempt s: mean %.0f median %.0f max %d' % (st.mean(el), st.median(el), max(el)))
print('per-attempt decode speed tok/s (median):', round(st.median(x['tokens']/x['elapsed_s'] for x in A if x['elapsed_s']), 2))
print('overall avg tok/s first token->scheduler end:', S['tokens'] / (S['gen_minutes']*60))
# server-reported KV capacity and speculative-decoding metrics (MTP)
kvc = [re.search(r'GPU KV cache size: ([\d,]+) tokens', l)[1] for l in vl if 'GPU KV cache size' in l]
print('vLLM-reported GPU KV cache size (tokens):', kvc, '| solver capacity line:', [s[-80:] for s in lines if 'server capacity' in s])
sd = [l for l in vl if 'SpecDecoding metrics' in l]
if sd:
    ml = [float(re.search(r'Mean acceptance length: ([\d.]+)', l)[1]) for l in sd if 'Mean acceptance length' in l]
    ar = [float(re.search(r'Avg Draft acceptance rate: ([\d.]+)%', l)[1]) for l in sd if 'Avg Draft acceptance rate' in l]
    print('spec decode samples', len(sd), 'mean acceptance length mean %.2f' % st.mean(ml) if ml else '', 'draft acceptance rate mean %.1f%%' % st.mean(ar) if ar else '')
    print('  last:', sd[-1][-250:])
else:
    print('no SpecDecoding metrics lines in vllm_server.log')
print('server cmd spec flags:', [s[:40] + ' ... ' + (re.search(r'--speculative-config \S+', s) or [''])[0] for s in lines if 'server cmd' in s])
