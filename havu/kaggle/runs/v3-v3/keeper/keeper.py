import yaml, subprocess, time, re, os, sys
ROOT='/home/vu/Workspace/ARC-AGI-2'
R=ROOT+'/experiments/E019_qwen_program_answers/router'
LOG=R+'/pool_manager.log'
EV=ROOT+'/kaggle/runs/v3-v3/keeper/events.txt'
PORTS={'au':30101,'jp':30102,'tx':30103,'au2':30104,'ca':30105}
d=yaml.safe_load(open(ROOT+'/ENDPOINTS.yaml'))
eps={e['name']:e for e in (d if isinstance(d,list) else d.get('endpoints',d))}
def now(): return time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
def log(m,ev=False):
    open(LOG,'a').write(f'{now()} keeper {m}\n')
    if ev: open(EV,'a').write(f'{now()} {m}\n')
def curl(url,hdr=None):
    c=['curl','-s','-m','15','-o','/dev/null','-w','%{http_code}']
    if hdr: c+=['-H',hdr]
    try: return subprocess.run(c+[url],capture_output=True,text=True,timeout=25).stdout.strip()
    except Exception: return '000'
def probe(p): return curl(f'http://127.0.0.1:{p}/get_load')=='200'
def direct(n): 
    e=eps[f'vastai-1x6000-{n}-qwen27b']
    return curl(f"http://{e['addr']}/get_load",'Authorization: Bearer '+e['token'])=='200'
def pid_on(p):
    o=subprocess.run(['ss','-ltnp',f'sport = :{p}'],capture_output=True,text=True).stdout
    m=re.search(r'users:\(\("ssh",pid=(\d+)',o)
    return int(m.group(1)) if m else None
fails={n:0 for n in PORTS}; dfail={n:None for n in PORTS}; alerted={n:False for n in PORTS}
end=time.time()+8*3600; last_hb=0; restarts=0
log('started (8h, 2-min cycle)')
while time.time()<end:
    t0=time.time(); st={}
    for n,p in PORTS.items():
        if probe(p):
            if fails[n]: log(f'{n} probe recovered after {fails[n]} fail(s)')
            fails[n]=0; dfail[n]=None; alerted[n]=False; st[n]='ok'; continue
        fails[n]+=1; st[n]=f'fail{fails[n]}'
        if fails[n]<2: continue
        if direct(n):
            dfail[n]=None; pid=pid_on(p)
            if pid:
                subprocess.run(['kill',str(pid)]); log(f'{n} tunnel hung (2 probe fails, box ok directly): killed ssh pid {pid} on :{p}')
                restarts+=1; act='killed hung ssh'
            else:
                r=subprocess.run([ROOT+'/.venv/bin/python',R+'/register_worker.py',f'vastai-1x6000-{n}-qwen27b'],capture_output=True,text=True,cwd=ROOT)
                log(f'{n} no listener on :{p}: ran register_worker rc={r.returncode}'); restarts+=1; act='reopened tunnel (register_worker)'
            ok=False
            for _ in range(6):
                time.sleep(5)
                if probe(p): ok=True;break
            log(f'{n} post-restart probe {"PASS" if ok else "FAIL"}')
            log(f'{n} {act}; probe {"passes" if ok else "STILL FAILING"}',ev=True)
            if ok: fails[n]=0
        else:
            if dfail[n] is None: dfail[n]=time.time(); log(f'{n} probe and direct both failing',ev=True)
            elif time.time()-dfail[n]>600 and not alerted[n]:
                alerted[n]=True; log(f'{n} BOX DOWN directly >10 min (tunnel probe also failing)',ev=True)
    if time.time()-last_hb>=1800:
        last_hb=time.time(); log('heartbeat '+' '.join(f'{k}={v}' for k,v in st.items())+f' restarts={restarts}')
    time.sleep(max(1,120-(time.time()-t0)))
log('ended (8h)',ev=True)
