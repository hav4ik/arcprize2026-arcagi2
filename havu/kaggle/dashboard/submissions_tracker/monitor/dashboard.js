'use strict';
let snapshot = null, selected = localStorage.getItem('arc-monitor-selected'), noteTab = 'findings', etag = null, loading = false;
let compared = (()=>{try{const ids=JSON.parse(localStorage.getItem('arc-monitor-compare'));return Array.isArray(ids)?new Set(ids.filter(id=>typeof id==='string')):null;}catch{return null;}})();
const palette=['#2455b8','#c65b16','#168369','#8950b5','#c33f65','#008a9c','#847025','#5268be','#9c4c36','#607d31','#99749e','#607e91'];
const darkPalette=['#80aaff','#f5a464','#75c8b4','#bea0e4','#ef91af','#6bc8d4','#d7c082','#a5b4fa','#e6a58b','#b0c47c','#c3a8ce','#a9c1d0'];
const darkMode=matchMedia('(prefers-color-scheme: dark)');
const $ = id => document.getElementById(id);
const esc = x => String(x ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const finite = x => typeof x === 'number' && Number.isFinite(x);
const fmt = (x, digits=1) => finite(x) ? x.toLocaleString(undefined,{minimumFractionDigits:digits,maximumFractionDigits:digits}) : '—';
const ago = seconds => !finite(seconds) ? 'unknown' : seconds < 90 ? `${Math.max(0,Math.round(seconds))}s ago` : seconds < 3600 ? `${Math.floor(seconds/60)}m ago` : `${fmt(seconds/3600)}h ago`;
const pill = state => `<span class="pill ${esc(state)}">${esc(state.replaceAll('-',' '))}</span>`;
const visibleRuns = () => (snapshot?.runs || []).filter(r => $('history').checked || !r.historical);
const comparisonRuns = () => [...(compared||[])].map(id=>snapshot?.runs.find(r=>r.id===id)).filter(Boolean);
const colorFor = id => (darkMode.matches?darkPalette:palette)[Math.max(0,[...(compared||[])].indexOf(id))%palette.length];
const dashFor = id => [...(compared||[])].indexOf(id)>=palette.length?'6 3':'';
const swatch = id => `<svg class="swatch" viewBox="0 0 24 8" aria-hidden="true"><line x1="1" x2="23" y1="4" y2="4" stroke="${colorFor(id)}" stroke-width="3" stroke-dasharray="${dashFor(id)}"/></svg>`;

function saveComparison() {
  localStorage.setItem('arc-monitor-compare',JSON.stringify([...(compared||[])]));renderRuns();renderDetail();
}

function setComparison(ids) {
  compared=new Set(ids.filter(Boolean));saveComparison();
}

function toggleComparison(id, checked) {
  if(!compared)compared=new Set();
  checked?compared.add(id):compared.delete(id);saveComparison();
}

function selectRun(id) {
  const previous=selected;
  if(compared===null || (compared.size===1&&compared.has(previous)))compared=new Set([id]);
  selected = id; localStorage.setItem('arc-monitor-selected', id); renderRuns(); renderDetail();
  localStorage.setItem('arc-monitor-compare',JSON.stringify([...compared]));
}

function renderSummary() {
  const runs = snapshot.runs.filter(r=>!r.historical);
  const live = runs.filter(r=>r.state==='running').length;
  const connected = snapshot.pods.filter(p=>p.connected).length;
  const rented=snapshot.pods.filter(p=>(p.desired_status||'RUNNING')==='RUNNING');
  const cost=rented.every(p=>finite(p.cost_per_hour))?rented.reduce((n,p)=>n+p.cost_per_hour,0):null;
  const attention = runs.filter(r=>['blocked','telemetry-stale','unverified','failed'].includes(r.state)).length;
  $('summary').innerHTML = [
    ['Solving now',live,`${connected} connected pod${connected===1?'':'s'} · ${snapshot.pods.length} reported`],
    ['Compute rental',finite(cost)?`$${fmt(cost,2)}/h`:'—','Currently reported running L4 pods; idle pods still incur rental'],
    ['Need attention',attention,'Blocked, stale, unverified or failed observations'],
    ['Manh V12 gold',`${fmt(snapshot.baseline.pass2,2)}%`,'Final pass@2 · 240 Kaggle-version tasks']
  ].map(([label,value,note])=>`<div class="stat"><span class="label">${esc(label)}</span><b>${esc(value)}</b><span>${esc(note)}</span></div>`).join('');
}

function renderPods() {
  $('pods').innerHTML = snapshot.pods.length ? snapshot.pods.map(p=>{
    const active = snapshot.runs.filter(r=>r.pod_id===p.id && ['running','starting','recovering','finishing','blocked','telemetry-stale'].includes(r.state));
    const queue = p.queue?.pending || [];
    return `<div class="pod"><div class="pod-head"><strong>${esc(p.name)}</strong>${pill(p.connected?'online':'offline')}</div>
      <div class="muted">${esc(p.id)} · observed ${esc(p.observed_at || 'never')}</div>
      <div class="gpu-row">${(p.gpus||[]).map(g=>`<div class="gpu">GPU ${fmt(g.index,0)}<b>${fmt(g.utilization_pct,0)}%</b>${fmt(g.used_mib/1024)} / ${fmt(g.total_mib/1024)} GiB</div>`).join('')}</div>
      <div>${active.length ? active.map(r=>esc(r.name)).join(' · ') : p.connected ? 'No experiment process is running' : 'Current process state unavailable'}</div>
      ${queue.length ? `<div class="muted">Queued: ${queue.map(esc).join(', ')}</div>` : ''}
      ${p.error ? `<div class="muted">${esc(p.error)}</div>`:''}</div>`;
  }).join('') : '<p class="muted">No L4 pods currently reported. Recorded experiment history is retained.</p>';
}

function renderRuns() {
  const runs = visibleRuns();
  if (!runs.some(r=>r.id===selected)) selected = runs.find(r=>r.state==='running')?.id || runs[0]?.id;
  if(compared===null)compared=new Set(selected?[selected]:[]);
  $('runs').innerHTML = runs.map(r=>{
    const s=r.summary||{};
    const score=r.holdout_locked ? 'Locked' : finite(s.score_pass2) ? fmt(s.score_pass2,2)+'%' : '—';
    return `<tr data-id="${esc(r.id)}" tabindex="0" class="${r.id===selected?'selected':''}">
      <td class="compare-cell"><input type="checkbox" data-compare="${esc(r.id)}" aria-label="Compare ${esc(r.name)}" ${compared.has(r.id)?'checked':''}></td>
      <td><strong>${esc(r.name)}</strong><small>${esc(r.pod_name||'')} · ${esc(r.source)}</small></td>
      <td>${pill(r.state)}<small>${esc(r.reason)}</small></td>
      <td>${r.tasks ? `${fmt(r.tasks,0)} tasks / ${fmt(r.pairs,0)} pairs`:'Count unverified'}<small>${esc(r.version)}</small></td>
      <td>${fmt(s.completed_attempts,0)} / ${fmt(s.expected_attempts,0)}<small>First ${fmt(s.first_finished,0)} · both ${fmt(s.both_finished,0)}</small></td>
      <td><span class="value">${score}</span><small>${r.holdout_locked?'Held until validation finishes':r.historical?'Recorded endpoint':'Two ranked answers · so far'}</small></td>
      <td><span class="value">${fmt(r.gen_tok_s,0)}</span></td>
      <td><span class="value">${fmt(r.cap_reserved_gib,2)}</span><small>GiB · lowest GPU${r.state==='running'?'':' · last reading'}</small></td>
      <td>${r.historical ? 'Archived' : esc(ago(r.telemetry_at ? Date.now()/1000-Date.parse(r.telemetry_at)/1000 : null))}<small>${esc(r.telemetry_at||'No timestamped sample')}</small></td>
    </tr>`;
  }).join('') || '<tr><td colspan="9">No experiment data yet. Collection continues automatically.</td></tr>';
  const previous = $('run-select').value;
  $('run-select').innerHTML = runs.map(r=>`<option value="${esc(r.id)}">${esc(r.name)}</option>`).join('');
  $('run-select').value=selected || previous;
}

function chart(id, groups, field, reference=null, domain=null) {
  const measured=groups.map(g=>({...g,points:(g.rows||[]).filter(p=>finite(p.h)&&finite(p[field])).sort((a,b)=>a.h-b.h)}));
  const points=measured.flatMap(g=>g.points);
  if (!points.length) { $(id).innerHTML=`<div class="empty">${groups.length?'No measured samples for the compared runs yet':'Select runs to compare'}</div>`; return; }
  const w=620,h=215,left=48,right=16,top=14,bottom=32;
  const xmax=Math.max(.2,...points.map(p=>p.h));
  let ymin=domain?domain[0]:0, ymax=domain?domain[1]:Math.max(...points.map(p=>p[field]),finite(reference)?reference:0)*1.13;
  ymax=Math.max(ymax,ymin+.1);
  const x=v=>left+(v/xmax)*(w-left-right),y=v=>h-bottom-(v-ymin)/(ymax-ymin)*(h-top-bottom);
  let grid='';for(let i=0;i<=4;i++){
    const v=ymin+(ymax-ymin)*i/4,xx=xmax*i/4;
    grid+=`<line class="grid" x1="${left}" x2="${w-right}" y1="${y(v)}" y2="${y(v)}"/><text text-anchor="end" x="${left-8}" y="${y(v)+3}">${fmt(v,ymax<5?2:0)}</text><text text-anchor="middle" x="${x(xx)}" y="${h-13}">${fmt(xx,1)}</text>`;
  }
  const paths=measured.filter(g=>g.points.length).map(g=>{
    let path='',last=null;for(const p of g.points){path+=`${!last||p.h-last.h>.35?'M':'L'}${x(p.h).toFixed(2)},${y(p[field]).toFixed(2)} `;last=p;}
    return `<path class="line" data-run="${esc(g.id)}" stroke="${g.color}" stroke-dasharray="${g.dash}" d="${path}"><title>${esc(g.name)}</title></path><circle cx="${x(last.h)}" cy="${y(last[field])}" r="3" fill="${g.color}"><title>${esc(g.name)} · ${fmt(last.h,2)} h: ${fmt(last[field],2)}</title></circle>`;
  }).join('');
  const ref=finite(reference)?`<line class="ref" x1="${left}" x2="${w-right}" y1="${y(reference)}" y2="${y(reference)}"><title>Reference: ${fmt(reference,2)}</title></line>`:'';
  const missing=measured.filter(g=>!g.points.length);
  $(id).innerHTML=`<svg viewBox="0 0 ${w} ${h}" role="img" aria-label="${esc(id)} over elapsed hours for ${groups.length} compared runs">${grid}${ref}${paths}<text x="${w-right}" y="${h-1}" text-anchor="end">elapsed hours per run</text></svg>${missing.length?`<div class="chart-missing" title="${esc(missing.map(g=>g.name).join(', '))}">No samples: ${missing.slice(0,3).map(g=>esc(g.name)).join(', ')}${missing.length>3?` · ${missing.length-3} more`:''}</div>`:''}`;
}

function renderComparison() {
  const runs=comparisonRuns();
  $('compare-count').textContent=`(${runs.length} selected)`;
  $('compare-options').innerHTML=snapshot.runs.map(r=>`<label class="compare-option"><input type="checkbox" data-compare="${esc(r.id)}" ${compared.has(r.id)?'checked':''}><span><strong>${esc(r.name)}</strong><small>${r.tasks?`${r.tasks} tasks`:'Unverified workload'} · ${esc(r.pod_name||'')}${r.historical?' · archived':''}</small></span></label>`).join('');
  $('compare-legend').innerHTML=runs.map(r=>`<button data-compare-remove="${esc(r.id)}" aria-label="Remove ${esc(r.name)} from comparison">${swatch(r.id)}<span>${esc(r.name)} <small>· ${r.tasks||'?'} tasks</small></span><span aria-hidden="true">×</span></button>`).join('');
  const sameCohort=runs.length>0&&runs[0].cohort&&runs.every(r=>r.cohort===runs[0].cohort);
  const counts=[...new Set(runs.map(r=>r.tasks||'?'))];
  let note=runs.length?`${runs.length} run${runs.length===1?'':'s'} · elapsed time is measured from each run’s own start. `:'Tick Compare in the table or open the run list above. ';
  if(runs.length>1&&!sameCohort)note+=`Workloads differ or are unverified (${counts.join(' / ')} tasks); a paired accuracy comparison requires identical task grids. `;
  if(runs.some(r=>r.historical))note+='Archived curves retain their recorded time origin. ';
  const reference=(runs.length===1||sameCohort)&&finite(runs[0]?.baseline_pass2)?runs[0].baseline_pass2:null;
  if(finite(reference))note+=`Dashed score line: V12 final on matching tasks (${fmt(reference,2)}%). `;
  if(compared.size>runs.length)note+=`${compared.size-runs.length} saved selection(s) currently unavailable. `;
  note+='Selections stay saved across automatic refresh and page reload.';
  $('compare-note').textContent=note;
  const groups=key=>runs.map(r=>({id:r.id,name:`${r.name} (${r.tasks||'?'} tasks)`,color:colorFor(r.id),dash:dashFor(r.id),rows:r.series?.[key]}));
  chart('chart-score',groups('score'),'score',reference,[0,100]);
  chart('chart-tps',groups('throughput'),'tokens');
  chart('chart-cap',groups('memory'),'cap_reserved_gib');
  chart('chart-ram',groups('memory'),'ram_gib',173,[0,185]);
}

function renderDetail() {
  const r=snapshot.runs.find(r=>r.id===selected);if(!r)return;
  $('run-title').textContent=r.name;
  $('run-meta').textContent=`${r.version} · ${r.tasks || '?'} tasks · ${r.scoring} · ${r.clock}`;
  const facts=[r.started_at ? `Started ${r.started_at}`:null, `Elapsed ${fmt(r.elapsed_h,2)} h`,
    `Running ${fmt(r.running,0)} · queued ${fmt(r.queued,0)}`,`KV ${fmt(finite(r.kv)?100*r.kv:null,0)}%`,
    r.expected_draws?`${r.expected_draws} independent draws / task${r.extra_retry_enabled?' + conditional retries':''} · at most 2 answers`:null,
    `Host RAM ${fmt(r.ram_gib)} GiB`,
    `Reported GPU prefix hits ${fmt(finite(r.prefix_hit)?100*r.prefix_hit:null,0)}% · CPU KV restores ${fmt(r.host_tok_s,0)} tok/s`,
    `OOMs ${fmt(r.ooms,0)} · allocator retries ${fmt(r.allocator_retries,0)}`,`Restarts ${fmt(r.restarts,0)}`,
    finite(r.baseline_pass2)?`V12 final on this run’s matching tasks: ${fmt(r.baseline_pass2,2)}%`:null,
    r.score_note || null,r.holdout_locked?'Holdout accuracy remains locked until the run completes':null];
  $('run-facts').innerHTML=facts.filter(Boolean).map(x=>`<span>${esc(x)}</span>`).join('');
  renderComparison();
  $('log').textContent=r.log_tail || 'No solver log recorded for this source.';
}

function renderNotes() {
  document.querySelectorAll('[data-tab]').forEach(b=>b.classList.toggle('active',b.dataset.tab===noteTab));
  const notes=snapshot.notes[noteTab] || [];
  $('notes').innerHTML=notes.length ? notes.map(n=>{
    if(typeof n==='string')return `<article class="note"><p>${esc(n)}</p></article>`;
    if(noteTab==='findings')return `<article class="note"><div class="meta">${esc(n.id)} · ${esc(n.area)} · ${esc(n.severity)} · research note</div><p>${esc(n.finding)}</p><div class="status">${esc(n.status)}</div></article>`;
    if(noteTab==='events')return `<article class="note"><div class="meta">Recorded UTC ${esc(n.utc)} · ${esc(n.run)} · hour ${esc(n.kaggle_h)} · ${esc(n.kind)}</div><p>${esc(n.event)}</p></article>`;
    if(noteTab==='submissions')return `<article class="note"><div class="meta">${esc(n.id)} · ${esc(n.date)}</div><p>${Object.entries(n).map(([k,v])=>`${esc(k)}: ${esc(typeof v==='object'?JSON.stringify(v):v)}`).join('<br>')}</p></article>`;
    return `<article class="note"><div class="meta">${esc(n.author || n.source || 'Research comment')} · ${esc(n.utc || n.date || '')}</div><p>${esc(n.text || n.comment || n.body || JSON.stringify(n))}</p></article>`;
  }).join('') : '<p class="muted">No additional comments recorded. Existing timeline comments and findings are preserved in the other tabs.</p>';
  $('archive').href=snapshot.archive;
}

function stamp(health={}) {
  if (!snapshot) return;
  const age=Date.now()/1000-snapshot.generated_epoch;
  $('stamp').textContent=`Snapshot ${snapshot.generated_at || 'unavailable'} · ${ago(age)}`;
  $('connection').textContent=health.refreshing?'Fetching pods…':age>180?'Snapshot is stale':'● Automatic monitoring';
  $('refresh').disabled=Boolean(health.refreshing);
  const errors=[...(snapshot.errors||[]),...(health.error?[health.error]:[]),...(health.collector_alive===false?['Collector is not running']:[])];
  $('error').hidden=!errors.length;$('error').textContent=errors.join(' · ');
}

async function load() {
  if(loading)return;loading=true;
  try {
    const r=await fetch('/api/snapshot',{cache:'no-store',headers:etag?{'If-None-Match':etag}:{}});
    if(r.status!==304){if(!r.ok)throw new Error(`HTTP ${r.status}`);snapshot=await r.json();etag=r.headers.get('ETag');
      if(snapshot.generated_at){renderSummary();renderPods();renderRuns();renderDetail();renderNotes();}
      else $('summary').textContent='Collecting the first pod observations…';}
    const health=await fetch('/api/health',{cache:'no-store'}).then(r=>r.json());stamp(health);
    window.__monitorReady=Boolean(snapshot?.generated_at);window.__monitorGeneration=snapshot?.generation;
  } catch(e) { $('connection').textContent='Connection unavailable';$('error').hidden=false;$('error').textContent=`Unable to fetch monitor data. Last readings are retained. ${e.message}`; }
  finally {loading=false;}
}
$('runs').addEventListener('click',e=>{if(e.target.closest('input,label,button'))return;const row=e.target.closest('[data-id]');if(row)selectRun(row.dataset.id);});
$('runs').addEventListener('keydown',e=>{if(e.target.closest('input,label,button'))return;if(e.key==='Enter'||e.key===' '){const row=e.target.closest('[data-id]');if(row){e.preventDefault();selectRun(row.dataset.id);}}});
$('runs').addEventListener('change',e=>{if(e.target.matches('[data-compare]'))toggleComparison(e.target.dataset.compare,e.target.checked);});
$('compare-options').addEventListener('change',e=>{if(e.target.matches('[data-compare]'))toggleComparison(e.target.dataset.compare,e.target.checked);});
$('compare-legend').addEventListener('click',e=>{const b=e.target.closest('[data-compare-remove]');if(b)toggleComparison(b.dataset.compareRemove,false);});
$('compare-selected').addEventListener('click',()=>setComparison([selected]));
$('compare-live').addEventListener('click',()=>setComparison(snapshot.runs.filter(r=>['running','starting','recovering','finishing','blocked','telemetry-stale'].includes(r.state)).map(r=>r.id)));
$('compare-gold').addEventListener('click',()=>{const gold=snapshot.runs.find(r=>r.name==='Manh V12 replica'||r.run_name==='m12_rerun');if(gold)toggleComparison(gold.id,true);});
$('compare-clear').addEventListener('click',()=>setComparison([]));
darkMode.addEventListener('change',()=>{if(snapshot)renderDetail();});
$('run-select').addEventListener('change',e=>selectRun(e.target.value));
$('history').checked=localStorage.getItem('arc-monitor-history')==='true';
$('history').addEventListener('change',()=>{localStorage.setItem('arc-monitor-history',$('history').checked);renderRuns();renderDetail();});
document.querySelectorAll('[data-tab]').forEach(b=>b.addEventListener('click',()=>{noteTab=b.dataset.tab;renderNotes();}));
$('refresh').addEventListener('click',async()=>{try{$('refresh').disabled=true;await fetch('/api/refresh',{method:'POST'});await load();}catch(e){$('error').hidden=false;$('error').textContent='Refresh request failed; automatic retries continue.';}});
load();setInterval(load,10000);
