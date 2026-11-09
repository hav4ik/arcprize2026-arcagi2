// Run with a dedicated Chromium profile and --remote-debugging-port=9235, then:
// node kaggle/dashboard/submissions_tracker/monitor/tests/compare_browser.mjs
// No dependencies beyond Node 22; uses the running dashboard and real browser interactions.
import assert from 'node:assert/strict';
import fs from 'node:fs';

const base=process.env.ARC_MONITOR_URL || 'http://127.0.0.1:8765';
const debug=process.env.ARC_CHROME_DEBUG || 'http://127.0.0.1:9235';
const pages=await(await fetch(debug+'/json')).json();
const ws=new WebSocket(pages.find(p=>p.type==='page').webSocketDebuggerUrl);
await new Promise((resolve,reject)=>{ws.onopen=resolve;ws.onerror=reject;});
let sequence=0;const pending=new Map(), errors=[];
ws.onmessage=e=>{
  const m=JSON.parse(e.data);
  if(m.id&&pending.has(m.id)){const p=pending.get(m.id);pending.delete(m.id);m.error?p.reject(m.error):p.resolve(m.result);}
  if(m.method==='Runtime.exceptionThrown')errors.push(m.params.exceptionDetails.text);
  if(m.method==='Log.entryAdded'&&m.params.entry.level==='error')errors.push(m.params.entry.text);
};
function call(method,params={}){
  return new Promise((resolve,reject)=>{
    const id=++sequence;pending.set(id,{resolve,reject});ws.send(JSON.stringify({id,method,params}));
    setTimeout(()=>{if(pending.has(id)){pending.delete(id);reject(Error('Timeout: '+method));}},15000).unref();
  });
}
async function ev(expression){
  const r=await call('Runtime.evaluate',{expression,returnByValue:true,awaitPromise:true});
  if(r.exceptionDetails)throw Error(JSON.stringify(r.exceptionDetails));return r.result.value;
}
async function until(expression, seconds=30){
  for(let i=0;i<seconds*4;i++){if(await ev(expression))return;await new Promise(r=>setTimeout(r,250));}
  throw Error('Timed out waiting for '+expression);
}
try{
  await call('Runtime.enable');await call('Log.enable');await call('Log.clear');await call('Page.enable');
  await call('Emulation.setDeviceMetricsOverride',{width:1440,height:1100,deviceScaleFactor:1,mobile:false});
  const start=Date.now();await call('Page.navigate',{url:base+'/local.html'});
  await until('window.__monitorReady && performance.timeOrigin>='+start);
  await ev('document.getElementById("compare-clear").click()');
  const ids=await ev(`snapshot.runs.filter(r=>!r.historical&&r.tasks===120&&r.series.score.length&&r.series.memory.length).slice(0,2).map(r=>r.id)`);
  assert.equal(ids.length,2,'Need two measured 120-task runs for this browser test');
  const selectedBefore=await ev('selected');
  for(const id of ids)await ev(`document.querySelector('#runs input[data-compare='+CSS.escape(${JSON.stringify(id)})+']').click()`);
  assert.equal(await ev('selected'),selectedBefore,'Comparison checkbox must not change inspected run');
  assert.deepEqual(await ev('[...compared]'),ids);
  assert.deepEqual(await ev(`[...document.querySelectorAll('.chart')].map(c=>c.querySelectorAll('path.line').length)`),[2,2,2,2]);
  assert.equal(await ev(`new Set([...document.querySelectorAll('#chart-score path.line')].map(p=>getComputedStyle(p).stroke)).size`),2);
  await call('Emulation.setEmulatedMedia',{features:[{name:'prefers-color-scheme',value:'dark'}]});
  await until(`getComputedStyle(document.querySelector('#chart-score path.line')).stroke==='rgb(128, 170, 255)'`);
  await call('Emulation.setEmulatedMedia',{features:[{name:'prefers-color-scheme',value:'light'}]});
  assert.equal(await ev(`document.getElementById('compare-note').textContent.includes('Workloads differ')`),false);
  await ev('document.getElementById("compare-gold").click()');
  assert.equal(await ev('compared.size'),3);
  assert.equal(await ev(`document.querySelectorAll('#chart-score path.line').length`),3);
  assert.equal(await ev(`document.getElementById('compare-note').textContent.includes('120 / 240')`),true);
  const saved=await ev('[...compared]');
  await ev('document.querySelector(\'[data-tab="events"]\').click()');
  const comments=await ev('document.querySelectorAll("#notes .note").length');
  await ev('document.getElementById("detail").scrollIntoView()');
  const screenshot=await call('Page.captureScreenshot',{format:'png'});
  fs.writeFileSync('/tmp/arc-monitor-comparison.png',Buffer.from(screenshot.data,'base64'));
  await call('Emulation.setDeviceMetricsOverride',{width:390,height:844,deviceScaleFactor:1,mobile:true});
  assert.equal(await ev('document.documentElement.scrollWidth>innerWidth'),false,'Mobile page overflow');
  const reload=Date.now();await call('Page.reload',{ignoreCache:true});
  await until('window.__monitorReady && performance.timeOrigin>='+reload);
  assert.deepEqual(await ev('[...compared]'),saved,'Selections must survive reload');
  await ev('document.querySelector(\'[data-tab="events"]\').click()');
  await until('!document.getElementById("refresh").disabled');
  const generation=await ev('window.__monitorGeneration');
  await ev('document.getElementById("refresh").click()');
  await until('window.__monitorGeneration!=='+generation,45);
  assert.deepEqual(await ev('[...compared]'),saved,'Selections must survive automatic refresh');
  assert.equal(await ev('document.querySelectorAll("#notes .note").length'),comments);
  await ev('document.getElementById("compare-clear").click()');
  assert.equal(await ev('document.querySelectorAll(".chart svg").length'),0);
  assert.equal(await ev('document.getElementById("chart-score").textContent'),'Select runs to compare');
  assert.deepEqual(errors,[],'Browser errors');
  console.log(JSON.stringify({twoRunOverlay:true,distinctColors:true,archivedGoldOverlay:true,mixedWorkloadLabel:true,
    persistedOnReload:true,persistedOnRefresh:true,darkModeColors:true,mobileOverflow:false,commentsPreserved:comments,consoleErrors:errors}));
}finally{ws.close();}
