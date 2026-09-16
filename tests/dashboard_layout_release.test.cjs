// SPDX-License-Identifier: Apache-2.0
// Requires jsdom: see tests/dashboard_layout_TESTING.md. Mocked geometry only.
const {JSDOM}=require('jsdom');
const fs=require('fs'), cp=require('child_process'), assert=require('node:assert/strict');
const repo=require('node:path').resolve(__dirname, '..');
const baseline=process.argv.includes('--baseline');
const read=p=>baseline?cp.execFileSync('git',['show','HEAD:'+p],{cwd:repo,encoding:'utf8'}):fs.readFileSync(repo+'/'+p,'utf8');
const source=read('omlx/admin/static/js/status-layout.js');
const dom=new JSDOM(`<div class="omlx-page-shell"><div id="status-layout">${['a','b','c'].map(id=>`<div class="omlx-card" data-card="${id}"><div class="omlx-drag-handle"></div><section>content</section></div>`).join('')}</div></div>`,{url:'https://dashboard.example',runScripts:'outside-only',pretendToBeVisual:true});
const w=dom.window, d=w.document, root=d.querySelector('#status-layout');
const rect=(x,y,width,height)=>({x,y,left:x,top:y,width,height,right:x+width,bottom:y+height});
Object.defineProperty(w,'innerWidth',{value:1400,configurable:true});
let resizeCallback;
w.ResizeObserver=class{constructor(cb){resizeCallback=cb}observe(){}};
w.scrollBy=()=>{};
root.getBoundingClientRect=()=>rect(0,0,1000,500);
const items=[...root.children];
items.forEach((el,i)=>{el.getBoundingClientRect=()=>rect(i===1?520:0,i===2?220:0,480,180);});
d.elementFromPoint=(x,y)=>items.find(el=>{const b=el.getBoundingClientRect();return x>=b.left&&x<=b.right&&y>=b.top&&y<=b.bottom})||root;
const order=()=>[...root.children].map(el=>el.dataset.card);
const fire=(target,type,x,y)=>{const e=new w.Event(type,{bubbles:true,cancelable:true});Object.assign(e,{clientX:x,clientY:y,pointerId:17,pointerType:'mouse',button:0,buttons:type==='pointerup'?0:1});target.dispatchEvent(e);};
const sleep=ms=>new Promise(r=>setTimeout(r,ms));
(async()=>{
 w.eval(source);w.dispatchEvent(new w.Event('load'));await sleep(90);
 const failures=[];const check=(label,fn)=>{try{fn();console.log('PASS',label)}catch(e){failures.push(label);console.log('FAIL',label,e.message.slice(0,250))}};
 const before=order();
 fire(items[0].querySelector('.omlx-drag-handle'),'pointerdown',10,10);
 fire(w,'pointermove',950,90);
 check('no DOM reorder before release',()=>assert.deepEqual(order(),before));
 check('visible insertion marker',()=>assert.ok(d.querySelector('[data-omlx-drop-indicator]')));
 const styles=items.map(el=>el.style.cssText);
 resizeCallback();await sleep(350);
 check('observer does not change frozen card styles',()=>assert.deepEqual(items.map(el=>el.style.cssText),styles));
 fire(w,'pointerup',950,90);
 check('one reorder on release',()=>assert.deepEqual(order(),['b','a','c']));
 check('persist committed order',()=>assert.deepEqual(JSON.parse(w.localStorage.getItem('omlx-status-layout-v4')).order,['b','a','c']));
 check('marker and drag state cleaned',()=>assert.equal(d.querySelector('[data-omlx-drop-indicator],.omlx-dragging'),null));
 const saved=w.localStorage.getItem('omlx-status-layout-v4');
 for(const how of ['pointercancel','escape','blur','outside']){
  const prior=order();fire(items[0].querySelector('.omlx-drag-handle'),'pointerdown',10,10);fire(w,'pointermove',40,350);
  if(how==='escape')w.dispatchEvent(new w.KeyboardEvent('keydown',{key:'Escape'}));
  else if(how==='blur')w.dispatchEvent(new w.Event('blur'));
  else if(how==='outside')fire(w,'pointerup',1200,600);
  else fire(w,'pointercancel',40,350);
  check(how+' cancels without saving',()=>{assert.deepEqual(order(),prior);assert.equal(w.localStorage.getItem('omlx-status-layout-v4'),saved);assert.equal(d.querySelector('.omlx-dragging'),null)});
 }
 check('natural heights restored',()=>items.forEach(el=>assert.equal(el.style.height,'')));
 const template=read('omlx/admin/templates/dashboard/_status.html');
 check('slider input only previews',()=>assert.match(template,/@input="previewLayoutMaxWidth\(/));
 check('slider change commits',()=>assert.match(template,/@change="setLayoutMaxWidth\(/));
 const dj=read('omlx/admin/static/js/dashboard.js');
 const methods=dj.slice(dj.indexOf('            setLayoutColumns(n)'),dj.indexOf('            shellQuote(value)'));
 const api=w.eval('({'+methods+'})');api.layoutSettings={columns:2,maxWidth:1740};
 const shell=d.querySelector('.omlx-page-shell');let oldWidth=shell.style.getPropertyValue('--omlx-max-width');
 check('preview method does not apply/persist',()=>{api.previewLayoutMaxWidth(1200);assert.equal(shell.style.getPropertyValue('--omlx-max-width'),oldWidth);assert.equal(w.localStorage.getItem('omlx-status-settings-v4'),null);assert.equal(api.layoutSettings.maxWidth,1200)});
 check('commit applies synchronously',()=>{api.setLayoutMaxWidth(1200);assert.equal(shell.style.getPropertyValue('--omlx-max-width'),'1200px');assert.equal(JSON.parse(w.localStorage.getItem('omlx-status-settings-v4')).maxWidth,1200)});
 console.log('Failures:',failures.length);w.close();process.exitCode=failures.length?1:0;
})().catch(e=>{console.error(e);w.close();process.exitCode=1});
