const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const crypto = require('node:crypto');
class Element {
  constructor(tag='div', text='') { Object.assign(this,{tag,textContent:text,children:[],events:{},value:'',dataset:{},attributes:{}}); }
  append(...children) { this.children.push(...children); }
  replaceChildren(...children) { this.children=children; }
  setAttribute(k,v) { this.attributes[k]=v; }
  addEventListener(k,v) { this.events[k]=v; }
  focus() {}
  add(child) { this.append(child); }
}
const flush = async () => { for(let i=0;i<8;i++) await new Promise(setImmediate); };
function boot({storage=new Map(), unavailable=[], statusFails=false}={}) {
  const nodes=Object.fromEntries(['compare-form','compare-prompt','run-models','comparison-results','comparison-status','saved-comparisons','comparison-notice','reload-models'].map(id=>[id,new Element()]));
  const pending=[];
  const ids=['qwen_base','sft_q3','rl_q3','closed_quality_bar'];
  const models=ids.map(id=>({id,label:id,model:id,model_version:id==='rl_q3'?'adapter:v6':null,available:!unavailable.includes(id),unavailable_reason:'Missing credential'}));
  const context=vm.createContext({document:{getElementById:id=>nodes[id],createElement:tag=>new Element(tag)},crypto,console,Date,Map,
    Option:class extends Element {constructor(text,value){super('option',text);this.value=value;}},
    localStorage:{getItem:k=>storage.get(k)||null,setItem:(k,v)=>storage.set(k,v)},setInterval:()=>0,clearInterval(){},
    fetch:async(url,options)=>{
      if(url==='/api/status') return {ok:!statusFails,text:async()=>JSON.stringify({models})};
      return new Promise(resolve=>pending.push({body:JSON.parse(options.body),resolve}));
    }});
  vm.runInContext(fs.readFileSync('src/bufogen/app/static/generation-client.js','utf8'), context);
  vm.runInContext(fs.readFileSync('src/bufogen/app/static/demo-models.js','utf8'),context);
  if(fs.existsSync('src/bufogen/app/static/compare.js')) vm.runInContext(fs.readFileSync('src/bufogen/app/static/compare.js','utf8'),context);
  const send=prompt=>{nodes['compare-prompt'].value=prompt; return nodes['compare-form'].events.submit?.({preventDefault(){}});};
  const complete=(index, overrides={})=>pending[index].resolve({ok:true,text:async()=>JSON.stringify({id:`row-${index}`,group_id:`group-${index}`,backend:pending[index].body.backend,prompt:pending[index].body.prompt,valid:true,png_url:`/api/files/${index}.png`,model_label:pending[index].body.backend,model_version:'actual:v6',latency_s:12,conversation_url:'https://example.com/trace',...overrides})});
  const allText=node=>[node.textContent,...node.children.map(allText)].join(' ');
  return {nodes,pending,send,complete,storage,text:()=>allText(nodes['comparison-results'])};
}
test('one prompt starts all four fresh executions and keeps output order as they finish',async()=>{
  const ui=boot();await flush();const sending=ui.send('Airport Bufo');await flush();
  assert.equal(ui.pending.length,4);
  assert.deepEqual(ui.pending.map(r=>r.body.backend),['qwen_base','sft_q3','rl_q3','closed_quality_bar']);
  assert.ok(ui.pending.every(r=>r.body.prompt==='Airport Bufo' && r.body.previous_id===null));
  assert.equal(ui.nodes['run-models'].disabled,true);
  await ui.send('Duplicate');assert.equal(ui.pending.length,4);
  ui.complete(3);await flush();
  assert.deepEqual(ui.nodes['comparison-results'].children.map(n=>n.dataset.backend),['qwen_base','sft_q3','rl_q3','closed_quality_bar']);
  assert.ok(ui.text().includes('Completed'));
  assert.equal(ui.nodes['comparison-results'].children[3].children.find(n=>n.className==='drawing-link').children[0].src, '/api/files/3.png');
  ui.pending[1].resolve({ok:false,text:async()=>'<html>Bad Gateway</html>'});await flush();
  assert.ok(ui.text().includes('Could not make this drawing'));
  ui.complete(0);ui.complete(2);await sending;
  assert.equal(ui.nodes['run-models'].disabled,false);
  assert.ok(ui.text().includes('actual:v6'));
  const restored=boot({storage:ui.storage});await flush();
  assert.ok(restored.text().includes('Completed'));assert.ok(restored.text().includes('Could not make this drawing'));
  assert.equal(restored.pending.length,0);
});

function descendants(node) { return [node, ...node.children.flatMap(descendants)]; }
function retryButton(ui, index=0) {
  return descendants(ui.nodes['comparison-results'].children[index]).find(n=>n.tag==='button' && n.textContent==='Retry this model');
}
test('an invalid drawing can retry alone while other models finish, without duplicate submissions',async()=>{
  const ui=boot();await flush();const sending=ui.send('GODZILLA BUFO');await flush();
  ui.complete(0,{valid:false,png_url:null,truncated_calls:2,tool_calls:0,error:'No drawing was rendered.'});await flush();
  assert.ok(ui.text().includes('output limit'));
  const retry=retryButton(ui);assert.ok(retry);
  const retrying=retry.events.click();await flush();
  await retry.events.click();assert.equal(ui.pending.length,5);
  assert.equal(ui.pending[4].body.backend,'qwen_base');
  assert.equal(ui.pending[4].body.prompt,'GODZILLA BUFO');
  assert.equal(ui.pending[4].body.previous_id,null);
  assert.notEqual(ui.pending[4].body.request_id,ui.pending[0].body.request_id);
  ui.complete(1);ui.complete(2);ui.complete(3);await sending;
  assert.equal(ui.nodes['run-models'].disabled,true);
  assert.ok(!ui.nodes['comparison-status'].textContent.startsWith('Finished'));
  ui.complete(4);await retrying;
  assert.equal(ui.nodes['run-models'].disabled,false);
  assert.ok(ui.text().includes('Attempt 2'));
  assert.ok(descendants(ui.nodes['comparison-results'].children[0]).some(n=>n.tag==='a' && n.textContent.includes('attempt 1') && n.href==='https://example.com/trace'));
  assert.equal(ui.nodes['comparison-results'].children[1].children.find(n=>n.className==='drawing-link').children[0].src,'/api/files/1.png');
});
test('saved failed and interrupted results have retry controls after reload',async()=>{
  const ui=boot();await flush();const sending=ui.send('Bufo');await flush();
  ui.pending[0].resolve({ok:false,text:async()=>'<html>Bad Gateway</html>'});await flush();
  const restored=boot({storage:ui.storage});await flush();
  assert.ok(retryButton(restored));assert.ok(retryButton(restored,1));
  const retrying=retryButton(restored).events.click();await flush();
  assert.equal(restored.pending.length,1);
  restored.complete(0);await retrying;
  assert.equal(retryButton(restored),undefined);
  ui.complete(1);ui.complete(2);ui.complete(3);await sending;
});
test('a background retry stays with its saved comparison and can fail and retry again',async()=>{
  const ui=boot();await flush();let sending=ui.send('First');await flush();
  ui.complete(0,{valid:false,png_url:null,error:'No drawing was rendered.'});
  ui.complete(1);ui.complete(2);ui.complete(3);await sending;
  const firstId=ui.nodes['saved-comparisons'].value;
  sending=ui.send('Second');await flush();
  const secondId=ui.nodes['saved-comparisons'].value;
  const choose=id=>{ui.nodes['saved-comparisons'].value=id;ui.nodes['saved-comparisons'].events.change({target:ui.nodes['saved-comparisons']});};
  choose(firstId);let retrying=retryButton(ui).events.click();await flush();
  choose(secondId);ui.pending[8].resolve({ok:false,text:async()=>JSON.stringify({detail:'Retry failed'})});await retrying;
  assert.equal(ui.nodes['compare-prompt'].value,'Second');assert.ok(!ui.text().includes('Retry failed'));
  choose(firstId);assert.ok(ui.text().includes('Retry failed'));
  retrying=retryButton(ui).events.click();await flush();assert.equal(ui.pending.length,10);
  choose(secondId);ui.complete(9);await retrying;
  assert.equal(ui.nodes['run-models'].disabled,true);
  ui.complete(4);ui.complete(5);ui.complete(6);ui.complete(7);await sending;
  assert.equal(ui.nodes['run-models'].disabled,false);
  choose(firstId);assert.ok(ui.text().includes('Attempt 3'));assert.equal(retryButton(ui),undefined);
});
test('unavailable models get their own error and available models still run',async()=>{
  const ui=boot({unavailable:['rl_q3']});await flush();const sending=ui.send('Bufo');await flush();
  assert.equal(ui.pending.length,3);assert.ok(ui.text().includes('Missing credential'));
  assert.equal(retryButton(ui,2).disabled,true);
  ui.complete(0);ui.complete(1);ui.complete(2);await sending;
});
test('reload marks unfinished executions interrupted without automatically submitting again',async()=>{
  const ui=boot();await flush();const sending=ui.send('Bufo');await flush();
  assert.equal(ui.pending.length,4);ui.complete(0);await flush();
  const restored=boot({storage:ui.storage});await flush();
  assert.equal(restored.pending.length,0);assert.ok(restored.text().includes('Interrupted'));assert.ok(restored.text().includes('Completed'));
  ui.complete(1);ui.complete(2);ui.complete(3);await sending;
});
test('status failure disables generation and provides a retry control',async()=>{
  const ui=boot({statusFails:true});await flush();
  assert.equal(ui.nodes['run-models'].disabled,true);
  assert.ok(ui.nodes['comparison-notice'].textContent.includes('Could not load models'));
  assert.equal(typeof ui.nodes['reload-models'].events.click,'function');
});

test('switching saved comparisons while drawing never loses or misroutes background results',async()=>{
  const ui=boot();await flush();let sending=ui.send('First');await flush();
  ui.complete(0);ui.complete(1);ui.complete(2);ui.complete(3);await sending;
  const firstId=ui.nodes['saved-comparisons'].value;
  sending=ui.send('Second');await flush();
  const secondId=ui.nodes['saved-comparisons'].value;
  assert.equal(ui.nodes['saved-comparisons'].disabled,false);
  const choose=id=>{ui.nodes['saved-comparisons'].value=id;ui.nodes['saved-comparisons'].events.change({target:ui.nodes['saved-comparisons']});};
  choose(firstId);assert.equal(ui.nodes['compare-prompt'].value,'First');
  ui.complete(4);await flush();assert.equal(ui.nodes['compare-prompt'].value,'First');
  choose(secondId);assert.equal(ui.nodes['compare-prompt'].value,'Second');
  ui.complete(5);await flush();assert.ok(ui.nodes['comparison-results'].children[1].children.some(n=>n.textContent.startsWith('Completed')));
  choose(firstId);ui.complete(6);ui.complete(7);await sending;
  assert.equal(ui.nodes['compare-prompt'].value,'First');
  choose(secondId);assert.ok(!ui.text().includes('Drawing…'));
});
