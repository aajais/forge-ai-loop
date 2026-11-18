// Exercise the shipped chat controller with a small DOM adapter and delayed HTTP responses.
// This checks request/navigation races; it is not a browser layout or clipboard test.
const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const crypto = require('node:crypto');

class Element {
  constructor(tag = 'div', text = '') {
    Object.assign(this, {tag, textContent:text, children:[], events:{}, style:{}, dataset:{}, value:'', className:'', attributes:{}});
    this.classList = {add: name => { this.className += ` ${name}`; }, toggle: () => {}};
  }
  append(...children) { this.children.push(...children); }
  replaceChildren(...children) { this.children = children; }
  setAttribute(key, value) { this.attributes[key] = value; }
  removeAttribute(key) { delete this[key]; }
  addEventListener(name, fn) { this.events[name] = fn; }
  add(child) { this.append(child); }
  focus() {}
  get childElementCount() { return this.children.length; }
  querySelectorAll(selector) {
    const match = node => selector.startsWith('.') ? node.className.split(' ').includes(selector.slice(1)) : node.tag === selector;
    return this.children.flatMap(child => [ ...(match(child) ? [child] : []), ...child.querySelectorAll(selector) ]);
  }
  querySelector(selector) { return this.querySelectorAll(selector)[0] || null; }
}
const flush = async () => { for (let i = 0; i < 8; i++) await new Promise(resolve => setImmediate(resolve)); };
function boot(clipboard = {writeText:async()=>{}}, routes = {}) {
  const html = fs.readFileSync('src/bufogen/app/static/index.html', 'utf8');
  const nodes = Object.fromEntries([...html.matchAll(/id="([^"]+)"/g)].map(m => [m[1], new Element()]));
  const document = {
    getElementById: id => nodes[id], createElement: tag => new Element(tag), addEventListener() {},
    querySelectorAll: selector => selector.startsWith('#') ? [nodes[selector.slice(1)]] : Object.values(nodes).flatMap(n => n.querySelectorAll(selector)),
  };
  const pending = [], timers = [];
  const context = vm.createContext({document, crypto, console, Date, Map, navigator:{clipboard}, FileReader:class {},
    Option:class extends Element { constructor(text,value) { super('option', text); this.value=value; } },
    localStorage:{getItem:()=>null,setItem(){}}, setInterval:fn=>{timers.push(fn);return timers.length;}, clearInterval(){},
    fetch: async (url, options) => {
      if (routes[url]) return routes[url]();
      if (url.startsWith('/api/generations/') && routes['/api/generations/']) return routes['/api/generations/']();
      if (url === '/api/status') return {ok:true,text:async()=>JSON.stringify({agents_url:'https://example.com',default_backend:'open_base',models:[
        {id:'open_base',label:'google/gemma-4-26B-A4B-it',model:'google/gemma-4-26B-A4B-it',available:true},
        {id:'closed_quality_bar',label:'gpt-6.1-sol',model:'gpt-6.1-sol',available:true},
        {id:'open_tuned',label:'ft-google/gemma-4-26B-A4B-it',model:'wandb-artifact:///team/project/adapter:v2',model_version:'wandb-artifact:///team/project/adapter:v2',tuning_method:'sft',available:true},
        {id:'qwen_base',label:'Qwen/Qwen3.8-27B',model:'Qwen/Qwen3.8-27B',available:true},
        {id:'sft_q3',label:'ft-Qwen/Qwen3.8-27B',model:'wandb-artifact:///team/project/qwen:v1',model_version:'wandb-artifact:///team/project/qwen:v1',available:true},
        {id:'nemotron_base',label:'nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B',model:'nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B',available:true},
        {id:'sft_n3',label:'ft-nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B',model:'wandb-artifact:///team/project/nemotron:v1',model_version:'wandb-artifact:///team/project/nemotron:v1',available:true},
        {id:'rl_q3',label:'rl-Qwen/Qwen3.8-27B',model:'wandb-artifact:///team/project/qwen:v6',model_version:'wandb-artifact:///team/project/qwen:v6',available:true},
        {id:'forge_proxy',label:'bufo-gen (routing alias; model pending)',model:'bufo-gen',routing_alias:'bufo-gen',available:true},
      ]})};
      if (url === '/api/history') return {ok:true,text:async()=>'[]'};
      return await new Promise(resolve => pending.push({body:JSON.parse(options.body),resolve}));
    },
  });
  if (fs.existsSync('src/bufogen/app/static/demo-models.js')) vm.runInContext(fs.readFileSync('src/bufogen/app/static/demo-models.js','utf8'), context);
  vm.runInContext(fs.readFileSync('src/bufogen/app/static/generation-client.js','utf8'), context);
  vm.runInContext(fs.readFileSync('src/bufogen/app/static/app.js','utf8'), context);
  const act = code => vm.runInContext(code,context);
  const send = text => { nodes.prompt.value=text; return nodes['prompt-form'].events.submit({preventDefault(){}}); };
  const complete = (index, id, group) => pending[index].resolve({ok:true,text:async()=>JSON.stringify({id,group_id:group,prompt:pending[index].body.prompt,backend:pending[index].body.backend,response_text:`Reply ${id}`,valid:true,png_url:`/${id}.png`})});
  return {nodes,pending,act,send,complete,timers};
}

test('switching chats and out-of-order completions retain each conversation and model', async () => {
  const ui = boot(); await flush();
  const first = ui.send('First drawing'); await flush();
  const firstKey = ui.act('active');
  assert.equal(ui.nodes['new-chat'].disabled, false);
  assert.equal(ui.nodes['conversation-select'].disabled, false);
  ui.nodes['new-chat'].events.click();
  assert.equal(ui.nodes.send.disabled, false);
  ui.act("selectModel('closed_quality_bar')");
  const second = ui.send('Second drawing'); await flush();
  assert.equal(ui.pending.length, 2);
  assert.equal(ui.pending[0].body.backend, 'qwen_base');
  assert.equal(ui.pending[1].body.backend, 'closed_quality_bar');
  ui.act(`openConversation(${JSON.stringify(firstKey)})`);
  assert.equal(ui.nodes.send.disabled, true);
  ui.complete(1,'two','group-two'); await second;
  assert.equal(ui.act('active'), firstKey, 'background completion must not steal focus');
  assert.equal(ui.nodes['conversation-title'].textContent, 'First drawing');
  ui.complete(0,'one','group-one'); await first;
  assert.equal(ui.act('active'), 'group-one');
  assert.equal(ui.nodes.send.disabled, false);
  ui.act("openConversation('group-two')");
  const followup = ui.send('Refine second'); await flush();
  assert.equal(ui.pending[2].body.previous_id, 'two');
  assert.equal(ui.pending[2].body.backend, 'closed_quality_bar');
  ui.complete(2,'three','group-two'); await followup;
});

test('model IDs and checkpoint versions remain visible without relabeling historical replies', async () => {
  const ui = boot(); await flush();
  const choices = ui.nodes['model-select'].children.map(node => node.textContent);
  assert.deepEqual(ui.nodes['model-select'].children.map(node => node.value), ['qwen_base', 'sft_q3', 'rl_q3', 'closed_quality_bar']);
  assert.ok(choices.every(label => !label.includes('Gemma') && !label.includes('Nemotron')));
  ui.act("selectModel('sft_q3')");
  assert.equal(ui.nodes['model-status'].textContent, 'ft-Qwen/Qwen3.8-27B');
  assert.equal(ui.nodes['model-status'].title, 'wandb-artifact:///team/project/qwen:v1');
  const node = ui.act(`assistantMessage({backend:'open_tuned',model_label:'ft-google/previous-base',model:'wandb-artifact:///team/project/adapter:v1',model_version:'wandb-artifact:///team/project/adapter:v1'})`);
  assert.equal(node.querySelector('.assistant-model').textContent, 'ft-google/previous-base');
  assert.equal(node.querySelector('.model-version').textContent, 'wandb-artifact:///team/project/adapter:v1');
  assert.equal(node.querySelector('details').querySelector('summary').textContent, 'Checkpoint');
  assert.notEqual(node.querySelector('details').open, true);
});

test('a failed background request preserves its draft without changing the active composer', async () => {
  const ui = boot(); await flush();
  const first = ui.send('Retry me'); await flush();
  const key = ui.act('active');
  ui.nodes['new-chat'].events.click(); ui.nodes.prompt.value = 'Still typing';
  ui.pending[0].resolve({ok:false,text:async()=>JSON.stringify({detail:'Service unavailable'})}); await first;
  assert.equal(ui.nodes.prompt.value, 'Still typing');
  assert.equal(ui.act('active'), null);
  ui.act(`openConversation(${JSON.stringify(key)})`);
  assert.equal(ui.nodes.prompt.value, 'Retry me');
  assert.equal(ui.nodes.notice.textContent, 'Service unavailable');
  assert.equal(ui.nodes.send.disabled, false);
});

test('emoji copy handler and web citations survive rendering without injecting HTML', async () => {
  let copied;
  const ui = boot({writeText:async value=>{copied=value;}}); await flush();
  const node = ui.act(`assistantMessage({valid:true, emoji_name:':bufo-concert:', web_searches:[
    {status:'ok',query:'concert',summary:'Stage details',sources:[
      {url:'https://example.com/event',title:'<script>event</script>'},
      {url:'javascript:alert(1)',title:'Unsafe'},
    ]}
  ]})`);
  const copy = node.querySelector('.copy-emoji'); await copy.events.click();
  assert.equal(copied, ':bufo-concert:');
  assert.equal(copy.textContent, 'Copied!');
  assert.equal(node.querySelector('.emoji-shortcode').textContent, ':bufo-concert:');
  const links = node.querySelector('.web-context').querySelectorAll('a');
  assert.equal(links.length, 1);
  assert.equal(links[0].textContent, '<script>event</script>');
  assert.equal(links[0].href, 'https://example.com/event');
});

test('a non-JSON 502 from /api/generate shows the friendly error and keeps the draft', async () => {
  const ui = boot(); await flush();
  const first = ui.send('Draw a frog'); await flush();
  ui.pending[0].resolve({ok:false,text:async()=>'<html>502 Bad Gateway</html>'}); await first;
  assert.equal(ui.nodes.notice.textContent, 'Could not make this drawing. Please try again.');
  assert.equal(ui.nodes.prompt.value, 'Draw a frog');
  assert.equal(ui.nodes.send.disabled, false);
  const second = ui.send('Draw a frog'); await flush();
  ui.pending[1].resolve({ok:true,text:async()=>'not json'}); await second;
  assert.equal(ui.nodes.notice.textContent, 'The studio returned an unreadable reply. Please try again.');
});

test('status failure leaves the model picker usable and retry reloads models', async () => {
  let healthy = false;
  const good = {agents_url:'https://example.com',default_backend:'open_base',models:[{id:'qwen_base',label:'base',model:'base',available:true}]};
  const ui = boot(undefined, {'/api/status': async () => healthy ? {ok:true,text:async()=>JSON.stringify(good)} : {ok:true,text:async()=>'<html>oops</html>'}});
  await flush();
  assert.equal(ui.nodes.notice.textContent, 'Could not load models. Refresh to retry.');
  const options = ui.nodes['model-select'].children;
  assert.deepEqual(options.map(o => o.textContent), ['Models unavailable', 'Retry loading models']);
  assert.equal(options[0].disabled, true);
  assert.equal(ui.nodes['model-select'].value, options[0].value, 'placeholder is the selected option');
  assert.ok(!options[1].disabled);
  assert.equal(ui.nodes['model-select'].disabled, false);
  healthy = true;
  ui.nodes['model-select'].value = options[1].value;
  ui.nodes['model-select'].events.change({target:ui.nodes['model-select']}); await flush();
  assert.deepEqual(ui.nodes['model-select'].children.map(o => o.textContent), ['base · Base']);
  assert.equal(ui.nodes.notice.textContent, '');
  assert.equal(ui.nodes.send.disabled, false);
});

test('failedThreads is available before the first draw', async () => {
  const ui = boot(); await flush();
  assert.doesNotThrow(() => ui.act('drawConversations()'));
  // Runtime access cannot expose a late declaration (init runs after the script body), so pin the order too.
  const source = fs.readFileSync('src/bufogen/app/static/app.js', 'utf8');
  assert.ok(source.indexOf('const failedThreads') < source.indexOf('function drawConversations'));
});

test('a 200 reply whose JSON is not a plain object is treated as unreadable', async () => {
  const ui = boot(); await flush();
  const replies = ['"ok"', '42', '[{"id":"x","group_id":"g"}]', 'true'];
  for (const [index, body] of replies.entries()) {
    const sending = ui.send('Draw a frog'); await flush();
    ui.pending[index].resolve({ok:true,text:async()=>body}); await sending;
    assert.equal(ui.nodes.notice.textContent, 'The studio returned an unreadable reply. Please try again.');
    assert.equal(ui.nodes.prompt.value, 'Draw a frog');
    assert.equal(ui.act('history.length'), 0);
  }
});

test('saved sidebar conversations are clickable immediately after loading', async () => {
  const ui = boot(undefined, {'/api/history': async () => ({ok:true,text:async()=>JSON.stringify([
    {id:'old',group_id:'saved',prompt:'Saved drawing',backend:'qwen_base',valid:true},
  ])})}); await flush();
  const button=ui.nodes['conversation-list'].children[0];
  assert.equal(button.disabled,false);
  const sending=ui.send('Keep drawing');await flush();
  ui.nodes['new-chat'].events.click();
  const saved=ui.nodes['conversation-list'].children.find(n=>n.title==='Saved drawing');
  assert.equal(saved.disabled,false); saved.events.click();
  assert.equal(ui.act('active'),'saved');
  ui.complete(0,'next','saved');await sending;
});

test('a rendered preview replaces drawing text and a polled completion clears the pending chat',async()=>{
  let progress;
  const ui=boot(undefined,{'/api/generations/':async()=>({ok:true,text:async()=>JSON.stringify(progress)})});await flush();
  const sending=ui.send('Preview me');await flush();
  progress={status:'rendered',row:{id:'preview',backend:'qwen_base',valid:true,png_url:'/api/generations/request/preview.png'}};
  await ui.timers.at(-1)();
  assert.equal(ui.nodes.thread.querySelector('.drawing').src,'/api/generations/request/preview.png');
  assert.equal(ui.nodes.thread.querySelector('.reply').textContent,'Image rendered — finishing the reply…');
  ui.timers[0]();
  assert.equal(ui.nodes.thread.querySelector('.reply').textContent,'Image rendered — finishing the reply…');
  progress={status:'complete',row:{id:'final',group_id:'final-group',prompt:'Preview me',backend:'qwen_base',valid:true,png_url:'/api/files/final.png'}};
  await ui.timers.at(-1)();await sending;
  assert.equal(ui.nodes.send.disabled,false);assert.equal(ui.act('pendingJobs.size'),0);
  assert.equal(ui.nodes.thread.querySelector('.drawing').src,'/api/files/final.png');
  ui.complete(0,'final','final-group');await flush();assert.equal(ui.act('history.length'),1);
});
