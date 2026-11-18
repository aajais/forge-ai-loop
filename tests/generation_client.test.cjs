const {test}=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const vm=require('node:vm');
const crypto=require('node:crypto');
const flush=async()=>{for(let i=0;i<6;i++)await new Promise(setImmediate);};
const response=body=>({ok:true,text:async()=>JSON.stringify(body)});
test('shows rendered previews and finishes from status even when the original response stalls',async()=>{
  let tick, status, posted; const previews=[];
  const context=vm.createContext({crypto,console,setInterval:fn=>{tick=fn;return 1;},clearInterval:()=>{},
    fetch:async(url,options)=>{
      if(options?.method==='POST'){posted=JSON.parse(options.body);return new Promise(()=>{});}
      return response(status);
    }});
  if(fs.existsSync('src/bufogen/app/static/generation-client.js'))vm.runInContext(fs.readFileSync('src/bufogen/app/static/generation-client.js','utf8'),context);
  assert.equal(vm.runInContext('typeof generateDrawing',context),'function');
  context.preview=row=>previews.push(row);
  const done=vm.runInContext("generateDrawing({prompt:'frog',backend:'qwen_base',previous_id:null},preview)",context);
  await flush();assert.match(posted.request_id,/^[a-f0-9]{32}$/);
  status={status:'rendered',row:{id:'preview',png_url:'/api/generations/x/preview.png'}};
  await tick();assert.equal(previews[0].id,'preview');
  status={status:'complete',row:{id:'final',group_id:'g',backend:'qwen_base',valid:true,png_url:'/api/files/final.png'}};
  await tick();assert.equal((await done).id,'final');
});
