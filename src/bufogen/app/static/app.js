const $ = id => document.getElementById(id);
let history = [], active = null, busy = false, loading = true;
const pendingJobs = new Map();
const threadErrors = new Map();
const failedThreads = new Map();
let models = [], selectedBackend = 'open_base';
const drafts = new Map();
const imageDrafts = new Map();
let attachment = null, readingImage = false, imageReadToken = 0;
function syncSend() { $('send').disabled = busy || readingImage || !currentModel()?.available; }
function showAttachment() {
  $('attachment-preview').hidden = !attachment;
  if (attachment) {
    $('attachment-image').src = attachment.data;
    $('attachment-name').textContent = attachment.name;
  } else $('attachment-image').removeAttribute('src');
  syncSend();
}
async function attachImage(file) {
  if (busy || !file) return;
  if (!['image/png', 'image/jpeg', 'image/webp'].includes(file.type)) { notice('Choose a PNG, JPEG, or WebP image.'); return; }
  if (file.size > 5 * 1024 * 1024) { notice('Images must be 5 MB or smaller.'); return; }
  const token = ++imageReadToken; readingImage = true; syncSend(); notice('');
  try {
    const data = await new Promise((resolve, reject) => {
      const reader = new FileReader(); reader.onload = () => resolve(reader.result);
      reader.onerror = () => reject(new Error('Could not read that image. Try attaching it again.'));
      reader.readAsDataURL(file);
    });
    if (token === imageReadToken) { attachment = {data, name: file.name || 'Pasted screenshot'}; showAttachment(); }
  } catch (error) { if (token === imageReadToken) notice(error.message); }
  finally { if (token === imageReadToken) { readingImage = false; syncSend(); } }
}
function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text) node.textContent = text;
  return node;
}
function notice(text) { $('notice').textContent = text; $('notice').hidden = !text; }
function resizeComposer() { $('prompt').style.height = 'auto'; $('prompt').style.height = `${Math.min($('prompt').scrollHeight, 160)}px`; }
function scrollBottom() { $('thread-scroll').scrollTop = $('thread-scroll').scrollHeight; }
function rows() { return history.filter(row => row.group_id === active); }
function remember() { try { localStorage.setItem('bufo-chat', active || 'new'); } catch (_) { /* Storage is optional. */ } }
function currentModel() { return models.find(model => model.id === selectedBackend); }
function selectModel(preferred) {
  selectedBackend = models.find(model => model.id === preferred && model.available)?.id || models.find(model => model.available)?.id || models[0]?.id || 'open_base';
  $('model-select').value = selectedBackend;
  const model = currentModel();
  $('active-model-label').textContent = model?.label || 'No model available';
  $('model-status').textContent = model?.available ? model.label : (model?.unavailable_reason || 'Could not load models. Refresh to retry.');
  $('model-status').title = model?.model_version || model?.model || '';
  $('status-dot').classList.toggle('ready', Boolean(model?.available));
  syncSend();
  try { localStorage.setItem('bufo-model', selectedBackend); } catch (_) { /* Storage is optional. */ }
}
function setBusy(value) {
  busy = value;
  for (const selector of ['#send', '#prompt', '#model-select', '#attach-image', '#remove-image', '#image-file', '[data-prompt]']) {
    document.querySelectorAll(selector).forEach(node => { node.disabled = value; });
  }
  $('new-chat').disabled = loading;
  $('conversation-select').disabled = loading;
  syncSend();
  $('prompt-form').setAttribute('aria-busy', String(value));
}
function drawConversations() {
  const grouped = new Map();
  // Reinsertion keeps recently continued conversations at the top.
  history.forEach(row => { const items = grouped.get(row.group_id) || []; items.push(row); grouped.delete(row.group_id); grouped.set(row.group_id, items); });
  failedThreads.forEach((prompt, id) => { if (!grouped.has(id)) grouped.set(id, [{prompt}]); });
  pendingJobs.forEach((job, id) => { if (!grouped.has(id)) grouped.set(id, [{prompt: job.prompt}]); });
  $('conversation-list').replaceChildren();
  $('conversation-select').replaceChildren(new Option('New conversation', ''));
  [...grouped.entries()].reverse().forEach(([id, items]) => {
    const title = items[0].prompt;
    const button = element('button', 'conversation-item'); button.type = 'button';
    button.setAttribute('aria-current', String(id === active)); button.title = title; button.disabled = loading;
    button.append(element('span', '', title), element('small', '', pendingJobs.has(id) ? 'Drawing…' : `${items.length} drawing${items.length === 1 ? '' : 's'}`));
    button.addEventListener('click', () => openConversation(id));
    $('conversation-list').append(button);
    $('conversation-select').add(new Option(title + (pendingJobs.has(id) ? ' · Drawing…' : ''), id));
  });
  $('conversation-select').value = active || '';
}
function assistantMessage(row) {
  const message = element('article', 'assistant-message');
  const avatar = element('img', 'avatar'); avatar.src = '/static/bufo-avatar.png'; avatar.alt = '';
  const body = element('div', 'assistant-body');
  const author = element('div', 'assistant-name', 'Bufo');
  if (row.backend) {
    const configured = models.find(model => model.id === row.backend);
    const label = element('span', 'assistant-model', row.model_label || row.model || configured?.label || 'Model pending');
    label.title = row.model_version || row.model || ''; author.append(label);
    if (row.model_version) {
      const checkpoint = element('details', 'model-checkpoint');
      checkpoint.append(element('summary', '', 'Checkpoint'), element('span', 'model-version', row.model_version));
      author.append(checkpoint);
    }
  }
  body.append(author, element('p', 'reply', row.response_text || (row.valid ? 'Here you go—your Bufo is ready!' : 'This drawing needs another pass.')));
  if (row.png_url) {
    const link = element('a', 'drawing-link'); link.href = row.png_url; link.target = '_blank'; link.rel = 'noreferrer'; link.setAttribute('aria-label', 'Open full-size drawing');
    const img = element('img', 'drawing'); img.src = row.png_url; img.alt = `Bufo drawing: ${row.prompt}`;
    img.addEventListener('load', () => { if ($('thread-scroll').scrollHeight - $('thread-scroll').scrollTop - $('thread-scroll').clientHeight < 450) scrollBottom(); });
    link.append(img); body.append(link);
  }
  const actions = element('div', 'actions');
  [['PNG', row.png_url], ['Download SVG', row.svg_url], ['View trace ↗', row.conversation_url]].forEach(([label, url]) => {
    if (!url) return;
    const link = element('a', '', label); link.href = url;
    if (label.startsWith('View')) { link.target = '_blank'; link.rel = 'noreferrer'; } else link.download = '';
    actions.append(link);
  });
  if (actions.childElementCount) body.append(actions);
  if (row.valid && row.emoji_name) {
    const naming = element('div', 'emoji-naming');
    const shortcode = element('code', 'emoji-shortcode', row.emoji_name);
    const copy = element('button', 'copy-emoji', 'Copy name'); copy.type = 'button';
    copy.setAttribute('aria-label', `Copy emoji name ${row.emoji_name}`);
    copy.addEventListener('click', async () => {
      try { await navigator.clipboard.writeText(row.emoji_name); copy.textContent = 'Copied!'; }
      catch (_) { copy.textContent = 'Select name to copy'; }
    });
    naming.append(shortcode, copy); body.append(naming);
  }
  (row.web_searches || []).forEach(search => {
    if (search.status !== 'ok') {
      body.append(element('p', 'web-context-note', 'Web context unavailable; drawn from your request.'));
      return;
    }
    const details = element('details', 'web-context');
    details.append(element('summary', '', `Web context · ${search.query}`));
    details.append(element('p', '', search.summary));
    const sources = element('ul');
    (search.sources || []).forEach(source => {
      if (!/^https?:\/\//i.test(source.url)) return;
      const item = element('li'), link = element('a', '', source.title || source.url);
      link.href = source.url; link.target = '_blank'; link.rel = 'noreferrer';
      item.append(link); sources.append(item);
    });
    details.append(sources); body.append(details);
  });
  if (row.error) body.append(element('p', 'drawing-error', row.error));
  message.append(avatar, body); return message;
}
function userMessage(prompt, image) {
  const message = element('div', 'user-message');
  if (image) { const img = element('img', 'user-image'); img.src = image; img.alt = 'Attached reference image'; message.append(img); }
  message.append(element('p', '', prompt)); return message;
}
function drawThread() {
  const items = rows(); $('thread').replaceChildren();
  const job = pendingJobs.get(active);
  $('welcome').hidden = Boolean(items.length || job);
  items.forEach(row => $('thread').append(userMessage(row.prompt, row.input_image_url), assistantMessage(row)));
  if (job) {
    $('thread').append(userMessage(job.prompt, job.attachment?.data));
    const pending = assistantMessage({...job.preview, response_text: job.preview ? 'Image rendered — finishing the reply…' : `Drawing your next version… ${Math.floor((Date.now() - job.start) / 1000)}s`, backend: job.backend});
    pending.classList.add('pending'); $('thread').append(pending);
  }
  $('conversation-title').textContent = items[0]?.prompt || job?.prompt || 'A little frog. A big idea.';
  $('prompt').placeholder = items.length ? 'Ask for a change, or paste a reference image…' : 'Describe your bufo, or paste an image…';
  $('context-hint').textContent = items.length ? 'Continuing this conversation' : 'A new idea starts here';
  scrollBottom();
}
function openConversation(id) {
  if (loading) return;
  drafts.set(active || 'new', $('prompt').value); imageDrafts.set(active || 'new', attachment);
  imageReadToken++; readingImage = false;
  active = id; remember(); attachment = imageDrafts.get(active || 'new') || null; showAttachment();
  selectModel(pendingJobs.get(active)?.backend || rows().at(-1)?.backend || selectedBackend);
  setBusy(pendingJobs.has(active));
  $('prompt').value = drafts.get(active || 'new') || ''; resizeComposer(); notice(threadErrors.get(active) || ''); drawConversations(); drawThread();
}
$('new-chat').addEventListener('click', () => { openConversation(null); $('prompt').focus(); });
$('conversation-select').addEventListener('change', event => openConversation(event.target.value || null));
$('model-select').addEventListener('change', async event => {
  if (event.target.value !== RETRY_MODELS) { selectModel(event.target.value); return; }
  notice(await loadStatus()); drawThread();
});
$('attach-image').addEventListener('click', () => $('image-file').click());
$('image-file').addEventListener('change', event => { attachImage(event.target.files[0]); event.target.value = ''; });
$('remove-image').addEventListener('click', () => { imageReadToken++; readingImage = false; attachment = null; showAttachment(); });
document.addEventListener('paste', event => {
  if (busy) return;
  const item = [...(event.clipboardData?.items || [])].find(item => item.kind === 'file' && item.type.startsWith('image/'));
  if (item) { event.preventDefault(); attachImage(item.getAsFile()); }
});
$('prompt').addEventListener('input', resizeComposer);
$('prompt').addEventListener('keydown', event => {
  if (event.key === 'Enter' && !event.shiftKey && !event.isComposing) { event.preventDefault(); if (!busy) $('prompt-form').requestSubmit(); }
});
document.querySelectorAll('[data-prompt]').forEach(button => button.addEventListener('click', () => { $('prompt').value = button.dataset.prompt; resizeComposer(); $('prompt').focus(); }));
$('prompt-form').addEventListener('submit', async event => {
  event.preventDefault(); const text = $('prompt').value.trim(); if (busy || readingImage || (!text && !attachment) || !currentModel()?.available) return;
  const prompt = text || 'Create a Bufo scene inspired by this image.';
  const previous = rows().at(-1);
  const key = active || `pending:${crypto.randomUUID()}`;
  const job = {prompt, text, attachment, backend: selectedBackend, start: Date.now()};
  drafts.delete(active || 'new'); imageDrafts.delete(active || 'new');
  active = key; remember(); pendingJobs.set(key, job); threadErrors.delete(key);
  attachment = null; $('prompt').value = ''; showAttachment(); resizeComposer();
  setBusy(true); notice(''); drawConversations(); drawThread();
  try {
    const row = await generateDrawing({prompt, image: job.attachment?.data || null, previous_id: previous?.id || null, backend: job.backend}, preview => {
      job.preview = preview;
      if (active === key) drawThread();
    });
    history.push(row); pendingJobs.delete(key); failedThreads.delete(key);
    // A completed background request must not change the chat being viewed.
    if (active === key) { active = row.group_id; remember(); setBusy(false); drawThread(); }
    drawConversations();
  } catch (error) {
    pendingJobs.delete(key);
    const message = error.message || 'Could not reach the studio. Check that the local server is running.';
    threadErrors.set(key, message); drafts.set(key, text); imageDrafts.set(key, job.attachment);
    // Keep a failed new thread discoverable so its unsent image/text can be retried.
    if (!history.some(row => row.group_id === key)) failedThreads.set(key, prompt);
    if (active === key) { setBusy(false); drawThread(); $('prompt').value = text; attachment = job.attachment; showAttachment(); resizeComposer(); notice(message); }
    drawConversations();
  } finally { if (active === key) $('prompt').focus(); }
});
setInterval(() => {
  const job = pendingJobs.get(active);
  const reply = $('thread').querySelector('.pending .reply');
  if (job && reply && !job.preview) reply.textContent = `Drawing your next version… ${Math.floor((Date.now() - job.start) / 1000)}s`;
}, 1000);

const RETRY_MODELS = '__retry_models__';
async function fetchJson(url) {
  const response = await fetch(url);
  const text = await response.text();
  let body = null; try { body = JSON.parse(text); } catch {}
  if (!response.ok || body === null) throw new Error(url);
  return body;
}
async function loadStatus() {
  try {
    const status = await fetchJson('/api/status');
    if (!Array.isArray(status.models)) throw new Error('status');
    $('weave-link').href = status.agents_url;
    models = demoModels(status.models);
    $('model-select').replaceChildren();
    models.forEach(model => {
      const option = new Option(`${model.label} · ${model.role}`, model.id);
      option.disabled = !model.available; option.title = model.unavailable_reason || model.model_version || model.model;
      $('model-select').add(option);
    });
    let preferred; try { preferred = localStorage.getItem('bufo-model'); } catch (_) { /* Storage is optional. */ }
    selectModel(rows().at(-1)?.backend || preferred || status.default_backend);
    return '';
  } catch (_) {
    models = [];
    // A lone selected option fires no change event when re-picked, so keep a disabled placeholder selected.
    const placeholder = new Option('Models unavailable', ''); placeholder.disabled = true;
    $('model-select').replaceChildren(placeholder, new Option('Retry loading models', RETRY_MODELS));
    $('model-select').value = '';
    $('active-model-label').textContent = 'No model available';
    $('model-status').textContent = 'Could not load models. Refresh to retry.';
    $('status-dot').classList.toggle('ready', false);
    syncSend();
    return 'Could not load models. Refresh to retry.';
  }
}
async function loadHistory() {
  try {
    const body = await fetchJson('/api/history');
    if (!Array.isArray(body)) throw new Error('history');
    history = body;
    let saved; try { saved = localStorage.getItem('bufo-chat'); } catch (_) { /* Storage is optional. */ }
    active = saved === 'new' ? null : (history.some(row => row.group_id === saved) ? saved : history.at(-1)?.group_id || null);
    return '';
  } catch (_) { return 'Could not load past drawings. Refresh to retry.'; }
}
async function init() {
  setBusy(true);
  // History is applied first so the saved conversation can pick the model; each load fails on its own.
  const historyProblem = await loadHistory(), statusProblem = await loadStatus();
  notice([statusProblem, historyProblem].filter(Boolean).join(' '));
  loading = false; setBusy(false);
  drawConversations(); drawThread();
}
init();
