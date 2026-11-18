const $ = id => document.getElementById(id);
const STORAGE_KEY = 'bufo-comparisons-v1';
let models = [], comparisons = [], current = null, running = false;
const timers = new Map();
function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text) node.textContent = text;
  return node;
}
function notice(text) { $('comparison-notice').textContent = text; $('comparison-notice').hidden = !text; }
function save() {
  try { localStorage.setItem(STORAGE_KEY, JSON.stringify(comparisons.slice(0, 12))); }
  catch (_) { notice('Browser storage is full or unavailable. Keep this page open; completed drawings are also saved in chat.'); }
}
function syncControls() {
  running = comparisons.some(item => item.slots.some(slot => slot.status === 'running'));
  const available = models.filter(model => model.available).length;
  $('run-models').disabled = running || !available;
  $('compare-prompt').disabled = running;
  $('saved-comparisons').disabled = false;
  $('reload-models').disabled = running;
  $('run-models').textContent = running ? 'Drawing…' : available && available < 4 ? `Run ${available} available models` : 'Run all 4 models';
  $('compare-form').setAttribute('aria-busy', String(running));
}
function drawSummary() {
  if (!current) return;
  const finished = current.slots.filter(slot => slot.status !== 'running').length;
  const succeeded = current.slots.filter(slot => slot.status === 'complete' && slot.row?.valid).length;
  $('comparison-status').textContent = finished < 4 ? `${finished} of 4 finished` : `Finished · ${succeeded} of 4 drawings rendered`;
}
function canRetry(slot) {
  return slot.status === 'failed' || slot.status === 'interrupted' || (slot.status === 'complete' && !slot.row?.valid);
}
async function retrySlot(comparison, slot) {
  if (!comparison || !canRetry(slot) || !models.some(model => model.id === slot.model.id && model.available)) return;
  slot.attempts = [...(slot.attempts || []), {
    id: slot.row?.id, conversation_url: slot.row?.conversation_url,
    error: slot.error || slot.row?.error, status: slot.status,
  }];
  slot.status = 'running'; slot.start = Date.now(); slot.error = ''; slot.row = null; slot.preview = false;
  save(); syncControls(); drawHistory();
  if (current === comparison) { drawResults(); drawSummary(); }
  await runSlot(comparison, slot);
}
function drawHistory() {
  $('saved-comparisons').replaceChildren(new Option('Choose a saved comparison', ''));
  comparisons.forEach(item => $('saved-comparisons').add(new Option(item.prompt + (item.slots.some(slot => slot.status === 'running') ? ' · Drawing…' : ''), item.id)));
  $('saved-comparisons').value = current?.id || '';
}
function addLink(parent, label, url) {
  // Only server file URLs and web trace links are valid output destinations.
  if (!url || !(url.startsWith('/api/files/') || url.startsWith('/api/generations/') || /^https?:\/\//i.test(url))) return;
  const link = element('a', '', label); link.href = url; link.target = '_blank'; link.rel = 'noreferrer';
  parent.append(link);
}
function drawSlot(slot, node) {
  node.replaceChildren(); node.dataset.backend = slot.model.id;
  node.append(element('p', 'comparison-role', slot.model.role), element('h2', '', slot.row?.model_label || slot.model.label));
  const version = slot.row?.model_version || slot.model.model_version || slot.row?.model || slot.model.model;
  const details = element('details', 'model-checkpoint');
  details.append(element('summary', '', 'Model / checkpoint'), element('span', 'model-version', version)); node.append(details);
  const state = element('p', 'execution-state'); node.append(state);
  if (slot.attempts?.length) {
    const attempts = element('div', 'actions');
    attempts.append(element('span', '', `Attempt ${slot.attempts.length + 1}`));
    slot.attempts.forEach((attempt, index) => addLink(attempts, `View attempt ${index + 1} trace ↗`, attempt.conversation_url));
    node.append(attempts);
  }
  if (slot.status === 'running' && !slot.preview) {
    state.textContent = `Drawing… ${Math.floor((Date.now() - slot.start) / 1000)}s`;
    timers.set(slot.model.id, {node: state, start: slot.start});
    return;
  }
  if (slot.status === 'running' && slot.preview) { state.textContent = 'Image rendered — finishing the reply…'; }
  else if (slot.status === 'waiting') { state.textContent = slot.model.available ? 'Ready' : 'Unavailable'; }
  else if (slot.status === 'complete') { state.textContent = slot.row.valid ? 'Completed' : 'Drawing needs another pass'; }
  else { state.textContent = slot.status === 'interrupted' ? 'Interrupted' : 'Failed'; }
  if (slot.error) node.append(element('p', 'drawing-error', slot.error));
  if (canRetry(slot)) {
    if (!slot.row?.valid && slot.row?.truncated_calls > 0 && slot.row?.tool_calls === 0) {
      node.append(element('p', 'drawing-error', 'The model reached its output limit before drawing. You can retry this model with the same prompt.'));
    }
    const retry = element('button', '', 'Retry this model'); retry.type = 'button';
    retry.disabled = !models.some(model => model.id === slot.model.id && model.available);
    const comparison = current;
    retry.addEventListener('click', () => retrySlot(comparison, slot));
    node.append(retry);
  }
  if (!slot.row) return;
  const row = slot.row;
  if (Number.isFinite(row.latency_s)) state.textContent += ` · ${row.latency_s.toFixed(1)}s`;
  if ((row.png_url?.startsWith('/api/files/') || row.png_url?.startsWith('/api/generations/'))) {
    const img = element('img', 'drawing'); img.src = row.png_url; img.alt = `${slot.model.label}: ${current.prompt}`;
    const link = element('a', 'drawing-link'); link.href = row.png_url; link.target = '_blank'; link.rel = 'noreferrer';
    link.append(img); node.append(link);
  }
  if (row.response_text) node.append(element('p', 'reply', row.response_text));
  if (row.error) node.append(element('p', 'drawing-error', row.error));
  const actions = element('div', 'actions');
  addLink(actions, 'Open PNG', row.png_url); addLink(actions, 'Download SVG', row.svg_url); addLink(actions, 'View trace ↗', row.conversation_url);
  node.append(actions);
}
function drawResults() {
  timers.clear(); $('comparison-results').replaceChildren();
  const slots = current?.slots || models.map(model => ({model, status: 'waiting', error: model.available ? '' : model.unavailable_reason}));
  for (const slot of slots) {
    const node = element('article', 'comparison-result');
    drawSlot(slot, node); $('comparison-results').append(node);
  }
}
async function loadModels() {
  try {
    const response = await fetch('/api/status');
    const body = JSON.parse(await response.text());
    if (!response.ok || !Array.isArray(body.models)) throw new Error('status');
    const configured = demoModels(body.models);
    models = DEMO_MODELS.map(choice => configured.find(model => model.id === choice.id) || {...choice, label: choice.id, available: false, unavailable_reason: 'Model is not configured in this studio.'});
    $('reload-models').hidden = true; notice('');
    $('comparison-status').textContent = `${models.filter(model => model.available).length} of 4 models ready`;
  } catch (_) {
    models = []; notice('Could not load models. Retry when the studio is available.');
    $('reload-models').hidden = false; $('comparison-status').textContent = 'Models unavailable';
  }
  syncControls(); drawResults();
}
async function runSlot(comparison, slot) {
  try {
    const row = await generateDrawing({prompt: comparison.prompt, backend: slot.model.id, previous_id: null}, preview => {
      slot.row = preview; slot.preview = true;
      if (current === comparison) {
        timers.delete(slot.model.id);
        drawSlot(slot, $('comparison-results').children[comparison.slots.indexOf(slot)]);
      }
    });
    if (!row || typeof row !== 'object' || Array.isArray(row) || !row.id || !row.group_id || row.backend !== slot.model.id) throw new Error('The studio returned an unreadable or mismatched reply. Please try again.');
    slot.row = row; slot.preview = false; slot.status = 'complete';
  } catch (error) {
    slot.status = 'failed'; slot.error = error.message || 'Could not reach the studio.';
  }
  save(); drawHistory(); syncControls();
  if (current === comparison) {
    timers.delete(slot.model.id);
    drawSlot(slot, $('comparison-results').children[comparison.slots.indexOf(slot)]);
    drawSummary();
  }
}
$('compare-form').addEventListener('submit', async event => {
  event.preventDefault(); const prompt = $('compare-prompt').value.trim();
  if (running || !prompt || !models.some(model => model.available)) return;
  running = true; notice('');
  current = {id: crypto.randomUUID(), prompt, slots: models.map(model => ({model, start: Date.now(), status: model.available ? 'running' : 'failed', error: model.available ? '' : model.unavailable_reason}))};
  comparisons.unshift(current); comparisons = comparisons.slice(0, 12);
  syncControls(); drawHistory(); drawResults(); save();
  $('comparison-status').textContent = 'Starting models…';
  const comparison = current;
  await Promise.all(comparison.slots.map((slot, index) => slot.status === 'running' ? runSlot(comparison, slot) : Promise.resolve()));
  syncControls();
  if (current === comparison) drawSummary();
  drawHistory();
});
$('saved-comparisons').addEventListener('change', event => {
  current = comparisons.find(item => item.id === event.target.value) || null;
  if (current) $('compare-prompt').value = current.prompt;
  drawResults(); $('comparison-status').textContent = current ? 'Saved comparison' : `${models.filter(model => model.available).length} of 4 models ready`;
});
$('reload-models').addEventListener('click', loadModels);
setInterval(() => { for (const timer of timers.values()) timer.node.textContent = `Drawing… ${Math.floor((Date.now() - timer.start) / 1000)}s`; }, 1000);
async function init() {
  try {
    const saved = JSON.parse(localStorage.getItem(STORAGE_KEY) || '[]');
    if (Array.isArray(saved)) comparisons = saved.filter(item => typeof item.prompt === 'string' && Array.isArray(item.slots) && item.slots.length === 4 && item.slots.every(slot => slot?.model?.id && slot.status)).slice(0, 12);
    comparisons.forEach(item => item.slots.forEach(slot => {
      if (slot.status === 'running') { slot.status = 'interrupted'; slot.error = 'Page reloaded before this result arrived. Check saved conversations in chat; the server may still be finishing it. This request was not resubmitted.'; }
    }));
    current = comparisons[0] || null;
    if (current) $('compare-prompt').value = current.prompt;
  } catch (_) { comparisons = []; }
  drawHistory(); drawResults(); await loadModels();
}
init();
