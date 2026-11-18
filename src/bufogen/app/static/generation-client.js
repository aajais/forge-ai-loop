// A rendered PNG can be ready before the model finishes its final reply.
// Poll the same request ID so a delayed POST response cannot hide a saved result.
function generateDrawing(payload, onPreview = () => {}) {
  const requestId = crypto.randomUUID().replaceAll('-', '');
  return new Promise((resolve, reject) => {
    let settled = false, polling = false, previewId = null;
    const finish = (error, row) => {
      if (settled) return;
      settled = true; clearInterval(timer);
      if (error) reject(error); else resolve(row);
    };
    const poll = async () => {
      if (settled || polling) return;
      polling = true;
      try {
        const response = await fetch(`/api/generations/${requestId}`, {
          cache: 'no-store',
          ...(typeof AbortSignal !== 'undefined' ? {signal: AbortSignal.timeout(10000)} : {}),
        });
        if (!response.ok || settled) return;
        const status = JSON.parse(await response.text());
        if (settled) return;
        if (status.status === 'complete') finish(null, status.row);
        else if (status.status === 'failed') finish(new Error(status.error || 'Generation failed.'));
        else if (status.status === 'rendered' && status.row?.id !== previewId) {
          previewId = status.row.id; onPreview(status.row);
        }
      } catch (_) { /* A transient status failure must not cancel inference. */ }
      finally { polling = false; }
    };
    const timer = setInterval(poll, 2000);
    fetch('/api/generate', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({...payload, request_id: requestId})})
      .then(async response => {
        let row; try { row = JSON.parse(await response.text()); } catch (_) {}
        if (!response.ok) throw new Error(typeof row?.detail === 'string' ? row.detail : 'Could not make this drawing. Please try again.');
        if (!row || typeof row !== 'object' || Array.isArray(row)) throw new Error('The studio returned an unreadable reply. Please try again.');
        finish(null, row);
      }).catch(error => finish(error));
  });
}
