import sys, os
if os.getenv("DIAG_TRANSFORMERS_PATH"):
    sys.path.insert(0, os.environ["DIAG_TRANSFORMERS_PATH"])
import hashlib, json, traceback
from pathlib import Path
import art
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(os.environ.get("DIAG_TOKENIZER_PATH", "/private/tmp/bufo-gemma4-tokenizer"))
template = tokenizer.chat_template
source = Path('runs/bufo-gemma-sft-rl-pilot-20261015-retry1/captures')
results = []
for path in sorted(source.glob('*.json')):
    capture = json.loads(path.read_text())
    if capture.get('status') != 'scored':
        continue
    trajectory = art.Trajectory(exchanges=capture['exchanges'], reward=capture['reward'])
    for label, selected in [('original', template), ('preserve_content_template_diagnostic', template.replace("strip_thinking(message['content'])", "message['content'] | trim").replace("strip_thinking(item['text'])", "item['text'] | trim"))]:
        row = {'capture': path.name, 'variant': label, 'capture_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
        try:
            value = trajectory.tokenize(multi_history=True, base_model='google/gemma-4-26B-A4B-it', tokenizer=tokenizer, chat_template=selected)
            row['status'] = 'ok'
            row['histories'] = len(value.histories) if hasattr(value, 'histories') else None
        except Exception as error:
            row.update(status='failed', error_type=type(error).__name__, error=str(error))
            tb = error.__traceback__
            while tb:
                if tb.tb_frame.f_code.co_name == '_tokenize_chat_view':
                    loc = tb.tb_frame.f_locals
                    row['failure_local'] = {k: loc.get(k) for k in ['message_index', 'part_index', 'part', 'sampled', 'search_cursor', 'sampled_bounds', 'content_bounds_proven']}
                    row['failing_text_prefix'] = str(loc.get('text', ''))[:90]
                    row['failing_text_length'] = len(str(loc.get('text', '')))
                tb = tb.tb_next
        results.append(row)
        print(json.dumps(row), flush=True)
Path('runs/forge-loop/rl-tokenization-diagnosis/results.json').write_text(json.dumps(results, indent=2)+'\n')
