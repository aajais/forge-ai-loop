"""Publish the complete contact-v2 development evidence; offline validation by default.

This evaluates tool feedback under a shared renderer. It is not an FT gain claim.
Only an explicit --publish imports W&B and starts one durable, non-retriable upload.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import statistics
import uuid

from bufogen.evaluation import benchmark
from bufogen.generation import BufoGeneration
from bufogen.settings import ROOT
from bufogen.settings import project_path

HERE = Path(__file__).resolve().parent
ARMS = ('open_base', 'closed_quality_bar')
MODELS = ('google/gemma-4-26B-A4B-it', 'gpt-6.1-sol')
PURPOSE = 'contact-v2 tool-feedback development evaluation; no fine-tuning gain claim'
LIMITATIONS = [
    'Assistant visual reviews, not human annotations; automated scores are uncalibrated.',
    'Neutral copies hid model IDs and original image paths. Reviewer 1 had prior aggregate v1 context.',
    'Other reviewers had prior contact-compiler involvement; reviewers are not independent of project development.',
    'Development prompts were previously inspected. This is not an untouched confirmation set.',
    'This compares base and teacher with contact-v2 tool feedback. No fine-tuned model is included.',
]


def read(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError) as exc:
        raise ValueError(f'Missing or invalid evidence: {path}') from exc


def jsonl(path):
    try:
        return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    except (OSError, ValueError) as exc:
        raise ValueError(f'Missing or invalid JSONL evidence: {path}') from exc


def sha(path):
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError as exc:
        raise ValueError(f'Missing evidence file: {path}') from exc


def experiment():
    spec = importlib.util.spec_from_file_location('contact_v2_publication_experiment', HERE/'experiment.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_reviews(directory, rows):
    """Require every neutral copied image and every assigned explicit decision."""
    directory = Path(directory).resolve()
    folder = directory/'visual-review'
    protocol_path = folder/'protocol.json'
    expected_protocol = {
        'version': 'neutral-visual-review-v1', 'attempts': 32, 'reviewers': 3,
        'benchmark_sha256': sha(directory/'benchmark.json'),
        'manifest_sha256': sha(directory/'manifest.jsonl'),
        'packet_source_sha256': sha(HERE/'neutral_review_packets.py'),
        'blinding': 'Model identities and original image paths omitted from reviewer inputs.',
    }
    if read(protocol_path) != expected_protocol:
        raise ValueError('Neutral review protocol differs from frozen benchmark/manifest/source')
    actual = {row['id']: row for row in rows}
    key_path = folder/'unblinding-key.json'
    key = read(key_path)
    if (len(rows) != 32 or len(actual) != 32 or not isinstance(key, list) or len(key) != 32
            or any(not isinstance(item, dict) or type(item.get('index')) is not int for item in key)
            or {item['index'] for item in key} != set(range(32))
            or {item.get('id') for item in key} != set(actual)):
        raise ValueError('Neutral key must uniquely cover all 32 attempted generations')
    indexed = {}
    for item in key:
        row = actual[item['id']]
        if any(item.get(field) != row[field] for field in ('id','prompt_id','backend','model')):
            raise ValueError('Neutral key differs from actual generation identity')
        indexed[item['index']] = row
    files = {str(path.relative_to(directory)): path for path in (protocol_path,key_path)}
    reviews = {}
    for reviewer in range(3):
        packet = folder/f'reviewer-{reviewer}'
        input_path, decision_path = packet/'inputs.json', packet/'decisions.jsonl'
        inputs, decisions = read(input_path), jsonl(decision_path)
        expected_indices = {i for i in range(32) if i % 3 == reviewer}
        for values in (inputs, decisions):
            if (not isinstance(values,list) or len(values) != len(expected_indices)
                    or any(not isinstance(item,dict) or type(item.get('index')) is not int for item in values)
                    or {item['index'] for item in values} != expected_indices):
                raise ValueError('Require complete, unique, correctly assigned review inputs and decisions')
        by_index = {item['index']: item for item in inputs}
        for index, item in by_index.items():
            row = indexed[index]
            original = row.get('png_path')
            copied = folder/'images'/f'image-{index+1:02d}.png' if original else None
            image_hash = sha(original) if original else None
            if (item.get('label') != f'Image {index+1:02d}' or item.get('prompt') != row['prompt']
                    or item.get('requirements') != row['benchmark_metadata']['requirements']
                    or item.get('png_path') != (str(copied) if copied else None)
                    or item.get('png_sha256') != image_hash
                    or (original and (image_hash != row['benchmark_metadata'].get('png_sha256')
                                      or image_hash != sha(copied)))):
                raise ValueError('Neutral copy, original capture, image bytes, prompt or checklist mismatch')
            if copied:
                files[str(copied.relative_to(directory))] = copied
        for decision in decisions:
            index = decision['index']; item = by_index[index]
            failed = decision.get('failed_requirements')
            if (decision.get('label') != item['label'] or decision.get('png_sha256') != item['png_sha256']
                    or decision.get('reviewer_kind') != 'assistant' or type(decision.get('passed')) is not bool
                    or not isinstance(decision.get('evidence'),str) or not decision['evidence'].strip()
                    or not isinstance(failed,list) or any(not isinstance(value,str) for value in failed)
                    or len(set(failed)) != len(failed) or not set(failed) <= set(item['requirements'])
                    or (decision['passed'] and (failed or not item['png_path']))
                    or (not item['png_path'] and failed != item['requirements'])):
                raise ValueError('Invalid explicit assistant review, PNG binding or failed requirements')
            reviews[indexed[index]['id']] = decision | {'reviewer_packet': f'reviewer-{reviewer}'}
        notes = packet/'review-notes.md'
        if not notes.is_file() or not notes.read_text().strip():
            raise ValueError('Missing reviewer notes and limitations')
        sheets = sorted(packet.glob('sheet-*.png'))
        expected_sheets = [packet/f'sheet-{i+1:02d}.png' for i in range((len(inputs)+3)//4)]
        if sheets != expected_sheets:
            raise ValueError('Missing or unexpected original review sheets')
        for path in (input_path,decision_path,notes,*sheets):
            sha(path)
            files[str(path.relative_to(directory))] = path
    return reviews, files, {
        'status': 'complete', 'reviewed_attempts': 32, 'required_attempts': 32,
        'reviewer_kind': 'assistant', 'blinding': 'neutral_png_paths_model_identity_hidden',
        'limitations': LIMITATIONS,
    }


def validate_attempts(directory, settings):
    """Validate all original attempts and terminal scores, without filtering failures."""
    config = settings['configuration']
    expected = {(prompt['id'],arm) for prompt in config['prompts'] for arm in ARMS}
    manifest, rows = jsonl(directory/'manifest.jsonl'), jsonl(directory/'scored.jsonl')
    for values in (manifest,rows):
        if (len(values) != 32 or len({row['id'] for row in values}) != 32
                or {(row['prompt_id'],row['backend']) for row in values} != expected):
            raise ValueError('Require all 32 unique attempts, including generation failures')
        benchmark._validate_rows([BufoGeneration(**row) for row in values],settings)
    originals = {row['id']:row for row in manifest}
    files = {}
    for row in rows:
        original = originals.get(row['id'])
        if original is None or {k:v for k,v in original.items() if k != 'scores'} != {
                k:v for k,v in row.items() if k != 'scores'}:
            raise ValueError('Scoring altered original generation evidence')
        metadata = row['benchmark_metadata']
        if metadata.get('status') not in {'completed','generation_error'}:
            raise ValueError('Generation attempt is not terminal')
        if row.get('png_path'):
            if sha(row['png_path']) != metadata.get('png_sha256'):
                raise ValueError('PNG differs from original captured hash')
            files[f'generations/{row["id"]}/image.png'] = Path(row['png_path'])
        if row.get('svg_path'):
            path = Path(row['svg_path'])
            if path.read_text() != row['svg']:
                raise ValueError('Saved SVG differs from generated SVG')
            files[f'generations/{row["id"]}/drawing.svg'] = path
        score = row.get('scores',{}).get('scene_quality',{})
        if (score.get('scorer_configuration') != config['scorer'] or score.get('generation_id') != row['id']
                or score.get('requirements') != metadata['requirements']
                or score.get('status') not in {'ok','judge_error','missing_image','prior_turn_failure'}):
            raise ValueError('Missing or mismatched terminal scene score')
        if score['status'] == 'ok':
            from bufogen.evaluation.scorers.scene_quality import scene_result
            normalized = scene_result(score,score['requirements'],threshold=config['scorer']['threshold'])
            if any(score.get(k) != v for k,v in normalized.items()):
                raise ValueError('Inconsistent score/checklist arithmetic or threshold')
        elif score.get('quality_failure') is False:
            raise ValueError('Unscored failure cannot be an automated quality pass')
        if row.get('trajectory'):
            trajectory = row['trajectory']; provenance = trajectory['provenance']
            if ({t['function']['name'] for t in trajectory['tools']} != {'render_scene'}
                    or provenance.get('prompt_version') != config['prompt_version']
                    or row['prompt_version'] != config['prompt_version']
                    or provenance.get('model') != row['model']):
                raise ValueError('Actual tool, prompt version or source model mismatch')
            if provenance.get('scene_path'):
                path = Path(provenance['scene_path'])
                if sha(path) != provenance.get('scene_sha256') or read(path) != provenance.get('scene'):
                    raise ValueError('Captured scene JSON changed')
                files[f'generations/{row["id"]}/scene.json'] = path
    summary = benchmark._summary([BufoGeneration(**row) for row in rows],settings)
    if summary != read(directory/'summary.json'):
        raise ValueError('Summary differs from all attempted generations')
    publication = read(directory/'weave-publication.json')
    feedback, evaluations = publication.get('feedback',{}), publication.get('evaluations',{})
    # Failed rows can receive a visual-generation call during benchmark._publish;
    # that fallback ID is saved in this publication record, not rewritten into rows.
    call_ids = [item.get('call_id') for item in feedback.values()]
    if (publication.get('errors') or set(feedback) != set(originals) or set(evaluations) != set(ARMS)
            or any(not isinstance(call_id,str) or not call_id.strip() for call_id in call_ids)
            or len(set(call_ids)) != 32
            or any(row.get('weave_call_id') and feedback[row['id']].get('call_id') != row['weave_call_id'] for row in rows)
            or any(item.get('scored_rows') != 16 or not isinstance(item.get('ref'),str)
                   or not item['ref'].startswith('weave:///') for item in evaluations.values())):
        raise ValueError('Native Weave feedback/evaluation coverage incomplete or misattributed')
    return rows, summary, files


def validate(directory):
    directory = Path(directory).resolve()
    runner = experiment()
    prompts, _, configuration, fixtures = runner.prepare(HERE/'prompts.json',list(ARMS))
    if len(prompts) != 16 or {arm['name']:arm['model'] for arm in configuration['arms']} != dict(zip(ARMS,MODELS)):
        raise ValueError('Expected the frozen 16-prompt base/teacher development comparison')
    protocol = read(directory/'experiment-protocol.json')
    expected_hashes = {'fixture-manifest.json': fixtures['manifest_sha256'],
        'fixture-assistant-review.json': fixtures['review_sha256'],
        'fixture-verifier.py': configuration['source_hashes'][str((HERE/'evaluate.py').relative_to(ROOT))]}
    expected_protocol = {'schema_version':1,'configuration':configuration,
        'configuration_hash':benchmark.digest(configuration), 'fixture_evidence':fixtures,
        'evidence_sha256':expected_hashes}
    if protocol != expected_protocol:
        raise ValueError('Frozen experiment protocol, fixture review or source configuration changed')
    files = {}
    for name,digest in expected_hashes.items():
        path = directory/'protocol-evidence'/name
        if sha(path) != digest:
            raise ValueError('Frozen protocol evidence changed')
        files[str(path.relative_to(directory))] = path
    settings = read(directory/'benchmark.json')
    if (settings['configuration'] != configuration or settings['configuration_hash'] != benchmark.digest(configuration)
            or settings['comparison_fingerprint'] != benchmark.comparison_fingerprint(configuration)):
        raise ValueError('Frozen benchmark settings/hash differ')
    rows, summary, generation_files = validate_attempts(directory,settings)
    reviews, review_files, review_status = validate_reviews(directory,rows)
    files.update(generation_files); files.update(review_files)
    for name in ('experiment-protocol.json','benchmark.json','manifest.jsonl','scored.jsonl','summary.json','weave-publication.json'):
        files[name] = directory/name
    for name,digest in configuration['source_hashes'].items():
        path = ROOT/name
        if sha(path) != digest:
            raise ValueError(f'Frozen source changed: {name}')
        files[f'sources/{name}'] = path
    for path in (Path(__file__).resolve(),HERE/'neutral_review_packets.py'):
        files[f'publication-code/{path.name}'] = path
    return {'rows':rows,'reviews':reviews,'review_status':review_status,'files':files,
            'hashes':{name:sha(path) for name,path in files.items()},'prompts':prompts,
            'configuration_hash':settings['configuration_hash'],'summary':summary}


def automatic_passed(row):
    return bool(row['valid'] and row['benchmark_metadata']['status'] == 'completed'
                and row['scores']['scene_quality'].get('quality_failure') is False)


def assistant_passed(row,review):
    return bool(row['valid'] and row['benchmark_metadata']['status'] == 'completed' and review['passed'])


def metrics(data):
    result = []
    for arm,model in zip(ARMS,MODELS,strict=True):
        rows = [row for row in data['rows'] if row['backend'] == arm]
        completed = [row for row in rows if row['benchmark_metadata']['status'] == 'completed']
        latencies = [row['latency_s'] for row in completed]
        automatic = sum(automatic_passed(row) for row in rows)
        reviewed = sum(assistant_passed(row,data['reviews'][row['id']]) for row in rows)
        combined = sum(automatic_passed(row) and assistant_passed(row,data['reviews'][row['id']]) for row in rows)
        result.append({'model_id':model,'attempts':len(rows),'automated_passes':automatic,
            'automated_pass_rate_all_attempts':automatic/len(rows),'assistant_passes':reviewed,
            'assistant_pass_rate_all_attempts':reviewed/len(rows),'combined_passes':combined,
            'combined_pass_rate_all_attempts':combined/len(rows),
            'generation_errors':sum(row['benchmark_metadata']['status'] != 'completed' for row in rows),
            'judge_errors':sum(row['scores']['scene_quality']['status']=='judge_error' for row in rows),
            'completed_attempts':len(completed),'completed_mean_latency_s':statistics.mean(latencies) if latencies else None,
            'completed_median_latency_s':statistics.median(latencies) if latencies else None,
            'total_recorded_latency_s':sum(row['latency_s'] for row in rows),
            'total_recorded_input_tokens':sum(row['input_tokens'] for row in rows),
            'total_recorded_output_tokens':sum(row['output_tokens'] for row in rows)})
    return result


def ensure_hashes(data):
    if {name:sha(path) for name,path in data['files'].items()} != data['hashes']:
        raise ValueError('Validated publication evidence changed before upload')


def tables(data,wandb):
    def image(row):
        return wandb.Image(row['png_path']) if row.get('png_path') else None
    pair_columns = ['prompt_id','prompt','requirements','base_model_id','teacher_model_id']
    fields = ['png','automated_pass','assistant_pass','combined_pass','generation_status','generation_error',
              'requirement_checks','assistant_evidence','trace']
    pair_columns += [f'{prefix}_{field}' for prefix in ('base','teacher') for field in fields]
    pairs = wandb.Table(columns=pair_columns)
    lookup = {(row['prompt_id'],row['backend']):row for row in data['rows']}
    for prompt in data['prompts']:
        values = [prompt['id'],prompt['prompt'],prompt['requirements'],*MODELS]
        for arm in ARMS:
            row = lookup[prompt['id'],arm]; review = data['reviews'][row['id']]
            a,b = automatic_passed(row),assistant_passed(row,review)
            values += [image(row),a,b,a and b,row['benchmark_metadata']['status'],row.get('error'),
                       row['scores']['scene_quality']['prompt_checks'],review['evidence'],row.get('conversation_url')]
        pairs.add_data(*values)
    attempts = wandb.Table(columns=['model_id','prompt_id','generation_id','png','generation_status','generation_error',
        'score_status','automated_pass','assistant_visual_pass','assistant_pass','combined_pass','requirement_checks',
        'assistant_evidence','assistant_failed_requirements','blind_label','reviewer_packet','recorded_input_tokens',
        'recorded_output_tokens','recorded_latency_s','tool_calls','prompt_version','trace'])
    for row in data['rows']:
        review = data['reviews'][row['id']]; score = row['scores']['scene_quality']
        a,b = automatic_passed(row),assistant_passed(row,review)
        attempts.add_data(row['model'],row['prompt_id'],row['id'],image(row),row['benchmark_metadata']['status'],
            row.get('error'),score['status'],a,review['passed'],b,a and b,score['prompt_checks'],review['evidence'],
            review['failed_requirements'],review['label'],review['reviewer_packet'],row['input_tokens'],row['output_tokens'],
            row['latency_s'],row['tool_calls'],row['prompt_version'],row.get('conversation_url'))
    summaries = metrics(data)
    summary_table = wandb.Table(columns=list(summaries[0]))
    for row in summaries:
        summary_table.add_data(*row.values())
    return {'paired_images':pairs,'all_32_attempts':attempts,'model_summary':summary_table}


def publish(directory,execute=False):
    directory = Path(directory).resolve()
    if not directory.is_dir():
        raise ValueError('Missing completed experiment directory')
    receipt_path = directory/'visualization.json'
    with benchmark._locked(directory):
        data = validate(directory)
        ensure_hashes(data)
        summary = metrics(data)
        if not execute:
            return {'offline_validation':'passed','network_calls':0,'attempts':32,'paired_rows':16,
                    'models':list(MODELS),'purpose':PURPOSE,'fine_tuning_gain_claim':False,
                    'assistant_review':data['review_status'],'model_summary':summary}
        if receipt_path.exists():
            prior = read(receipt_path)
            if prior.get('evidence_sha256') != data['hashes']:
                raise ValueError('Prior publication evidence differs; no duplicate upload attempted')
            if prior.get('status') == 'published':
                return prior
            raise RuntimeError('Publication already started; reconcile its saved run ID before retrying')
        receipt = {'schema_version':1,'status':'publishing','run_id':uuid.uuid4().hex[:8],
            'purpose':PURPOSE,'attempts':32,'paired_rows':16,'models':list(MODELS),
            'configuration_hash':data['configuration_hash'],'fine_tuning_gain_claim':False,'promotion':False,
            'evidence_sha256':data['hashes'],'assistant_review':data['review_status'],
            'model_summary':summary,'limitations':LIMITATIONS,
            'accounting_note':'All attempts count. Recorded usage/latency may omit failed remote requests; completed latency is separately reported.'}
        with receipt_path.open('x') as stream:
            json.dump(receipt,stream,indent=2,allow_nan=False); stream.flush(); os.fsync(stream.fileno())
    import wandb
    visualization = tables(data,wandb)
    ensure_hashes(data)
    entity,project = project_path().split('/')
    # Plain JSON only: do not insert SDK objects or SDK-managed config proxies into receipts.
    config = json.loads(json.dumps(receipt,allow_nan=False))
    with wandb.init(entity=entity,project=project,id=receipt['run_id'],resume='never',
                    name=f'Contact-v2 tool feedback: {MODELS[0]} / {MODELS[1]}',
                    job_type='tool-feedback-evaluation',config=config) as run:
        run.log(visualization)
        for row in summary:
            for key,value in row.items():
                if key != 'model_id':run.summary[f'{row["model_id"]}/{key}'] = value
        artifact = wandb.Artifact('bufo-contact-v2-development',type='evaluation')
        for name,path in sorted(data['files'].items()):
            artifact.add_file(str(path),name=name)
        saved = run.log_artifact(artifact); saved.wait()
        run_url,artifact_ref = str(run.url),f'{entity}/{project}/{saved.name}'
    ensure_hashes(data)
    receipt.update(status='published',run_url=run_url,artifact_ref=artifact_ref)
    with benchmark._locked(directory):
        current = read(receipt_path)
        if current['run_id'] != receipt['run_id'] or current['status'] != 'publishing':
            raise RuntimeError('Publication receipt changed; reconcile saved run before any retry')
        benchmark._write_json(receipt_path,receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,default=ROOT/'runs/forge-contact-pilot/v2-development')
    parser.add_argument('--publish',action='store_true')
    args = parser.parse_args()
    print(json.dumps(publish(args.directory,execute=args.publish),indent=2,allow_nan=False))


if __name__ == '__main__':
    main()
