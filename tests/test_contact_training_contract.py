"""Synthetic SDK captures exercise training gates; these are never training data."""
import importlib.util
import json
import sys
from copy import deepcopy
from pathlib import Path

import pytest
from agents import function_tool
from PIL import Image

from bufogen.curation.trajectories import capture_trajectory, tool_schemas
from bufogen.generation import BufoGeneration

DIRECTORY = Path(__file__).resolve().parents[1] / 'docs/forge-contact-pilot'
sys.path.insert(0, str(DIRECTORY))
import contact_agent
from scene_renderer import Scene, compile_scene


def adapter():
    path = DIRECTORY / 'training_contract.py'
    assert path.exists(), 'training-only contact adapter not implemented'
    if 'contact_training_contract_test' not in sys.modules:
        spec=importlib.util.spec_from_file_location('contact_training_contract_test',path)
        mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
    return sys.modules['contact_training_contract_test']


def captured(tmp_path):
    import hashlib
    async def render_scene(scene: Scene, emoji_name: str = '') -> str:
        """Compile the complete contact scene, render its PNG, and name the Slack emoji."""
        raise AssertionError('Synthetic test capture; no model or rendering call')
    tools=tool_schemas([function_tool(render_scene)])
    scene=Scene.model_validate({'bufo':{'x':160,'y':100,'scale':2}}).model_dump(mode='json')
    compiled=compile_scene(scene)
    sp=tmp_path/'example.scene.json';sp.write_text(json.dumps(scene,indent=2)+'\n')
    vp=tmp_path/'example.svg';vp.write_text(compiled['svg'])
    pp=tmp_path/'example.png';Image.new('RGBA',(512,512),(0,0,0,0)).save(pp)
    final='Here is your Bufo!';prompt='Draw Bufo standing.'
    result={'valid':True,'error':None,'image_ready':True,'emoji_name':':bufo-standing:',
            **{k:v for k,v in compiled.items() if k!='svg'}}
    items=[{'role':'user','content':prompt},
        {'type':'function_call','name':'render_scene','call_id':'captured-call-1',
         'arguments':json.dumps({'scene':scene,'emoji_name':':bufo-standing:'})},
        {'type':'function_call_output','call_id':'captured-call-1','output':json.dumps(result)},
        {'type':'message','role':'assistant','content':[{'type':'output_text','text':final}]}]
    row=BufoGeneration(id='generation-1',prompt_id='train-1',prompt=prompt,backend='closed_quality_bar',
        model='gpt-6.1-sol',group_id='group-1',svg=compiled['svg'],valid=True,latency_s=1,
        svg_path=str(vp),png_path=str(pp),tool_calls=1,emoji_name=':bufo-standing:',
        response_text=final,prompt_version=contact_agent.prompt_version())
    row.trajectory=capture_trajectory(items=items,system_instructions=contact_agent.instructions(),tools=tools,
        provenance={'model':row.model,'base_model':row.model,'tuning_method':'none',
            'prompt_id':row.prompt_id,'group_id':row.group_id,'prompt_version':row.prompt_version,
            'tool_contract':contact_agent.TOOL_CONTRACT,'scene':scene,'scene_path':str(sp),
            'scene_sha256':hashlib.sha256(sp.read_bytes()).hexdigest(),
            'compiler_verification':'geometry_only'},
        final_output=final,history_authentic=True,expected_input=items[:1])
    row.benchmark_metadata={'png_sha256':hashlib.sha256(pp.read_bytes()).hexdigest()}
    review={'schema_version':1,'source_generation_id':row.id,'prompt_id':row.prompt_id,
        'decision':'accept','reviewer_kind':'assistant','reviewer_id':'test-reviewer',
        'source_ref':str(pp),'evidence':'Synthetic test-only explicit review.',
        'prompt_family':'standing','partition':'train','reviewed_at':'2026-10-20T00:00:00+00:00',
        'png_sha256':hashlib.sha256(pp.read_bytes()).hexdigest(),'visual_quality_pass':True}
    spec={'instructions':contact_agent.instructions(),'tools':tools,
          'prompt_version':row.prompt_version,'source_hashes':contact_agent.source_hashes()}
    return row,review,spec


def test_export_preserves_actual_actions_and_binds_reviewed_artifacts(tmp_path):
    row,review,spec=captured(tmp_path); original=deepcopy(row.trajectory)
    c=adapter().ContactTrainingContract(**spec)
    example=c.export(row,review)
    assert example['messages']==original['messages']
    assert example['tools']==original['tools']
    assert example['messages'][2]['tool_calls'][0]['function']['name']=='render_scene'
    assert example['partition']=='train'
    receipt=example['metadata']['contact_training_receipt']
    assert receipt['tool_call_id']=='captured-call-1'
    assert receipt['png_sha256']==review['png_sha256']
    assert receipt['scene_sha256']==original['provenance']['scene_sha256']
    assert receipt['contract_sha256']==c.contract_sha256
    assert row.trajectory==original and row.training_review is None


def test_caller_frozen_later_agent_version_is_supported_without_import_dispatch(tmp_path):
    row,review,spec=captured(tmp_path)
    spec.update(instructions='Frozen v2 collection instructions.',
                prompt_version='contact-v2-collection',tool_contract='bufo-contact-scene-v2')
    row.prompt_version=spec['prompt_version']
    row.trajectory['system_instructions']=spec['instructions']
    row.trajectory['messages'][0]['content']=spec['instructions']
    row.trajectory['provenance'].update(prompt_version=spec['prompt_version'],
                                       tool_contract=spec['tool_contract'])
    c=adapter().ContactTrainingContract(**spec)
    spec['tools'][0]['function']['description']='Mutation after construction'
    snapshot=c.specification
    snapshot['tools'].clear()
    example=c.export(row,review)
    assert example['prompt_version']=='contact-v2-collection'
    assert example['metadata']['contact_training_receipt']['tool_contract']=='bufo-contact-scene-v2'
    assert example['tools']==row.trajectory['tools']


def test_ambiguous_duplicate_json_arguments_are_rejected(tmp_path):
    row,review,spec=captured(tmp_path);c=adapter().ContactTrainingContract(**spec)
    function=row.trajectory['messages'][2]['tool_calls'][0]['function']
    function['arguments']=function['arguments'][:-1]+', "emoji_name": ":other:"}'
    with pytest.raises(ValueError,match='Invalid captured JSON'):c.export(row,review)


def test_hook_validates_forge_mapping_and_restores_both_imported_aliases(tmp_path):
    import bufogen.curation.trajectories as trajectories
    import bufogen.integrations.forge as forge
    import bufogen.workflows.forge_loop as workflow
    row,review,spec=captured(tmp_path);c=adapter().ContactTrainingContract(**spec)
    old_t,old_f=trajectories._validate_messages,forge._validate_messages
    example=c.export(row,review)
    with c.activate():
        assert trajectories._validate_messages is forge._validate_messages
        mapped=workflow.reviewed_entries([example])
        assert mapped[0]['split']=='train'
        assert mapped[0]['messages'][2]['tool_calls'][0]['function']['name']=='render_scene'
        bad=deepcopy(example);bad['messages'][3]['tool_call_id']='orphan'
        with pytest.raises(ValueError): workflow.reviewed_entries([bad])
    assert trajectories._validate_messages is old_t and forge._validate_messages is old_f
    with pytest.raises(ValueError,match='render_svg'): forge.reviewed_entries([example])
    with pytest.raises(RuntimeError,match='test interruption'):
        with c.activate(): raise RuntimeError('test interruption')
    assert trajectories._validate_messages is old_t and forge._validate_messages is old_f


@pytest.mark.parametrize('kind',[
    'mixed_model','prior_history','diagnostics','wrong_model','wrong_contract','wrong_prompt',
    'orphan_result','second_call','raw_json_reply','failed_result','changed_geometry',
    'changed_arguments','changed_schema','changed_instructions','rejected_review','heldout_review',
    'wrong_png_hash','wrong_review_image','image_input','generation_error',
])
def test_export_rejects_corrupted_or_ineligible_capture(tmp_path,kind):
    row,review,spec=captured(tmp_path);c=adapter().ContactTrainingContract(**spec)
    t=row.trajectory;m=t['messages'];p=t['provenance']
    if kind=='mixed_model':p['source_models'].append('google/gemma-4-26B-A4B-it')
    elif kind=='prior_history':p['prior_conversation']=True
    elif kind=='diagnostics':t['diagnostics']=['incomplete SDK trajectory']
    elif kind=='wrong_model':row.model='google/gemma-4-26B-A4B-it'
    elif kind=='wrong_contract':p['tool_contract']='bufo-svg'
    elif kind=='wrong_prompt':row.prompt='Different request'
    elif kind=='orphan_result':m[3]['tool_call_id']='other'
    elif kind=='second_call':m[2]['tool_calls'].append(deepcopy(m[2]['tool_calls'][0]))
    elif kind=='raw_json_reply':m[-1]['content']='{"scene":{}}';p['final_output']=m[-1]['content']
    elif kind in {'failed_result','changed_geometry'}:
        result=json.loads(m[3]['content'])
        if kind=='failed_result':result['valid']=False
        else:result['anchors']['bufo.left_foot']=[0,0]
        m[3]['content']=json.dumps(result)
    elif kind=='changed_arguments':
        args=json.loads(m[2]['tool_calls'][0]['function']['arguments']);args['scene']['bufo']['x']=170
        m[2]['tool_calls'][0]['function']['arguments']=json.dumps(args)
    elif kind=='changed_schema':t['tools'][0]['function']['description']='Changed'
    elif kind=='changed_instructions':t['system_instructions']='Changed';m[0]['content']='Changed'
    elif kind=='rejected_review':review['decision']='reject';review['visual_quality_pass']=False
    elif kind=='heldout_review':review['partition']='heldout'
    elif kind=='wrong_png_hash':review['png_sha256']='0'*64
    elif kind=='wrong_review_image':review['source_ref']=str(tmp_path/'other.png')
    elif kind=='image_input':row.input_image_path=str(tmp_path/'input.png')
    elif kind=='generation_error':row.error='interrupted final completion'
    with pytest.raises(ValueError):c.export(row,review)


@pytest.mark.parametrize('artifact',['scene','svg','png'])
def test_saved_artifact_corruption_is_rejected(tmp_path,artifact):
    row,review,spec=captured(tmp_path);c=adapter().ContactTrainingContract(**spec)
    path=Path(row.trajectory['provenance']['scene_path'] if artifact=='scene' else getattr(row,artifact+'_path'))
    path.write_bytes(b'changed')
    with pytest.raises(ValueError):c.export(row,review)


def test_replacement_png_and_updated_review_do_not_match_original_capture(tmp_path):
    import hashlib
    row,review,spec=captured(tmp_path);c=adapter().ContactTrainingContract(**spec)
    Image.new('RGBA',(512,512),(255,0,0,255)).save(row.png_path)
    review['png_sha256']=hashlib.sha256(Path(row.png_path).read_bytes()).hexdigest()
    with pytest.raises(ValueError,match='PNG'):c.export(row,review)


def test_missing_original_png_capture_hash_is_rejected(tmp_path):
    row,review,spec=captured(tmp_path);c=adapter().ContactTrainingContract(**spec)
    row.benchmark_metadata={}
    with pytest.raises(ValueError,match='PNG'):c.export(row,review)


def test_source_hash_mismatch_and_nested_hook_fail_closed(tmp_path):
    row,review,spec=captured(tmp_path)
    spec['source_hashes']['docs/forge-contact-pilot/scene_renderer.py']='0'*64
    with pytest.raises(ValueError):adapter().ContactTrainingContract(**spec)
    _,_,spec=captured(tmp_path);c=adapter().ContactTrainingContract(**spec)
    with c.activate():
        with pytest.raises(RuntimeError,match='active'): 
            with c.activate():pass


async def test_hook_refuses_async_serving_context(tmp_path):
    _,_,spec=captured(tmp_path);c=adapter().ContactTrainingContract(**spec)
    with pytest.raises(RuntimeError,match='synchronous'):
        with c.activate():pass
