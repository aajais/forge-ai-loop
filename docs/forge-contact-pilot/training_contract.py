"""Strict contact SFT qualification, isolated from the frozen inference harness.

Activate only in a dedicated synchronous training CLI. The temporary dispatch
validates the real render_scene contract; it never renames or reconstructs tools.
No network operations, dataset imports, or training submissions occur here.
"""
from __future__ import annotations

import asyncio
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import sys
import threading

import jsonschema
from PIL import Image

import bufogen.integrations.forge as forge
import bufogen.curation.trajectories as trajectories
from bufogen.curation.curate import _validate_review
from bufogen.rendering.emoji import emoji_shortcode
from bufogen.settings import ROOT

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scene_renderer import Scene, compile_scene

TEACHER = 'gpt-6.1-sol'
_PATCH_LOCK = threading.Lock()
_ACTIVE = None


def _encoded(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _read(path):
    try:
        return Path(path).read_bytes()
    except (OSError, TypeError) as exc:
        raise ValueError('Required captured artifact is missing or unreadable') from exc


def _json(text):
    def unique(pairs):
        out={}
        for key,value in pairs:
            if key in out:
                raise ValueError('Duplicate JSON key in captured tool data')
            out[key]=value
        return out
    try:
        return json.loads(text, object_pairs_hook=unique,
                          parse_constant=lambda value: (_ for _ in ()).throw(ValueError('Nonfinite JSON')))
    except (ValueError, TypeError, UnicodeDecodeError) as exc:
        raise ValueError('Invalid captured JSON') from exc


class ContactTrainingContract:
    """Bind caller-frozen inference settings to reviewed one-render teacher captures.

    The caller supplies exact settings from the collection receipt. This class
    supports later agent versions without importing one particular agent module.
    Corpus counts and held-out exclusion remain the preparation layer's job.
    """

    def __init__(self, *, instructions, tools, prompt_version, source_hashes,
                 tool_contract='bufo-contact-scene-v1'):
        if not isinstance(instructions,str) or not instructions.strip():
            raise ValueError('Frozen instructions required')
        if not isinstance(prompt_version,str) or not prompt_version.strip():
            raise ValueError('Frozen prompt version required')
        if not isinstance(tool_contract,str) or not re.fullmatch(r'bufo-contact-scene-v\d+',tool_contract):
            raise ValueError('Explicit versioned contact tool contract required')
        if not isinstance(tools,list) or len(tools)!=1:
            raise ValueError('Exactly one frozen render_scene tool required')
        tool=tools[0]
        function=tool.get('function',{})
        if (tool.get('type')!='function' or function.get('name')!='render_scene'
                or function.get('strict') is not True or not isinstance(function.get('parameters'),dict)):
            raise ValueError('Expected exact strict render_scene wire schema')
        schema=function['parameters']
        if (schema.get('type')!='object' or schema.get('additionalProperties') is not False
                or set(schema.get('properties',{}))!={'scene','emoji_name'}
                or set(schema.get('required',[]))!={'scene','emoji_name'}):
            raise ValueError('Frozen render_scene schema must require scene and emoji_name only')
        def local_refs(value):
            if isinstance(value,dict):
                if '$ref' in value and not value['$ref'].startswith('#/'):
                    raise ValueError('External schema references forbidden')
                for item in value.values():local_refs(item)
            elif isinstance(value,list):
                for item in value:local_refs(item)
        local_refs(schema)
        try:jsonschema.Draft202012Validator.check_schema(schema)
        except jsonschema.SchemaError as exc:raise ValueError('Invalid frozen tool schema') from exc
        required={'docs/forge-contact-pilot/scene_renderer.py','src/bufogen/trajectories.py',
                  'src/bufogen/prompting.py','src/bufogen/svg.py','src/bufogen/render.py',
                  'src/bufogen/sandbox.py','scripts/render_worker.py',
                  'assets/bufo_style/fullbody.svg','assets/bufo_style/neutral-example.svg'}
        current_required = {
            name.replace('src/bufogen/trajectories.py', 'src/bufogen/curation/trajectories.py')
                .replace('src/bufogen/prompting.py', 'src/bufogen/agents/prompting.py')
                .replace('src/bufogen/svg.py', 'src/bufogen/rendering/svg.py')
                .replace('src/bufogen/render.py', 'src/bufogen/rendering/renderer.py')
                .replace('src/bufogen/sandbox.py', 'src/bufogen/integrations/sandbox.py')
            for name in required
        }
        if not isinstance(source_hashes,dict) or not (
            required <= set(source_hashes) or current_required <= set(source_hashes)
        ):
            raise ValueError('Frozen compiler, renderer and character dependency hashes required')
        for name,digest in source_hashes.items():
            path=Path(name)
            if (path.is_absolute() or not (ROOT/path).resolve().is_relative_to(ROOT.resolve())
                    or not isinstance(digest,str) or not re.fullmatch('[0-9a-f]{64}',digest)):
                raise ValueError('Invalid frozen source hash entry')
        training_paths=[Path(__file__), ROOT/'src/bufogen/integrations/forge.py',ROOT/'src/bufogen/workflows/forge_loop.py',
                        ROOT/'src/bufogen/curation/curate.py']
        self._spec=_encoded({'instructions':instructions,'tools':tools,'prompt_version':prompt_version,
            'tool_contract':tool_contract,'source_hashes':source_hashes,
            'training_source_hashes':{str(p.resolve().relative_to(ROOT.resolve())):_sha(_read(p)) for p in training_paths}})
        self.contract_sha256=_sha(self._spec)
        self._assert_current()

    @property
    def specification(self):
        return json.loads(self._spec)

    def _assert_current(self):
        spec=self.specification
        for name,digest in {**spec['source_hashes'],**spec['training_source_hashes']}.items():
            if _sha(_read(ROOT/name))!=digest:
                raise ValueError(f'Frozen source changed: {name}')

    def validate_messages(self,messages,tools):
        """Validate a clean actual exchange; returns compiled data for artifact binding."""
        spec=self.specification
        if tools!=spec['tools']:
            raise ValueError('Captured tool schema differs from frozen contact contract')
        if (not isinstance(messages,list) or len(messages)!=5
                or [m.get('role') for m in messages]!=['system','user','assistant','tool','assistant']):
            raise ValueError('Require one complete user/tool/final-reply exchange without repair targets')
        if messages[0]!={'role':'system','content':spec['instructions']}:
            raise ValueError('Captured instructions differ from frozen contract')
        user,action,result_message,final=messages[1:]
        if not isinstance(user.get('content'),str) or not user['content'].strip() or user.get('tool_calls'):
            raise ValueError('Require a single text-only user request')
        calls=action.get('tool_calls')
        if not isinstance(calls,list) or len(calls)!=1 or action.get('content') not in (None,''):
            raise ValueError('Require exactly one authentic render_scene action')
        call=calls[0];function=call.get('function',{})
        if (call.get('type')!='function' or function.get('name')!='render_scene'
                or not isinstance(call.get('id'),str) or not call['id']
                or result_message.get('tool_call_id')!=call['id']):
            raise ValueError('Wrong tool name or unmatched tool call/result')
        arguments=_json(function.get('arguments'))
        try:jsonschema.Draft202012Validator(spec['tools'][0]['function']['parameters']).validate(arguments)
        except jsonschema.ValidationError as exc:raise ValueError('Arguments violate frozen wire schema') from exc
        normalized=Scene.model_validate(arguments['scene']).model_dump(mode='json')
        compiled=compile_scene(normalized)
        if compiled['warnings']:
            raise ValueError('Accepted contact scene has geometry warnings')
        result=_json(result_message.get('content'))
        expected={'valid':True,'error':None,'image_ready':True,
            'emoji_name':emoji_shortcode(arguments['emoji_name'],user['content']),
            **{k:v for k,v in compiled.items() if k!='svg'}}
        if result!=expected:
            raise ValueError('Final successful tool result differs from actual compiled geometry')
        if (result_message.get('tool_calls') or final.get('tool_calls')
                or not isinstance(final.get('content'),str) or not final['content'].strip()
                or '<svg' in final['content'].lower() or final['content'].lstrip().startswith(('{','['))):
            raise ValueError('Require the actual final text reply, not raw SVG/scene JSON')
        return {'scene':normalized,'compiled':compiled,'tool_result':result,'tool_call_id':call['id']}

    @contextmanager
    def activate(self):
        """Temporarily route both imported validators; never run in serving/async code."""
        global _ACTIVE
        try:asyncio.get_running_loop()
        except RuntimeError:pass
        else:raise RuntimeError('Contact training dispatch requires a synchronous isolated process')
        if threading.current_thread() is not threading.main_thread():
            raise RuntimeError('Contact training dispatch requires the main synchronous thread')
        if not _PATCH_LOCK.acquire(blocking=False):
            raise RuntimeError('A contact training dispatch is already active')
        old_t,old_f=trajectories._validate_messages,forge._validate_messages
        try:
            if _ACTIVE is not None or any(getattr(f,'_contact_training_dispatch',False) for f in (old_t,old_f)):
                raise RuntimeError('A contact training dispatch is already active')
            self._assert_current()
            def dispatch(messages,tools):
                self._assert_current()
                return self.validate_messages(messages,tools)
            dispatch._contact_training_dispatch=True
            trajectories._validate_messages=forge._validate_messages=dispatch
            _ACTIVE=self
            try:yield self
            finally:
                trajectories._validate_messages=old_t
                forge._validate_messages=old_f
                _ACTIVE=None
        finally:_PATCH_LOCK.release()

    def export(self,row,review):
        """Qualify local evidence then reuse existing authentic-trajectory export checks."""
        self._assert_current();spec=self.specification
        _validate_review(review,row)
        if (review.get('reviewer_kind')!='assistant' or review.get('decision')!='accept'
                or review.get('visual_quality_pass') is not True or review.get('partition')!='train'):
            raise ValueError('Explicit accepted assistant train review required')
        if (row.model!=TEACHER or row.error or not row.valid or row.composition_warnings
                or row.tool_calls!=1 or row.parent_id or row.input_image_path
                or row.prompt_version!=spec['prompt_version']):
            raise ValueError('Generation is not a clean single-turn teacher target')
        trajectory=row.trajectory or {};provenance=trajectory.get('provenance',{})
        if (provenance.get('tool_contract')!=spec['tool_contract']
                or provenance.get('compiler_verification')!='geometry_only'):
            raise ValueError('Captured contact/compiler provenance mismatch')
        checked=self.validate_messages(trajectory.get('messages'),trajectory.get('tools'))
        if row.prompt!=trajectory['messages'][1]['content']:
            raise ValueError('Captured user request differs from generation prompt')
        if row.emoji_name!=checked['tool_result']['emoji_name']:
            raise ValueError('Emoji name differs from actual tool result')
        scene_bytes=_read(provenance.get('scene_path'))
        scene_hash=_sha(scene_bytes)
        if (scene_hash!=provenance.get('scene_sha256')
                or _json(scene_bytes)!=checked['scene'] or provenance.get('scene')!=checked['scene']):
            raise ValueError('Scene arguments/provenance/saved artifact mismatch')
        svg_bytes=_read(row.svg_path)
        try:svg=svg_bytes.decode()
        except UnicodeDecodeError as exc:raise ValueError('Invalid saved SVG encoding') from exc
        if svg!=row.svg or svg!=checked['compiled']['svg']:
            raise ValueError('Saved SVG differs from captured scene compilation')
        png_bytes=_read(row.png_path);png_hash=_sha(png_bytes)
        if (png_hash!=review.get('png_sha256')
                or png_hash!=(row.benchmark_metadata or {}).get('png_sha256')
                or Path(review['source_ref']).resolve()!=Path(row.png_path).resolve()):
            raise ValueError('Original capture, explicit review, and actual PNG must match')
        try:
            with Image.open(row.png_path) as image:
                if image.format!='PNG' or image.size!=(512,512):
                    raise ValueError('Reviewed image must be the actual 512px PNG')
                image.verify()
        except (OSError,SyntaxError) as exc:raise ValueError('Invalid reviewed PNG') from exc
        if _ACTIVE is self:
            example=trajectories.export_training_example(row)
        else:
            with self.activate():example=trajectories.export_training_example(row)
        receipt={'contract_sha256':self.contract_sha256,'tool_contract':spec['tool_contract'],
            'tool_call_id':checked['tool_call_id'],'scene_sha256':scene_hash,'svg_sha256':_sha(svg_bytes),
            'png_sha256':png_hash,'review_sha256':_sha(_encoded(review)),
            'tool_schema_sha256':_sha(_encoded(spec['tools'])),
            'source_hashes':spec['source_hashes'],'training_source_hashes':spec['training_source_hashes']}
        example.update(partition='train',prompt_family=review['prompt_family'],training_review=deepcopy(review),
                       origin='distillation',metadata={'contact_training_receipt':receipt})
        return example
