"""Geometry and trust-boundary behavior of the experimental shared compiler."""
import importlib.util
import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

MODULE = Path(__file__).resolve().parents[1] / 'docs/forge-contact-pilot/scene_renderer.py'


def compiler():
    assert MODULE.exists(), 'contact compiler not implemented yet'
    if 'contact_scene_renderer_test' not in sys.modules:
        spec = importlib.util.spec_from_file_location('contact_scene_renderer_test', MODULE)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
    return sys.modules['contact_scene_renderer_test']


def scene(objects=None, hands=None):
    return {'bufo': {'x': 150, 'y': 80, 'scale': 2}, 'objects': objects or [], 'hands': hands or []}


def obj(name='prop', **changes):
    result = {'id': name, 'svg': '<rect x="0" y="0" width="20" height="30" fill="#246"/>',
              'anchors': [{'name': 'grip', 'x': 10, 'y': 15}],
              'transform': {'x': 370, 'y': 250, 'angle': 0, 'scale': 1}}
    result.update(changes)
    return result


def test_rotated_scaled_attachment_keeps_shape_and_meets_foot():
    c = compiler()
    result = c.compile_scene(scene([obj(transform={'x': 0, 'y': 0, 'angle': 90, 'scale': 2},
        attach={'anchor': 'grip', 'to': 'bufo.right_foot', 'dx': 0, 'dy': 0})]))
    assert result['anchors']['prop.grip'] == pytest.approx([342, 428])
    assert result['transforms']['prop'] == pytest.approx({'x': 372, 'y': 408, 'angle': 90, 'scale': 2})
    assert result['contacts'][0]['residual'] < 1e-9
    assert result['verification'] == 'geometry_only'
    assert 'semantic_pass' not in result
    assert 'bufo-character' in result['svg']


def test_surface_tool_hand_chain_resolves_out_of_order():
    c = compiler()
    tool = obj('tool', anchors=[{'name': 'tip', 'x': 0, 'y': 0}, {'name': 'grip', 'x': 0, 'y': -40}],
               attach={'anchor': 'tip', 'to': 'table.contact'})
    table = obj('table', anchors=[{'name': 'contact', 'x': 0, 'y': 0}],
                transform={'x': 420, 'y': 340, 'angle': 0, 'scale': 1})
    result = c.compile_scene(scene([tool, table], [{'side': 'right', 'to': 'tool.grip'}]))
    assert result['anchors']['tool.tip'] == [420, 340]
    assert result['anchors']['tool.grip'] == [420, 300]
    assert result['anchors']['bufo.right_hand'] == [420, 300]
    nodes = {n.get('id'): n for n in ET.fromstring(result['svg']).iter() if n.get('id')}
    assert nodes['rig-arm-right'].get('d').startswith('M366 304 ')
    assert nodes['rig-arm-right'].get('d').endswith('420 300')


def test_two_hands_reference_same_prop_without_moving_it():
    result = compiler().compile_scene(scene([obj(anchors=[{'name': 'a','x': 0,'y': 0},
        {'name': 'b','x': 80,'y': 0}], transform={'x': 220,'y': 340,'angle': 0,'scale': 1})],
        [{'side': 'left','to': 'prop.a'}, {'side': 'right','to': 'prop.b'}]))
    assert result['anchors']['bufo.left_hand'] == [220,340]
    assert result['anchors']['bufo.right_hand'] == [300,340]
    assert result['transforms']['prop']['x'] == 220


def test_body_relative_clothing_and_explicit_foreground_order():
    result = compiler().compile_scene(scene([
        obj('shirt', frame='bufo', layer='clothing', transform={'x':0,'y':90,'angle':0,'scale':1}),
        obj('screen', layer='foreground'), obj()], [{'side':'right','to':'prop.grip'}]))
    assert result['anchors']['shirt.grip'] == [170,290]
    root = ET.fromstring(result['svg'])
    ids = [x.get('id') for x in root if x.get('id')]
    assert ids.index('rig-body') < ids.index('object-shirt') < ids.index('rig-arm-right')
    assert ids.index('rig-arm-right') < ids.index('object-prop') < ids.index('rig-hand-right') < ids.index('object-screen')


@pytest.mark.parametrize('objects,match', [
    ([obj(attach={'anchor':'grip','to':'absent.point'})], 'Unknown'),
    ([obj(attach={'anchor':'absent','to':'bufo.left_foot'})], 'anchor'),
    ([obj('a',attach={'anchor':'grip','to':'b.grip'}), obj('b',attach={'anchor':'grip','to':'a.grip'})], 'cycle'),
    ([obj('bufo')], 'reserved'),
    ([obj(),obj()], 'Duplicate'),
])
def test_rejects_unresolvable_or_ambiguous_graph(objects,match):
    with pytest.raises(ValueError,match=match): compiler().compile_scene(scene(objects))


@pytest.mark.parametrize('svg', [
    '<script>alert(1)</script>', '<rect width="2" height="2" onclick="bad()"/>',
    '<use href="https://example.com/x.svg#x"/>', '<!DOCTYPE svg [<!ENTITY x "oops">]><text>&x;</text>',
    '<?external x?><rect width="2" height="2"/>',
    '<g id="bufo-fullbody"><path d="M0 0"/></g>',
    '<use href="#bufo-character"/>', '<rect width="2" height="2" fill="url(#missing)"/>',
    '<g id="a"><use href="#a"/></g>', '<g transform="translate(NaN 1)"/>',
])
def test_unsafe_or_reserved_fragments_rejected(svg):
    with pytest.raises(ValueError): compiler().compile_scene(scene([obj(svg=svg)]))


def test_fragment_ids_are_namespaced_without_cross_object_references():
    svg = '<defs><linearGradient id="paint"><stop offset="0" stop-color="red"/></linearGradient></defs><rect width="10" height="10" fill="url(#paint)"/>'
    result = compiler().compile_scene(scene([obj('a',svg=svg), obj('b',svg=svg)]))
    root = ET.fromstring(result['svg'])
    ids = [n.get('id') for n in root.iter() if n.get('id')]
    assert len(ids) == len(set(ids))
    assert 'fragment-1-a-paint' in ids and 'fragment-1-b-paint' in ids
    fills = {n.get('fill') for n in root.iter()}
    assert 'url(#fragment-1-a-paint)' in fills and 'url(#fragment-1-b-paint)' in fills


def test_unreachable_hand_rejected_instead_of_stretching():
    with pytest.raises(ValueError,match='reach'):
        compiler().compile_scene(scene([obj(transform={'x':0,'y':0,'angle':0,'scale':1})],
                                        [{'side':'right','to':'prop.grip'}]))


@pytest.mark.parametrize('value', [math.nan, math.inf, -math.inf])
def test_nonfinite_transforms_are_rejected(value):
    with pytest.raises(ValueError):
        compiler().compile_scene(scene([obj(transform={'x':value,'y':0,'angle':0,'scale':1})]))


def test_negative_scale_and_duplicate_hands_are_rejected():
    with pytest.raises(ValueError):
        compiler().compile_scene(scene([obj(transform={'x':0,'y':0,'angle':0,'scale':-1})]))
    with pytest.raises(ValueError,match='hand'):
        compiler().compile_scene(scene([obj()], [{'side':'right','to':'prop.grip'}]*2))


def test_body_crop_and_offcanvas_anchor_are_geometry_warnings():
    data=scene([obj(transform={'x':550,'y':50,'angle':0,'scale':1})]); data['bufo']['x']=400
    result=compiler().compile_scene(data)
    assert any('Bufo' in x for x in result['warnings'])
    assert any('prop.grip' in x for x in result['warnings'])


def test_serving_schema_avoids_regex_but_runtime_still_validates_identifiers():
    import json

    from agents.strict_schema import ensure_strict_json_schema
    c=compiler()
    schema=ensure_strict_json_schema(c.Scene.model_json_schema())
    assert '"pattern"' not in json.dumps(schema)
    with pytest.raises(ValueError): c.compile_scene(scene([obj('bad.name')]))
    with pytest.raises(ValueError): c.compile_scene(scene([obj(attach={'anchor':'grip','to':'bad reference'})]))


def test_nonfinite_svg_numeric_expression_is_rejected():
    with pytest.raises(ValueError):
        compiler().compile_scene(scene([obj(svg='<path d="M1e999 0 L0 1"/>')]))


def test_hyphenated_object_and_fragment_names_cannot_collide():
    def fragment(ident):
        return f'<defs><linearGradient id="{ident}"><stop offset="0" stop-color="red"/></linearGradient></defs><rect width="10" height="10" fill="url(#{ident})"/>'
    out=compiler().compile_scene(scene([obj('a',svg=fragment('b-paint')),obj('a-b',svg=fragment('paint'))]))
    root=ET.fromstring(out['svg'])
    ids=[n.get('id') for n in root.iter() if n.get('id')]
    assert len(ids)==len(set(ids)), 'Object/local ID boundary must be unambiguous'
    groups={n.get('id'):n for n in root if n.get('id')}
    fills=[]
    for name in ['a','a-b']:
        fills.append(next(n.get('fill') for n in groups['object-'+name].iter() if n.tag.endswith('rect')))
    assert fills[0]!=fills[1]


def test_connected_arm_has_distinct_outline_for_visibility_on_body():
    out=compiler().compile_scene(scene([obj()], [{'side':'right','to':'prop.grip'}]))
    root=ET.fromstring(out['svg']); paths=[n for n in root if n.tag.endswith('path')]
    skin=next(n for n in paths if n.get('id')=='rig-arm-right')
    outlines=[n for n in paths if n.get('d')==skin.get('d') and n.get('stroke')!=skin.get('stroke')]
    assert len(outlines)==1
    assert float(outlines[0].get('stroke-width'))>float(skin.get('stroke-width'))


def test_crossed_arm_approaches_bar_from_above_instead_of_hiding_long_segment():
    import re
    data={'bufo':{'x':190,'y':120,'scale':1.8},'objects':[obj('bar',
        svg='<rect width="120" height="18" fill="#667fb6"/>',
        anchors=[{'name':'grip','x':110,'y':9}],
        transform={'x':230,'y':350,'angle':0,'scale':1})],
        'hands':[{'side':'left','to':'bar.grip','bend':'inward'}]}
    out=compiler().compile_scene(data)
    arm=next(n for n in ET.fromstring(out['svg']) if n.get('id')=='rig-arm-left')
    sx,sy,cx,cy,ex,ey=map(float,re.findall(r'-?\d+(?:\.\d+)?',arm.get('d')))
    # At 80% of the arm curve, the wrist approach must still be above the bar's
    # top edge. The old curve disappeared behind the prop much earlier.
    y=.04*sy+.32*cy+.64*ey
    assert y < 350
    assert [ex,ey]==[340,359]
