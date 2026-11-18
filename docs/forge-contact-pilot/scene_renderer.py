"""Experimental prompt-blind SVG compiler: explicit contacts, no scene presets."""
from __future__ import annotations

import math
import re
import xml.etree.ElementTree as ET
from typing import Annotated, Literal

from pydantic import AfterValidator, BaseModel, ConfigDict, Field

from bufogen.agents.prompting import assemble_character
from bufogen.rendering.svg import validate_svg

VERSION = 'bufo-contact-v1'
NS = 'http://www.w3.org/2000/svg'
ET.register_namespace('', NS)
Number = Annotated[float, Field(ge=-4096, le=4096, allow_inf_nan=False)]


def _name(value):
    if not re.fullmatch(r'[A-Za-z][A-Za-z0-9_-]{0,31}', value):
        raise ValueError('Names must begin with a letter and contain only letters, digits, _ or -')
    return value


def _reference(value):
    parts = value.split('.')
    if len(parts) != 2:
        raise ValueError('Anchor reference must be object.anchor')
    for part in parts:
        _name(part)
    return value


# Runtime validators retain strict identifiers without unsupported serving regexes.
Name = Annotated[str, Field(min_length=1, max_length=32), AfterValidator(_name)]
Reference = Annotated[str, Field(min_length=3, max_length=65), AfterValidator(_reference)]
SCALE = Annotated[float, Field(gt=0, le=8, allow_inf_nan=False)]


class Input(BaseModel):
    model_config = ConfigDict(extra='forbid')


class Placement(Input):
    x: Number
    y: Number
    scale: SCALE = 1


class Transform(Input):
    x: Number = 0
    y: Number = 0
    angle: Annotated[float, Field(ge=-360, le=360, allow_inf_nan=False)] = 0
    scale: SCALE = 1


class Anchor(Input):
    name: Name
    x: Number
    y: Number


class Attachment(Input):
    anchor: Name
    to: Reference
    dx: Number = 0
    dy: Number = 0


class SceneObject(Input):
    id: Name
    svg: Annotated[str, Field(min_length=1, max_length=16000)]
    anchors: Annotated[list[Anchor], Field(max_length=16)] = Field(default_factory=list)
    transform: Transform = Field(default_factory=Transform)
    frame: Literal['canvas', 'bufo'] = 'canvas'
    layer: Literal['background', 'clothing', 'prop', 'foreground'] = 'prop'
    attach: Attachment | None = None


class Hand(Input):
    side: Literal['left', 'right']
    to: Reference
    bend: Literal['outward', 'inward'] = 'outward'
    angle: Annotated[float, Field(ge=-360, le=360, allow_inf_nan=False)] = 0
    width: Annotated[float, Field(ge=4, le=40, allow_inf_nan=False)] = 14


class Scene(Input):
    bufo: Placement
    objects: Annotated[list[SceneObject], Field(max_length=24)] = Field(default_factory=list)
    hands: Annotated[list[Hand], Field(max_length=2)] = Field(default_factory=list)


def _n(x):
    return f'{x:.8g}'


def _element(tag, **attributes):
    return ET.Element(f'{{{NS}}}{tag}', {k.replace('_', '-'): str(v) for k, v in attributes.items()})


def _rotate(x, y, angle, scale):
    c, s = math.cos(math.radians(angle)), math.sin(math.radians(angle))
    return (scale * (x * c - y * s), scale * (x * s + y * c))


def _point(t, x, y):
    a, b = _rotate(x, y, t['angle'], t['scale'])
    return [t['x'] + a, t['y'] + b]


def _fragment(obj: SceneObject):
    # Reject declarations before parsing. The shared sanitizer remains authoritative.
    if re.search(r'<[!?]', obj.svg):
        raise ValueError(f'{obj.id}: XML declarations, comments and processing instructions are forbidden in fragments')
    wrapped = f'<svg xmlns="{NS}" xmlns:xlink="http://www.w3.org/1999/xlink">{obj.svg}</svg>'
    valid, error = validate_svg(wrapped)
    if not valid:
        raise ValueError(f'{obj.id}: {error}')
    root = ET.fromstring(wrapped)
    nodes = list(root.iter())
    if len(nodes) > 300:
        raise ValueError(f'{obj.id}: too many fragment elements')
    ids = {}
    for node in nodes:
        ident = node.get('id')
        if ident:
            if ident.startswith(('bufo-', 'rig-', 'fragment-', 'object-')):
                raise ValueError(f'{obj.id}: reserved ID {ident}')
            if ident in ids:
                raise ValueError(f'{obj.id}: duplicate fragment ID {ident}')
            ids[ident] = node
        if node is not root and node.tag.split('}')[-1] == 'svg':
            raise ValueError(f'{obj.id}: nested svg is unsupported; supply a fragment')
        for key, value in node.attrib.items():
            if key.split('}')[-1] not in {'id','fill','stroke','font-family'} and re.search(
                r'(?<![A-Za-z])(?:nan|[+-]?inf(?:inity)?)(?![A-Za-z])', value, re.I
            ):
                raise ValueError(f'{obj.id}: nonfinite fragment value')
            if key.split('}')[-1] in {
                'd', 'points', 'transform', 'x', 'y', 'x1', 'y1', 'x2', 'y2',
                'cx', 'cy', 'r', 'rx', 'ry', 'width', 'height', 'stroke-width',
            }:
                numbers = re.findall(r'[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?', value)
                if any(not math.isfinite(float(n)) or abs(float(n)) > 100000 for n in numbers):
                    raise ValueError(f'{obj.id}: fragment numeric value exceeds bounded range')
    links = {}
    for node in nodes:
        refs = []
        for key, value in list(node.attrib.items()):
            ref = value[1:] if key.split('}')[-1] == 'href' else None
            match = re.fullmatch(r'url\(#([A-Za-z_][\w.-]*)\)', value)
            if match:
                ref = match[1]
            if ref is not None:
                if ref not in ids:
                    raise ValueError(f'{obj.id}: unknown or reserved fragment reference {ref}')
                refs.append(ids[ref])
                node.set(key, ('#' if not match else 'url(#') + f'fragment-{len(obj.id)}-{obj.id}-{ref}' + (')' if match else ''))
        links[node] = list(node) + refs
    # Bounded expansion catches cyclic use/mask/gradient references and reuse bombs.
    def visit(node, stack, budget):
        if node in stack:
            raise ValueError(f'{obj.id}: cyclic fragment reference')
        if len(stack) > 32 or budget[0] > 4000:
            raise ValueError(f'{obj.id}: fragment expansion limit exceeded')
        budget[0] += 1
        for child in links[node]:
            visit(child, stack | {node}, budget)
    visit(root, set(), [0])
    for ident, node in ids.items():
        node.set('id', f'fragment-{len(obj.id)}-{obj.id}-{ident}')
    return list(root)


def _arm(hand, start, end, scale):
    dx, dy = end[0] - start[0], end[1] - start[1]
    distance = math.hypot(dx, dy)
    if distance > 90 * scale or distance < 3 * scale:
        raise ValueError(f'{hand.side} hand target outside reachable range (3–90 Bufo-local units)')
    outward = -1 if hand.side == 'left' else 1
    if hand.bend == 'inward':
        outward *= -1
    # A single bounded quadratic control point keeps the limb connected.
    cx = (start[0] + end[0]) / 2 + outward * min(16 * scale, distance * .25)
    bow = min(12 * scale, distance * .15)
    # Inward/crossed arms approach grips from above, rather than running a long
    # distal segment behind a foreground prop and appearing detached.
    cy = (start[1] + end[1]) / 2 + (bow if hand.bend == 'outward' else -bow)
    d = f'M{_n(start[0])} {_n(start[1])} Q{_n(cx)} {_n(cy)} {_n(end[0])} {_n(end[1])}'
    arm = _element('path', id=f'rig-arm-{hand.side}', d=d, fill='none',
        stroke='#577143', stroke_width=_n(10 * scale), stroke_linecap='round')
    outline = _element('path', id=f'rig-arm-outline-{hand.side}', d=d, fill='none',
        stroke='#303f21', stroke_width=_n(12 * scale), stroke_linecap='round')
    fingers = _element('g', id=f'rig-hand-{hand.side}',
        transform=f'translate({_n(end[0])} {_n(end[1])}) rotate({_n(hand.angle)})')
    # The grip axis is local x; three visible curved fingers straddle the anchor.
    width = hand.width
    for offset in (-3, 0, 3):
        y = offset * scale * .65
        fingers.append(_element('path', d=f'M{_n(-width/2)} {_n(y)} Q0 {_n(y+3*scale)} {_n(width/2)} {_n(y)}',
            fill='none', stroke='#303f21', stroke_width=_n(3.4*scale), stroke_linecap='round'))
        fingers.append(_element('path', d=f'M{_n(-width/2)} {_n(y)} Q0 {_n(y+3*scale)} {_n(width/2)} {_n(y)}',
            fill='none', stroke='#73895a', stroke_width=_n(2.2*scale), stroke_linecap='round'))
    return [outline, arm], fingers


def compile_scene(scene: Scene | dict) -> dict:
    """Compile real geometry. Raises ValueError; never grades prompt/visual quality.

    Angles are degrees, positive visually clockwise. Hand width is canvas pixels.
    Object transforms/anchors in the result are canvas-space. Body-local frames
    inherit Bufo translation/scale. Attachment offsets are canvas pixels.
    """
    scene = Scene.model_validate(scene)
    objects = {}
    fragments = {}
    if sum(len(o.svg.encode()) for o in scene.objects) > 60000:
        raise ValueError('Total fragment size exceeds 60KB')
    for obj in scene.objects:
        if obj.id == 'bufo' or obj.id.startswith(('rig-', 'fragment-', 'object-')):
            raise ValueError(f'Object id {obj.id} is reserved')
        if obj.id in objects:
            raise ValueError(f'Duplicate object id {obj.id}')
        names = [a.name for a in obj.anchors]
        if len(names) != len(set(names)):
            raise ValueError(f'{obj.id}: duplicate anchor name')
        objects[obj.id] = obj
        fragments[obj.id] = _fragment(obj)
    b = scene.bufo
    body_t = {'x':b.x,'y':b.y,'angle':0,'scale':b.scale}
    anchors = {f'bufo.{name}': _point(body_t,*point) for name,point in {
        'left_shoulder':(8,112), 'right_shoulder':(108,112),
        'left_foot':(20,174), 'right_foot':(96,174), 'origin':(0,0)}.items()}
    transforms = {}
    contacts = []
    def resolve(ident, stack):
        if ident in transforms or ident == 'bufo':
            return
        if ident in stack:
            raise ValueError(f'Attachment cycle involving {ident}')
        if ident not in objects:
            raise ValueError(f'Unknown object {ident}')
        obj = objects[ident]
        t = obj.transform.model_dump()
        if obj.frame == 'bufo':
            t['x'], t['y'] = _point(body_t,t['x'],t['y'])
            t['scale'] *= b.scale
        if obj.attach:
            a = obj.attach
            resolve(a.to.split('.')[0], stack | {ident})
            if a.to not in anchors:
                raise ValueError(f'Unknown anchor {a.to}')
            local = next((p for p in obj.anchors if p.name == a.anchor),None)
            if local is None:
                raise ValueError(f'{ident}: missing attachment anchor {a.anchor}')
            rx, ry = _rotate(local.x,local.y,t['angle'],t['scale'])
            t['x'] = anchors[a.to][0] + a.dx - rx
            t['y'] = anchors[a.to][1] + a.dy - ry
        transforms[ident] = t
        for a in obj.anchors:
            anchors[f'{ident}.{a.name}'] = _point(t,a.x,a.y)
        if obj.attach:
            a = obj.attach
            p,q = anchors[f'{ident}.{a.anchor}'],anchors[a.to]
            contacts.append({'source':f'{ident}.{a.anchor}','target':a.to,'offset':[a.dx,a.dy],
                             'residual':math.hypot(p[0]-q[0]-a.dx,p[1]-q[1]-a.dy)})
    for ident in objects:
        resolve(ident,set())
    root = _element('svg', viewBox='0 0 512 512', width='512', height='512')
    groups = {k:[] for k in ('background','clothing','prop','foreground')}
    for ident,obj in objects.items():
        t=transforms[ident]
        group=_element('g', id=f'object-{ident}',
            transform=f'translate({_n(t["x"])} {_n(t["y"])}) rotate({_n(t["angle"])}) scale({_n(t["scale"])})')
        group.extend(fragments[ident]); groups[obj.layer].append(group)
    root.extend(groups['background'])
    root.append(_element('use',id='rig-body',href='#bufo-fullbody',
        transform=f'translate({_n(b.x)} {_n(b.y)}) scale({_n(b.scale)})'))
    root.extend(groups['clothing'])
    arms, fingers, sides = [], [], set()
    for hand in scene.hands:
        if hand.side in sides:
            raise ValueError(f'Duplicate {hand.side} hand')
        sides.add(hand.side)
        if hand.to not in anchors or hand.to.startswith('bufo.'):
            raise ValueError(f'Unknown or invalid hand target {hand.to}; use an object anchor')
        start,end=anchors[f'bufo.{hand.side}_shoulder'],anchors[hand.to]
        arm,finger=_arm(hand,start,end,b.scale)
        arms.extend(arm); fingers.append(finger)
        anchors[f'bufo.{hand.side}_hand']=end.copy()
        contacts.append({'source':f'bufo.{hand.side}_hand','target':hand.to,'offset':[0,0],'residual':0.0})
    root.extend(arms); root.extend(groups['prop']); root.extend(fingers); root.extend(groups['foreground'])
    svg=assemble_character(ET.tostring(root,encoding='unicode'))
    valid,error=validate_svg(svg)
    if not valid:
        raise ValueError(f'Compiled SVG rejected: {error}')
    body_bounds=[b.x,b.y,b.x+128*b.scale,b.y+176*b.scale]
    warnings=[]
    if min(body_bounds[:2])<0 or max(body_bounds[2:])>512:
        warnings.append('Bufo bounds extend outside the canvas; change placement/scale for full-body framing')
    for name,(x,y) in anchors.items():
        if not 0<=x<=512 or not 0<=y<=512:
            warnings.append(f'Anchor {name} is outside the canvas')
    return {'version':VERSION,'svg':svg,'transforms':transforms,'anchors':anchors,'contacts':contacts,
            'bufo_bounds':body_bounds,'warnings':warnings,'verification':'geometry_only',
            'bounds_scope':'Bufo bounds and named anchors only; arbitrary fragment path bounds/occlusion are not verified'}
