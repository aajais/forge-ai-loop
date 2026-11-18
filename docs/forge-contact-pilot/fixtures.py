"""Hand-authored engineering fixtures. Not model outputs or training targets."""
from __future__ import annotations

import asyncio
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parent))
from scene_renderer import compile_scene
from bufogen.evaluation.benchmark import _source_hashes
from bufogen.rendering.renderer import Renderer
from bufogen.settings import ROOT

OUT=ROOT/'runs/forge-contact-pilot/fixtures'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def obj(name,svg,anchors=None,xy=(0,0),angle=0,scale=1,**kwargs):
    return {'id':name,'svg':svg,'anchors':[{'name':k,'x':v[0],'y':v[1]} for k,v in (anchors or {}).items()],
            'transform':{'x':xy[0],'y':xy[1],'angle':angle,'scale':scale},**kwargs}


def hand(side,to,**kwargs):
    return {'side':side,'to':to,**kwargs}


def build_fixtures():
    fixtures=[]
    def add(name,objects,hands,behavior):
        fixtures.append({'id':name,'scene':{'bufo':{'x':190,'y':120,'scale':1.8},
            'objects':[obj('background','<rect width="512" height="512" fill="#f1f4ee"/>',layer='background'),*objects],
            'hands':hands},'expected_behavior':behavior})
    mug='<path d="M0 6 L42 6 L40 48 Q20 58 2 48Z" fill="#eab552" stroke="#534323" stroke-width="3"/><path d="M0 14 C-27 6 -27 44 1 38" fill="none" stroke="#ae742e" stroke-width="8"/><ellipse cx="21" cy="6" rx="21" ry="5" fill="#6d432c"/>'
    add('01-single-grip',[obj('cup',mug,{'handle':(-16,26)},(445,275))],
        [hand('right','cup.handle',angle=90)],'Right arm ends at mug handle; fingers overlay handle.')
    add('02-rotated-scaled-prop',[obj('cup',mug,{'handle':(-16,26)},(120,260),angle=-25,scale=.85)],
        [hand('left','cup.handle',angle=65,width=12)],'Left hand meets the rotated scaled mug handle without changing mug shape.')
    hammer='<path d="M0 -75 L0 0" stroke="#9d663b" stroke-width="12"/><path d="M-20 -6 L25 -6 L25 9 L-20 9Z" fill="#758090" stroke="#323942" stroke-width="3"/>'
    add('03-target-tool-hand-chain',[
        obj('hammer',hammer,{'tip':(25,0),'grip':(0,-58)},attach={'anchor':'tip','to':'target.contact'}),
        obj('target','<rect x="0" y="0" width="55" height="30" fill="#bc8a52"/><path d="M0 0 L0 -12" stroke="#535961" stroke-width="6"/>',{'contact':(0,-12)},(440,380))],
        [hand('right','hammer.grip',angle=0)],'Out-of-order DAG resolves hammer head contact to target then hand to grip.')
    add('04-two-hands-one-prop',[obj('bar','<rect x="0" y="-8" width="180" height="16" rx="7" fill="#ba804c" stroke="#654625" stroke-width="3"/>',{'left':(10,0),'right':(170,0)},(205,365))],
        [hand('left','bar.left',angle=90),hand('right','bar.right',angle=90)],'Both hands grip opposite ends of one unmodified horizontal prop.')
    add('05-crossed-arms',[obj('bar','<rect width="120" height="18" rx="5" fill="#667fb6"/>',{'a':(10,9),'b':(110,9)},(230,350))],
        [hand('left','bar.b',bend='inward',angle=90),hand('right','bar.a',bend='inward',angle=90)],'Explicit inward bend creates crossing connected arms with distinct endpoints.')
    add('06-body-local-clothing',[
        obj('shirt','<path d="M8 95 L106 95 L108 142 Q62 159 9 142Z" fill="#b96d5e" stroke="#5a3430" stroke-width="2"/><path d="M58 96 L58 146" stroke="#e9c7b2" stroke-width="2"/>',frame='bufo',layer='clothing'),
        obj('cup',mug,{'handle':(-16,26)},(445,275))],
        [hand('right','cup.handle',angle=90)],'Clothing inherits body placement/scale; connected arm and grip remain in foreground.')
    add('07-deliberate-occlusion',[
        obj('cup',mug,{'handle':(-16,26)},(445,275)),
        obj('screen','<rect width="125" height="145" rx="8" fill="#445770"/><text x="14" y="75" font-size="14" fill="white">OCCLUDED</text>',xy=(380,250),layer='foreground')],
        [hand('right','cup.handle',angle=90)],'Negative visual fixture: explicit foreground must obscure hand/cup; compiler must not repair visibility.')
    add('08-foot-attachment',[
        obj('board','<rect x="0" y="0" width="180" height="13" rx="6" fill="#4c7994"/><circle cx="30" cy="22" r="9" fill="#333"/><circle cx="150" cy="22" r="9" fill="#333"/>',{'foot':(25,0)},attach={'anchor':'foot','to':'bufo.left_foot'})],[],
        'Board top anchor is attached to canonical left foot without moving Bufo.')
    add('09-rotated-two-contact-chain',[
        obj('wand','<path d="M0 -65 L0 0" stroke="#725c3c" stroke-width="10"/><circle r="8" fill="#d8b64b"/>',{'tip':(0,0),'grip':(0,-55)},angle=30,attach={'anchor':'tip','to':'disc.edge'}),
        obj('disc','<ellipse rx="35" ry="13" fill="#7aafa1" stroke="#345e50" stroke-width="3"/>',{'edge':(-35,0)},(460,365))],
        [hand('right','wand.grip',angle=30)],'Tool tip meets target edge while preserving chosen 30-degree rotation; hand tracks transformed grip.')
    add('10-novel-whisk',[
        obj('whisk','<path d="M0 -65 L0 -25 M0 -25 C-25 -6 -14 10 0 10 C14 10 25 -6 0 -25 M0 -25 L0 10" fill="none" stroke="#8a8e96" stroke-width="5"/>',{'tip':(0,10),'grip':(0,-50)},angle=-15,attach={'anchor':'tip','to':'bowl.surface'}),
        obj('bowl','<path d="M-42 0 Q-35 45 0 45 Q35 45 42 0Z" fill="#dfbb7e"/><ellipse rx="42" ry="10" fill="#eadcc5"/>',{'surface':(0,0)},(435,365))],
        [hand('right','whisk.grip',angle=-15)],'Novel arbitrary whisk fragment uses named tip/grip anchors and a declared target surface.')
    add('11-raised-freeform-prop',[
        obj('kite','<path d="M0 -50 L32 0 L0 35 L-32 0Z" fill="#d28f9f" stroke="#72414c" stroke-width="3"/><path d="M0 -50 L0 85 M-32 0 L32 0" fill="none" stroke="#806948" stroke-width="3"/>',{'handle':(0,85)},(135,150),angle=12)],
        [hand('left','kite.handle',angle=12)],'Raised left arm attaches to rotated arbitrary prop; no object catalog lookup.')
    gradient='<defs><linearGradient id="color"><stop offset="0" stop-color="#51abce"/><stop offset="1" stop-color="#9a6bba"/></linearGradient></defs><path d="M0 -25 L9 -8 L28 -8 L14 5 L19 25 L0 14 L-19 25 L-14 5 L-28 -8 L-9 -8Z" fill="url(#color)" stroke="#45516a" stroke-width="3"/>'
    add('12-local-reference-isolation',[
        obj('star1',gradient,{'grip':(0,14)},(125,285),angle=-15),
        obj('star2',gradient,{'grip':(0,14)},(440,285),angle=15)],
        [hand('left','star1.grip',angle=-15),hand('right','star2.grip',angle=15)],
        'Two arbitrary fragments reuse local gradient IDs without collision; both grips remain attached.')
    return fixtures


async def main():
    from PIL import Image,ImageDraw
    OUT.mkdir(parents=True,exist_ok=True)
    source_hashes = _source_hashes()
    compiler_sha = sha(Path(__file__).with_name('scene_renderer.py'))
    fixture_generator_sha = sha(__file__)
    renderer=Renderer(); rows=[]
    try:
        for item in build_fixtures():
            ident=item['id']; folder=OUT/ident; folder.mkdir(exist_ok=True)
            scene_path=folder/'scene.json'; svg_path=folder/'scene.svg'; png_path=folder/'scene.png'
            scene_text=json.dumps(item['scene'],indent=2)+'\n'; compiled=compile_scene(item['scene'])
            scene_path.write_text(scene_text); svg_path.write_text(compiled['svg'])
            (folder/'geometry.json').write_text(json.dumps({k:v for k,v in compiled.items() if k!='svg'},indent=2)+'\n')
            result=await renderer.render(compiled['svg'])
            if not result.valid or not result.png_bytes: raise RuntimeError(f'{ident}: {result.error}')
            png_path.write_bytes(result.png_bytes)
            rows.append({'id':ident,'scene_path':str(scene_path),'svg_path':str(svg_path),'png_path':str(png_path),
                'scene_sha256':sha(scene_path),'svg_sha256':sha(svg_path),'png_sha256':sha(png_path),
                'compiler_sha256':compiler_sha,'source_hashes':source_hashes,
                'fixture_generator_sha256':fixture_generator_sha,'expected_behavior':item['expected_behavior'],
                'provenance':'hand_authored_engineering_fixture','render_backend':'cwsandbox'})
            print(ident,'rendered',flush=True)
    finally: await renderer.close()
    if _source_hashes() != source_hashes or sha(Path(__file__).with_name('scene_renderer.py')) != compiler_sha:
        raise RuntimeError('Fixture dependencies changed during rendering; do not accept these images')
    (OUT/'manifest.json').write_text(json.dumps(rows,indent=2)+'\n')
    for page in range(3):
        canvas=Image.new('RGB',(1120,1230),'white'); draw=ImageDraw.Draw(canvas)
        for i,row in enumerate(rows[page*4:page*4+4]):
            x=(i%2)*560; y=(i//2)*615
            image=Image.open(row['png_path']).convert('RGBA')
            canvas.paste(image,(x+20,y+30),image)
            draw.text((x+20,y+7),row['id'],fill='black')
            small=image.resize((64,64),Image.Resampling.LANCZOS)
            canvas.paste(small,(x+20,y+545),small)
            draw.text((x+100,y+564),'64px preview',fill='black')
        canvas.save(OUT/f'sheet-{page+1}.png')
    print(json.dumps({'fixtures':len(rows),'manifest':str(OUT/'manifest.json')}))


if __name__=='__main__': asyncio.run(main())
