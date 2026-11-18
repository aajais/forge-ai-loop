"""Create local image-review sheets; emits no acceptance decisions."""
import argparse
import hashlib
import json
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--families", nargs="+", required=True)
    parser.add_argument("--name", required=True)
    args=parser.parse_args()
    root=Path("runs/forge-gap/teacher-curriculum")
    rows=[json.loads(s) for s in (root/"manifest.jsonl").read_text().splitlines()]
    selected=sorted((r for r in rows if r["benchmark_metadata"]["family"] in args.families),key=lambda r:r["prompt_id"])
    out=Path("runs/forge-gap/review")/args.name
    out.mkdir(parents=True,exist_ok=True)
    inputs=[]
    for i,r in enumerate(selected):
        png=Path(r["png_path"]) if r.get("png_path") else None
        inputs.append({"index":i,"source_generation_id":r["id"],"prompt_id":r["prompt_id"],"prompt":r["prompt"],
                       "family":r["benchmark_metadata"]["family"],"partition":r["benchmark_metadata"]["split"],
                       "requirements":r["benchmark_metadata"]["requirements"],"png_path":str(png) if png else None,
                       "png_sha256":hashlib.sha256(png.read_bytes()).hexdigest() if png and png.exists() else None,
                       "valid":r["valid"],"conversation_url":r.get("conversation_url")})
    (out/"inputs.json").write_text(json.dumps(inputs,indent=2)+"\n")
    font=ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc",16)
    for start in range(0,len(inputs),6):
        batch=inputs[start:start+6]
        canvas=Image.new("RGB",(1080,2160),"#eef0ed")
        draw=ImageDraw.Draw(canvas)
        for n,item in enumerate(batch):
            x=(n%2)*540; y=(n//2)*720
            draw.text((x+10,y+5),f"{item['index']}  {item['prompt_id']}",font=font,fill="black")
            if item["png_path"] and Path(item["png_path"]).exists():
                im=Image.open(item["png_path"]).convert("RGBA")
                im.thumbnail((512,512))
                canvas.paste(im,(x+14,y+32),im)
            else:
                draw.text((x+20,y+200),"NO GENERATED IMAGE",font=font,fill="red")
            checks=" | ".join(item["requirements"])
            for line_num,line in enumerate(textwrap.wrap(checks,width=61)):
                draw.text((x+10,y+550+line_num*19),line,font=font,fill="black")
        canvas.save(out/f"sheet-{start//6:02d}.png")
    print(json.dumps({"directory":str(out),"images":len(inputs),"sheets":(len(inputs)+5)//6}))

if __name__=="__main__":
    main()
