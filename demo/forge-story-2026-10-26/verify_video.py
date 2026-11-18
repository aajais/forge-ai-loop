"""Decode the completed demo, inventory tracks, and extract chapter frames for review."""
import json
from pathlib import Path
import subprocess
from PIL import Image, ImageDraw
ROOT=Path(__file__).resolve().parent
video=ROOT/'forge-bufo-story.mp4'
qa=ROOT/'qa'; qa.mkdir(exist_ok=True)
probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_format','-show_streams','-show_chapters','-of','json',str(video)]))
(qa/'probe.json').write_text(json.dumps(probe,indent=2)+'\n')
subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-i',str(video),'-map','0:v','-map','0:a','-f','null','-'],check=True)
levels=subprocess.run(['ffmpeg','-hide_banner','-i',str(video),'-vn','-af','volumedetect','-f','null','-'],capture_output=True,text=True,check=True)
(qa/'audio-levels.txt').write_text(levels.stderr)
frames=[]
for i,ch in enumerate(probe['chapters']):
    at=float(ch['start_time'])+min(8,(float(ch['end_time'])-float(ch['start_time']))/2)
    dest=qa/f'chapter-{i+1:02}.jpg'
    subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-ss',str(at),'-i',str(video),'-frames:v','1','-vf','scale=960:600','-y',str(dest)],check=True)
    im=Image.open(dest); frames.append((im,ch['tags']['title']))
sheet=Image.new('RGB',(1920,4*630),'#ffffff');draw=ImageDraw.Draw(sheet)
for i,(im,title) in enumerate(frames):
    x=(i%2)*960;y=(i//2)*630
    sheet.paste(im,(x,y));draw.text((x+12,y+608),title,fill='#15243b')
sheet.save(qa/'chapters.jpg')
print(json.dumps({'duration_s':probe['format']['duration'],'bytes':probe['format']['size'],'tracks':[s['codec_name'] for s in probe['streams']],'chapters':len(probe['chapters']),'decode':'passed'},indent=2))
print('\n'.join(x for x in levels.stderr.splitlines() if 'mean_volume' in x or 'max_volume' in x))
