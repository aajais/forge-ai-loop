"""Assemble the verified local capture with synthesized narration and captions.

Source ranges exclude setup, loading, browser chrome and unrelated applications.
Short scenes hold their last frame while narration finishes; no UI is fabricated.
"""
from pathlib import Path
import json
import math
import subprocess

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
NOTES = [
    "Run | Serverless Inference + Sandboxes | Recorded Bufo app conversation",
    "Validation example | Rendering succeeds, but a requested object is unclear.",
    "Agent Lens | Inspect the request, model, tools, and rendered PNG.",
    "911 reviewed gpt-6.1-sol trajectories | Automated + assistant review | Human confirmation pending",
    "Qwen/Qwen3.8-27B | All assistant turns | 1 epoch | Versioned dataset and checkpoint",
    "48 validation prompts | Base: 17 passes | SFT 1 epoch: 30 | SFT 3 epochs: 27 | RL: 30",
    "Same validation prompt | ft-Qwen/Qwen3.8-27B, checkpoint v1 | One example, cohort measured separately",
    "Candidate selected for the next turn | Existing image retains its generating model | Continue the AI loop",
]


def run(args):
    subprocess.run(args, check=True)


def duration(path):
    return float(json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "json", str(path)]))["format"]["duration"])


def stamp(seconds):
    ms = round(seconds * 1000)
    return f"{ms//3600000:02}:{ms//60000%60:02}:{ms//1000%60:02},{ms%1000:03}"


def main():
    scenes = json.loads((ROOT / "narration.json").read_text())
    shots = json.loads((ROOT / "shots.json").read_text())
    if len(shots) != len(scenes):
        raise SystemExit("Every narration scene needs a verified recording range.")
    for scene, shot in zip(scenes, shots):
        if shot["scene"] != scene["id"] or not shot.get("verified") or not shot.get("source"):
            raise SystemExit(f"Missing verified footage: {scene['id']}")
        if shot["end_s"] <= shot["start_s"] or not (ROOT / shot["source"]).exists():
            raise SystemExit(f"Invalid footage: {scene['id']}")
        if duration(ROOT / (scene["id"] + ".aiff")) <= 1:
            raise SystemExit(f"Narration missing: {scene['id']}")
    titles = [scene["title"] for scene in scenes]
    work = ROOT / "edit"
    work.mkdir(exist_ok=True)
    font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 30)
    bold = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 34)
    small = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 26)
    elapsed = 0.0
    subtitles, chapters, manifest, segments = [], [";FFMETADATA1"], [], []
    for i, (scene, shot) in enumerate(zip(scenes, shots)):
        start, end = shot["start_s"], shot["end_s"]
        raw = ROOT / shot["source"]
        audio = ROOT / f"{scene['id']}.aiff"
        speech = duration(audio)
        length = math.ceil((speech + 1.1) * 24) / 24
        overlay = Image.new("RGBA", (2560, 1600), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        draw.rectangle((0, 0, 2560, 79), fill="#132137")
        draw.text((38, 23), "CoreWeave Forge", font=bold, fill="white")
        draw.text((410, 25), titles[i], font=font, fill="#BBD1FF")
        draw.text((2395, 26), f"0{i+1} / {len(scenes):02}", font=small, fill="white")
        draw.rectangle((0, 1505, 2560, 1600), fill="#132137")
        draw.rectangle((0, 1505, round(2560*(i+1)/len(scenes)), 1510), fill="#466AFF")
        draw.text((38, 1540), NOTES[i], font=small, fill="white")
        overlay_path = work / f"{scene['id']}-overlay.png"
        overlay.save(overlay_path)
        dest = work / f"{scene['id']}.mp4"
        filters = (
            f"[0:v]crop={shot['crop']},scale=2560:1424,setsar=1,"
            "pad=2560:1600:0:80:color=0x132137,fps=24,"
            f"tpad=stop_mode=clone:stop_duration={length}[screen];"
            "[screen][2:v]overlay=0:0,format=yuv420p[v];"
            "[1:a]loudnorm=I=-16:TP=-1.5:LRA=11,aresample=48000,"
            "adelay=500:all=1,apad[a]"
        )
        print(f"Rendering {scene['id']} ({length:.1f}s)", flush=True)
        run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
             "-ss", str(start), "-t", str(end-start), "-i", str(raw),
             "-i", str(audio), "-loop", "1", "-i", str(overlay_path),
             "-filter_complex", filters, "-map", "[v]", "-map", "[a]",
             "-t", str(length), "-c:v", "libx264", "-preset", "veryfast",
             "-crf", "21", "-threads", "4", "-c:a", "aac", "-b:a", "160k",
             "-ar", "48000", "-ac", "2", "-movflags", "+faststart", str(dest)])
        segments.append(dest)
        words = scene["text"].split()
        for offset in range(0, len(words), 12):
            chunk = words[offset:offset+12]
            begin = elapsed + 0.5 + speech * offset / len(words)
            finish = elapsed + 0.5 + speech * (offset+len(chunk)) / len(words)
            subtitles.append(f"{len(subtitles)+1}\n{stamp(begin)} --> {stamp(finish)}\n{' '.join(chunk)}\n")
        chapters.extend(["[CHAPTER]", "TIMEBASE=1/1000", f"START={round(elapsed*1000)}",
                         f"END={round((elapsed+length)*1000)}", f"title={titles[i]}"])
        manifest.append({"scene":scene["id"], "source":shot["source"], "source_start_s":start,
                         "source_end_s":end, "output_start_s":elapsed,
                         "output_duration_s":length, "last_frame_hold_s":max(0,length-(end-start))})
        elapsed += length
    captions = ROOT / "forge-bufo-story.srt"
    captions.write_text("\n".join(subtitles))
    metadata = work / "chapters.ffmetadata"
    metadata.write_text("\n".join(chapters)+"\n")
    concat = work / "segments.txt"
    concat.write_text("\n".join(f"file '{p.name}'" for p in segments)+"\n")
    result = ROOT / "forge-bufo-story.mp4"
    run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "concat",
         "-safe", "0", "-i", str(concat), "-i", str(captions), "-i", str(metadata),
         "-map", "0:v", "-map", "0:a", "-map", "1:0", "-map_metadata", "2",
         "-map_chapters", "2", "-c:v", "copy", "-c:a", "copy", "-c:s", "mov_text",
         "-metadata:s:s:0", "language=eng", "-metadata:s:s:0", "title=English",
         "-metadata", "title=Bufo | The CoreWeave Forge AI loop",
         "-metadata", "comment=Edited screen recording with synthesized narration. Short scenes hold their final frame. Validation snapshot: October 24, 2026.",
         "-movflags", "+faststart", str(result)])
    (ROOT / "edit-manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(f"Created {result} ({duration(result):.1f}s)", flush=True)


if __name__ == "__main__":
    main()
