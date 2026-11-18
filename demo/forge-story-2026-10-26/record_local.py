"""Run in the user's macOS Terminal to capture the prepared Forge walkthrough.

Records one selected display, without microphone audio, for at most seven minutes.
Press q in this Terminal to finish sooner. No files are uploaded.
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parent


def duration(path):
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", str(path)],
        capture_output=True, text=True, check=True,
    )
    return float(json.loads(result.stdout).get("format", {}).get("duration", 0))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, help="Display index from ffmpeg's device list")
    parser.add_argument("--seconds", type=int, default=420, help="Maximum capture duration (default 420)")
    parser.add_argument("--check", action="store_true", help="List displays without recording")
    args = parser.parse_args()
    if not 1 <= args.seconds <= 900:
        parser.error("--seconds must be between 1 and 900")
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        raise SystemExit("ffmpeg and ffprobe must be installed.")

    discovery = subprocess.run(
        ["ffmpeg", "-hide_banner", "-f", "avfoundation", "-list_devices", "true", "-i", ""],
        capture_output=True, text=True,
    )
    displays = dict(
        (int(index), name.strip())
        for index, name in re.findall(r"\[(\d+)\]\s+(Capture screen[^\n]+)", discovery.stderr)
    )
    if not displays:
        raise SystemExit(
            "No capture display is exposed to ffmpeg. No recording started. "
            "Run this from your normal macOS Terminal; check its Screen Recording access if needed."
        )
    for index, name in displays.items():
        print(f"Display {index}: {name}")
    if args.check:
        return
    device = args.device
    if device is None and len(displays) == 1:
        device = next(iter(displays))
    if device not in displays:
        raise SystemExit("Choose the display containing Chrome with --device INDEX.")

    # Synthesize locally; validate actual audio rather than trusting say's exit status.
    audio_ready = True
    for scene in json.loads((ROOT / "narration.json").read_text()):
        output = ROOT / (scene["id"] + ".aiff")
        if output.exists() and duration(output) > 1:
            continue
        result = subprocess.run([
            "say", "-v", "Samantha", "-r", "150", "-f", str(ROOT / (scene["id"] + ".txt")),
            "-o", str(output),
        ])
        if result.returncode or not output.exists() or duration(output) <= 1:
            audio_ready = False
            print(f"Narration unavailable for {scene['id']}; raw capture can still proceed.")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = ROOT / f"forge-demo-raw-{stamp}.mov"
    print(f"\nRecording only {displays[device]}, without microphone audio.")
    print("Put Chrome on that display with the Forge demo visible.")
    input("Press Enter to start, then tell Codex 'recording started'. Press q here to stop early. ")
    receipt = {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "display": displays[device], "device_index": device,
        "output": str(output), "maximum_duration_s": args.seconds,
        "microphone_recorded": False, "narration_audio_valid": audio_ready,
        "status": "starting",
    }
    receipt_path = ROOT / f"capture-{stamp}.json"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    progress_path = ROOT / f"capture-{stamp}.progress"
    process = subprocess.Popen([
        "ffmpeg", "-hide_banner", "-n", "-f", "avfoundation", "-framerate", "30",
        "-capture_cursor", "1", "-capture_mouse_clicks", "1", "-i", f"{device}:none",
        "-t", str(args.seconds), "-an", "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "20", "-pix_fmt", "yuv420p",
        "-progress", str(progress_path),
        str(output),
    ])
    launched = time.monotonic()
    stop_reason = None
    try:
        while process.poll() is None:
            progress = progress_path.read_text() if progress_path.exists() else ""
            frames = re.findall(r"^frame=(\d+)$", progress, re.MULTILINE)
            if receipt["status"] == "starting" and any(int(frame) > 0 for frame in frames):
                receipt["status"] = "recording"
                receipt["first_frame_observed_at_utc"] = datetime.now(timezone.utc).isoformat()
                receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
                print("\nFrames are arriving. Recording is active.", flush=True)
            elapsed = time.monotonic() - launched
            if receipt["status"] == "starting" and elapsed > 30:
                stop_reason = "No frames received within 30 seconds. Check Screen & System Audio Recording permission for the app hosting this terminal, then restart that app."
                break
            if elapsed > args.seconds + 45:
                stop_reason = "Recorder exceeded its wall-clock limit."
                break
            time.sleep(0.25)
    except KeyboardInterrupt:
        stop_reason = "Stopped by user."
    finally:
        if process.poll() is None:
            process.send_signal(signal.SIGINT)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
    receipt["returncode"] = process.returncode
    if stop_reason:
        receipt["stop_reason"] = stop_reason
        print(f"\n{stop_reason}")
    try:
        receipt["duration_s"] = duration(output) if output.exists() and output.stat().st_size else 0
    except (subprocess.CalledProcessError, ValueError):
        receipt["duration_s"] = 0
    receipt["status"] = "captured" if receipt["duration_s"] > 1 else "failed"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    print(f"\n{receipt['status']}: {output}")
    if receipt["status"] != "captured":
        sys.exit(1)


if __name__ == "__main__":
    main()
