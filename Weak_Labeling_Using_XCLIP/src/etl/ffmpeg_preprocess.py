# Video Preprocessing using ffmpeg

from pathlib import Path
import subprocess

def ffmpeg_preprocess_video(
    in_video: Path,
    out_video: Path,
    fps: int,
    resize_shorter_side: int = 224,
    crf: int = 23
):
    out_video.parent.mkdir(parents=True, exist_ok=True)

    scale = f"scale='if(gt(iw,ih),-2,{resize_shorter_side})':'if(gt(iw,ih),{resize_shorter_side},-2)'"
    vf = f"fps={fps},{scale}"

    cmd = [
        "ffmpeg", "-y",
        "-i", str(in_video),
        "-vf", vf,
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", str(crf),
        "-an",
        str(out_video)
    ]
    subprocess.run(cmd, check=True)
