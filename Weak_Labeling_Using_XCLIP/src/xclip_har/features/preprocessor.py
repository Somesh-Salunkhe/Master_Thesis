import subprocess
from pathlib import Path

def preprocess_video(
    in_video: str | Path,
    out_video: str | Path,
    fps: int,
    resize_shorter_side: int = 224,
    crf: int = 23
):
    """
    Uses FFmpeg to resize and re-encode video with specific FPS.
    """
    in_video = Path(in_video)
    out_video = Path(out_video)
    out_video.parent.mkdir(parents=True, exist_ok=True)

    # Maintain aspect ratio while resizing shorter side
    scale = f"scale='if(gt(iw,ih),-2,{resize_shorter_side})':'if(gt(iw,ih),{resize_shorter_side},-2)'"
    vf = f"fps={fps},{scale}"

    cmd = [
        "ffmpeg", "-y",
        "-i", str(in_video),
        "-vf", vf,
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", str(crf),
        "-an", # No audio
        str(out_video)
    ]
    
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
