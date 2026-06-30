from __future__ import annotations

import os
import subprocess
from pathlib import Path


def parse_segments_file(segments_file: str | Path) -> list[tuple[float, float]]:
    segments: list[tuple[float, float]] = []
    for line in Path(segments_file).read_text(encoding="utf-8").splitlines():
        value = line.strip().rstrip(",")
        if not value:
            continue
        start, end = map(float, value.strip("()").split(", "))
        segments.append((start, end))
    return segments


def remove_video_segments_with_ffmpeg(
    video_path: str | Path,
    segments_to_remove: list[tuple[float, float]],
    output_video: str | Path,
) -> None:
    if not Path(video_path).exists():
        raise FileNotFoundError(f"video not found: {video_path}")
    total_duration = float(
        subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(video_path),
            ],
            stderr=subprocess.STDOUT,
            text=True,
        ).strip()
    )
    retain_segments: list[tuple[float, float]] = []
    last_end = 0.0
    for start, end in segments_to_remove:
        if start > last_end:
            retain_segments.append((last_end, start))
        last_end = end
    if last_end < total_duration:
        retain_segments.append((last_end, total_duration))

    temp_dir = Path(output_video).with_suffix("")
    temp_dir = temp_dir.parent / f"{temp_dir.name}_parts"
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_clips: list[Path] = []

    for index, (start, end) in enumerate(retain_segments):
        temp_clip_path = temp_dir / f"temp_clip_{index}.mp4"
        temp_clips.append(temp_clip_path)
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(video_path),
                "-ss",
                str(start),
                "-to",
                str(end),
                "-c",
                "copy",
                str(temp_clip_path),
            ],
            check=True,
        )

    file_list_path = temp_dir / "file_list.txt"
    with file_list_path.open("w", encoding="utf-8") as handle:
        for clip_path in temp_clips:
            normalized_path = os.path.abspath(clip_path).replace("\\", "/")
            handle.write(f"file '{normalized_path}'\n")

    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(file_list_path),
            "-c",
            "copy",
            str(output_video),
        ],
        check=True,
    )
