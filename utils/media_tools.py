from __future__ import annotations

import subprocess
from pathlib import Path


def run_ffmpeg(command: list[str]) -> None:
    subprocess.run(command, check=True)


def extract_audio_from_mp4(mp4_file_path: str | Path, output_audio_file_path: str | Path) -> None:
    run_ffmpeg(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(mp4_file_path),
            "-q:a",
            "0",
            "-map",
            "a",
            str(output_audio_file_path),
        ]
    )


def burn_srt_subtitles(
    video_path: str | Path,
    srt_path: str | Path,
    output_path: str | Path,
    margin_v: int = 40,
) -> None:
    subtitle_filter = (
        f"subtitles={srt_path}:"
        f"force_style='MarginV={margin_v},Fontsize=13,Fontname=MingLiU,"
        "PrimaryColour=&H4FA5FF,Outline=0.85,OutlineColour=&HFFFFFFFF,"
        "Shadow=0.55,ShadowColour=&H00000000,LetterSpacing=1.2,Bold=1'"
    )
    run_ffmpeg(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(video_path),
            "-vf",
            subtitle_filter,
            "-c:v",
            "mpeg4",
            "-q:v",
            "4",
            "-c:a",
            "aac",
            "-strict",
            "experimental",
            str(output_path),
        ]
    )


def change_video_speed(
    input_video_path: str | Path,
    output_video_path: str | Path,
    speed_factor: float,
) -> None:
    video_filter = f"setpts={1 / speed_factor}*PTS"
    audio_filter = (
        f"atempo={speed_factor}"
        if speed_factor <= 2.0
        else f"atempo=2.0,atempo={speed_factor / 2.0}"
    )
    run_ffmpeg(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(input_video_path),
            "-vf",
            video_filter,
            "-filter:a",
            audio_filter,
            "-c:v",
            "mpeg4",
            "-q:v",
            "5",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            str(output_video_path),
        ]
    )


def merge_video_with_bgm(
    video_path: str | Path,
    audio_path: str | Path,
    output_path: str | Path,
    video_volume: float = 1.0,
    bgm_volume: float = 0.15,
) -> None:
    filter_complex = (
        f"[0:a]volume={video_volume}[v_audio];"
        f"[1:a]aloop=loop=-1:size=0:start=0,volume={bgm_volume}[bgm];"
        "[v_audio][bgm]amix=inputs=2:duration=longest[aout]"
    )
    run_ffmpeg(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(video_path),
            "-i",
            str(audio_path),
            "-filter_complex",
            filter_complex,
            "-map",
            "0:v",
            "-map",
            "[aout]",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-shortest",
            str(output_path),
        ]
    )
