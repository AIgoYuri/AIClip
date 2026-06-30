"""AICutPipeline —— 全流程编排"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.subtitle_tools import (
    extract_empty_segments,
    generate_json,
    generate_srt,
    transcribe_audio,
    update_json_from_txt,
)
from utils.cut_tools import remove_video_segments_with_ffmpeg
from utils.highlight_tools import build_highlight_candidates_from_transcript
from utils.media_tools import (
    burn_srt_subtitles,
    change_video_speed,
    extract_audio_from_mp4,
    merge_video_with_bgm,
)


class AICutPipeline:
    """对单条素材执行全流程处理"""

    def __init__(self, workspace_root: str | Path) -> None:
        self.root = Path(workspace_root)
        self.root.mkdir(parents=True, exist_ok=True)

    def extract_audio(self, video_path: str | Path, name: str = "audio.mp3") -> Path:
        out = self.root / name
        extract_audio_from_mp4(video_path, out)
        return out

    def transcribe(
        self,
        audio_path: str | Path,
        srt_name: str = "subtitles.srt",
        json_name: str = "subtitles.json",
        model_name: str = "base",
        language: str = "zh",
    ) -> tuple[Path, Path, list[dict]]:
        segments = transcribe_audio(audio_path, model_name=model_name, language=language)
        srt = self.root / srt_name
        jsn = self.root / json_name
        generate_srt(segments, srt)
        generate_json(segments, jsn)
        return srt, jsn, segments

    def rewrite_json(self, txt_path: str | Path, json_path: str | Path) -> None:
        update_json_from_txt(txt_path, json_path)

    def build_highlight(self, transcript_text: str, ollama_model: str) -> str:
        return build_highlight_candidates_from_transcript(transcript_text, ollama_model)

    def cut_empty(self, video_path: str | Path, json_path: str | Path, out_name: str = "cut.mp4") -> Path:
        out = self.root / out_name
        segments = extract_empty_segments(json_path)
        remove_video_segments_with_ffmpeg(video_path, segments, out)
        return out

    def burn_subtitles(self, video_path: str | Path, srt_path: str | Path, out_name: str = "subtitled.mp4") -> Path:
        out = self.root / out_name
        burn_srt_subtitles(video_path, srt_path, out)
        return out

    def speed_up(self, video_path: str | Path, factor: float, out_name: str = "speed.mp4") -> Path:
        out = self.root / out_name
        change_video_speed(video_path, out, factor)
        return out

    def add_bgm(
        self,
        video_path: str | Path,
        bgm_path: str | Path,
        out_name: str = "bgm.mp4",
        video_volume: float = 1.0,
        bgm_volume: float = 0.15,
    ) -> Path:
        out = self.root / out_name
        merge_video_with_bgm(video_path, bgm_path, out, video_volume=video_volume, bgm_volume=bgm_volume)
        return out
