from __future__ import annotations

import json
from pathlib import Path


def transcribe_audio_whisper(audio_path: str | Path, model_name: str = "medium") -> list[dict]:
    import whisper

    model = whisper.load_model(model_name)
    result = model.transcribe(str(audio_path))
    return result["segments"]


def format_srt_time(total_seconds: float) -> str:
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds_int = int(total_seconds % 60)
    milliseconds = int((total_seconds - int(total_seconds)) * 1000)
    return f"{hours:02}:{minutes:02}:{seconds_int:02},{milliseconds:03}"


def generate_srt(subtitles_segments: list[dict], srt_path: str | Path, max_chars: int = 15) -> None:
    lines: list[str] = []
    idx = 1
    for segment in subtitles_segments:
        start_time = segment["start"]
        end_time = segment["end"]
        text = segment["text"].strip()
        while len(text) > max_chars:
            split_index = text.rfind(" ", 0, max_chars)
            if split_index == -1:
                split_index = max_chars
            line = text[:split_index].strip()
            text = text[split_index:].strip()
            segment_duration = end_time - start_time
            line_duration = segment_duration * (len(line) / max(1, len(segment["text"])))
            line_end_time = start_time + line_duration
            lines.extend(
                [
                    str(idx),
                    f"{format_srt_time(start_time)} --> {format_srt_time(line_end_time)}",
                    line,
                    "",
                ]
            )
            idx += 1
            start_time = line_end_time
        if text:
            lines.extend(
                [
                    str(idx),
                    f"{format_srt_time(start_time)} --> {format_srt_time(end_time)}",
                    text,
                    "",
                ]
            )
            idx += 1
    Path(srt_path).write_text("\n".join(lines), encoding="utf-8")


def generate_json(subtitles_segments: list[dict], json_path: str | Path, max_chars: int = 25) -> None:
    subtitles: list[dict] = []
    line_number = 1
    for segment in subtitles_segments:
        start_time = segment["start"]
        end_time = segment["end"]
        text = segment["text"].strip()
        while len(text) > max_chars:
            split_index = text.rfind(" ", 0, max_chars)
            if split_index == -1:
                split_index = max_chars
            line = text[:split_index].strip()
            text = text[split_index:].strip()
            line_end_time = start_time + (end_time - start_time) * (len(line) / max(1, len(segment["text"])))
            subtitles.append({"start": round(start_time, 2), "end": round(line_end_time, 2), "text": f"{line_number}: {line}"})
            start_time = line_end_time
            line_number += 1
        if text:
            subtitles.append({"start": round(start_time, 2), "end": round(end_time, 2), "text": f"{line_number}: {text}"})
            line_number += 1
    Path(json_path).write_text(json.dumps(subtitles, ensure_ascii=False, indent=4), encoding="utf-8")


def update_json_from_txt(txt_path: str | Path, json_path: str | Path) -> None:
    subtitles = json.loads(Path(json_path).read_text(encoding="utf-8"))
    updates: dict[int, str] = {}
    for line in Path(txt_path).read_text(encoding="utf-8").splitlines():
        if ": " in line:
            index, text = line.split(": ", 1)
            updates[int(index)] = text.strip()
    for subtitle in subtitles:
        index = int(subtitle["text"].split(": ")[0])
        if index in updates:
            subtitle["text"] = f"{index}: {updates[index]}"
    Path(json_path).write_text(json.dumps(subtitles, ensure_ascii=False, indent=4), encoding="utf-8")


def extract_empty_segments(json_path: str | Path) -> list[tuple[float, float]]:
    data = json.loads(Path(json_path).read_text(encoding="utf-8"))
    empty_segments: list[tuple[float, float]] = []
    for item in data:
        text = item["text"].strip()
        if text and (text.endswith(":") or text.isdigit()):
            empty_segments.append((item["start"], item["end"]))
    return empty_segments
