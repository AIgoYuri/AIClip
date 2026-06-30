"""语音转写 & 字幕工具：支持 Whisper / Qwen3-ASR 双引擎"""
from __future__ import annotations
from pathlib import Path
import json
import os

# ── 转写引擎 ──────────────────────────────────────────

_WHISPER_AVAILABLE: bool | None = None


def _check_whisper() -> bool:
    global _WHISPER_AVAILABLE
    if _WHISPER_AVAILABLE is not None:
        return _WHISPER_AVAILABLE
    try:
        import whisper  # noqa: F401
        _WHISPER_AVAILABLE = True
    except ImportError:
        _WHISPER_AVAILABLE = False
    return _WHISPER_AVAILABLE


def transcribe_audio(
    audio_path: str | Path,
    model_name: str = "base",
    language: str = "zh",
    device: str = "cpu",
) -> list[dict]:
    """通用转写入口 —— 自动选择可用引擎（Whisper 优先，可扩展 Qwen）"""
    if _check_whisper():
        return _transcribe_whisper(audio_path, model_name, language, device)
    raise RuntimeError(
        "没有可用的 ASR 引擎。请安装 openai-whisper：\n"
        "  pip install openai-whisper"
    )


def _transcribe_whisper(
    audio_path: str | Path,
    model_name: str = "base",
    language: str = "zh",
    device: str = "cpu",
) -> list[dict]:
    import whisper  # delayed import

    os.environ["CUDA_VISIBLE_DEVICES"] = "" if device == "cpu" else os.environ.get("CUDA_VISIBLE_DEVICES", "")
    model = whisper.load_model(model_name, device=device)
    result = model.transcribe(str(audio_path), language=language)
    # whisper 返回的 segments 已包含 start / end / text 字段
    return [
        {"start": seg["start"], "end": seg["end"], "text": seg["text"].strip()}
        for seg in result["segments"]
    ]


# ── SRT 生成 ──────────────────────────────────────────


def format_srt_time(total_seconds: float) -> str:
    h = int(total_seconds // 3600)
    m = int((total_seconds % 3600) // 60)
    s = int(total_seconds % 60)
    ms = int((total_seconds - int(total_seconds)) * 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"


def generate_srt(segments: list[dict], srt_path: str | Path, max_chars: int = 15) -> None:
    lines: list[str] = []
    idx = 1
    for seg in segments:
        start, end, text = seg["start"], seg["end"], seg["text"]
        while len(text) > max_chars:
            split = text.rfind(" ", 0, max_chars)
            if split == -1:
                split = max_chars
            line, text = text[:split].strip(), text[split:].strip()
            ratio = len(line) / max(1, len(seg["text"]))
            line_end = start + (end - start) * ratio
            lines.extend([str(idx), f"{format_srt_time(start)} --> {format_srt_time(line_end)}", line, ""])
            idx += 1
            start = line_end
        if text:
            lines.extend([str(idx), f"{format_srt_time(start)} --> {format_srt_time(end)}", text, ""])
            idx += 1
    Path(srt_path).write_text("\n".join(lines), encoding="utf-8")


# ── JSON 字幕生成 ─────────────────────────────────────


def generate_json(segments: list[dict], json_path: str | Path, max_chars: int = 25) -> None:
    items: list[dict] = []
    line_no = 1
    for seg in segments:
        start, end, text = seg["start"], seg["end"], seg["text"]
        while len(text) > max_chars:
            split = text.rfind(" ", 0, max_chars)
            if split == -1:
                split = max_chars
            line, text = text[:split].strip(), text[split:].strip()
            ratio = len(line) / max(1, len(seg["text"]))
            line_end = start + (end - start) * ratio
            items.append({"start": round(start, 2), "end": round(line_end, 2), "text": f"{line_no}: {line}"})
            start = line_end
            line_no += 1
        if text:
            items.append({"start": round(start, 2), "end": round(end, 2), "text": f"{line_no}: {text}"})
            line_no += 1
    Path(json_path).write_text(json.dumps(items, ensure_ascii=False, indent=4), encoding="utf-8")


def update_json_from_txt(txt_path: str | Path, json_path: str | Path) -> None:
    """用人工编辑后的 txt 更新 json 字幕内容"""
    subtitles = json.loads(Path(json_path).read_text(encoding="utf-8"))
    updates: dict[int, str] = {}
    for line in Path(txt_path).read_text(encoding="utf-8").splitlines():
        if ": " in line:
            idx, txt = line.split(": ", 1)
            updates[int(idx)] = txt.strip()
    for item in subtitles:
        idx = int(item["text"].split(": ")[0])
        if idx in updates:
            item["text"] = f"{idx}: {updates[idx]}"
    Path(json_path).write_text(json.dumps(subtitles, ensure_ascii=False, indent=4), encoding="utf-8")


def extract_empty_segments(json_path: str | Path) -> list[tuple[float, float]]:
    """从 JSON 字幕中检测空白/无效段的起止时间"""
    data = json.loads(Path(json_path).read_text(encoding="utf-8"))
    empty: list[tuple[float, float]] = []
    for item in data:
        t = item["text"].strip()
        if t and (t.endswith(":") or t.isdigit()):
            empty.append((item["start"], item["end"]))
    return empty
