from __future__ import annotations

import json
import subprocess
from pathlib import Path


def run_ollama_model(input_text: str, model_name: str) -> str:
    command = ["ollama", "run", model_name]
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
    )
    output, error = process.communicate(input_text)
    if process.returncode != 0:
        raise RuntimeError(error.strip() or "ollama failed")
    return output.strip()


def summarize_text_file(input_file: str | Path, output_file: str | Path, model_name: str) -> None:
    text = Path(input_file).read_text(encoding="utf-8")
    prompt = "请重组以下内容，去除冗余信息，只保留有意义的部分：\n" + text
    summary = run_ollama_model(prompt, model_name)
    Path(output_file).write_text(summary, encoding="utf-8")


def build_highlight_candidates_from_transcript(
    transcript_text: str,
    model_name: str,
    max_candidates: int = 8,
) -> str:
    prompt = (
        "你是一个直播切片助手。请从下面的直播转写文本中挑选最适合做短视频切片的高光片段。"
        "输出要求：按重要性排序，说明推荐理由，并尽量指出适合保留的句子段落。\n"
        f"最多输出 {max_candidates} 条候选。\n\n"
        f"{transcript_text}"
    )
    return run_ollama_model(prompt, model_name)
