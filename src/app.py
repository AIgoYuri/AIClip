"""AICut — 分步式视频智能裁剪 API"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "output"
TEMPLATE_PATH = ROOT / "src" / "index.html"
HOST = "127.0.0.1"
PORT = 3801

_tasks: dict[str, dict] = {}
_lock = threading.Lock()
_model_status: dict[str, dict] = {}  # whisper model cache status


def _new_task(task_type: str, filename: str) -> str:
    tid = uuid.uuid4().hex[:12]
    with _lock:
        _tasks[tid] = {
            "id": tid, "type": task_type, "filename": filename,
            "status": "queued", "progress": 0, "message": "等待处理",
            "created_at": time.time(), "result": None, "error": None,
        }
    return tid


def _update_task(tid: str, **kw) -> None:
    with _lock:
        if tid in _tasks:
            _tasks[tid].update(kw)


def _get_task(tid: str) -> dict | None:
    with _lock:
        return _tasks.get(tid)


# ── Whisper 模型管理 ─────────────────────────────────

WHISPER_MODELS = {
    "tiny":    {"size_mb": 72,   "desc": "最快，精度最低"},
    "base":    {"size_mb": 139,  "desc": "快速，基础精度"},
    "small":   {"size_mb": 466,  "desc": "平衡速度与精度"},
    "medium":  {"size_mb": 1460, "desc": "较高精度，较慢"},
    "large":   {"size_mb": 2870, "desc": "最高精度，最慢"},
}

def _whisper_cache_dir() -> Path:
    return Path(os.environ.get(
        "WHISPER_CACHE_DIR",
        os.path.join(os.path.expanduser("~"), ".cache", "whisper")
    ))


def _check_whisper_models() -> dict[str, dict]:
    """检查所有 whisper 模型的缓存状态"""
    cache = _whisper_cache_dir()
    result = {}
    for name, info in WHISPER_MODELS.items():
        pt = cache / f"{name}.pt"
        if pt.exists():
            actual_mb = pt.stat().st_size / 1024 / 1024
            expected = info["size_mb"]
            # 允许 5% 误差，低于 80% 视为损坏
            if actual_mb >= expected * 0.8:
                result[name] = {"status": "ok", "size_mb": round(actual_mb, 1), "path": str(pt)}
            else:
                result[name] = {"status": "corrupt", "size_mb": round(actual_mb, 1),
                                "expected_mb": expected, "path": str(pt)}
        else:
            result[name] = {"status": "missing", "size_mb": 0}
    return result


def _delete_corrupted_models() -> list[str]:
    """删除损坏的 whisper 模型缓存，返回已删除列表"""
    deleted = []
    for name, info in _check_whisper_models().items():
        if info["status"] == "corrupt":
            try:
                os.remove(info["path"])
                deleted.append(name)
                print(f"[WHISPER] 已删除损坏模型: {name}.pt ({info['size_mb']} MB, 预期 {info['expected_mb']} MB)")
            except OSError as e:
                print(f"[WHISPER] 删除失败: {name}.pt — {e}")
    return deleted


def _preload_whisper_model(model_name: str) -> tuple[bool, str]:
    """预下载一个 whisper 模型（阻塞，约 30s-2min）"""
    try:
        import whisper
        print(f"[WHISPER] 正在下载模型: {model_name} ...")
        whisper.load_model(model_name, device="cpu")
        return True, f"模型 {model_name} 就绪"
    except Exception as e:
        msg = str(e)
        # SHA256 错误 → 删除损坏文件
        if "SHA256" in msg or "checksum" in msg:
            cache = _whisper_cache_dir()
            bad = cache / f"{model_name}.pt"
            bad.unlink(missing_ok=True)
            # 重试一次
            try:
                import whisper
                whisper.load_model(model_name, device="cpu")
                return True, f"模型 {model_name} 就绪（重试成功）"
            except Exception as e2:
                return False, str(e2)
        return False, msg


def _warmup_whisper():
    """启动时：清理损坏缓存，预下载 base 模型（异步）"""
    import time as _time
    deleted = _delete_corrupted_models()
    if deleted:
        print(f"[WHISPER] ✅ 已清理 {len(deleted)} 个损坏模型: {deleted}")

    models = _check_whisper_models()
    missing_or_corrupt = [n for n, i in models.items() if i["status"] in ("missing", "corrupt")]

    if missing_or_corrupt:
        print(f"[WHISPER] 需要下载模型: {missing_or_corrupt}")
        # 只预下载 base（最小可用模型），其他按需下载
        if "base" in missing_or_corrupt:
            print("[WHISPER] 开始预下载 base 模型（约 139 MB，首次需要 1-2 分钟）...")
            ok, msg = _preload_whisper_model("base")
            if ok:
                print(f"[WHISPER] ✅ {msg}")
            else:
                print(f"[WHISPER] ❌ base 下载失败: {msg}")
    else:
        print(f"[WHISPER] ✅ 所有模型缓存正常: {[n for n,i in models.items() if i['status']=='ok']}")

    # 更新全局状态
    global _model_status
    _model_status = _check_whisper_models()


@asynccontextmanager
async def _lifespan(_app):
    for p in (DATA_DIR, OUTPUT_DIR):
        p.mkdir(parents=True, exist_ok=True)

    # 后台预热 whisper 模型（不阻塞服务启动）
    threading.Thread(target=_warmup_whisper, daemon=True).start()
    yield


app = FastAPI(title="AICut", lifespan=_lifespan)

# ── 工具 ──────────────────────────────────────────────

def _run(cmd: list[str], timeout: int | None = None) -> tuple[bool, str]:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="ignore", timeout=timeout)
    except FileNotFoundError:
        return False, "command not found"
    except subprocess.TimeoutExpired:
        return False, "timeout"
    return r.returncode == 0, (r.stdout or "") + (r.stderr or "")


def _ffprobe_duration(path: Path) -> float:
    try:
        r = subprocess.run(["ffprobe", "-v", "error", "-show_entries","format=duration",
            "-of","default=noprint_wrappers=1:nokey=1", str(path)], capture_output=True, text=True, timeout=30)
        return float(r.stdout.strip())
    except Exception:
        return 0.0


def _ffprobe_streams(path: Path) -> dict:
    """获取视频的流信息（有无视频/音频、编码等）"""
    info = {"has_video": False, "has_audio": False, "video_codec": "", "audio_codec": ""}
    try:
        r = subprocess.run([
            "ffprobe", "-v", "error", "-show_entries",
            "stream=codec_type,codec_name", "-of", "json", str(path)
        ], capture_output=True, text=True, timeout=30)
        data = json.loads(r.stdout)
        for s in data.get("streams", []):
            if s.get("codec_type") == "video":
                info["has_video"] = True
                info["video_codec"] = s.get("codec_name", "")
            elif s.get("codec_type") == "audio":
                info["has_audio"] = True
                info["audio_codec"] = s.get("codec_name", "")
    except Exception:
        pass
    return info


def _safe_filename(name: str) -> str:
    """只保留文件名，避免浏览器上传的路径片段污染 data 目录。"""
    cleaned = Path(name or "upload.mp4").name.strip()
    cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", cleaned)
    return cleaned or f"upload_{uuid.uuid4().hex[:8]}.mp4"


def _format_time(sec: float) -> str:
    m, s = divmod(int(sec), 60)
    h, m = divmod(m, 60)
    return f"{h:02}:{m:02}:{s:02}" if h else f"{m:02}:{s:02}"


def _extract_audio(video: Path, audio: Path) -> None:
    subprocess.run(["ffmpeg","-y","-i",str(video),"-q:a","0","-map","a",str(audio)], check=True, capture_output=True, timeout=600)


def _make_browser_audio(video: Path) -> dict:
    """为前端预览生成独立 AAC 音轨，作为 video 内嵌音频不可播时的兜底。"""
    info = _ffprobe_streams(video)
    result = {"ok": False, "filename": "", "path": "", "error": ""}
    if not info["has_audio"]:
        return result

    audio = video.with_name(f"{video.stem}.audio.m4a")
    tmp = video.with_name(f".{video.stem}.audio_tmp_{uuid.uuid4().hex[:8]}.m4a")
    try:
        subprocess.run([
            "ffmpeg", "-y",
            "-i", str(video),
            "-map", "0:a:0",
            "-vn",
            "-c:a", "aac", "-b:a", "160k", "-ac", "2",
            "-movflags", "+faststart",
            str(tmp),
        ], check=True, capture_output=True, timeout=600)
        if not tmp.exists():
            result["error"] = "ffmpeg 未生成独立音轨"
            return result
        audio.unlink(missing_ok=True)
        tmp.rename(audio)
        result.update({"ok": True, "filename": audio.name, "path": str(audio)})
    except subprocess.CalledProcessError as e:
        tmp.unlink(missing_ok=True)
        err = e.stderr.decode(errors="ignore")[-600:] if e.stderr else str(e)
        result["error"] = err
        print(f"[AUDIO-FALLBACK] ffmpeg 失败: {err}")
    except Exception as e:
        tmp.unlink(missing_ok=True)
        result["error"] = str(e)
        print(f"[AUDIO-FALLBACK] 异常: {e}")
    return result


def _ensure_browser_compatible(video: Path) -> dict:
    """
    确保上传素材在浏览器中有稳定画面；有音频时保证音频也可播放。

    策略：
      - 必须有视频流，否则拒绝作为视频素材
      - 只取第一路视频和第一路音频，避免封面图/字幕/数据流被错误映射
      - 视频不兼容时转 H.264 + yuv420p，音频不兼容时转 AAC
      - 输出统一为 MP4，并前置 moov atom（-movflags +faststart）
    """
    info = _ffprobe_streams(video)
    result = {
        "ok": True,
        "path": str(video),
        "filename": video.name,
        "has_video": info["has_video"],
        "has_audio": info["has_audio"],
        "video_codec": info["video_codec"],
        "audio_codec": info["audio_codec"],
        "source_video_codec": info["video_codec"],
        "source_audio_codec": info["audio_codec"],
        "converted": False,
        "error": "",
    }

    if not info["has_video"]:
        result["ok"] = False
        result["error"] = "未检测到视频流，请上传包含画面的文件"
        return result

    final_path = video.with_suffix(".mp4")
    if final_path != video and final_path.exists():
        final_path = video.with_name(f"{video.stem}_{uuid.uuid4().hex[:8]}.mp4")
    tmp = video.with_name(f".{video.stem}.browser_tmp_{uuid.uuid4().hex[:8]}.mp4")
    try:
        needs_video_recode = info["video_codec"] != "h264"
        needs_audio_recode = info["has_audio"] and info["audio_codec"] != "aac"
        cmd = [
            "ffmpeg", "-y",
            "-i", str(video),
            "-map", "0:v:0",
            "-movflags", "+faststart",
        ]

        if needs_video_recode:
            cmd += ["-c:v", "libx264", "-preset", "fast", "-crf", "22", "-pix_fmt", "yuv420p"]
        else:
            cmd += ["-c:v", "copy"]

        if info["has_audio"]:
            cmd += ["-map", "0:a:0"]
            if needs_audio_recode:
                cmd += ["-c:a", "aac", "-b:a", "160k", "-ac", "2"]
            else:
                cmd += ["-c:a", "copy"]
        else:
            cmd += ["-an"]

        cmd.append(str(tmp))

        print(
            f"[UPLOAD] normalize video={info['video_codec']}, audio={info['audio_codec'] or 'none'}, "
            f"recode_v={needs_video_recode}, recode_a={needs_audio_recode}"
        )
        subprocess.run(cmd, check=True, capture_output=True, timeout=600)

        if not tmp.exists():
            result["ok"] = False
            result["error"] = "ffmpeg 未生成输出文件"
            return result

        # ffprobe 验证输出文件
        out_info = _ffprobe_streams(tmp)
        print(f"[UPLOAD] output video={out_info['video_codec']}, audio={out_info['audio_codec'] or 'none'}")
        if not out_info["has_video"]:
            tmp.unlink(missing_ok=True)
            result["ok"] = False
            result["error"] = "兼容处理后未检测到视频流"
            return result

        # 用新文件替换原文件
        old_size = video.stat().st_size
        if final_path == video:
            video.unlink()
        else:
            video.unlink(missing_ok=True)
        tmp.rename(final_path)
        result["converted"] = True
        result["path"] = str(final_path)
        result["filename"] = final_path.name
        result["has_video"] = out_info["has_video"]
        result["has_audio"] = out_info["has_audio"]
        result["video_codec"] = out_info["video_codec"]
        result["audio_codec"] = out_info["audio_codec"]
        result["size_before"] = old_size
        result["size_after"] = final_path.stat().st_size

    except subprocess.CalledProcessError as e:
        tmp.unlink(missing_ok=True)
        err = e.stderr.decode(errors="ignore")[-600:] if e.stderr else str(e)
        result["ok"] = False
        result["error"] = err
        print(f"[AUDIO] ffmpeg 失败: {err}")
    except Exception as e:
        tmp.unlink(missing_ok=True)
        result["ok"] = False
        result["error"] = str(e)
        print(f"[AUDIO] 异常: {e}")

    return result


# ── 波形 ──────────────────────────────────────────────

@app.get("/api/waveform/{filename:path}")
def api_waveform(filename: str):
    """返回音频波形数据（最多 2000 个采样点）"""
    video = DATA_DIR / filename
    if not video.exists():
        return JSONResponse({"error":"file not found"}, status_code=404)
    raw = video.with_suffix(".raw")
    try:
        subprocess.run(["ffmpeg","-y","-i",str(video),"-ac","1","-ar","22050","-f","s16le","-t","600",str(raw)],
            check=True, capture_output=True, timeout=120)
        import numpy as np
        data = np.fromfile(raw, dtype=np.int16).astype(np.float32)
        target = 2000
        if len(data) > target:
            idx = np.linspace(0, len(data)-1, target).astype(int)
            data = data[idx]
        amps = np.abs(data)
        mx = amps.max() or 1
        return JSONResponse({"waveform": (amps / mx).round(4).tolist()})
    except Exception as e:
        return JSONResponse({"waveform": [], "error": str(e)})
    finally:
        raw.unlink(missing_ok=True)


# ── 状态 & 模型 ──────────────────────────────────────

@app.get("/api/status")
def api_status():
    models = _check_whisper_models()
    ok_count = sum(1 for m in models.values() if m["status"] == "ok")
    return JSONResponse({
        "whisper_models": models,
        "whisper_ready": ok_count >= 1,
        "data_dir": str(DATA_DIR),
        "output_dir": str(OUTPUT_DIR),
    })


@app.post("/api/models/prepare")
def api_prepare_model(model: str = Form("base")):
    """预下载/修复指定 whisper 模型"""
    if model not in WHISPER_MODELS:
        return JSONResponse({"error": f"未知模型: {model}，可选: {list(WHISPER_MODELS)}"}, status_code=400)

    current = _check_whisper_models().get(model, {})
    if current.get("status") == "ok":
        return JSONResponse({"model": model, "status": "already_ready", "size_mb": current["size_mb"]})

    tid = _new_task("prepare_model", model)
    def _run():
        try:
            _update_task(tid, progress=20, message=f"正在下载 {model} 模型…")
            ok, msg = _preload_whisper_model(model)
            if ok:
                info = _check_whisper_models().get(model, {})
                _update_task(tid, progress=100, status="completed", message=msg,
                    result={"model": model, "size_mb": info.get("size_mb", 0)})
            else:
                _update_task(tid, status="error", error=msg)
        except Exception as e:
            _update_task(tid, status="error", error=str(e))

    threading.Thread(target=_run, daemon=True).start()
    return JSONResponse({"task_id": tid, "model": model})


# ── 上传 ──────────────────────────────────────────────

@app.post("/api/upload")
async def api_upload(file: UploadFile = File(...)):
    DATA_DIR.mkdir(exist_ok=True)
    original_name = _safe_filename(file.filename)
    target = DATA_DIR / original_name
    with target.open("wb") as buf:
        shutil.copyfileobj(file.file, buf)

    # ── 浏览器兼容性处理 ──
    # 使用 ffmpeg 确保：H.264 视频 + AAC 音频 + moov atom 前置（faststart）
    # 这三个条件保证了所有主流浏览器都能正常播放音频和视频
    compat = _ensure_browser_compatible(target)
    target = Path(compat.get("path") or target)

    if not compat.get("ok", True):
        target.unlink(missing_ok=True)
        return JSONResponse({
            "error": compat.get("error") or "视频兼容处理失败",
            "filename": original_name,
            "has_video": compat.get("has_video", False),
            "has_audio": compat.get("has_audio", False),
            "video_codec": compat.get("video_codec", ""),
            "audio_codec": compat.get("audio_codec", ""),
            "compat_ok": False,
        }, status_code=400)

    dur = _ffprobe_duration(target)
    final_info = _ffprobe_streams(target)
    audio_preview = _make_browser_audio(target) if final_info["has_audio"] else {"ok": False}

    return JSONResponse({
        "filename": target.name,
        "original_filename": original_name,
        "size_bytes": target.stat().st_size,
        "duration": round(dur, 1),
        "duration_str": _format_time(dur),
        "has_audio": final_info["has_audio"],
        "audio_codec": final_info["audio_codec"],
        "has_video": final_info["has_video"],
        "video_codec": final_info["video_codec"],
        "audio_converted": compat.get("converted", False),
        "audio_preview_ok": audio_preview.get("ok", False),
        "audio_preview_filename": audio_preview.get("filename", ""),
        "audio_preview_error": audio_preview.get("error", ""),
        "original_audio_codec": compat.get("source_audio_codec", ""),
        "original_video_codec": compat.get("source_video_codec", ""),
        "compat_ok": compat.get("ok", True),
        "compat_error": compat.get("error", ""),
    })


# ── Step 1: 裁剪 ─────────────────────────────────────

def _bg_cut(tid: str, video_path: Path, segments: list, speed: float):
    tmp = video_path.parent / f".{tid}"
    try:
        tmp.mkdir(exist_ok=True)
        _update_task(tid, progress=10, message="裁剪片段中…")
        clips = []
        for i, (s, e) in enumerate(segments):
            c = tmp / f"c{i:03d}.mp4"
            clips.append(c)
            subprocess.run(["ffmpeg","-y","-i",str(video_path),"-ss",str(s),"-to",str(e),"-c","copy",str(c)],
                check=True, capture_output=True, timeout=600)

        _update_task(tid, progress=40, message="合并片段…")
        merged = video_path.with_suffix(f".cut{video_path.suffix}")
        fl = tmp / "list.txt"
        with open(fl, "w") as f:
            for c in clips:
                f.write(f"file '{c.as_posix()}'\n")
        subprocess.run(["ffmpeg","-y","-f","concat","-safe","0","-i",str(fl),"-c","copy",
            "-movflags","+faststart",str(merged)],
            check=True, capture_output=True, timeout=600)

        dur = _ffprobe_duration(merged)

        if speed != 1.0:
            _update_task(tid, progress=60, message=f"变速 {speed}x…")
            sp = merged.with_suffix(f".sp{speed}{merged.suffix}")
            af = f"atempo={speed}" if speed <= 2 else f"atempo=2.0,atempo={speed/2}"
            subprocess.run(["ffmpeg","-y","-i",str(merged),"-vf",f"setpts={1/speed}*PTS",
                "-filter:a",af,"-c:v","mpeg4","-q:v","5","-c:a","aac","-b:a","192k",
                "-movflags","+faststart",str(sp)],
                check=True, capture_output=True, timeout=600)
            merged.unlink(); merged = sp
            dur = _ffprobe_duration(merged)

        _update_task(tid, progress=80, message="生成浏览器预览资产…")
        compat = _ensure_browser_compatible(merged)
        if compat.get("ok"):
            merged = Path(compat.get("path") or merged)
            dur = _ffprobe_duration(merged)
        audio_preview = _make_browser_audio(merged) if _ffprobe_streams(merged)["has_audio"] else {"ok": False}

        _update_task(tid, progress=100, status="completed", message="裁剪完成",
            result={
                "cut_path": str(merged),
                "duration": round(dur,1),
                "duration_str": _format_time(dur),
                "audio_preview_ok": audio_preview.get("ok", False),
                "audio_preview_filename": audio_preview.get("filename", ""),
                "audio_preview_error": audio_preview.get("error", ""),
            })
    except Exception as e:
        _update_task(tid, status="error", error=str(e))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


@app.post("/api/cut")
async def api_cut(filename: str = Form(...), segments: str = Form(...), speed: float = Form(1.0)):
    video = DATA_DIR / filename
    if not video.exists():
        return JSONResponse({"error":"文件不存在"}, status_code=404)
    parsed = []
    for part in re.split(r"[,，\s]+", segments.strip()):
        m = re.match(r"(\d+\.?\d*)\s*[-~至到]\s*(\d+\.?\d*)", part.strip())
        if m:
            s, e = float(m.group(1)), float(m.group(2))
            if s < e: parsed.append((s, e))
    if not parsed:
        return JSONResponse({"error":"时间段无效"}, status_code=400)
    tid = _new_task("cut", filename)
    threading.Thread(target=_bg_cut, args=(tid, video, parsed, speed), daemon=True).start()
    return JSONResponse({"task_id": tid, "segments": len(parsed)})


# ── Step 2: 转录 ─────────────────────────────────────

def _bg_transcribe(tid: str, video_path: Path, asr_model: str, language: str):
    try:
        _update_task(tid, progress=10, message="提取音频…")
        audio = video_path.with_suffix(".mp3")
        _extract_audio(video_path, audio)

        _update_task(tid, progress=30, message=f"Whisper {asr_model} 转写中…")
        from src.subtitle_tools import transcribe_audio, generate_srt, generate_json
        segs = transcribe_audio(str(audio), model_name=asr_model, language=language, device="cpu")

        srt = video_path.with_suffix(".srt")
        jsn = video_path.with_suffix(".json")
        generate_srt(segs, srt)
        generate_json(segs, jsn)
        text = "\n".join(s["text"] for s in segs)

        audio.unlink(missing_ok=True)
        _update_task(tid, progress=100, status="completed",
            message=f"转写完成：{len(segs)} 条",
            result={"segments_count": len(segs), "transcript": text, "srt_path": str(srt), "json_path": str(jsn)})
    except Exception as e:
        _update_task(tid, status="error", error=str(e))


@app.post("/api/transcribe")
async def api_transcribe(video_path: str = Form(...), asr_model: str = Form("base"), language: str = Form("zh")):
    vp = Path(video_path)
    if not vp.exists():
        return JSONResponse({"error":"视频文件不存在"}, status_code=404)
    tid = _new_task("transcribe", vp.name)
    threading.Thread(target=_bg_transcribe, args=(tid, vp, asr_model, language), daemon=True).start()
    return JSONResponse({"task_id": tid})


# ── Step 3: 字幕实时预览（截图模式） ──────────────────


# ASS 颜色格式：&HAABBGGRR
_ALIGN_MAP = {
    "bl": 1, "bc": 2, "br": 3,
    "ml": 4, "mc": 5, "mr": 6,
    "tl": 7, "tc": 8, "tr": 9,
}


def _html_color_to_ass(html_color: str) -> str:
    """将 #RRGGBB 转为 ASS 格式 &H00BBGGRR"""
    c = html_color.strip().lstrip("#")
    if not c or len(c) < 6:
        return "&H00FFFFFF"
    r, g, b = c[0:2], c[2:4], c[4:6]
    return f"&H00{b}{g}{r}"


@app.post("/api/preview-subs")
async def api_preview_subs(
    video_path: str = Form(...),
    srt_path: str = Form(...),
    font_size: int = Form(16),
    font_color: str = Form("#FFFFFF"),
    outline_color: str = Form("#000000"),
    outline_width: float = Form(0.85),
    shadow_depth: float = Form(0.55),
    bold: bool = Form(True),
    italic: bool = Form(False),
    alignment: str = Form("bc"),   # bl/bc/br/ml/mc/mr/tl/tc/tr
    margin_v: int = Form(40),
    seek_time: float = Form(3.0),   # 取第几秒的截图
):
    """生成一帧字幕截图（极速，1-2秒），直接返回图片"""
    vp, sp = Path(video_path), Path(srt_path)
    if not vp.exists() or not sp.exists():
        return JSONResponse({"error": "文件不存在"}, status_code=404)

    align = _ALIGN_MAP.get(alignment, 2)
    primary = _html_color_to_ass(font_color)
    outline_col = _html_color_to_ass(outline_color)
    fs = max(8, min(72, font_size))
    ow = max(0, min(5, outline_width))
    sd = max(0, min(5, shadow_depth))
    b = "1" if bold else "0"
    it = "1" if italic else "0"

    style = (
        f"FontName=MingLiU,FontSize={fs},"
        f"PrimaryColour={primary},OutlineColour={outline_col},"
        f"Bold={b},Italic={it},"
        f"Outline={ow},Shadow={sd},"
        f"Alignment={align},MarginV={margin_v}"
    )

    # ── 关键：用相对路径避免 filter 解析冒号 ──
    # 把 SRT 复制到 DATA_DIR（视频也在那里），使用简单文件名
    # subprocess cwd=DATA_DIR，filter 里只写文件名 → 没有 D: 盘符问题
    TEMP_SRT_NAME = "_sub_preview.srt"
    temp_srt = DATA_DIR / TEMP_SRT_NAME
    try:
        shutil.copy2(sp, temp_srt)
        # 验证 SRT 内容
        srt_size = temp_srt.stat().st_size
        if srt_size < 10:
            return JSONResponse({"error": f"字幕文件内容异常（仅 {srt_size} 字节）"}, status_code=400)
        print(f"[PREVIEW-SUBS] SRT 已复制: {srt_size} 字节, seek={seek_time}s, video={vp.name}")
    except Exception as e:
        return JSONResponse({"error": f"无法读取字幕文件: {e}"}, status_code=500)

    out = DATA_DIR / f"sub_prev_{os.getpid()}.png"  # 不用 . 开头避免隐藏文件

    # 检查 seek_time 是否超过视频时长
    dur = _ffprobe_duration(vp)
    if dur > 0 and seek_time >= dur:
        seek_time = max(0, dur - 0.5)
        print(f"[PREVIEW-SUBS] seek_time 超出视频时长({dur}s)，调整为 {seek_time}s")

    cmd = [
        "ffmpeg", "-y",
        "-ss", str(seek_time),
        "-i", str(vp),
        "-vframes", "1",
        "-vf", f"subtitles={TEMP_SRT_NAME}:force_style='{style}'",
        str(out),
    ]
    print(f"[PREVIEW-SUBS] 运行: {' '.join(cmd)}")

    try:
        subprocess.run(
            cmd, check=True, capture_output=True, timeout=30, cwd=str(DATA_DIR)
        )

        if not out.exists():
            return JSONResponse({"error": "截图生成失败"}, status_code=500)

        from fastapi.responses import FileResponse as FRImg
        resp = FRImg(
            out,
            media_type="image/png",
            headers={"Cache-Control": "no-cache"},
        )
        return resp
    except subprocess.CalledProcessError as e:
        full = e.stderr.decode(errors="ignore") if e.stderr else str(e)
        # ffmpeg 版本信息太长，取最后 1500 字符（真正的错误在末尾）
        tail = full[-1500:].strip()
        # 同时在控制台打印完整错误便于调试
        print(f"[PREVIEW-SUBS] ❌ ffmpeg 退出码 {e.returncode}")
        print(f"[PREVIEW-SUBS]   命令: {' '.join(e.cmd)}" if hasattr(e, 'cmd') else "")
        print(f"[PREVIEW-SUBS]   错误尾: {tail}")
        return JSONResponse({"error": f"字幕渲染失败(exit {e.returncode}): {tail}"}, status_code=500)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
    finally:
        temp_srt.unlink(missing_ok=True)
        # 清理预览截图（延迟 5 秒，以免响应还没返回）
        if out.exists():
            threading.Timer(5.0, lambda: out.unlink(missing_ok=True)).start()


# ── BGM 扫描 ─────────────────────────────────────────# ── BGM 扫描 ─────────────────────────────────────────

@app.get("/api/scan-bgm")
def api_scan_bgm(folder: str = ""):
    folder = folder.strip()
    if not folder or not Path(folder).exists():
        return JSONResponse({"files": [], "error": "目录不存在"})
    files = []
    for f in sorted(Path(folder).iterdir()):
        if f.suffix.lower() in (".mp3", ".wav", ".flac", ".m4a"):
            files.append({"name": f.name, "path": str(f), "size": f.stat().st_size})
    return JSONResponse({"files": files})


# ── BGM 预览 ─────────────────────────────────────────

@app.post("/api/preview-bgm")
async def api_preview_bgm(
    video_path: str = Form(...), bgm_path: str = Form(""),
    bgm_volume: float = Form(0.30), video_volume: float = Form(1.0),
    duration: int = Form(10), start_at: float = Form(0.0),
):
    vp = Path(video_path)
    if not vp.exists():
        return JSONResponse({"error":"视频不存在"}, status_code=404)

    bgm_exists = bgm_path and Path(bgm_path).exists()
    print(f"[BGM-PREVIEW] 请求: video={vp.name}, bgm={bgm_path}, bgm_exists={bgm_exists}, vol={bgm_volume}")

    tid = _new_task("preview_bgm", vp.name)

    def _run():
        try:
            _update_task(tid, progress=20, message="混音预览中…")
            out = vp.with_suffix(f".bgm_prev{vp.suffix}")
            snippet = vp.with_suffix(".snippet.mp4")
            subprocess.run(["ffmpeg","-y","-i",str(vp),"-ss",str(start_at),"-t",str(duration),str(snippet)],
                check=True, capture_output=True, timeout=120)

            if bgm_path and Path(bgm_path).exists():
                # 使用 aloop 循环 BGM（和用户源代码一致，已验证稳定）
                fc = (
                    f"[0:a]volume={video_volume}[va];"
                    f"[1:a]aloop=loop=-1:size=0:start=0,volume={bgm_volume}[bgm];"
                    "[va][bgm]amix=inputs=2:duration=longest[aout]"
                )
                print(f"[BGM-PREVIEW] bgm={bgm_path}, vol={bgm_volume}, video_vol={video_volume}")
                subprocess.run([
                    "ffmpeg","-y",
                    "-i",str(snippet),"-i",str(bgm_path),
                    "-filter_complex", fc,
                    "-map","0:v","-map","[aout]",
                    "-c:v","copy","-c:a","aac",
                    "-shortest",
                    "-movflags","+faststart",str(out),
                ], check=True, capture_output=True, timeout=300)
                print(f"[BGM-PREVIEW] ✅ 生成成功: {out.name}")
            else:
                shutil.copy2(snippet, out)
            snippet.unlink(missing_ok=True)
            _update_task(tid, progress=100, status="completed", message="BGM 预览已就绪",
                result={"preview_path": str(out), "duration": duration})
        except subprocess.CalledProcessError as e:
            err = e.stderr.decode(errors="ignore")[-800:] if e.stderr else str(e)
            print(f"[BGM-PREVIEW] ❌ ffmpeg 失败: {err}")
            _update_task(tid, status="error", error=f"ffmpeg 失败: {err}")
        except Exception as e:
            print(f"[BGM-PREVIEW] ❌ 异常: {e}")
            _update_task(tid, status="error", error=str(e))

    threading.Thread(target=_run, daemon=True).start()
    return JSONResponse({"task_id": tid})


# ── Step 4: 导出 ─────────────────────────────────────

def _bg_export(tid, video_path, srt_path, bgm_path, speed, bgm_vol, video_vol, burn_subs, export_format="mp4"):
    tmp = video_path.parent / f".{tid}"
    try:
        tmp.mkdir(exist_ok=True)
        current = video_path

        # 非 mp4 格式：直接输出
        if export_format == "mp3":
            _update_task(tid, progress=10, message="提取音频…")
            out = OUTPUT_DIR / f"{video_path.stem}_final.mp3"
            subprocess.run(["ffmpeg","-y","-i",str(current),"-vn","-c:a","libmp3lame","-q:a","2",str(out)],
                check=True, capture_output=True, timeout=600)
            dur = _ffprobe_duration(out) if out.exists() else 0
            _update_task(tid, progress=100, status="completed", message=f"导出完成：{out.name}",
                result={"output_path": str(out), "output_name": out.name, "duration": round(dur,1)})
            return

        if export_format == "gif":
            _update_task(tid, progress=10, message="生成 GIF…")
            out = OUTPUT_DIR / f"{video_path.stem}_final.gif"
            subprocess.run(["ffmpeg","-y","-i",str(current),"-vf","fps=10,scale=480:-1:flags=lanczos","-loop","0",str(out)],
                check=True, capture_output=True, timeout=600)
            dur = _ffprobe_duration(out) if out.exists() else 0
            _update_task(tid, progress=100, status="completed", message=f"导出完成：{out.name}",
                result={"output_path": str(out), "output_name": out.name, "duration": round(dur,1)})
            return

        if speed != 1.0:
            _update_task(tid, progress=10, message=f"变速 {speed}x…")
            sp = tmp / f"sp{speed}{current.suffix}"
            af = f"atempo={speed}" if speed <= 2 else f"atempo=2.0,atempo={speed/2}"
            subprocess.run(["ffmpeg","-y","-i",str(current),"-vf",f"setpts={1/speed}*PTS",
                "-filter:a",af,"-c:v","mpeg4","-q:v","5","-c:a","aac","-b:a","192k",
                "-movflags","+faststart",str(sp)],
                check=True, capture_output=True, timeout=600)
            current = sp

        if bgm_path and Path(bgm_path).exists():
            _update_task(tid, progress=30, message="混入 BGM…")
            bg = tmp / f"bgm{current.suffix}"
            fc = f"[0:a]volume={video_vol}[va];[1:a]aloop=loop=-1:size=0:start=0,volume={bgm_vol}[bgm];[va][bgm]amix=inputs=2:duration=longest[aout]"
            subprocess.run(["ffmpeg","-y","-i",str(current),"-i",str(bgm_path),
                "-filter_complex",fc,"-map","0:v","-map","[aout]","-c:v","mpeg4","-q:v","5","-c:a","aac",
                "-movflags","+faststart",str(bg)],
                check=True, capture_output=True, timeout=600)
            current.unlink(); current = bg

        if burn_subs and srt_path and Path(srt_path).exists():
            _update_task(tid, progress=50, message="烧录字幕…")
            sb = tmp / f"sub{current.suffix}"
            style = "FontSize=13,FontName=MingLiU,PrimaryColour=&H4FA5FF,Outline=0.85,Shadow=0.55,Bold=1,Alignment=2"
            # 复制 SRT 到临时目录，用相对路径避免 filter 冒号解析问题
            temp_srt = tmp / "_sub.srt"
            shutil.copy2(srt_path, temp_srt)
            subprocess.run(["ffmpeg","-y","-i",str(current),"-vf",f"subtitles=_sub.srt:force_style='{style}'",
                "-c:v","mpeg4","-q:v","5","-c:a","aac",
                "-movflags","+faststart",str(sb)],
                check=True, capture_output=True, timeout=600, cwd=str(tmp))
            current.unlink(); current = sb

        _update_task(tid, progress=80, message="生成最终文件…")
        final = OUTPUT_DIR / f"{video_path.stem}_final{current.suffix}"
        shutil.copy2(current, final)
        dur = _ffprobe_duration(final)
        _update_task(tid, progress=100, status="completed", message=f"导出完成：{final.name}",
            result={"output_path": str(final), "output_name": final.name, "duration": round(dur,1)})
    except Exception as e:
        _update_task(tid, status="error", error=str(e))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


@app.post("/api/export")
async def api_export(
    video_path: str = Form(...), srt_path: str = Form(""),
    bgm_path: str = Form(""), speed: float = Form(1.0),
    bgm_volume: float = Form(0.30), video_volume: float = Form(1.0),
    burn_subtitles: bool = Form(True),
    export_format: str = Form("mp4"),
):
    vp = Path(video_path)
    if not vp.exists():
        return JSONResponse({"error":"视频不存在"}, status_code=404)
    tid = _new_task("export", vp.name)
    threading.Thread(target=_bg_export, args=(tid, vp, srt_path, bgm_path, speed, bgm_volume, video_volume, burn_subtitles, export_format), daemon=True).start()
    return JSONResponse({"task_id": tid})


# ── 任务查询 ─────────────────────────────────────────

@app.get("/api/task/{task_id}")
def api_task(task_id: str):
    t = _get_task(task_id)
    return JSONResponse(t) if t else JSONResponse({"error":"not found"}, status_code=404)


@app.get("/api/tasks")
def api_tasks():
    with _lock:
        return JSONResponse({"tasks": list(_tasks.values())})


# ── 视频服务 — 用 FileResponse 原生支持 Range ─────

@app.get("/api/video/{filename:path}")
async def api_video(filename: str):
    """视频文件服务"""
    safe_name = filename.replace("\\", "/").lstrip("/")
    # cache_bust 可能为 ?t=xxx
    if "?" in safe_name:
        safe_name = safe_name.split("?")[0]
    path = DATA_DIR / safe_name
    if not path.exists() or not path.is_file():
        return JSONResponse({"error": "not found"}, status_code=404)

    suffix = path.suffix.lower()
    media_type = {
        ".mp4": "video/mp4", ".mov": "video/quicktime",
        ".avi": "video/x-msvideo", ".mkv": "video/x-matroska",
        ".webm": "video/webm", ".m4v": "video/mp4",
        ".m4a": "audio/mp4", ".aac": "audio/aac", ".mp3": "audio/mpeg",
    }.get(suffix, "application/octet-stream")

    return FileResponse(
        path,
        media_type=media_type,
        filename=path.name,
        headers={
            "Accept-Ranges": "bytes",
            "Cache-Control": "no-cache, no-store, must-revalidate",
        },
    )


# ── 打开输出文件夹 ─────────────────────────────────────

@app.post("/api/open-output")
def api_open_output():
    """在文件管理器中打开输出目录"""
    import subprocess as _sp
    try:
        _sp.run(["explorer", str(OUTPUT_DIR)], check=True, timeout=10)
        return JSONResponse({"ok": True})
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)})


# ── 视频缩略图（用于片段预览）─────────────────────────────

@app.get("/api/thumbnail")
def api_thumbnail(video: str = "", time: float = 0):
    """返回视频指定时间点的一帧缩略图"""
    vp = Path(video)
    if not vp.exists():
        vp = DATA_DIR / video
    if not vp.exists():
        return JSONResponse({"error":"not found"}, status_code=404)
    out = DATA_DIR / f".thumb_{os.getpid()}.jpg"
    try:
        subprocess.run([
            "ffmpeg","-y","-ss",str(time),"-i",str(vp),
            "-vframes","1","-s","320x180","-q:v","10",str(out),
        ], check=True, capture_output=True, timeout=30)
        if not out.exists():
            return JSONResponse({"error":"generate failed"}, status_code=500)
        resp = FileResponse(out, media_type="image/jpeg", headers={"Cache-Control":"no-cache"})
        return resp
    except Exception as e:
        return JSONResponse({"error":str(e)}, status_code=500)
    finally:
        threading.Timer(5.0, lambda: out.unlink(missing_ok=True)).start()


# ── 首页 ──────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def index():
    return TEMPLATE_PATH.read_text(encoding="utf-8")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT)
