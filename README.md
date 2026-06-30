# AIClip — 智能直播切片工具

## 背景

六年之约。

一位主播和他的观众之间有个约定——用六年的时间，一起做点有意思的事。

这个项目就是为此而生。

AIClip 是一个本地运行的直播切片工具。它让你可以轻松地把长直播视频，自动裁剪成一个个高光片段，配上字幕，加上背景音乐，导出成品。

不需要上传到任何云端，一切都在你自己电脑上运行。

---

## 快速开始（小白版）

> 你只需要会双击鼠标就行。

**双击 start.bat**

把整个文件夹解压后，找到 `start.bat`，双击它。

脚本会自动：
- 检测 conda → 自动创建 `aiclip` 环境（已存在则直接使用）
- 检测 ffmpeg
- 安装所需的依赖包（已安装则跳过）
- 启动网页服务 + 自动打开浏览器

> 首次运行约 10-15 分钟（下载 PyTorch + Whisper 依赖）
> 之后每次双击，5 秒就进入浏览器 🚀

---

## 手动安装（给懂技术的朋友）

### 环境要求

| 组件 | 版本要求 |
|------|---------|
| Python | ≥ 3.10（推荐 3.13）|
| pip | 最新版 |
| ffmpeg | ≥ 6.0，需 `--enable-libass` |

### 安装步骤

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 下载 ffmpeg（完整版，不是精简版）
#    从 https://www.gyan.dev/ffmpeg/builds/
#    下载 ffmpeg-release-full.7z
#    解压后把 ffmpeg.exe 和 ffprobe.exe 放到项目目录

# 3. 启动
python -m uvicorn src.app:app --host 127.0.0.1 --port 3801 --reload

# 4. 打开浏览器访问
#     http://127.0.0.1:3801
```

### 一键检查

如果遇到问题，双击 `check_env.bat`，它会告诉你哪个环节没配置好。

---

## 功能

- 🎬 **视频上传** — 支持 MP4 / MOV / AVI
- 🎤 **语音识别** — 基于 Whisper，自动转写为字幕
- 📝 **字幕编辑** — 调整字体、颜色、大小、位置、5 套预设一键切换
- ✂️ **智能裁剪** — 波形可视化，快捷键（Space/←→/Delete），吸附对齐
- 🎵 **背景音乐** — 自定义 BGM，调节音量比例
- 🚀 **多格式导出** — MP4 视频 / MP3 音频 / GIF 动图

---

## 技术栈

| 模块 | 技术 |
|------|------|
| 后端 | FastAPI + Uvicorn |
| 前端 | 纯 HTML + CSS + JS |
| 语音识别 | OpenAI Whisper |
| 视频处理 | ffmpeg |
| 运行环境 | Python 3.13 |

---

## 项目结构

```
AIClip/
├── start.bat            # 一键启动（双击运行）
├── check_env.bat        # 环境检查
├── requirements.txt     # 依赖清单
├── data/                # 上传的视频
│   └── 视频保存的位置.txt
├── output/              # 导出的成品
│   └── 视频导出目录.txt
├── src/                 # 源代码
│   ├── app.py           # 主程序
│   ├── index.html       # 网页界面
│   ├── subtitle_tools.py
│   └── pipeline.py
└── utils/               # 辅助工具
```

---

## License

MIT
