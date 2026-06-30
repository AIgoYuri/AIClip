@echo off
cd /d %~dp0
echo Starting AIClip...
python -m uvicorn src.app:app --host 127.0.0.1 --port 3801 --reload
pause
