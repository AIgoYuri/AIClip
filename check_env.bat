@echo off
chcp 65001 >nul
title AICut Environment Check
echo =============================================
echo   AICut Environment Check
echo =============================================
echo.
set ERR=0

:: 1. Python
python --version >nul 2>&1
if %errorlevel% equ 0 (
    python --version 2>&1
) else (
    echo [FAIL] Python not found in PATH
    set ERR=1
)

:: 2. pip
pip --version >nul 2>&1
if %errorlevel% equ 0 (
    echo [PASS] pip is ready
) else (
    echo [FAIL] pip not found
    set ERR=1
)

:: 3. ffmpeg
ffmpeg -version >nul 2>&1
if %errorlevel% neq 0 (
    echo [FAIL] ffmpeg not found in PATH
    set ERR=1
) else (
    ffmpeg -version 2>&1 | findstr "libass" >nul
    if %errorlevel% equ 0 (
        echo [PASS] ffmpeg with libass
    ) else (
        echo [WARN] ffmpeg missing libass
    )
)

:: 4. Python dependencies
python -c "import whisper; print('[PASS] Whisper', whisper.__version__)" 2>&1
if %errorlevel% neq 0 ( echo [FAIL] whisper missing & set ERR=1 )

python -c "import fastapi; print('[PASS] FastAPI', fastapi.__version__)" 2>&1
if %errorlevel% neq 0 ( echo [FAIL] fastapi missing & set ERR=1 )

python -c "import uvicorn; print('[PASS] Uvicorn', uvicorn.__version__)" 2>&1
if %errorlevel% neq 0 ( echo [FAIL] uvicorn missing & set ERR=1 )

:: 5. Whisper model cache
python -c "import os,glob; cache=os.path.expanduser('~/.cache/whisper'); files=glob.glob(cache+'/*.pt'); [print('[PASS] Model:', os.path.basename(f), '(' + str(round(os.path.getsize(f)/1024/1024)) + ' MB)') for f in files] or print('[INFO] No cached model (auto-download on first use)')" 2>&1

:: 6. Project files
if exist "%~dp0src\app.py" (
    if exist "%~dp0src\index.html" (
        echo [PASS] Project files ready
    )
) else (
    echo [FAIL] Run this script from AICut/opensource/
    set ERR=1
)

echo.
if %ERR% equ 0 (
    echo =============================================
    echo   All checks passed! Ready to start.
    echo =============================================
    echo.
    echo   Start:  python -m uvicorn src.app:app --host 127.0.0.1 --port 3801 --reload
    echo   Or:     double-click start.bat
    echo   Open:   http://127.0.0.1:3801
) else (
    echo =============================================
    echo   Some checks failed. Fix errors above.
    echo =============================================
)
echo.
pause
