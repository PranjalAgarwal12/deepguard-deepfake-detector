@echo off
REM ════════════════════════════════════════════
REM  DeepGuard — Windows Startup Script
REM  Usage: double-click start.bat or run in CMD
REM ════════════════════════════════════════════

echo.
echo  DeepGuard - Deepfake Detection System
echo  ======================================
echo.

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo  ERROR: Python not found. Install from python.org
    pause
    exit /b 1
)
echo  Python found.

REM Create venv
if not exist "venv" (
    echo  Creating virtual environment...
    python -m venv venv
)

REM Activate
call venv\Scripts\activate.bat
echo  Virtual environment activated.

REM Install deps
echo  Installing dependencies...
pip install -q -r backend\requirements.txt
echo  Dependencies installed.

REM Check model
if exist "backend\model\deepfake_detector.h5" (
    echo  Model found.
) else (
    echo  WARNING: Model NOT found at backend\model\deepfake_detector.h5
    echo  Run: python backend\model\train.py
)

REM Start API
echo.
echo  Starting API on http://localhost:8000
echo  Open frontend\index.html in your browser
echo.

cd backend
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

pause
