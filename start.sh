#!/bin/bash
# ════════════════════════════════════════════
#  DeepGuard — Local Development Startup
#  Usage: bash start.sh
# ════════════════════════════════════════════

set -e

echo ""
echo "╔══════════════════════════════════════╗"
echo "║     DeepGuard — Starting Up…         ║"
echo "╚══════════════════════════════════════╝"
echo ""

# 1. Check Python
if ! command -v python3 &> /dev/null; then
    echo "✗ Python 3 not found. Install from python.org"
    exit 1
fi
echo "✔  Python: $(python3 --version)"

# 2. Create venv if not exists
if [ ! -d "venv" ]; then
    echo "  Creating virtual environment…"
    python3 -m venv venv
fi

# 3. Activate
source venv/bin/activate
echo "✔  Virtual environment activated"

# 4. Install dependencies
echo "  Installing backend dependencies…"
pip install -q -r backend/requirements.txt
echo "✔  Dependencies installed"

# 5. Check model
MODEL="backend/model/deepfake_detector.h5"
if [ -f "$MODEL" ]; then
    echo "✔  Model found: $MODEL"
else
    echo "⚠  Model NOT found at $MODEL"
    echo "   → Run: python backend/model/train.py"
    echo "   → Or place a pretrained .h5 file there"
    echo ""
fi

# 6. Start API
echo ""
echo "▶  Starting FastAPI server on http://localhost:8000"
echo "   API Docs → http://localhost:8000/docs"
echo "   Frontend → Open frontend/index.html in browser"
echo ""
echo "   Press Ctrl+C to stop."
echo ""

cd backend && uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
