#!/usr/bin/env python3
"""
DeepGuard Quick Start Script
Run this from the project root to start the API.
"""
import os
import sys
import subprocess

print("=" * 55)
print("  🛡️  DEEPGUARD — DEEPFAKE DETECTION SYSTEM")
print("=" * 55)

# Check Python version
if sys.version_info < (3, 9):
    print("❌ Python 3.9+ required")
    sys.exit(1)

print(f"✅ Python {sys.version.split()[0]}")

# Check if model exists
model_path = os.path.join("backend", "model", "deepfake_detector.h5")
if not os.path.exists(model_path):
    print(f"\n⚠️  Model not found at: {model_path}")
    print("   Creating demo model (requires internet for ImageNet weights)...")
    result = subprocess.run(
        [sys.executable, os.path.join("backend", "model", "create_demo_model.py")],
        capture_output=False
    )
    if result.returncode != 0:
        print("❌ Failed to create demo model. Install requirements first:")
        print("   pip install -r requirements.txt")
        sys.exit(1)
else:
    print(f"✅ Model found: {model_path}")

print("\n🚀 Starting FastAPI backend on http://localhost:8000")
print("   API docs: http://localhost:8000/api/docs")
print("   Frontend: open frontend/index.html in your browser")
print("\n   Press CTRL+C to stop\n")
print("=" * 55 + "\n")

os.chdir(os.path.dirname(os.path.abspath(__file__)))

subprocess.run([
    sys.executable, "-m", "uvicorn",
    "backend.api.main:app",
    "--reload",
    "--host", "0.0.0.0",
    "--port", "8000",
    "--log-level", "info",
])
