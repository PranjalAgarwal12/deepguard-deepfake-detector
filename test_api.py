"""
========================================================
  DeepGuard — Quick Test Script
  Tests the API with sample images.
  Run: python test_api.py
  Requires: backend running on localhost:8000
========================================================
"""

import os
import sys
import json
import urllib.request
from pathlib import Path

API_BASE = "http://localhost:8000"


def check_health():
    print("\n[1/3] Health Check…")
    try:
        with urllib.request.urlopen(f"{API_BASE}/health") as r:
            data = json.loads(r.read())
        status = data.get("status", "unknown")
        model_ready = data.get("model_ready", False)
        print(f"  Status      : {status}")
        print(f"  Model Ready : {model_ready}")
        return model_ready
    except Exception as e:
        print(f"  ✗ API not reachable: {e}")
        print("  → Make sure backend is running: uvicorn backend.api.main:app --reload")
        return False


def test_with_url_image():
    """Download a sample face image and test prediction."""
    print("\n[2/3] Testing /predict endpoint…")

    # A small public-domain face image for testing
    img_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/1/14/Gatto_europeo4.jpg/220px-Gatto_europeo4.jpg"
    img_path = Path("/tmp/test_image.jpg")

    try:
        urllib.request.urlretrieve(img_url, img_path)
        print(f"  Downloaded test image → {img_path}")
    except Exception as e:
        print(f"  ✗ Could not download test image: {e}")
        print("  → Place a face image at /tmp/test_image.jpg and re-run")
        return

    test_predict(img_path)


def test_predict(img_path: Path):
    import urllib.request
    import http.client
    import mimetypes
    import uuid

    print(f"\n  Sending {img_path.name} to /predict…")

    boundary = uuid.uuid4().hex
    with open(img_path, "rb") as f:
        img_data = f.read()

    content_type, _ = mimetypes.guess_type(str(img_path))
    content_type = content_type or "image/jpeg"

    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{img_path.name}"\r\n'
        f"Content-Type: {content_type}\r\n\r\n"
    ).encode() + img_data + f"\r\n--{boundary}--\r\n".encode()

    req = urllib.request.Request(
        f"{API_BASE}/predict",
        data=body,
        headers={
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "Content-Length": str(len(body)),
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req) as r:
            result = json.loads(r.read())
        print("\n  ╔══════════════════════════════════╗")
        print(f"  ║  LABEL      : {result['label']:<20}║")
        print(f"  ║  CONFIDENCE : {result['confidence']*100:.1f}%{' '*15}║")
        print(f"  ║  REAL PROB  : {result['real_prob']*100:.1f}%{' '*15}║")
        print(f"  ║  FAKE PROB  : {result['fake_prob']*100:.1f}%{' '*15}║")
        print(f"  ║  LATENCY    : {result['latency_ms']} ms{' '*12}║")
        print("  ╚══════════════════════════════════╝")
        print(f"\n  Verdict: {result['verdict']}")
    except urllib.error.HTTPError as e:
        body = e.read().decode()
        print(f"  ✗ HTTP {e.code}: {body}")
    except Exception as e:
        print(f"  ✗ Error: {e}")


def print_curl_examples():
    print("\n[3/3] cURL Examples:")
    print("""
  # Health check
  curl http://localhost:8000/health

  # Predict
  curl -X POST http://localhost:8000/predict \\
       -F "file=@/path/to/image.jpg"

  # Predict + Grad-CAM
  curl -X POST http://localhost:8000/gradcam \\
       -F "file=@/path/to/image.jpg"

  # Interactive Swagger UI
  open http://localhost:8000/docs
""")


if __name__ == "__main__":
    print("=" * 50)
    print("  DeepGuard API Test Suite")
    print("=" * 50)

    model_ready = check_health()
    if model_ready:
        test_with_url_image()
    else:
        print("\n  ⚠  Model not ready — skipping prediction test")
        print("  → Train the model first: python backend/model/train.py")

    print_curl_examples()
    print("\n✅  Test run complete.\n")
