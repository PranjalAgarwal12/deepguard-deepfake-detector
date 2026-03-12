from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import time
import uuid
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils.predictor import DeepfakePredictor
from utils.logger import setup_logger

logger = setup_logger(__name__)
router = APIRouter()
predictor = DeepfakePredictor()


@router.post("/predict")
async def predict(file: UploadFile = File(...)):
    allowed_types = {"image/jpeg", "image/png", "image/webp", "image/jpg"}
    if file.content_type not in allowed_types:
        raise HTTPException(status_code=400, detail="Invalid file type.")

    image_bytes = await file.read()
    if len(image_bytes) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large. Max 10MB.")

    start_time = time.time()
    result = predictor.predict(image_bytes)
    latency_ms = round((time.time() - start_time) * 1000, 2)

    prediction = result["prediction"]
    confidence = result["confidence"]

    # Match the frontend's expected format
    is_real = prediction == "Real"
    real_prob = confidence if is_real else 1.0 - confidence
    fake_prob = 1.0 - real_prob
    label = "REAL" if is_real else "FAKE"

    if is_real:
        verdict = "This image appears authentic. No significant manipulation artifacts detected."
    else:
        verdict = "This image shows signs of AI manipulation. Facial inconsistencies detected."

    return JSONResponse({
        "label":      label,
        "prediction": prediction,
        "confidence": round(confidence, 4),
        "real_prob":  round(real_prob, 4),
        "fake_prob":  round(fake_prob, 4),
        "verdict":    verdict,
        "request_id": str(uuid.uuid4())[:8],
        "latency_ms": latency_ms,
        "filename":   file.filename,
    })


@router.post("/gradcam")
async def gradcam(file: UploadFile = File(...)):
    # For now return same as predict (gradcam needs extra setup)
    return await predict(file)