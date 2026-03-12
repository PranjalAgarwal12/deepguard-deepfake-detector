import os
import sys
import time
import logging

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from model.predict import predict, generate_gradcam, load_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="DeepGuard API",
    description="Deepfake Image Detection using MobileNetV2",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    try:
        load_model()
        logger.info("Model pre-loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


@app.get("/api/health")
async def health():
    return {"status": "ok", "model": "MobileNetV2-v1"}


@app.post("/api/predict")
async def predict_endpoint(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        result = predict(image_bytes)
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/gradcam")
async def gradcam_endpoint(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        gradcam_b64 = generate_gradcam(image_bytes)
        return JSONResponse(content={"gradcam_image": gradcam_b64})
    except Exception as e:
        logger.error(f"Grad-CAM error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/analyze")
async def analyze_endpoint(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()

        # Get prediction
        result = predict(image_bytes)

        # Get Grad-CAM
        try:
            gradcam_b64 = generate_gradcam(image_bytes)
            result["gradcam_image"] = gradcam_b64
        except Exception as e:
            logger.warning(f"Grad-CAM skipped: {e}")
            result["gradcam_image"] = None

        return JSONResponse(content=result)

    except Exception as e:
        logger.error(f"Analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
