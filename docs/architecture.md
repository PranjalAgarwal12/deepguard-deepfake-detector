# Technical Architecture — DeepGuard

## System Workflow

```
┌─────────────────────────────────────────────────────┐
│                  USER INTERFACE                      │
│  (frontend/index.html — HTML + CSS + JavaScript)     │
│                                                      │
│  1. User drags/drops or selects image               │
│  2. Preview shown with file info                    │
│  3. Click "Analyze" → POST to /api/analyze          │
│  4. Loading animation with step indicators          │
│  5. Result card: verdict, confidence, Grad-CAM      │
└───────────────────────┬─────────────────────────────┘
                        │ HTTP POST multipart/form-data
                        ▼
┌─────────────────────────────────────────────────────┐
│                 FASTAPI BACKEND                      │
│  (backend/api/main.py)                               │
│                                                      │
│  POST /api/analyze                                   │
│  ├─ Validate file type & size                       │
│  ├─ Call predict() from predict.py                  │
│  ├─ Call gradcam_to_bytes() from predict.py         │
│  ├─ Generate analysis text                          │
│  └─ Return JSON with base64 Grad-CAM                │
└───────────────────────┬─────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────┐
│              INFERENCE ENGINE                        │
│  (backend/model/predict.py)                          │
│                                                      │
│  preprocess_image()                                  │
│  ├─ Decode bytes → PIL Image                        │
│  ├─ Resize to 224×224                               │
│  └─ Normalize to [0, 1]                             │
│                                                      │
│  predict()                                           │
│  ├─ Load cached model (singleton)                   │
│  ├─ model.predict(img_array)                        │
│  ├─ Apply threshold (0.5)                           │
│  └─ Return label, confidence, risk                  │
│                                                      │
│  compute_gradcam()                                   │
│  ├─ Build gradient model                            │
│  ├─ GradientTape → compute gradients                │
│  ├─ Pool gradients → weight feature maps            │
│  ├─ Generate heatmap (COLORMAP_JET)                 │
│  └─ Overlay on original image (alpha blend)        │
└───────────────────────┬─────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────┐
│              EFFICIENTNETB3 MODEL                    │
│  (backend/model/deepfake_detector.h5)                │
│                                                      │
│  Input: (1, 224, 224, 3)                            │
│  ↓                                                   │
│  EfficientNetB3 backbone (ImageNet pretrained)      │
│  ├─ 7 MBConv blocks                                 │
│  ├─ Squeeze-Excitation modules                      │
│  └─ ~12M parameters                                 │
│  ↓                                                   │
│  GlobalAveragePooling2D → (1, 1536)                 │
│  ↓                                                   │
│  BatchNorm → Dense(512) → Dropout(0.4)              │
│  ↓                                                   │
│  Dense(256) → Dropout(0.3)                          │
│  ↓                                                   │
│  Dense(1, sigmoid) → P(Deepfake) ∈ [0, 1]          │
└─────────────────────────────────────────────────────┘
```

## Dataset Structure

```
dataset/
├── train/          (70% of data)
│   ├── real/       Real face images
│   └── fake/       Deepfake images
├── val/            (15% of data)
│   ├── real/
│   └── fake/
└── test/           (15% of data)
    ├── real/
    └── fake/
```

## Training Pipeline

```
Raw Dataset (FaceForensics++ / DFDC / Kaggle)
          ↓
prepare_dataset.py
  ├─ Face detection (Haar cascade)
  ├─ Face cropping with padding
  ├─ Resize to 224×224
  ├─ 70/15/15 split
  └─ Save as JPEG (quality 95)
          ↓
train.py — Phase 1 (10 epochs)
  ├─ EfficientNetB3 frozen
  ├─ Train only Dense head
  ├─ LR = 1e-4
  └─ EarlyStopping (patience=5)
          ↓
train.py — Phase 2 (10 epochs)
  ├─ Unfreeze top 30 EfficientNet layers
  ├─ Fine-tune with LR = 1e-5
  └─ ReduceLROnPlateau
          ↓
Saved Model: deepfake_detector.h5
```

## API Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/api/health` | Health check |
| POST | `/api/predict` | Predict Real/Fake |
| POST | `/api/gradcam` | Get Grad-CAM image |
| POST | `/api/analyze` | Full analysis + Grad-CAM |

## Response Schema

```json
{
  "prediction": "Deepfake",
  "confidence": 0.92,
  "probability_real": 0.08,
  "probability_fake": 0.92,
  "risk_level": "High",
  "processing_time_ms": 183.4,
  "model_version": "EfficientNetB3-v1",
  "gradcam_image": "data:image/jpeg;base64,...",
  "analysis_details": "This image has been classified..."
}
```
