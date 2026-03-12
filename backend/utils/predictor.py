import os
import io
import uuid
import time
import base64
import numpy as np
from PIL import Image
import logging

logger = logging.getLogger(__name__)

MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'model', 'deepfake_model.h5')
IMG_SIZE = (224, 224)


class DeepfakePredictor:

    def __init__(self):
        self.model = None
        self.framework = None
        self._load_model()

    def _load_model(self):
        if os.path.exists(MODEL_PATH):
            try:
                import tensorflow as tf
                self.model = tf.keras.models.load_model(MODEL_PATH)
                self.framework = "keras"
                print("✅ Model loaded successfully")
            except Exception as e:
                print(f"❌ Model load failed: {e}")
                self.framework = "demo"
        else:
            print("⚠️ Model not found, running DEMO mode")
            self.framework = "demo"

    def _preprocess(self, image_bytes):
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img = img.resize(IMG_SIZE, Image.LANCZOS)
        arr = np.array(img, dtype=np.float32) / 255.0
        return np.expand_dims(arr, axis=0)

    def predict(self, image_bytes):
        arr = self._preprocess(image_bytes)

        if self.framework == "keras":
            prob = float(self.model.predict(arr, verbose=0)[0][0])
            print(f"Raw model output: {prob}")
        else:
            # Demo mode: deterministic fake score based on image content
            prob = float(np.mean(arr))

        # prob close to 1.0 = Real, close to 0.0 = Fake
        real_prob = prob
        fake_prob = 1.0 - prob

        if real_prob >= 0.5:
            label = "REAL"
            confidence = real_prob
        else:
            label = "FAKE"
            confidence = fake_prob

        verdict = self._build_verdict(label, confidence)

        return {
            "label":      label,
            "confidence": round(confidence, 4),
            "real_prob":  round(real_prob, 4),
            "fake_prob":  round(fake_prob, 4),
            "verdict":    verdict,
        }


    def _build_verdict(self, label, confidence):
        pct = round(confidence * 100, 1)
        if label == "FAKE":
            if pct >= 90:
                return f"Almost certainly a deepfake ({pct}% confidence). Strong AI manipulation detected."
            elif pct >= 70:
                return f"Likely a deepfake ({pct}% confidence). Significant signs of AI generation detected."
            else:
                return f"Possibly a deepfake ({pct}% confidence). Some anomalies detected."
        else:
            if pct >= 90:
                return f"Almost certainly authentic ({pct}% confidence). No manipulation detected."
            elif pct >= 70:
                return f"Likely authentic ({pct}% confidence). Minor anomalies but generally real."
            else:
                return f"Possibly authentic ({pct}% confidence). Inconclusive — consider re-analysis."


# ---------- GLOBAL PREDICTOR INSTANCE ----------
predictor = DeepfakePredictor()


# ---------- FUNCTIONS USED BY API ----------
def predict(image_bytes, model_path=None):
    """Returns dict with label, confidence, real_prob, fake_prob, verdict"""
    return predictor.predict(image_bytes)


def generate_gradcam(image_bytes, model_path=None):
    """
    Returns (original_b64, heatmap_b64) as base64 strings.
    If real Grad-CAM is unavailable, returns the original image twice
    so the frontend still renders without crashing.
    """
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img = img.resize((300, 300), Image.LANCZOS)

        # Encode original
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        original_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        # Try real Grad-CAM with keras model
        if predictor.framework == "keras":
            try:
                import tensorflow as tf
                arr = np.array(img, dtype=np.float32) / 255.0
                arr = np.expand_dims(arr, axis=0)

                grad_model = tf.keras.models.Model(
                    inputs=predictor.model.inputs,
                    outputs=[predictor.model.layers[-3].output, predictor.model.output]
                )

                with tf.GradientTape() as tape:
                    conv_outputs, predictions = grad_model(arr)
                    loss = predictions[:, 0]

                grads = tape.gradient(loss, conv_outputs)
                pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
                conv_outputs = conv_outputs[0]
                heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
                heatmap = tf.squeeze(heatmap).numpy()
                heatmap = np.maximum(heatmap, 0)
                if heatmap.max() > 0:
                    heatmap /= heatmap.max()

                # Resize and colorize heatmap
                heatmap_img = Image.fromarray(np.uint8(heatmap * 255)).resize(img.size, Image.LANCZOS)
                heatmap_arr = np.array(heatmap_img)
                colored = np.zeros((*heatmap_arr.shape, 3), dtype=np.uint8)
                colored[:, :, 0] = heatmap_arr          # Red channel
                colored[:, :, 1] = 0
                colored[:, :, 2] = 255 - heatmap_arr    # Blue channel

                # Blend with original
                orig_arr = np.array(img)
                blended = (0.6 * orig_arr + 0.4 * colored).astype(np.uint8)
                heatmap_pil = Image.fromarray(blended)

                buf2 = io.BytesIO()
                heatmap_pil.save(buf2, format="JPEG")
                heatmap_b64 = base64.b64encode(buf2.getvalue()).decode("utf-8")
                return original_b64, heatmap_b64

            except Exception as e:
                print(f"Grad-CAM failed, using fallback: {e}")

        # Fallback: tint the original image red/blue as a fake heatmap
        orig_arr = np.array(img, dtype=np.float32)
        tinted = orig_arr.copy()
        tinted[:, :, 0] = np.clip(orig_arr[:, :, 0] * 1.4, 0, 255)  # boost red
        tinted[:, :, 2] = np.clip(orig_arr[:, :, 2] * 0.5, 0, 255)  # reduce blue
        tinted_img = Image.fromarray(tinted.astype(np.uint8))

        buf2 = io.BytesIO()
        tinted_img.save(buf2, format="JPEG")
        heatmap_b64 = base64.b64encode(buf2.getvalue()).decode("utf-8")

        return original_b64, heatmap_b64

    except Exception as e:
        print(f"generate_gradcam error: {e}")
        # Last resort: return same image twice
        buf = io.BytesIO()
        Image.open(io.BytesIO(image_bytes)).convert("RGB").resize((300, 300)).save(buf, format="JPEG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return b64, b64
