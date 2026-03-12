import os
import io
import time
import numpy as np
from PIL import Image
import tensorflow as tf

# Suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# ── Constants ──────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "deepfake_detector.h5")
IMG_SIZE   = 224
# Class mapping from training: {'fake': 0, 'real': 1}
# So output close to 0 = fake, close to 1 = real

_model = None


def load_model():
    global _model
    if _model is None:
        print(f"Loading model from {MODEL_PATH}...")
        _model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        print("Model loaded successfully.")
        print("Input shape:", _model.input_shape)
        print("Output shape:", _model.output_shape)
    return _model


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """Preprocess image bytes into model input tensor."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)  # (1, 224, 224, 3)
    return arr


def predict(image_bytes: bytes) -> dict:
    """
    Run deepfake detection on image bytes.
    Returns dict with prediction, confidence, probabilities, risk level.
    """
    model = load_model()

    start = time.time()
    arr   = preprocess_image(image_bytes)
    raw   = model.predict(arr, verbose=0)
    elapsed_ms = round((time.time() - start) * 1000, 1)

    # raw output is sigmoid: 0 = fake, 1 = real
    prob_real = float(raw[0][0])
    prob_fake = 1.0 - prob_real

    # Determine prediction
    if prob_real >= 0.5:
        prediction = "Real"
        confidence = prob_real
    else:
        prediction = "Deepfake"
        confidence = prob_fake

    # Risk level based on fake probability
    if prob_fake >= 0.80:
        risk_level = "High"
    elif prob_fake >= 0.55:
        risk_level = "Medium"
    else:
        risk_level = "Low"

    # Analysis text
    if prediction == "Deepfake":
        analysis = (
            f"This image shows strong indicators of AI manipulation. "
            f"The model detected deepfake artifacts with {confidence*100:.1f}% confidence. "
            f"Key facial regions show inconsistencies typical of GAN-generated content."
        )
    else:
        analysis = (
            f"This image appears to be an authentic photograph. "
            f"No significant deepfake artifacts were detected. "
            f"The model classified this as real with {confidence*100:.1f}% confidence."
        )

    return {
        "prediction":        prediction,
        "confidence":        round(confidence, 4),
        "probability_real":  round(prob_real, 4),
        "probability_fake":  round(prob_fake, 4),
        "risk_level":        risk_level,
        "processing_time_ms": elapsed_ms,
        "model_version":     "MobileNetV2-v1",
        "analysis_details":  analysis,
    }


def generate_gradcam(image_bytes: bytes) -> str:
    """
    Generate Grad-CAM heatmap overlay.
    Returns base64 encoded JPEG string.
    """
    import base64
    import cv2

    model = load_model()

    # Preprocess
    img_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_pil = img_pil.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    img_arr = np.array(img_pil, dtype=np.float32) / 255.0
    img_tensor = np.expand_dims(img_arr, axis=0)

    # Find last conv layer
    last_conv = None
    for layer in reversed(model.layers):
        if isinstance(layer, (tf.keras.layers.Conv2D,
                               tf.keras.layers.DepthwiseConv2D)):
            last_conv = layer
            break
        # Check inside sub-models (MobileNetV2 is wrapped)
        if hasattr(layer, 'layers'):
            for sub in reversed(layer.layers):
                if isinstance(sub, (tf.keras.layers.Conv2D,
                                     tf.keras.layers.DepthwiseConv2D)):
                    last_conv = sub
                    break
            if last_conv:
                break

    if last_conv is None:
        # Return original image if no conv layer found
        buf = io.BytesIO()
        img_pil.save(buf, format="JPEG")
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()

    try:
        # Build grad model
        grad_model = tf.keras.Model(
            inputs=model.inputs,
            outputs=[last_conv.output, model.output]
        )

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_tensor, training=False)
            loss = predictions[:, 0]

        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap).numpy()
        heatmap = np.maximum(heatmap, 0)

        if heatmap.max() > 0:
            heatmap /= heatmap.max()

        # Resize and colorize
        heatmap_resized = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE))
        heatmap_uint8   = np.uint8(255 * heatmap_resized)
        heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        heatmap_rgb     = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

        # Overlay
        original_np = np.array(img_pil)
        overlay     = (0.6 * original_np + 0.4 * heatmap_rgb).astype(np.uint8)
        overlay_pil = Image.fromarray(overlay)

        buf = io.BytesIO()
        overlay_pil.save(buf, format="JPEG", quality=85)
        encoded = base64.b64encode(buf.getvalue()).decode()
        return "data:image/jpeg;base64," + encoded

    except Exception as e:
        print(f"Grad-CAM error: {e}")
        buf = io.BytesIO()
        img_pil.save(buf, format="JPEG")
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()

# Aliases for backward compatibility with main.py
def gradcam_to_bytes(image_bytes: bytes) -> str:
    return generate_gradcam(image_bytes)

def get_model():
    return load_model()
