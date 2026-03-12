"""
Create a demo model for testing without full dataset training.
This creates an EfficientNetB3 model with random weights for UI/API testing.
Replace with your trained model for real predictions.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB3

IMG_SIZE = 224
MODEL_PATH = os.path.join(os.path.dirname(__file__), "deepfake_detector.h5")


def create_demo_model():
    """Create and save a demo model (random weights) for API testing."""
    print("Creating demo EfficientNetB3 model...")

    base_model = EfficientNetB3(
        include_top=False,
        weights="imagenet",
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
    )
    base_model.trainable = False

    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(512, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = keras.Model(inputs, outputs, name="DeepfakeDetector_Demo")
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    model.save(MODEL_PATH)
    print(f"Demo model saved to: {MODEL_PATH}")
    print("\nNOTE: This demo model has ImageNet pretrained weights but no deepfake-specific training.")
    print("      To get accurate predictions, train with train.py on a real dataset.")
    return model


if __name__ == "__main__":
    create_demo_model()
