"""
Deepfake Detection Model Training Script
Uses EfficientNetB3 with Transfer Learning for binary classification (Real vs Fake)
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import json

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS_FROZEN = 10       # Train only the head
EPOCHS_FINETUNE = 10     # Unfreeze top layers and fine-tune
LEARNING_RATE = 1e-4
FINE_TUNE_LR = 1e-5
DATASET_DIR = "../dataset"
MODEL_SAVE_PATH = "deepfake_detector.h5"
WEIGHTS_SAVE_PATH = "deepfake_detector_weights.weights.h5"

# ─────────────────────────────────────────────
# DATA GENERATORS WITH AUGMENTATION
# ─────────────────────────────────────────────
def create_data_generators():
    """Create train, validation, and test data generators."""

    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
        fill_mode="nearest",
        validation_split=0.15,
    )

    val_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=0.15,
    )

    test_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_generator = train_datagen.flow_from_directory(
        os.path.join(DATASET_DIR, "train"),
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="binary",
        subset="training",
        shuffle=True,
    )

    val_generator = val_datagen.flow_from_directory(
        os.path.join(DATASET_DIR, "train"),
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="binary",
        subset="validation",
        shuffle=False,
    )

    test_generator = test_datagen.flow_from_directory(
        os.path.join(DATASET_DIR, "test"),
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="binary",
        shuffle=False,
    )

    return train_generator, val_generator, test_generator


# ─────────────────────────────────────────────
# MODEL ARCHITECTURE
# ─────────────────────────────────────────────
def build_model(num_classes=1):
    """
    Build EfficientNetB3-based deepfake detector.
    - Load EfficientNetB3 pretrained on ImageNet
    - Freeze base layers
    - Replace classification head for binary output
    """
    # Load pretrained base (no top layers)
    base_model = EfficientNetB3(
        include_top=False,
        weights="imagenet",
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
    )

    # Freeze ALL base layers initially
    base_model.trainable = False

    # Build model
    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))

    # Preprocessing built-in to EfficientNet (expects 0-255 OR 0-1)
    x = base_model(inputs, training=False)

    # Global pooling instead of flatten (reduces overfitting)
    x = layers.GlobalAveragePooling2D()(x)

    # Classification head
    x = layers.BatchNormalization()(x)
    x = layers.Dense(512, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.3)(x)

    # Binary output: sigmoid for Real (0) vs Fake (1)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = keras.Model(inputs, outputs, name="DeepfakeDetector")
    return model, base_model


# ─────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────
def compile_and_train(model, train_gen, val_gen, epochs, lr, phase_name="Phase"):
    """Compile and train the model."""

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
            keras.metrics.AUC(name="auc"),
        ],
    )

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.3,
            patience=3,
            min_lr=1e-7,
            verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=f"best_model_{phase_name}.h5",
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
    ]

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1,
    )

    return history


def unfreeze_top_layers(base_model, num_layers=30):
    """Unfreeze the top N layers of the base model for fine-tuning."""
    base_model.trainable = True
    for layer in base_model.layers[:-num_layers]:
        layer.trainable = False
    print(f"Unfrozen top {num_layers} layers for fine-tuning.")


# ─────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────
def evaluate_model(model, test_gen):
    """Evaluate model and print detailed metrics."""
    print("\n" + "=" * 50)
    print("MODEL EVALUATION ON TEST SET")
    print("=" * 50)

    # Predictions
    y_pred_proba = model.predict(test_gen, verbose=1)
    y_pred = (y_pred_proba > 0.5).astype(int).flatten()
    y_true = test_gen.classes

    # Classification report
    class_names = list(test_gen.class_indices.keys())
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print(cm)

    # Save metrics
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred)),
        "f1_score": float(f1_score(y_true, y_pred)),
        "class_indices": test_gen.class_indices,
    }

    with open("model_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nFinal Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision:       {metrics['precision']:.4f}")
    print(f"Recall:          {metrics['recall']:.4f}")
    print(f"F1-Score:        {metrics['f1_score']:.4f}")

    return metrics


def plot_training_history(history1, history2=None):
    """Plot training and validation accuracy/loss curves."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Combine histories if fine-tuning occurred
    acc = history1.history["accuracy"]
    val_acc = history1.history["val_accuracy"]
    loss = history1.history["loss"]
    val_loss = history1.history["val_loss"]

    if history2:
        acc += history2.history["accuracy"]
        val_acc += history2.history["val_accuracy"]
        loss += history2.history["loss"]
        val_loss += history2.history["val_loss"]

    epochs_range = range(len(acc))

    axes[0].plot(epochs_range, acc, label="Training Accuracy", color="#00d4ff")
    axes[0].plot(epochs_range, val_acc, label="Validation Accuracy", color="#ff6b6b")
    axes[0].set_title("Model Accuracy", fontsize=14)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs_range, loss, label="Training Loss", color="#00d4ff")
    axes[1].plot(epochs_range, val_loss, label="Validation Loss", color="#ff6b6b")
    axes[1].set_title("Model Loss", fontsize=14)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("training_history.png", dpi=150, bbox_inches="tight")
    print("Training history plot saved as training_history.png")
    plt.close()


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  DEEPFAKE DETECTION MODEL TRAINING")
    print("  Using EfficientNetB3 + Transfer Learning")
    print("=" * 60)

    # GPU check
    gpus = tf.config.list_physical_devices("GPU")
    print(f"\nGPUs Available: {len(gpus)}")
    if gpus:
        tf.config.experimental.set_memory_growth(gpus[0], True)

    # Load data
    print("\n[1/4] Loading dataset...")
    train_gen, val_gen, test_gen = create_data_generators()
    print(f"  Train samples: {train_gen.samples}")
    print(f"  Val samples:   {val_gen.samples}")
    print(f"  Test samples:  {test_gen.samples}")
    print(f"  Classes:       {train_gen.class_indices}")

    # Build model
    print("\n[2/4] Building model...")
    model, base_model = build_model()
    model.summary()

    # Phase 1: Train only the classification head
    print("\n[3/4] Phase 1 — Training classification head (base frozen)...")
    history1 = compile_and_train(
        model, train_gen, val_gen,
        epochs=EPOCHS_FROZEN,
        lr=LEARNING_RATE,
        phase_name="phase1",
    )

    # Phase 2: Fine-tune top layers
    print("\n[4/4] Phase 2 — Fine-tuning top layers...")
    unfreeze_top_layers(base_model, num_layers=30)
    history2 = compile_and_train(
        model, train_gen, val_gen,
        epochs=EPOCHS_FINETUNE,
        lr=FINE_TUNE_LR,
        phase_name="phase2",
    )

    # Evaluate
    metrics = evaluate_model(model, test_gen)

    # Save final model
    model.save(MODEL_SAVE_PATH)
    model.save_weights(WEIGHTS_SAVE_PATH)
    print(f"\nModel saved to: {MODEL_SAVE_PATH}")
    print(f"Weights saved to: {WEIGHTS_SAVE_PATH}")

    # Plot training history
    plot_training_history(history1, history2)

    print("\n✅ Training complete!")
    return model, metrics


if __name__ == "__main__":
    main()
