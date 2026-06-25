"""
Model factory module.
Provides a unified interface to build any model in the project.
"""
import sys
from pathlib import Path

import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config
from .cnn import build_cnn_model
from .transfer_learning import get_transfer_learning_model


def get_model(
    model_name: str = "CustomCNN",
    input_shape: tuple = None,
    num_classes: int = None,
    compile_model: bool = True,
    learning_rate: float = 1e-3,
) -> tf.keras.Model:
    """
    Factory function to build and compile any supported model.
    """
    if input_shape is None:
        input_shape = config.INPUT_SHAPE
    if num_classes is None:
        num_classes = config.NUM_CLASSES

    # Build the model
    if model_name == "CustomCNN":
        model = build_cnn_model(input_shape, num_classes)
    else:
        model = get_transfer_learning_model(model_name, input_shape, num_classes)

    # Compile if requested
    if compile_model:
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

    return model


MODEL_REGISTRY = {
    "CustomCNN": {
        "builder": build_cnn_model,
        "description": "Custom CNN architecture from scratch (3.9M params)",
        "best_for": "Baseline comparison, understanding CNN fundamentals",
    },
    "MobileNetV2": {
        "builder": lambda: get_transfer_learning_model("MobileNetV2"),
        "description": "MobileNetV2 transfer learning (3.5M params)",
        "best_for": "Production deployment, mobile/web apps",
    },
    "EfficientNetB0": {
        "builder": lambda: get_transfer_learning_model("EfficientNetB0"),
        "description": "EfficientNetB0 transfer learning (5.3M params)",
        "best_for": "Best accuracy-to-size ratio",
    },
    "ResNet50": {
        "builder": lambda: get_transfer_learning_model("ResNet50"),
        "description": "ResNet50 transfer learning (25.6M params)",
        "best_for": "Maximum accuracy (with sufficient data)",
    },
}
