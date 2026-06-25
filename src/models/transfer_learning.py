"""
Transfer Learning Models for Leaf Disease Classification.

Why Transfer Learning?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Transfer learning leverages models pre-trained on ImageNet (1.4M images,
1000 classes). These models have already learned rich feature representations
(generic edges, textures, shapes, color patterns) that transfer well to
plant disease detection.

Benefits:
1. **Faster training**: Pre-trained weights provide a head start
2. **Better performance**: Especially with limited data
3. **Improved generalization**: Learned features are more robust

Models Compared:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

MobileNetV2 (Sandler et al., 2018):
  - Uses inverted residuals and linear bottlenecks
  - Depthwise separable convolutions for efficiency
  - Excellent accuracy/parameter ratio
  - ~3.5M parameters
  - Best for: Production deployment, edge devices

EfficientNetB0 (Tan & Le, 2019):
  - Neural architecture search optimized
  - Compound scaling of depth/width/resolution
  - State-of-the-art accuracy for its size
  - ~5.3M parameters
  - Best for: Accuracy-focused applications

ResNet50 (He et al., 2016):
  - Residual connections (skip connections)
  - Solves vanishing gradient in deep networks
  - Proven robust across many domains
  - ~25.6M parameters
  - Best for: Large datasets, high accuracy needs

Strategy:
  1. Load pre-trained model without top (include_top=False)
  2. Freeze base layers to preserve learned features
  3. Add custom classification head (GAP → Dense → Dropout → Softmax)
  4. Train only the head initially
  5. Optionally fine-tune top layers with very low learning rate
"""
import sys
from pathlib import Path

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.applications import (
    MobileNetV2,
    EfficientNetB0,
    ResNet50,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config


def _add_classification_head(base_model: tf.keras.Model, num_classes: int, freeze_base: bool = True) -> tf.keras.Model:
    """
    Add a custom classification head on top of a pre-trained base model.
    """
    base_model.trainable = not freeze_base

    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, kernel_initializer="he_normal",
                     kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(num_classes, activation="softmax",
                     kernel_initializer="glorot_uniform")(x)

    model = models.Model(inputs=base_model.input, outputs=x,
                         name=f"{base_model.name}_transfer")

    return model


def build_mobilenetv2(
    input_shape: tuple = None,
    num_classes: int = None,
) -> tf.keras.Model:
    """Build MobileNetV2-based transfer learning model."""
    if input_shape is None:
        input_shape = config.INPUT_SHAPE
    if num_classes is None:
        num_classes = config.NUM_CLASSES

    base_model = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=input_shape,
    )

    model = _add_classification_head(base_model, num_classes, freeze_base=config.FREEZE_BASE_LAYERS)
    return model


def build_efficientnet(
    input_shape: tuple = None,
    num_classes: int = None,
) -> tf.keras.Model:
    """Build EfficientNetB0-based transfer learning model."""
    if input_shape is None:
        input_shape = config.INPUT_SHAPE
    if num_classes is None:
        num_classes = config.NUM_CLASSES

    base_model = EfficientNetB0(
        weights="imagenet",
        include_top=False,
        input_shape=input_shape,
    )

    model = _add_classification_head(base_model, num_classes, freeze_base=config.FREEZE_BASE_LAYERS)
    return model


def build_resnet50(
    input_shape: tuple = None,
    num_classes: int = None,
) -> tf.keras.Model:
    """Build ResNet50-based transfer learning model."""
    if input_shape is None:
        input_shape = config.INPUT_SHAPE
    if num_classes is None:
        num_classes = config.NUM_CLASSES

    base_model = ResNet50(
        weights="imagenet",
        include_top=False,
        input_shape=input_shape,
    )

    model = _add_classification_head(base_model, num_classes, freeze_base=config.FREEZE_BASE_LAYERS)
    return model


def get_transfer_learning_model(
    model_name: str,
    input_shape: tuple = None,
    num_classes: int = None,
) -> tf.keras.Model:
    """Factory function to get any transfer learning model by name."""
    if input_shape is None:
        input_shape = config.INPUT_SHAPE
    if num_classes is None:
        num_classes = config.NUM_CLASSES

    builders = {
        "MobileNetV2": build_mobilenetv2,
        "EfficientNetB0": build_efficientnet,
        "ResNet50": build_resnet50,
    }

    if model_name not in builders:
        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Available: {list(builders.keys())}"
        )

    return builders[model_name](input_shape, num_classes)


# For fine-tuning: unfreeze some base layers
def prepare_fine_tuning(
    model: tf.keras.Model,
    unfreeze_from: int = 100,
    learning_rate: float = 1e-5,
) -> tf.keras.Model:
    """
    Prepare a model for fine-tuning by unfreezing the top base layers.

    Fine-tuning strategy:
    1. First train the classification head (frozen base)
    2. Then unfreeze top layers for joint training
    3. Use very low learning rate (1e-5) to avoid destroying
       pre-trained features (the "catastrophic forgetting" problem)
    """
    # Unfreeze the base model
    base_model = model.layers[1]  # The base model is typically at index 1
    base_model.trainable = True

    # Freeze early layers, unfreeze later layers
    for layer in base_model.layers[:unfreeze_from]:
        layer.trainable = False

    # Recompile with lower learning rate
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model
