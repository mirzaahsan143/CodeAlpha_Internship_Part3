"""
Custom CNN architecture for leaf disease classification.

Architecture Design Rationale:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Input Layer (224×224×3):
  Standard input size for pre-trained model compatibility. 224×224 provides
  sufficient resolution to capture disease patterns (lesions, spots, discoloration)
  while keeping computational costs manageable.

Convolution Blocks:
  Each block: Conv2D → BatchNorm → ReLU → MaxPooling → Dropout
  
  - Conv2D (32→64→128→256 filters): Progressive increase in feature maps.
    Early layers capture low-level features (edges, textures, color blobs).
    Deeper layers capture high-level disease-specific patterns.
  
  - Kernel Size (3×3): Optimal for capturing local leaf textures. Smaller than
    5×5 but stacks of 3×3 convs have equivalent receptive field with fewer
    parameters (Simonyan & Zisserman, 2014).
  
  - Batch Normalization: Accelerates training by normalizing layer inputs.
    Reduces internal covariate shift, allows higher learning rates, and
    provides regularization (Ioffe & Szegedy, 2015).
  
  - MaxPooling (2×2): Reduces spatial dimensions by 75%, making the model
    spatially invariant and reducing overfitting. Each pooling layer doubles
    the receptive field.

  - Dropout (25%): Randomly drops 25% of neurons during training. This prevents
    co-adaptation of feature detectors and acts as an ensemble method
    (Srivastava et al., 2014).

Dense Layers (512→256→NUM_CLASSES):
  - 512 units: High capacity for combining learned features into disease
    predictions. Dropout (50%) applied here as dense layers are most prone
    to overfitting.
  - 256 units: Intermediate representation before final classification.
  - Softmax output: Multi-class probability distribution across disease classes.

Total params: ~3.9M - Balanced between capacity and overfitting risk.
"""
import sys
from pathlib import Path

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config


def conv_block(
    x: tf.Tensor,
    filters: int,
    kernel_size: tuple = (3, 3),
    pool_size: tuple = (2, 2),
    dropout_rate: float = 0.25,
    use_batch_norm: bool = True,
) -> tf.Tensor:
    """A convolutional block: Conv2D → BatchNorm → ReLU → MaxPool → Dropout."""
    x = layers.Conv2D(
        filters,
        kernel_size,
        padding="same",
        kernel_initializer="he_normal",
        kernel_regularizer=regularizers.l2(1e-4),
    )(x)

    if use_batch_norm:
        x = layers.BatchNormalization()(x)

    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(pool_size)(x)
    x = layers.Dropout(dropout_rate)(x)
    return x


def build_cnn_model(
    input_shape: tuple = config.INPUT_SHAPE,
    num_classes: int = None,
    config_override: dict = None,
) -> tf.keras.Model:
    """
    Build the custom CNN architecture.

    Architecture Diagram:
    ┌─────────────────────────────────────────────────────┐
    │ Input (224×224×3)                                    │
    ├─────────────────────────────────────────────────────┤
    │ ConvBlock 1: 32 filters, 3×3, BN, ReLU, 2×2 Pool   │ → 112×112×32
    │ ConvBlock 2: 64 filters, 3×3, BN, ReLU, 2×2 Pool   │ → 56×56×64
    │ ConvBlock 3: 128 filters, 3×3, BN, ReLU, 2×2 Pool  │ → 28×28×128
    │ ConvBlock 4: 256 filters, 3×3, BN, ReLU, 2×2 Pool  │ → 14×14×256
    ├─────────────────────────────────────────────────────┤
    │ GlobalAveragePooling2D                                │ → 256
    │ Dense(512) + BN + ReLU + Dropout(0.5)                │ → 512
    │ Dense(256) + BN + ReLU + Dropout(0.3)                │ → 256
    │ Dense(num_classes) + Softmax                          │ → NUM_CLASSES
    └─────────────────────────────────────────────────────┘
    """
    if num_classes is None:
        num_classes = config.NUM_CLASSES
    if config_override is None:
        config_override = config.CNN_CONFIG

    inputs = layers.Input(shape=input_shape)

    x = inputs

    # Convolutional blocks with progressively increasing filters
    for i, filters in enumerate(config_override["conv_filters"]):
        x = conv_block(
            x,
            filters=filters,
            kernel_size=config_override["kernel_size"],
            pool_size=config_override["pool_size"],
            dropout_rate=0.25,
            use_batch_norm=config_override["batch_norm"],
        )

    # Global average pooling instead of Flatten
    # GAP reduces parameters vs Flatten + Dense, acts as a structural regularizer
    x = layers.GlobalAveragePooling2D()(x)

    # Dense layers for classification
    for i, units in enumerate(config_override["dense_units"]):
        x = layers.Dense(
            units,
            kernel_initializer="he_normal",
            kernel_regularizer=regularizers.l2(1e-4),
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Dropout(config_override["dropout_rate"] if i == 0 else 0.3)(x)

    # Output layer
    outputs = layers.Dense(
        num_classes,
        activation="softmax",
        kernel_initializer="glorot_uniform",
    )(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="CustomCNN")

    return model


if __name__ == "__main__":
    # Quick architecture test
    import numpy as np
    from config import NUM_CLASSES

    # Set test classes
    NUM_CLASSES = 15

    model = build_cnn_model()
    model.summary()

    # Test forward pass
    dummy_input = np.random.randn(1, 224, 224, 3).astype(np.float32)
    output = model.predict(dummy_input, verbose=0)
    print(f"\n✅ Model output shape: {output.shape}")
    print(f"✅ Output (should sum to 1.0): {output.sum():.4f}")
