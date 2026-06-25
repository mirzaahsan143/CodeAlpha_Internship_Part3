"""
Data augmentation module for leaf disease images.
Applies controlled transformations to increase dataset diversity
and improve model generalization.
"""
import sys
from pathlib import Path

import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import AUGMENTATION, IMG_SIZE


def get_augmentation_pipeline(training: bool = True) -> tf.keras.Sequential:
    """
    Create a data augmentation pipeline using TensorFlow's Keras layers.

    Data augmentation is crucial for:
    1. **Increasing effective dataset size** - Creates diverse training samples
       from limited data, reducing overfitting.
    2. **Improving generalization** - Exposes the model to varied leaf orientations,
       lighting conditions, and perspectives seen in real-world scenarios.
    3. **Building invariance** - Teaches the model that a disease looks the same
       regardless of rotation, brightness, or slight perspective changes.

    Augmentation choices explained:
    - **Rotation (±30°)**: Leaves in the wild appear at any angle. This builds
      rotation invariance into the model.
    - **Width/Height Shift (±20%)**: Diseases may appear anywhere on the leaf;
      shifting simulates different disease positions.
    - **Shear (±15%)**: Simulates leaf curvature and perspective distortion.
    - **Zoom (±20%)**: Simulates varying distances from the camera to the leaf.
    - **Horizontal Flip**: Realistic since leaves are symmetric along the vertical axis.
    - **Vertical Flip**: DISABLED because leaves have a natural orientation (stem
      at bottom). Vertical flipping would create unrealistic training samples.
    - **Brightness Adjustment (0.8–1.2)**: Simulates different lighting conditions
      (shade, direct sunlight, overcast).
    - **Fill Mode 'nearest'**: When transformations create empty border pixels,
      this fills them with the nearest valid pixel value, avoiding artificial
      edge artifacts.

    Reference: Shorten, C., & Khoshgoftaar, T.M. (2019). A survey on image data
    augmentation for deep learning. Journal of Big Data.
    """
    if not training:
        return tf.keras.Sequential([
            layers.Resizing(IMG_SIZE[0], IMG_SIZE[1]),
            layers.Rescaling(1./255),
        ])

    return tf.keras.Sequential([
        # First, resize all inputs to a standard size
        layers.Resizing(IMG_SIZE[0] + 32, IMG_SIZE[1] + 32),
        # Random transformations
        layers.RandomRotation(AUGMENTATION["rotation_range"] / 360.0),
        layers.RandomTranslation(
            AUGMENTATION["width_shift_range"],
            AUGMENTATION["height_shift_range"],
        ),
        layers.RandomZoom(AUGMENTATION["zoom_range"]),
        layers.RandomFlip("horizontal"),
        layers.RandomBrightness(AUGMENTATION["brightness_range"]),
        # Crop back to target size
        layers.CenterCrop(IMG_SIZE[0], IMG_SIZE[1]),
        # Normalize pixel values
        layers.Rescaling(1./255),
    ])


def get_test_preprocessing() -> tf.keras.Sequential:
    """
    Minimal preprocessing for validation/test data.
    Only resize and normalize - no random augmentations.
    """
    return tf.keras.Sequential([
        layers.Resizing(IMG_SIZE[0], IMG_SIZE[1]),
        layers.Rescaling(1./255),
    ])


def create_tf_dataset(
    file_paths,
    labels,
    batch_size: int = 32,
    training: bool = True,
    cache: bool = True,
    shuffle_buffer: int = 1000,
):
    """
    Create a tf.data.Dataset pipeline for efficient training.

    The tf.data pipeline provides:
    - Parallel file loading
    - Prefetching for GPU/CPU overlap
    - On-the-fly augmentation
    - Efficient shuffling without loading all data into memory
    """
    AUTOTUNE = tf.data.AUTOTUNE

    def load_and_preprocess(image_path, label):
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.convert_image_dtype(image, tf.float32)
        return image, label

    # Create dataset from file paths and labels
    dataset = tf.data.Dataset.from_tensor_slices((file_paths, labels))
    dataset = dataset.map(load_and_preprocess, num_parallel_calls=AUTOTUNE)

    # Apply augmentation
    aug_pipeline = get_augmentation_pipeline(training=training)

    def apply_augmentation(image, label):
        image = aug_pipeline(tf.expand_dims(image, 0))
        image = tf.squeeze(image, 0)
        return image, label

    dataset = dataset.map(apply_augmentation, num_parallel_calls=AUTOTUNE)

    if training:
        dataset = dataset.shuffle(shuffle_buffer)

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(AUTOTUNE)

    if cache and not training:
        dataset = dataset.cache()

    return dataset
