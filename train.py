"""
Training script - Train a specific model on the leaf disease dataset.
Usage: python train.py [--model CustomCNN|MobileNetV2|EfficientNetB0|ResNet50]
"""
import sys
import argparse
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import tensorflow as tf

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    DATA_DIR, IMG_SIZE, BATCH_SIZE, INITIAL_LR, EPOCHS,
    MODELS_DIR, PLOTS_DIR,
    # CLASS_NAMES / NUM_CLASSES are NOT imported here because they start empty;
    # they are dynamically set via config.CLASS_NAMES after dataset.investigate().
)
from src.data.dataset import LeafDataset
from src.data.splits import create_dataset_splits
from src.data.augmentation import create_tf_dataset
from src.models.factory import get_model
from src.training.trainer import ModelTrainer


def main():
    parser = argparse.ArgumentParser(description="Train a leaf disease detection model")
    parser.add_argument("--model", type=str, default="CustomCNN",
                        choices=["CustomCNN", "MobileNetV2", "EfficientNetB0", "ResNet50"],
                        help="Model architecture to train")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Max epochs")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size")
    parser.add_argument("--lr", type=float, default=INITIAL_LR, help="Learning rate")
    args = parser.parse_args()

    # Setup GPU
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✅ Using {len(gpus)} GPU(s)")

    # Load dataset
    print("\n📂 Loading dataset...")
    dataset = LeafDataset(DATA_DIR)
    dataset.investigate()

    # Update class config
    import config
    config.CLASS_NAMES = dataset.class_names
    config.NUM_CLASSES = dataset.num_classes

    # Get clean paths
    clean_paths = dataset.get_clean_file_paths()

    # Create splits
    train_paths, val_paths, test_paths, train_labels, val_labels, test_labels = \
        create_dataset_splits(clean_paths)

    # Create data pipelines
    train_ds = create_tf_dataset(train_paths, train_labels, args.batch_size, training=True)
    val_ds = create_tf_dataset(val_paths, val_labels, args.batch_size, training=False)
    test_ds = create_tf_dataset(test_paths, test_labels, args.batch_size, training=False)

    # Build model
    print(f"\n🔧 Building {args.model}...")
    model = get_model(args.model, learning_rate=args.lr)

    # Train
    trainer = ModelTrainer(model, args.model)
    history = trainer.train(train_ds, val_ds, epochs=args.epochs)

    # Save model
    model_path = MODELS_DIR / f"{args.model}_final.h5"
    model.save(str(model_path))
    print(f"\n✅ Model saved to {model_path}")

    # Evaluate
    print(f"\n📊 Evaluating on test set...")
    test_metrics = trainer.evaluate(test_ds)
    print(f"\n  Test Accuracy: {test_metrics['test_accuracy']:.4f}")
    print(f"  Test Loss: {test_metrics['test_loss']:.4f}")


if __name__ == "__main__":
    main()
