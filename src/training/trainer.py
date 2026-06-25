"""
Professional model training module.

Training strategy:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Phase 1 - Initial Training (Transfer Learning):
  - Freeze base model layers
  - Train classification head only
  - Learning rate: 1e-3
  - Epochs: Up to 50 with early stopping
  - Purpose: Train the classifier while leveraging pre-trained features

Phase 2 - Fine-Tuning (if enabled):
  - Unfreeze top layers of base model
  - Very low learning rate: 1e-5
  - Epochs: Up to 20
  - Purpose: Adapt pre-trained features to leaf disease domain

Phase 3 - Full Training (Custom CNN):
  - Train all layers from initialized weights
  - Learning rate: 1e-3 with ReduceLROnPlateau
  - Epochs: Up to 100 with early stopping

Key Techniques:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. Early Stopping:
   - Monitors validation loss
   - Stops training when no improvement for N epochs
   - Restores best weights
   - Prevents overfitting and saves time

2. ReduceLROnPlateau:
   - Reduces LR by factor 0.5 when validation loss plateaus
   - Allows model to converge to finer optima
   - More effective than fixed LR schedules

3. Model Checkpoint:
   - Saves best weights based on validation accuracy
   - Enables recovery and deployment of best model
   - Only saves weights (not full model) to save space

4. CSV Logging:
   - Logs all metrics to CSV for later analysis
   - Enables plotting training curves post-hoc
   - Useful for experiment tracking
"""
import sys
import time
import json
from pathlib import Path
from typing import Optional, Callable, Dict, Any

import numpy as np
import tensorflow as tf
from tensorflow.keras import callbacks

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import (
    EPOCHS, BATCH_SIZE, INITIAL_LR, MIN_LR,
    EARLY_STOPPING_PATIENCE, REDUCE_LR_PATIENCE, REDUCE_LR_FACTOR,
    MODELS_DIR, LOGS_DIR, CLASS_NAMES, NUM_CLASSES,
)


class ModelTrainer:
    """
    Professional model trainer with callbacks, logging, and checkpointing.
    """

    def __init__(
        self,
        model: tf.keras.Model,
        model_name: str = "model",
        log_dir: Path = LOGS_DIR,
        checkpoint_dir: Path = MODELS_DIR,
    ):
        self.model = model
        self.model_name = model_name
        self.log_dir = Path(log_dir)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.history = None
        self.training_time = 0

        # Create subdirectories
        self.model_log_dir = self.log_dir / model_name
        self.model_checkpoint_dir = self.checkpoint_dir
        self.model_log_dir.mkdir(parents=True, exist_ok=True)
        self.model_checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def _get_callbacks(self) -> list:
        """
        Create comprehensive training callbacks.

        Returns a list of Keras callbacks for professional training.
        """
        callbacks_list = []

        # 1. Early Stopping
        callbacks_list.append(
            callbacks.EarlyStopping(
                monitor="val_loss",
                patience=EARLY_STOPPING_PATIENCE,
                restore_best_weights=True,
                verbose=1,
                mode="min",
                min_delta=1e-4,
            )
        )

        # 2. ReduceLROnPlateau
        callbacks_list.append(
            callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=REDUCE_LR_FACTOR,
                patience=REDUCE_LR_PATIENCE,
                min_lr=MIN_LR,
                verbose=1,
                mode="min",
            )
        )

        # 3. Model Checkpoint (save best weights only)
        checkpoint_path = str(
            self.model_checkpoint_dir / f"{self.model_name}_best.weights.h5"
        )
        callbacks_list.append(
            callbacks.ModelCheckpoint(
                filepath=checkpoint_path,
                monitor="val_accuracy",
                save_best_only=True,
                save_weights_only=True,
                mode="max",
                verbose=1,
            )
        )

        # 4. CSV Logger
        csv_path = self.model_log_dir / "training_log.csv"
        callbacks_list.append(
            callbacks.CSVLogger(
                filename=str(csv_path),
                separator=",",
                append=False,
            )
        )

        return callbacks_list

    def train(
        self,
        train_dataset,
        val_dataset,
        epochs: int = EPOCHS,
        class_weights: Optional[Dict] = None,
        verbose: int = 1,
    ) -> Dict[str, Any]:
        """
        Train the model with professional callbacks and monitoring.

        Args:
            train_dataset: tf.data.Dataset for training
            val_dataset: tf.data.Dataset for validation
            epochs: Maximum number of epochs
            class_weights: Optional dictionary mapping class indices to weights
            verbose: Verbosity level (0=quiet, 1=progress bar, 2=one line/epoch)

        Returns:
            Training history dictionary
        """
        print(f"\n{'='*70}")
        print(f"🚀 TRAINING: {self.model_name}")
        print(f"{'='*70}")
        print(f"  Epochs: {epochs}")
        print(f"  Batch Size: {BATCH_SIZE}")
        print(f"  Initial LR: {INITIAL_LR}")
        print(f"  Training samples: (see below)")
        print(f"{'='*70}\n")

        # Get callbacks
        callbacks_list = self._get_callbacks()

        # Start timer
        start_time = time.time()

        # Train
        self.history = self.model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=epochs,
            callbacks=callbacks_list,
            class_weight=class_weights,
            verbose=verbose,
        )

        # End timer
        self.training_time = time.time() - start_time

        # Save training summary
        self._save_training_summary()

        print(f"\n✅ Training complete! Time: {self.training_time:.2f}s")
        print(f"  Best val_accuracy: {max(self.history.history['val_accuracy']):.4f}")
        print(f"  Best val_loss: {min(self.history.history['val_loss']):.4f}")

        return self.history.history

    def _save_training_summary(self) -> None:
        """Save training summary as JSON."""
        if self.history is None:
            return

        summary = {
            "model_name": self.model_name,
            "training_time_seconds": self.training_time,
            "epochs_trained": len(self.history.history["loss"]),
            "best_val_accuracy": float(max(self.history.history["val_accuracy"])),
            "best_val_loss": float(min(self.history.history["val_loss"])),
            "best_train_accuracy": float(max(self.history.history["accuracy"])),
            "best_train_loss": float(min(self.history.history["loss"])),
            "final_lr": float(self.history.history.get("lr", [0])[-1]),
        }

        summary_path = self.model_log_dir / "training_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

    def evaluate(self, test_dataset) -> Dict[str, float]:
        """
        Evaluate the trained model on the test set.

        Returns:
            Dictionary of test metrics
        """
        print(f"\n{'='*70}")
        print(f"📊 EVALUATING: {self.model_name}")
        print(f"{'='*70}")

        results = self.model.evaluate(test_dataset, verbose=1)

        metrics = {
            "test_loss": float(results[0]),
            "test_accuracy": float(results[1]),
        }

        print(f"\n  Test Loss: {metrics['test_loss']:.4f}")
        print(f"  Test Accuracy: {metrics['test_accuracy']:.4f}")

        return metrics

    def predict(self, dataset):
        """Generate predictions for a dataset."""
        return self.model.predict(dataset, verbose=1)
