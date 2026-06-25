"""
Leaf Disease Detection System - Main Entry Point
CodeAlpha Data Science Internship Project

Orchestrates the complete pipeline from dataset investigation to model evaluation.
"""
import sys
import argparse
import json
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import tensorflow as tf

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import config
from config import (
    DATA_DIR,
    IMG_SIZE, BATCH_SIZE, INITIAL_LR, EPOCHS,
    MODELS_DIR, PLOTS_DIR, RESULTS_DIR, REPORTS_DIR,
    TRANSFER_LEARNING_MODELS, CNN_CONFIG,
)
from src.data.dataset import LeafDataset
from src.data.splits import create_dataset_splits, get_split_summary
from src.data.augmentation import create_tf_dataset
from src.models.factory import get_model
from src.training.trainer import ModelTrainer
from src.utils.metrics import ClassificationMetrics
from src.utils.visualization import Visualizer


def setup_gpu():
    """Configure GPU for optimal performance."""
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"✅ Using {len(gpus)} GPU(s)")
        except RuntimeError as e:
            print(f"⚠️  GPU config error: {e}")
    else:
        print("ℹ️  No GPU found, using CPU")


def phase_1_2_dataset_investigation(viz: Visualizer) -> LeafDataset:
    """Phase 1 & 2: Dataset investigation and EDA."""
    print("\n" + "█"*70)
    print("█  PHASES 1 & 2: DATASET INVESTIGATION & EDA")
    print("█"*70)

    # Initialize dataset
    dataset = LeafDataset(DATA_DIR)

    # Phase 1: Dataset investigation
    summary = dataset.investigate()

    # Save class names to config module for other modules to access
    config.CLASS_NAMES = dataset.class_names
    config.NUM_CLASSES = dataset.num_classes

    # Save summary as JSON
    summary_path = REPORTS_DIR / "dataset_investigation.json"
    # Convert non-serializable items
    serializable_summary = summary.copy()
    serializable_summary["corrupted_images"] = [
        {"file": f, "error": e} for f, e in summary["corrupted_images"]
    ]
    serializable_summary["duplicate_images"] = [
        {"duplicate": d, "original": o} for d, o in summary["duplicate_images"]
    ]
    with open(summary_path, "w") as f:
        json.dump(serializable_summary, f, indent=2, default=str)
    print(f"\n✅ Investigation summary saved to {summary_path}")

    # Generate dataset report
    dataset.generate_report(REPORTS_DIR / "dataset_report.md")

    # Phase 2: EDA
    viz.create_eda_report(dataset, config.CLASS_NAMES)

    return dataset


def phase_3_4_preprocessing(dataset: LeafDataset):
    """Phase 3 & 4: Data preprocessing and splits."""
    print("\n" + "█"*70)
    print("█  PHASES 3 & 4: DATA PREPROCESSING & SPLITS")
    print("█"*70)

    # Get clean file paths (remove corrupted and duplicate)
    print("\n🔧 Cleaning dataset...")
    clean_paths = dataset.get_clean_file_paths()
    total_clean = sum(len(v) for v in clean_paths.values())
    total_original = sum(len(v) for v in dataset.file_paths.values())
    print(f"  Removed {total_original - total_clean} problematic images")
    print(f"  Clean images: {total_clean:,}")

    # Create stratified splits
    train_paths, val_paths, test_paths, train_labels, val_labels, test_labels = \
        create_dataset_splits(clean_paths)

    # Show split distribution
    split_df = get_split_summary(train_labels, val_labels, test_labels, config.CLASS_NAMES)
    print("\nSplit distribution:")
    print(split_df.to_string(index=False))

    # Save split info
    split_info = {
        "train_count": len(train_paths),
        "val_count": len(val_paths),
        "test_count": len(test_paths),
        "class_names": config.CLASS_NAMES,
        "num_classes": config.NUM_CLASSES,
    }
    with open(REPORTS_DIR / "split_info.json", "w") as f:
        json.dump(split_info, f, indent=2)

    return train_paths, val_paths, test_paths, train_labels, val_labels, test_labels


def phase_5_6_7_8_model_pipeline(
    train_paths, val_paths, test_paths,
    train_labels, val_labels, test_labels,
    dataset: LeafDataset,
    viz: Visualizer,
    train_custom_cnn: bool = True,
    train_transfer: bool = True,
):
    """Phases 5-8: Model development, training, and evaluation."""
    all_results = {}

    # Create tf.data datasets
    print("\n🔧 Creating data pipelines...")
    train_ds = create_tf_dataset(train_paths, train_labels, BATCH_SIZE, training=True)
    val_ds = create_tf_dataset(val_paths, val_labels, BATCH_SIZE, training=False)
    test_ds = create_tf_dataset(test_paths, test_labels, BATCH_SIZE, training=False)

    models_to_train = []

    if train_custom_cnn:
        models_to_train.append("CustomCNN")

    if train_transfer:
        models_to_train.extend(TRANSFER_LEARNING_MODELS)

    for model_name in models_to_train:
        print(f"\n{'█'*70}")
        print(f"█  MODEL: {model_name}")
        print(f"{'█'*70}")

        # Phase 5 & 6: Build model
        print(f"\n🔧 Building {model_name}...")
        model = get_model(model_name, learning_rate=INITIAL_LR)

        # Phase 7: Train
        trainer = ModelTrainer(model, model_name)
        history = trainer.train(train_ds, val_ds, epochs=EPOCHS)

        # Save model
        model_path = MODELS_DIR / f"{model_name}_final.h5"
        model.save(str(model_path))
        print(f"\n✅ Model saved to {model_path}")

        # Phase 8: Evaluate
        print(f"\n📊 Evaluating {model_name}...")
        test_metrics = trainer.evaluate(test_ds)

        # Get predictions
        y_prob = model.predict(test_ds, verbose=1)
        y_pred = np.argmax(y_prob, axis=1)

        # Compute classification metrics
        metrics_calc = ClassificationMetrics(test_labels, y_pred, y_prob)
        all_metrics = metrics_calc.get_all_metrics()
        all_metrics["test_loss"] = test_metrics["test_loss"]
        all_metrics["test_accuracy"] = test_metrics["test_accuracy"]
        all_metrics["training_time"] = trainer.training_time
        all_results[model_name] = all_metrics

        metrics_calc.print_summary(config.CLASS_NAMES)

        # Save per-class metrics
        per_class = metrics_calc.get_per_class_metrics(config.CLASS_NAMES)
        per_class.to_csv(RESULTS_DIR / f"{model_name}_per_class_metrics.csv")

        # Visualizations
        cm = metrics_calc.get_confusion_matrix()
        viz.plot_training_history(history, model_name)
        viz.plot_confusion_matrix(cm, config.CLASS_NAMES, model_name)

        # ROC curves
        viz.plot_roc_curves(test_labels, y_prob, config.CLASS_NAMES, model_name)

        # Phase 9: Error analysis
        viz.plot_error_analysis(
            test_labels, y_pred,
            test_paths, config.CLASS_NAMES, model_name
        )

        # Save predictions
        pred_df = {
            "true_label": test_labels.tolist(),
            "predicted_label": y_pred.tolist(),
            "true_class": [config.CLASS_NAMES[i] for i in test_labels],
            "predicted_class": [config.CLASS_NAMES[i] for i in y_pred],
            "confidence": np.max(y_prob, axis=1).tolist(),
        }
        import pandas as pd
        pd.DataFrame(pred_df).to_csv(RESULTS_DIR / f"{model_name}_predictions.csv", index=False)

    # Model comparison
    if len(all_results) > 1:
        viz.plot_model_comparison(all_results)

    # Save all results
    with open(RESULTS_DIR / "all_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "="*70)
    print("🎉 TRAINING & EVALUATION COMPLETE!")
    print("="*70)
    for name, metrics in all_results.items():
        print(f"  {name:<20} Acc: {metrics['test_accuracy']:.4f}  "
              f"F1: {metrics['f1_macro']:.4f}  "
              f"Time: {metrics['training_time']:.1f}s")
    print("="*70)

    return all_results


def main():
    """Main execution pipeline."""
    parser = argparse.ArgumentParser(
        description="Leaf Disease Detection System - Complete Pipeline"
    )
    parser.add_argument(
        "--skip-eda", action="store_true",
        help="Skip EDA and dataset investigation"
    )
    parser.add_argument(
        "--cnn-only", action="store_true",
        help="Train only the custom CNN model"
    )
    parser.add_argument(
        "--transfer-only", action="store_true",
        help="Train only transfer learning models"
    )
    parser.add_argument(
        "--epochs", type=int, default=EPOCHS,
        help=f"Maximum training epochs (default: {EPOCHS})"
    )
    parser.add_argument(
        "--batch-size", type=int, default=BATCH_SIZE,
        help=f"Batch size (default: {BATCH_SIZE})"
    )

    args = parser.parse_args()
    global EPOCHS, BATCH_SIZE
    EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size

    # Setup
    print("="*70)
    print("🌿 LEAF DISEASE DETECTION SYSTEM")
    print("   CodeAlpha Data Science Internship Project")
    print("="*70)
    setup_gpu()

    # Initialize visualizer
    viz = Visualizer(PLOTS_DIR)

    # Phase 1 & 2
    if not args.skip_eda:
        dataset = phase_1_2_dataset_investigation(viz)
    else:
        dataset = LeafDataset(DATA_DIR)
        dataset.investigate()

    # Phase 3 & 4
    train_paths, val_paths, test_paths, train_labels, val_labels, test_labels = \
        phase_3_4_preprocessing(dataset)

    # Phases 5-9
    all_results = phase_5_6_7_8_model_pipeline(
        train_paths, val_paths, test_paths,
        train_labels, val_labels, test_labels,
        dataset, viz,
        train_custom_cnn=not args.transfer_only,
        train_transfer=not args.cnn_only,
    )

    print("\n✅ ALL PHASES COMPLETE!")
    print(f"📁 Results saved to: {RESULTS_DIR}")
    print(f"📁 Plots saved to: {PLOTS_DIR}")
    print(f"📁 Models saved to: {MODELS_DIR}")
    print(f"📁 Reports saved to: {REPORTS_DIR}")


if __name__ == "__main__":
    main()
