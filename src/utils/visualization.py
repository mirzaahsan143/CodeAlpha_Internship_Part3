"""
Professional visualization utilities for EDA and model evaluation.
"""
import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import PLOT_STYLE, COLOR_PALETTE, FIGURE_DPI, PLOTS_DIR, CLASS_NAMES

# Set style
try:
    plt.style.use(PLOT_STYLE)
except Exception:
    plt.style.use("ggplot")


class Visualizer:
    """
    Professional visualization suite.
    Generates publication-quality figures for reports and presentations.
    """

    def __init__(self, output_dir: Path = PLOTS_DIR):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _save(self, filename: str, dpi: int = FIGURE_DPI):
        """Save figure to output directory."""
        path = self.output_dir / filename
        plt.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
        plt.close()
        print(f"  ✅ Saved: {path}")

    # ─── EDA VISUALIZATIONS ─────────────────────────────────────────────

    def plot_class_distribution(
        self,
        class_counts: Dict[str, int],
        healthy_classes: List[str],
    ) -> None:
        """
        Plot class distribution with healthy/diseased color coding.

        This visualization helps identify:
        - Class imbalance (some classes have significantly more samples)
        - Potential biases in the dataset
        - Rare disease classes that may need special handling
        """
        fig, axes = plt.subplots(1, 2, figsize=(18, 8))

        classes = sorted(class_counts.keys())
        counts = [class_counts[c] for c in classes]
        colors = [
            "limegreen" if c in healthy_classes else "tomato"
            for c in classes
        ]

        # Bar plot
        ax = axes[0]
        bars = ax.barh(range(len(classes)), counts, color=colors, edgecolor="white")
        ax.set_yticks(range(len(classes)))
        ax.set_yticklabels([c.replace("_", " ") for c in classes], fontsize=9)
        ax.set_xlabel("Number of Images", fontsize=12)
        ax.set_title("Class Distribution in PlantVillage Dataset", fontsize=14, fontweight="bold")

        # Add count labels
        for i, (bar, count) in enumerate(zip(bars, counts)):
            ax.text(count + 20, bar.get_y() + bar.get_height()/2,
                    f"{count:,}", va="center", fontsize=8)

        # Legend
        legend_patches = [
            mpatches.Patch(color="limegreen", label=f"Healthy ({len(healthy_classes)})"),
            mpatches.Patch(color="tomato", label=f"Diseased ({len(classes) - len(healthy_classes)})"),
        ]
        ax.legend(handles=legend_patches, loc="lower right")

        # Pie chart
        ax = axes[1]
        healthy_count = sum(class_counts[c] for c in healthy_classes)
        diseased_count = sum(class_counts[c] for c in classes if c not in healthy_classes)
        ax.pie(
            [healthy_count, diseased_count],
            labels=[f"Healthy\n{healthy_count:,}", f"Diseased\n{diseased_count:,}"],
            autopct="%1.1f%%",
            colors=["limegreen", "tomato"],
            startangle=90,
            explode=(0.05, 0.05),
            textprops={"fontsize": 12, "fontweight": "bold"},
        )
        ax.set_title("Healthy vs Diseased Distribution", fontsize=14, fontweight="bold")

        plt.tight_layout()
        self._save("class_distribution.png")

    def plot_dimension_distributions(
        self, dataset, class_names: List[str]
    ) -> None:
        """
        Plot distributions of image dimensions and aspect ratios.

        Understanding image dimensions is critical for:
        - Choosing appropriate resize dimensions
        - Detecting anomalies (very different sizes may indicate corrupted data)
        - Determining if aspect ratio preservation is needed
        """
        widths, heights, aspect_ratios, classes = [], [], [], []

        for cls in class_names:
            for fpath in dataset.file_paths[cls]:
                try:
                    with Image.open(fpath) as img:
                        w, h = img.size
                        widths.append(w)
                        heights.append(h)
                        aspect_ratios.append(w / h if h > 0 else 1)
                        classes.append(cls)
                except Exception:
                    continue

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # Width distribution
        axes[0, 0].hist(widths, bins=50, color="steelblue", edgecolor="white", alpha=0.8)
        axes[0, 0].set_xlabel("Width (pixels)", fontsize=11)
        axes[0, 0].set_ylabel("Frequency", fontsize=11)
        axes[0, 0].set_title("Image Width Distribution", fontsize=13, fontweight="bold")
        axes[0, 0].axvline(np.mean(widths), color="red", linestyle="--", label=f"Mean: {np.mean(widths):.0f}")
        axes[0, 0].legend()

        # Height distribution
        axes[0, 1].hist(heights, bins=50, color="coral", edgecolor="white", alpha=0.8)
        axes[0, 1].set_xlabel("Height (pixels)", fontsize=11)
        axes[0, 1].set_ylabel("Frequency", fontsize=11)
        axes[0, 1].set_title("Image Height Distribution", fontsize=13, fontweight="bold")
        axes[0, 1].axvline(np.mean(heights), color="red", linestyle="--", label=f"Mean: {np.mean(heights):.0f}")
        axes[0, 1].legend()

        # Aspect ratio distribution
        axes[0, 2].hist(aspect_ratios, bins=50, color="mediumseagreen", edgecolor="white", alpha=0.8)
        axes[0, 2].set_xlabel("Aspect Ratio (W/H)", fontsize=11)
        axes[0, 2].set_ylabel("Frequency", fontsize=11)
        axes[0, 2].set_title("Aspect Ratio Distribution", fontsize=13, fontweight="bold")
        axes[0, 2].axvline(np.mean(aspect_ratios), color="red", linestyle="--", label=f"Mean: {np.mean(aspect_ratios):.2f}")
        axes[0, 2].legend()

        # Scatter: Width vs Height
        axes[1, 0].scatter(widths, heights, alpha=0.3, c="purple", s=5)
        axes[1, 0].set_xlabel("Width (pixels)", fontsize=11)
        axes[1, 0].set_ylabel("Height (pixels)", fontsize=11)
        axes[1, 0].set_title("Width vs Height", fontsize=13, fontweight="bold")
        axes[1, 0].set_aspect("equal")

        # Box plot by class
        df = pd.DataFrame({"class": classes, "width": widths, "height": heights})
        class_widths = [df[df["class"] == c]["width"].values for c in class_names[:10]]
        bp = axes[1, 1].boxplot(class_widths, labels=[c[:12] for c in class_names[:10]],
                                 patch_artist=True)
        for patch in bp["boxes"]:
            patch.set_facecolor("lightblue")
        axes[1, 1].set_xticklabels([c[:12] for c in class_names[:10]], rotation=45, fontsize=8)
        axes[1, 1].set_ylabel("Width (pixels)", fontsize=11)
        axes[1, 1].set_title("Width Distribution by Class (first 10)", fontsize=13, fontweight="bold")

        # Pixel intensity analysis - actual analysis
        axes[1, 2].axis("off")

        plt.tight_layout()
        self._save("dimension_distributions.png")

    def plot_sample_grid(
        self, dataset, class_names: List[str], samples_per_class: int = 4
    ) -> None:
        """
        Display representative sample images from each class.

        Purpose: Visual inspection of dataset quality and diversity.
        """
        n_classes = len(class_names)
        n_cols = min(5, n_classes)
        n_rows = ((n_classes * samples_per_class) + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, n_rows * 4))
        axes = axes.flatten()

        idx = 0
        for cls in class_names:
            files = dataset.file_paths[cls][:samples_per_class]
            for fpath in files:
                try:
                    img = Image.open(fpath)
                    axes[idx].imshow(img)
                    axes[idx].axis("off")
                    plant_name = cls.replace("___", "\n").replace("_", " ")
                    axes[idx].set_title(plant_name, fontsize=8)
                except Exception:
                    axes[idx].text(0.5, 0.5, "Corrupted", ha="center", va="center")
                    axes[idx].axis("off")
                idx += 1

        # Hide unused axes
        for i in range(idx, len(axes)):
            axes[i].axis("off")

        plt.suptitle("Sample Images from Each Class", fontsize=16, fontweight="bold", y=0.98)
        plt.tight_layout()
        self._save("sample_images_grid.png")

    # ─── TRAINING VISUALIZATIONS ───────────────────────────────────────

    def plot_training_history(
        self,
        history: Dict,
        model_name: str = "Model",
    ) -> None:
        """
        Plot training and validation accuracy/loss curves.

        These curves reveal:
        - Overfitting: diverging train/val curves
        - Underfitting: both curves still improving
        - Learning rate issues: oscillating loss
        - Optimal stopping point
        """
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        epochs = range(1, len(history["accuracy"]) + 1)

        # Accuracy plot
        ax = axes[0]
        ax.plot(epochs, history["accuracy"], "b-", label="Training Accuracy", linewidth=2)
        ax.plot(epochs, history["val_accuracy"], "r-", label="Validation Accuracy", linewidth=2)
        ax.fill_between(epochs, history["accuracy"], history["val_accuracy"],
                         alpha=0.1, color="gray")
        ax.set_xlabel("Epochs", fontsize=12)
        ax.set_ylabel("Accuracy", fontsize=12)
        ax.set_title(f"{model_name} - Accuracy Curves", fontsize=14, fontweight="bold")
        ax.legend(loc="lower right", fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)

        # Loss plot
        ax = axes[1]
        ax.plot(epochs, history["loss"], "b-", label="Training Loss", linewidth=2)
        ax.plot(epochs, history["val_loss"], "r-", label="Validation Loss", linewidth=2)
        ax.fill_between(epochs, history["loss"], history["val_loss"],
                         alpha=0.1, color="gray")
        ax.set_xlabel("Epochs", fontsize=12)
        ax.set_ylabel("Loss", fontsize=12)
        ax.set_title(f"{model_name} - Loss Curves", fontsize=14, fontweight="bold")
        ax.legend(loc="upper right", fontsize=11)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        self._save(f"{model_name}_training_curves.png")

    # ─── EVALUATION VISUALIZATIONS ─────────────────────────────────────

    def plot_confusion_matrix(
        self,
        cm: np.ndarray,
        class_names: List[str],
        model_name: str = "Model",
        normalize: bool = True,
    ) -> None:
        """
        Plot confusion matrix with color-coding.

        The confusion matrix reveals:
        - Which classes are commonly confused (off-diagonal values)
        - Per-class accuracy (diagonal values)
        - Systematic errors (e.g., specific disease pairs)
        """
        if normalize:
            cm_norm = cm.astype("float") / (cm.sum(axis=1)[:, np.newaxis] + 1e-10)
            fmt = ".2f"
            vmax = 1.0
        else:
            cm_norm = cm
            fmt = "d"
            vmax = cm.max()

        # Shorten class names for display
        short_names = [c.replace("___", "\n").replace("_", " ")
                       for c in class_names]

        fig, ax = plt.subplots(figsize=(16, 14))

        sns.heatmap(
            cm_norm,
            annot=True,
            fmt=fmt,
            cmap="Blues",
            xticklabels=short_names,
            yticklabels=short_names,
            ax=ax,
            vmin=0,
            vmax=vmax,
            linewidths=0.5,
            cbar_kws={"shrink": 0.8, "label": "Proportion" if normalize else "Count"},
        )

        ax.set_xlabel("Predicted Label", fontsize=14)
        ax.set_ylabel("True Label", fontsize=14)
        ax.set_title(f"{model_name} - Confusion Matrix", fontsize=16, fontweight="bold")
        plt.xticks(rotation=45, ha="right", fontsize=8)
        plt.yticks(rotation=0, fontsize=8)

        plt.tight_layout()
        self._save(f"{model_name}_confusion_matrix.png")

    def plot_roc_curves(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        class_names: List[str],
        model_name: str = "Model",
    ) -> None:
        """
        Plot ROC curves for each class (one-vs-rest).

        ROC curves show the trade-off between true positive rate and
        false positive rate. AUC close to 1.0 indicates excellent
        class separation capability.
        """
        from sklearn.metrics import roc_curve, auc

        n_classes = len(class_names)
        y_true_bin = np.zeros((len(y_true), n_classes))
        for i in range(n_classes):
            y_true_bin[:, i] = (y_true == i).astype(int)

        fig, ax = plt.subplots(figsize=(12, 10))

        colors = plt.cm.viridis(np.linspace(0, 1, n_classes))

        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
            roc_auc = auc(fpr, tpr)
            ax.plot(
                fpr, tpr, color=colors[i], lw=2,
                label=f"{class_names[i][:25]} (AUC = {roc_auc:.3f})",
            )

        # Diagonal line (random classifier)
        ax.plot([0, 1], [0, 1], "k--", lw=2, label="Random (AUC = 0.500)")

        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.02])
        ax.set_xlabel("False Positive Rate", fontsize=13)
        ax.set_ylabel("True Positive Rate", fontsize=13)
        ax.set_title(f"{model_name} - ROC Curves (One-vs-Rest)", fontsize=15, fontweight="bold")
        ax.legend(loc="lower right", fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        self._save(f"{model_name}_roc_curves.png")

    def plot_error_analysis(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        file_paths: List[str],
        class_names: List[str],
        model_name: str = "Model",
    ) -> None:
        """
        Analyze and visualize misclassified images.

        Understanding misclassifications helps identify:
        - Classes that are inherently similar (need more data or different features)
        - Potential data labeling errors
        - Model blind spots
        """
        misclassified = np.where(y_true != y_pred)[0]

        print(f"\n{'='*70}")
        print(f"🔍 ERROR ANALYSIS: {model_name}")
        print(f"{'='*70}")
        print(f"  Total misclassified: {len(misclassified)} / {len(y_true)} "
              f"({len(misclassified)/len(y_true)*100:.2f}%)")

        # Per-class error rates
        print("\n  Per-class Error Rates:")
        print(f"  {'Class':<45} {'Errors':<10} {'Rate':<10}")
        print(f"  {'-'*65}")
        for i, cls in enumerate(class_names):
            class_indices = np.where(y_true == i)[0]
            if len(class_indices) > 0:
                class_errors = np.sum(y_pred[class_indices] != y_true[class_indices])
                error_rate = class_errors / len(class_indices)
                print(f"  {cls:<45} {class_errors:<10} {error_rate:.4f}")

        # Confusion pairs
        print("\n  Top Confusion Pairs (True → Predicted):")
        confusion_pairs = {}
        for i in misclassified:
            pair = (class_names[y_true[i]], class_names[y_pred[i]])
            confusion_pairs[pair] = confusion_pairs.get(pair, 0) + 1

        sorted_pairs = sorted(confusion_pairs.items(), key=lambda x: -x[1])
        for (true_cls, pred_cls), count in sorted_pairs[:10]:
            print(f"  {true_cls:<40} → {pred_cls:<35} [{count} errors]")

        # Plot misclassified examples
        if len(misclassified) > 0:
            n_examples = min(16, len(misclassified))
            fig, axes = plt.subplots(4, 4, figsize=(16, 16))
            axes = axes.flatten()

            for i in range(n_examples):
                idx = misclassified[i]
                try:
                    img_path = file_paths[idx]
                    img = Image.open(img_path)
                    axes[i].imshow(img)
                    true_name = class_names[y_true[idx]].replace("_", " ")
                    pred_name = class_names[y_pred[idx]].replace("_", " ")
                    axes[i].set_title(
                        f"True: {true_name[:20]}\nPred: {pred_name[:20]}",
                        fontsize=8, color="red",
                    )
                except Exception:
                    axes[i].text(0.5, 0.5, "Error loading", ha="center", va="center")
                axes[i].axis("off")

            for i in range(n_examples, len(axes)):
                axes[i].axis("off")

            plt.suptitle(f"{model_name} - Misclassified Examples", fontsize=14, fontweight="bold")
            plt.tight_layout()
            self._save(f"{model_name}_error_analysis.png")

    # ─── MODEL COMPARISON ──────────────────────────────────────────────

    def plot_model_comparison(
        self, results: Dict[str, Dict]
    ) -> None:
        """
        Create a comprehensive comparison bar chart across models.

        Compares: Accuracy, Precision, Recall, F1-Score, Training Time
        """
        model_names = list(results.keys())
        metrics = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]

        fig, axes = plt.subplots(1, 2, figsize=(16, 7))

        # Performance metrics comparison
        ax = axes[0]
        x = np.arange(len(model_names))
        width = 0.2

        colors = ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D"]
        for i, metric in enumerate(metrics):
            values = [results[m].get(metric, 0) for m in model_names]
            bars = ax.bar(x + i * width, values, width, label=metric.replace("_", " ").title(),
                          color=colors[i], edgecolor="white")
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f"{val:.3f}", ha="center", va="bottom", fontsize=8)

        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(model_names, fontsize=11)
        ax.set_ylabel("Score", fontsize=12)
        ax.set_title("Model Performance Comparison", fontsize=14, fontweight="bold")
        ax.legend(loc="lower right", fontsize=10)
        ax.set_ylim(0, 1.15)
        ax.grid(True, alpha=0.3, axis="y")

        # Training time comparison
        ax = axes[1]
        times = [results[m].get("training_time", 0) for m in model_names]
        bar_colors = [colors[i % len(colors)] for i in range(len(model_names))]
        bars = ax.barh(model_names, times, color=bar_colors, edgecolor="white")
        for bar, val in zip(bars, times):
            ax.text(bar.get_width() + 5, bar.get_y() + bar.get_height()/2,
                    f"{val:.1f}s", va="center", fontsize=11)

        ax.set_xlabel("Training Time (seconds)", fontsize=12)
        ax.set_title("Training Time Comparison", fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="x")

        plt.tight_layout()
        self._save("model_comparison.png")

    def plot_pixel_intensity_analysis(
        self, dataset, class_names: List[str], num_samples: int = 500
    ) -> None:
        """
        Analyze pixel intensity distributions and RGB channel characteristics.

        This analysis reveals:
        - Overall brightness distribution (under/over-exposure)
        - Color balance across the dataset
        - Channel-wise intensity patterns that may indicate
          dataset-wide color biases
        """
        all_pixels_r, all_pixels_g, all_pixels_b = [], [], []
        sampled = 0

        for cls in class_names:
            for fpath in dataset.file_paths[cls]:
                if sampled >= num_samples:
                    break
                try:
                    with Image.open(fpath) as img:
                        img = img.convert("RGB")
                        arr = np.array(img, dtype=np.float32)
                        all_pixels_r.extend(arr[:, :, 0].flatten())
                        all_pixels_g.extend(arr[:, :, 1].flatten())
                        all_pixels_b.extend(arr[:, :, 2].flatten())
                        sampled += 1
                except Exception:
                    continue
            if sampled >= num_samples:
                break

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # RGB histograms
        colors = ["red", "green", "blue"]
        channels = [all_pixels_r, all_pixels_g, all_pixels_b]
        channel_names = ["Red Channel", "Green Channel", "Blue Channel"]

        for i, (ax, channel, color, name) in enumerate(
            zip(axes[0], channels, colors, channel_names)
        ):
            ax.hist(channel, bins=256, color=color, alpha=0.7, range=(0, 255))
            ax.set_xlabel("Pixel Intensity", fontsize=10)
            ax.set_ylabel("Frequency", fontsize=10)
            ax.set_title(f"{name}\nMean={np.mean(channel):.1f}, Std={np.std(channel):.1f}",
                         fontsize=11, fontweight="bold")
            ax.axvline(np.mean(channel), color="black", linestyle="--", linewidth=1)

        # Combined RGB histogram
        ax = axes[1, 0]
        ax.hist(all_pixels_r, bins=256, color="red", alpha=0.4, label="Red", range=(0, 255))
        ax.hist(all_pixels_g, bins=256, color="green", alpha=0.4, label="Green", range=(0, 255))
        ax.hist(all_pixels_b, bins=256, color="blue", alpha=0.4, label="Blue", range=(0, 255))
        ax.set_xlabel("Pixel Intensity", fontsize=11)
        ax.set_ylabel("Frequency", fontsize=11)
        ax.set_title("Combined RGB Channel Distributions", fontsize=13, fontweight="bold")
        ax.legend()

        # Mean intensity per channel
        ax = axes[1, 1]
        means = [np.mean(all_pixels_r), np.mean(all_pixels_g), np.mean(all_pixels_b)]
        stds = [np.std(all_pixels_r), np.std(all_pixels_g), np.std(all_pixels_b)]
        x_pos = [0, 1, 2]
        bars = ax.bar(x_pos, means, yerr=stds, color=["red", "green", "blue"],
                      alpha=0.7, capsize=10, edgecolor="black")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(["Red", "Green", "Blue"], fontsize=12)
        ax.set_ylabel("Mean Pixel Intensity", fontsize=11)
        ax.set_title("Mean RGB Channel Intensities", fontsize=13, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y")
        for bar, mean, std in zip(bars, means, stds):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 1,
                    f"{mean:.1f}\n±{std:.1f}", ha="center", fontsize=9)

        # Grayscale intensity (average of RGB)
        ax = axes[1, 2]
        gray_pixels = (np.array(all_pixels_r) + np.array(all_pixels_g) + np.array(all_pixels_b)) / 3.0
        ax.hist(gray_pixels, bins=100, color="gray", alpha=0.7, edgecolor="black", range=(0, 255))
        ax.axvline(np.mean(gray_pixels), color="darkred", linestyle="--",
                   linewidth=2, label=f"Mean: {np.mean(gray_pixels):.1f}")
        ax.axvline(np.median(gray_pixels), color="darkblue", linestyle=":",
                   linewidth=2, label=f"Median: {np.median(gray_pixels):.1f}")
        ax.set_xlabel("Intensity (Grayscale)", fontsize=11)
        ax.set_ylabel("Frequency", fontsize=11)
        ax.set_title("Overall Brightness Distribution", fontsize=13, fontweight="bold")
        ax.legend()

        plt.tight_layout()
        self._save("pixel_intensity_analysis.png")

    def create_eda_report(self, dataset, class_names: List[str]) -> None:
        """
        Create comprehensive EDA report with all visualizations.
        """
        print("\n" + "="*70)
        print("📈 PHASE 2: EXPLORATORY DATA ANALYSIS")
        print("="*70)

        # 1. Class distribution
        print("\n1️⃣  Class Distribution Analysis...")
        class_counts = dataset.class_counts
        healthy_classes = [
            c for c in class_names
            if any(kw.lower() in c.lower() for kw in ["healthy"])
        ]
        self.plot_class_distribution(class_counts, healthy_classes)

        # 2. Sample images
        print("\n2️⃣  Sample Images Grid...")
        self.plot_sample_grid(dataset, class_names)

        # 3. Dimension analysis
        print("\n3️⃣  Dimension Analysis...")
        self.plot_dimension_distributions(dataset, class_names)

        # 4. Pixel intensity analysis
        print("\n4️⃣  Pixel Intensity & RGB Analysis...")
        self.plot_pixel_intensity_analysis(dataset, class_names)

        print("\n✅ EDA complete! All plots saved to:", self.output_dir)
