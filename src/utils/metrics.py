"""
Evaluation metrics computation module.
Computes accuracy, precision, recall, F1-score, and confusion matrix.
"""
import sys
from pathlib import Path
from typing import Tuple, Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
)


class ClassificationMetrics:
    """
    Comprehensive classification metrics calculator.

    Metrics Explained:
    - **Accuracy**: (TP + TN) / (TP + TN + FP + FN). Overall correctness.
    - **Precision**: TP / (TP + FP). When the model predicts a disease, how often
      is it correct? Important for avoiding false alarms.
    - **Recall**: TP / (TP + FN). Of all actual disease cases, how many did the
      model catch? Critical for not missing diseased plants.
    - **F1-Score**: Harmonic mean of precision and recall. Balances both concerns.
    """

    def __init__(self, y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray = None):
        """
        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            y_prob: Predicted probabilities (for ROC AUC)
        """
        self.y_true = np.array(y_true)
        self.y_pred = np.array(y_pred)
        self.y_prob = y_prob

    def get_all_metrics(self) -> Dict[str, float]:
        """Compute all classification metrics."""
        metrics = {
            "accuracy": float(accuracy_score(self.y_true, self.y_pred)),
            "precision_macro": float(precision_score(self.y_true, self.y_pred, average="macro", zero_division=0)),
            "recall_macro": float(recall_score(self.y_true, self.y_pred, average="macro", zero_division=0)),
            "f1_macro": float(f1_score(self.y_true, self.y_pred, average="macro", zero_division=0)),
            "precision_weighted": float(precision_score(self.y_true, self.y_pred, average="weighted", zero_division=0)),
            "recall_weighted": float(recall_score(self.y_true, self.y_pred, average="weighted", zero_division=0)),
            "f1_weighted": float(f1_score(self.y_true, self.y_pred, average="weighted", zero_division=0)),
        }
        return metrics

    def get_per_class_metrics(self, class_names: List[str]) -> pd.DataFrame:
        """Get precision, recall, f1 per class."""
        report = classification_report(
            self.y_true, self.y_pred,
            target_names=class_names,
            output_dict=True,
            zero_division=0,
        )
        df = pd.DataFrame(report).transpose()
        return df

    def get_confusion_matrix(self) -> np.ndarray:
        """Get the confusion matrix."""
        return confusion_matrix(self.y_true, self.y_pred)

    def get_roc_auc(self, num_classes: int) -> Dict[str, float]:
        """
        Compute ROC AUC scores per class (one-vs-rest).

        Note: ROC AUC is only meaningful for multi-class when probability
        predictions are available.
        """
        if self.y_prob is None:
            return {}

        try:
            if num_classes == 2:
                auc_scores = {"roc_auc": float(roc_auc_score(self.y_true, self.y_prob[:, 1]))}
            else:
                # One-vs-Rest
                y_true_bin = np.zeros((len(self.y_true), num_classes))
                for i in range(num_classes):
                    y_true_bin[:, i] = (self.y_true == i).astype(int)

                auc_scores = {}
                for i in range(num_classes):
                    try:
                        auc_scores[f"roc_auc_class_{i}"] = float(
                            roc_auc_score(y_true_bin[:, i], self.y_prob[:, i])
                        )
                    except ValueError:
                        auc_scores[f"roc_auc_class_{i}"] = 0.5

                auc_scores["roc_auc_macro"] = np.mean(
                    [v for k, v in auc_scores.items()]
                )

            return auc_scores
        except Exception as e:
            print(f"  ⚠️  Could not compute ROC AUC: {e}")
            return {}

    def print_summary(self, class_names: List[str]) -> None:
        """Print a formatted summary of all metrics."""
        metrics = self.get_all_metrics()
        per_class = self.get_per_class_metrics(class_names)

        print("\n" + "="*70)
        print("📊 CLASSIFICATION REPORT")
        print("="*70)
        print(f"\n  Accuracy          : {metrics['accuracy']:.4f}")
        print(f"  Precision (macro) : {metrics['precision_macro']:.4f}")
        print(f"  Recall (macro)    : {metrics['recall_macro']:.4f}")
        print(f"  F1-Score (macro)  : {metrics['f1_macro']:.4f}")
        print()

        print("Per-Class Performance:")
        print("-"*70)
        for cls_name in class_names:
            if cls_name in per_class.index:
                row = per_class.loc[cls_name]
                print(f"  {cls_name:<45} P:{row['precision']:.3f}  "
                      f"R:{row['recall']:.3f}  F1:{row['f1-score']:.3f}")
        print("-"*70)
