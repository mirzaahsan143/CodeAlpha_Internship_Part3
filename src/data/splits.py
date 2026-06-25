"""
Dataset splitting module.
Creates stratified train/validation/test splits to ensure
balanced class representation across all partitions.
"""
import sys
import random
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import TRAIN_RATIO, VAL_RATIO, TEST_RATIO, RANDOM_SEED, BATCH_SIZE


def create_dataset_splits(
    class_paths: Dict[str, List[Path]],
    val_size: float = VAL_RATIO,
    test_size: float = TEST_RATIO,
    random_seed: int = RANDOM_SEED,
) -> Tuple[list, list, list, list, list, list]:
    """
    Create stratified train/validation/test splits.

    Split ratio rationale (70/15/15):
    - **70% Training**: Provides sufficient samples for the model to learn
      meaningful features. The PlantVillage dataset has ~20K images, so 70%
      (~14K images) gives deep learning models enough data to converge.
    - **15% Validation**: Large enough to reliably detect overfitting and
      tune hyperparameters. With ~3K images, validation metrics have low variance.
    - **15% Test**: Held-out set for final evaluation. ~3K images provide
      statistically significant performance estimates.

    Stratification ensures each split maintains the original class distribution,
    which is critical because:
    - Some classes (e.g., Potato___healthy with 152 images) are much smaller
      than others (e.g., Tomato__Tomato_YellowLeaf__Curl_Virus with 3,209 images).
    - Without stratification, a random split could accidentally put all
      samples from a rare class into the test set.

    Returns:
        train_paths, val_paths, test_paths, train_labels, val_labels, test_labels
    """
    # Flatten paths and labels
    all_paths = []
    all_labels = []

    # Create label mapping
    class_names = sorted(class_paths.keys())
    label_map = {name: idx for idx, name in enumerate(class_names)}

    for cls in class_names:
        for fpath in class_paths[cls]:
            all_paths.append(str(fpath))
            all_labels.append(label_map[cls])

    all_paths = np.array(all_paths)
    all_labels = np.array(all_labels)

    # First split: separate training from temp (val + test)
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        all_paths,
        all_labels,
        test_size=(val_size + test_size),
        random_state=random_seed,
        stratify=all_labels,
    )

    # Second split: separate validation from test
    val_ratio_of_temp = val_size / (val_size + test_size)
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths,
        temp_labels,
        test_size=(1 - val_ratio_of_temp),
        random_state=random_seed,
        stratify=temp_labels,
    )

    # Print split summary
    print("\n" + "="*70)
    print("📊 DATASET SPLIT SUMMARY")
    print("="*70)
    print(f"  Training Set   : {len(train_paths):,} images ({len(train_paths)/len(all_paths)*100:.1f}%)")
    print(f"  Validation Set : {len(val_paths):,} images ({len(val_paths)/len(all_paths)*100:.1f}%)")
    print(f"  Test Set       : {len(test_paths):,} images ({len(test_paths)/len(all_paths)*100:.1f}%)")
    print(f"  Total          : {len(all_paths):,} images")
    print("="*70)

    return train_paths, val_paths, test_paths, train_labels, val_labels, test_labels


def get_split_summary(
    train_labels, val_labels, test_labels, class_names
) -> pd.DataFrame:
    """
    Get a DataFrame showing class distribution across splits.
    Useful for verifying stratified split quality.
    """
    data = []
    for cls_idx, cls_name in enumerate(class_names):
        data.append({
            "class": cls_name,
            "train": int(np.sum(train_labels == cls_idx)),
            "val": int(np.sum(val_labels == cls_idx)),
            "test": int(np.sum(test_labels == cls_idx)),
            "total": int(np.sum(train_labels == cls_idx)
                          + np.sum(val_labels == cls_idx)
                          + np.sum(test_labels == cls_idx)),
        })
    return pd.DataFrame(data)
