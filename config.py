"""
Configuration file for Leaf Disease Detection Project
All hyperparameters, paths, and settings are centralized here.
"""
import os
from pathlib import Path

# Project Base Path
BASE_DIR = Path(__file__).resolve().parent

# ─── Dataset Paths ───────────────────────────────────────────────────────────
# The PlantVillage dataset - adjust DATA_DIR to point to your archive location
# The archive is typically at the same level as the project directory
DATA_DIR = BASE_DIR.parent / "archive" / "PlantVillage" / "PlantVillage"
# Fallback to outer directory if inner doesn't exist
DATA_DIR_ALT = BASE_DIR / "archive" / "PlantVillage"

# Output directories
RESULTS_DIR = BASE_DIR / "results"
MODELS_DIR = RESULTS_DIR / "models"
PLOTS_DIR = RESULTS_DIR / "plots"
LOGS_DIR = RESULTS_DIR / "logs"
REPORTS_DIR = BASE_DIR / "reports"
ASSETS_DIR = BASE_DIR / "assets"

# Ensure directories exist
for d in [RESULTS_DIR, MODELS_DIR, PLOTS_DIR, LOGS_DIR, REPORTS_DIR, ASSETS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ─── Image Parameters ────────────────────────────────────────────────────────
IMG_WIDTH = 224          # Target width for model input
IMG_HEIGHT = 224         # Target height for model input
IMG_CHANNELS = 3         # RGB
IMG_SIZE = (IMG_WIDTH, IMG_HEIGHT)
INPUT_SHAPE = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

# ─── Dataset Split Ratios ────────────────────────────────────────────────────
TRAIN_RATIO = 0.70       # 70% for training
VAL_RATIO = 0.15         # 15% for validation
TEST_RATIO = 0.15        # 15% for testing
RANDOM_SEED = 42         # For reproducibility

# ─── Training Hyperparameters ───────────────────────────────────────────────
BATCH_SIZE = 32
EPOCHS = 100
INITIAL_LR = 1e-3        # Initial learning rate
MIN_LR = 1e-6            # Minimum learning rate
WEIGHT_DECAY = 1e-4      # L2 regularization factor

# ─── Early Stopping ─────────────────────────────────────────────────────────
EARLY_STOPPING_PATIENCE = 10
REDUCE_LR_PATIENCE = 5
REDUCE_LR_FACTOR = 0.5

# ─── Data Augmentation ──────────────────────────────────────────────────────
AUGMENTATION = {
    "rotation_range": 30,
    "width_shift_range": 0.2,
    "height_shift_range": 0.2,
    "shear_range": 0.15,
    "zoom_range": 0.2,
    "horizontal_flip": True,
    "vertical_flip": False,  # Not appropriate for leaf images
    "brightness_range": (0.8, 1.2),
    "fill_mode": "nearest",
}

# ─── CNN Architecture ────────────────────────────────────────────────────────
CNN_CONFIG = {
    "conv_filters": [32, 64, 128, 256],
    "kernel_size": (3, 3),
    "pool_size": (2, 2),
    "dropout_rate": 0.5,
    "dense_units": [512, 256],
    "batch_norm": True,
}

# ─── Transfer Learning ──────────────────────────────────────────────────────
# Models to compare
TRANSFER_LEARNING_MODELS = [
    "MobileNetV2",
    "EfficientNetB0",
    "ResNet50",
]

# Fine-tuning configuration
FREEZE_BASE_LAYERS = True
FINE_TUNE_EPOCHS = 20
UNFREEZE_LAYERS_FROM = 100  # Unfreeze layers from this index for fine-tuning

# ─── Class Names (auto-detected, but cached here) ───────────────────────────
# These will be populated automatically by the dataset analysis script
CLASS_NAMES = []
NUM_CLASSES = 0

# ─── Healthy vs Diseased Classification ─────────────────────────────────────
HEALTHY_KEYWORDS = ["healthy"]
# Classes containing these keywords will be classified as "healthy"

# ─── Logging ─────────────────────────────────────────────────────────────────
LOG_LEVEL = "INFO"
TENSORBOARD_LOG_DIR = LOGS_DIR / "tensorboard"

# ─── Visualization ──────────────────────────────────────────────────────────
PLOT_STYLE = "seaborn-v0_8-darkgrid"
COLOR_PALETTE = "viridis"
FIGURE_DPI = 150
