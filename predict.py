"""
Prediction script - Load trained model and predict disease from new leaf images.
Usage: python predict.py --image path/to/leaf.jpg [--model CustomCNN]
"""
import sys
import argparse
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import tensorflow as tf
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import config
from config import IMG_SIZE, MODELS_DIR   # ← Fix: these were used but never imported


def _get_class_names():
    """Get class names from config, with fallback."""
    if not config.CLASS_NAMES:
        # Auto-detect from dataset
        from src.data.dataset import LeafDataset
        dataset = LeafDataset(config.DATA_DIR)
        dataset.investigate()
        config.CLASS_NAMES = dataset.class_names
        config.NUM_CLASSES = dataset.num_classes
    return config.CLASS_NAMES, config.NUM_CLASSES


def load_trained_model(model_name: str = "CustomCNN"):
    """Load a trained model from the models directory."""
    model_path = MODELS_DIR / f"{model_name}_final.h5"
    if not model_path.exists():
        # Try .weights.h5
        model_path = MODELS_DIR / f"{model_name}_best.weights.h5"
        if not model_path.exists():
            raise FileNotFoundError(
                f"No trained model found for '{model_name}'. "
                f"Train one first using: python train.py --model {model_name}"
            )
        # Need to build model first since we have weights only
        from src.models.factory import get_model
        model = get_model(model_name)
        model.load_weights(str(model_path))
    else:
        model = tf.keras.models.load_model(str(model_path))

    print(f"✅ Loaded model: {model_name}")
    return model


def preprocess_image(image_path: str) -> np.ndarray:
    """
    Preprocess a single image for prediction.

    Steps:
    1. Load image
    2. Convert to RGB (handles grayscale or RGBA)
    3. Resize to model input size
    4. Normalize pixel values to [0, 1]
    5. Add batch dimension
    """
    img = Image.open(image_path)

    # Convert to RGB if necessary
    if img.mode != "RGB":
        img = img.convert("RGB")

    # Resize
    img = img.resize(IMG_SIZE, Image.Resampling.LANCZOS)

    # Convert to array and normalize
    img_array = np.array(img, dtype=np.float32) / 255.0

    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    return img_array


def predict_disease(
    image_path: str,
    model_name: str = "CustomCNN",
    model=None,
):
    """
    Predict disease from a leaf image.
    """
    class_names, n_classes = _get_class_names()

    if model is None:
        model = load_trained_model(model_name)

    # Preprocess
    img_array = preprocess_image(image_path)

    # Predict
    predictions = model.predict(img_array, verbose=0)[0]
    predicted_class_idx = np.argmax(predictions)
    confidence = float(predictions[predicted_class_idx])

    # Get top-3 predictions
    top_3_indices = np.argsort(predictions)[::-1][:3]
    top_3 = [
        {
            "class": class_names[i],
            "confidence": float(predictions[i]),
        }
        for i in top_3_indices
    ]

    result = {
        "image_path": image_path,
        "predicted_class": class_names[predicted_class_idx],
        "predicted_class_idx": int(predicted_class_idx),
        "confidence": confidence,
        "top_3_predictions": top_3,
        "all_confidences": {class_names[i]: float(predictions[i]) for i in range(n_classes)},
    }

    return result


def display_prediction(result: dict):
    """Display prediction results in a formatted way."""
    print("\n" + "="*60)
    print("🌿 LEAF DISEASE PREDICTION RESULTS")
    print("="*60)
    print(f"  Image: {result['image_path']}")
    print(f"  Predicted Disease: {result['predicted_class']}")
    print(f"  Confidence: {result['confidence']:.2%}")
    print("-"*60)
    print("  Top-3 Predictions:")
    for i, pred in enumerate(result["top_3_predictions"], 1):
        print(f"    {i}. {pred['class']:50s} {pred['confidence']:.2%}")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description="Predict leaf disease from image")
    parser.add_argument("--image", type=str, required=True,
                        help="Path to leaf image file")
    parser.add_argument("--model", type=str, default="CustomCNN",
                        choices=["CustomCNN", "MobileNetV2", "EfficientNetB0", "ResNet50"],
                        help="Trained model to use")
    parser.add_argument("--no-display", action="store_true",
                        help="Don't display the image, just print results")
    args = parser.parse_args()

    # Load model
    model = load_trained_model(args.model)

    # Predict
    result = predict_disease(args.image, args.model, model)

    # Display results
    display_prediction(result)

    # Save results to file
    import json
    output_path = Path(args.image).parent / f"prediction_{Path(args.image).stem}.json"
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n📄 Results saved to: {output_path}")

    # Show image with prediction overlay
    if not args.no_display:
        try:
            import matplotlib.pyplot as plt
            img = plt.imread(args.image)
            plt.figure(figsize=(8, 8))
            plt.imshow(img)
            plt.axis("off")
            title = f"Prediction: {result['predicted_class']}\nConfidence: {result['confidence']:.2%}"
            plt.title(title, fontsize=14, fontweight="bold", pad=20)
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"  ⚠️  Could not display image: {e}")


if __name__ == "__main__":
    main()
