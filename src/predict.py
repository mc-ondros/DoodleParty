"""
Prediction and evaluation script for QuickDraw classifier.
"""

import argparse
import numpy as np
import pickle
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns


def load_model_and_mapping(model_path, data_dir="data/processed"):
    """Load trained model and class mapping."""
    model = keras.models.load_model(model_path)
    
    with open(Path(data_dir) / "class_mapping.pkl", 'rb') as f:
        class_mapping = pickle.load(f)
    
    idx_to_class = {v: k for k, v in class_mapping.items()}
    
    return model, idx_to_class


def predict_image(model, idx_to_class, image_path, threshold=0.5):
    """
    Predict if a single image belongs to the dataset (binary classification).
    
    Args:
        model: Trained model
        idx_to_class: Class mapping
        image_path: Path to image file
        threshold: Decision threshold (default 0.5)
    
    Returns:
        class_name: 'positive' or 'negative'
        confidence: Confidence score (0-1)
        probability: Raw model output
    """
    # Load and preprocess image
    img = Image.open(image_path).convert('L')  # Convert to grayscale
    img = img.resize((28, 28))
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)
    
    # Predict
    probability = model.predict(img_array, verbose=0)[0][0]
    
    # Determine class based on threshold
    if probability >= threshold:
        class_name = 'positive (in-distribution)'
        confidence = probability
    else:
        class_name = 'negative (out-of-distribution)'
        confidence = 1 - probability
    
    return class_name, confidence, probability


def evaluate_model(model_path, data_dir="data/processed"):
    """
    Evaluate binary classification model on test set.
    Generate confusion matrix and classification metrics.
    """
    print("Loading model and data...")
    model, idx_to_class = load_model_and_mapping(model_path, data_dir)
    
    # Load test data
    X_test = np.load(Path(data_dir) / "X_test.npy")
    y_test = np.load(Path(data_dir) / "y_test.npy")
    
    print(f"Test set size: {len(X_test)}")
    print(f"Positive samples: {(y_test==1).sum()}, Negative samples: {(y_test==0).sum()}")
    
    # Evaluate
    print("\nEvaluating model...")
    loss, accuracy, auc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test AUC: {auc:.4f}")
    
    # Get predictions
    print("Getting predictions...")
    y_pred_proba = model.predict(X_test, verbose=0).flatten()
    y_pred = (y_pred_proba >= 0.5).astype(int)
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(
        y_test, y_pred,
        target_names=['negative (OOD)', 'positive (ID)']
    ))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        xticklabels=['negative (OOD)', 'positive (ID)'],
        yticklabels=['negative (OOD)', 'positive (ID)'],
        cmap='Blues'
    )
    plt.title('Binary Classification - Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    output_path = Path(model_path).parent / "confusion_matrix.png"
    plt.savefig(output_path, dpi=100)
    print(f"✓ Confusion matrix saved to {output_path}")


def predict_batch(model_path, image_dir, data_dir="data/processed"):
    """Make predictions for all images in a directory."""
    model, idx_to_class = load_model_and_mapping(model_path, data_dir)
    
    image_dir = Path(image_dir)
    image_files = list(image_dir.glob("*.png")) + list(image_dir.glob("*.jpg"))
    
    print(f"Found {len(image_files)} images\n")
    
    results = []
    for image_file in image_files:
        try:
            class_name, confidence, all_probs = predict_image(model, idx_to_class, image_file)
            print(f"{image_file.name}: {class_name} ({confidence:.2%})")
            results.append({
                'file': image_file.name,
                'predicted_class': class_name,
                'confidence': confidence
            })
        except Exception as e:
            print(f"Error processing {image_file.name}: {e}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Binary classifier for doodle detection")
    parser.add_argument("--model", required=True, help="Path to trained model")
    parser.add_argument("--data-dir", default="data/processed", help="Directory with processed data")
    parser.add_argument("--threshold", type=float, default=0.5, help="Classification threshold")
    
    # Prediction modes
    subparsers = parser.add_subparsers(dest='mode', help='Prediction mode')
    
    # Single image prediction
    single_parser = subparsers.add_parser('single', help='Predict single image')
    single_parser.add_argument("--image", required=True, help="Path to image file")
    
    # Evaluate
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate on test set')
    
    # Batch prediction
    batch_parser = subparsers.add_parser('batch', help='Predict batch of images')
    batch_parser.add_argument("--image-dir", required=True, help="Directory with images")
    
    args = parser.parse_args()
    
    if args.mode == 'single':
        model, idx_to_class = load_model_and_mapping(args.model, args.data_dir)
        class_name, confidence, probability = predict_image(model, idx_to_class, args.image, args.threshold)
        print(f"\nPredicted: {class_name}")
        print(f"Confidence: {confidence:.2%}")
        print(f"Raw probability: {probability:.4f}")
    
    elif args.mode == 'evaluate':
        evaluate_model(args.model, args.data_dir)
    
    elif args.mode == 'batch':
        predict_batch(args.model, args.image_dir, args.data_dir)
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
