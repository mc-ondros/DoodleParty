"""
Prediction and evaluation script for QuickDraw classifier.

Handles model inference, batch predictions, and evaluation metrics
for binary classification tasks.

Related:
- src/core/models.py (model architectures)
- scripts/train.py (model training)
- src/data/augmentation.py (data normalization)

Exports:
- load_model_and_mapping, predict_image, evaluate_model, predict_batch
"""

import argparse
import numpy as np
import pickle
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

import tensorflow as tf
import keras
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# Import normalization from data pipeline
from data_pipeline import normalize_image, normalize_batch

# Import region-based detection
from src.core.patch_extraction import (
    SlidingWindowDetector,
    AggregationStrategy,
    DetectionResult,
    visualize_detections
)


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
    
    # Apply per-image normalization (CRITICAL - must match training!)
    img_array = normalize_image(img_array)
    
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


def predict_image_region_based(
    model,
    idx_to_class,
    image_path,
    patch_size=(128, 128),
    stride=None,
    min_content_ratio=0.05,
    max_patches=16,
    early_stopping=True,
    aggregation_strategy=AggregationStrategy.MAX,
    threshold=0.5,
    visualize=False,
    save_visualization_path=None
):
    """
    Predict using region-based detection with sliding window.
    
    This approach is more robust against content dilution attacks where
    inappropriate content is mixed with innocent content.
    
    Args:
        model: Trained model
        idx_to_class: Class mapping
        image_path: Path to image file
        patch_size: Size of patches to extract (default 128x128)
        stride: Sliding window stride (None = no overlap)
        min_content_ratio: Minimum content to analyze a patch
        max_patches: Maximum number of patches to analyze
        early_stopping: Whether to stop on first positive detection
        aggregation_strategy: Strategy for combining patch predictions
        threshold: Decision threshold (default 0.5)
        visualize: Whether to create a visualization
        save_visualization_path: Path to save visualization
    
    Returns:
        class_name: 'positive' or 'negative'
        confidence: Confidence score (0-1)
        detection_result: Detailed DetectionResult object
    """
    # Load and preprocess image
    img = Image.open(image_path).convert('L')  # Convert to grayscale
    img_array = np.array(img, dtype=np.float32) / 255.0
    
    # Apply per-image normalization
    img_array = normalize_image(img_array)
    
    # Add channel dimension
    if len(img_array.shape) == 2:
        img_array = np.expand_dims(img_array, axis=-1)
    
    # Create sliding window detector
    detector = SlidingWindowDetector(
        model=model,
        patch_size=patch_size,
        stride=stride,
        min_content_ratio=min_content_ratio,
        max_patches=max_patches,
        early_stopping=early_stopping,
        early_stop_threshold=0.9,
        aggregation_strategy=aggregation_strategy,
        classification_threshold=threshold
    )
    
    # Perform detection (use batch inference for efficiency)
    detection_result = detector.detect_batch(img_array)
    
    # Determine class based on result
    if detection_result.is_positive:
        class_name = 'positive (in-distribution)'
        confidence = detection_result.confidence
    else:
        class_name = 'negative (out-of-distribution)'
        confidence = 1 - detection_result.confidence
    
    # Optionally visualize
    if visualize or save_visualization_path:
        visualize_detections(
            img_array,
            detection_result,
            save_path=save_visualization_path
        )
    
    return class_name, confidence, detection_result


def predict_batch_region_based(
    model_path,
    image_dir,
    data_dir="data/processed",
    patch_size=(128, 128),
    stride=None,
    aggregation_strategy=AggregationStrategy.MAX,
    save_visualizations=False
):
    """
    Make region-based predictions for all images in a directory.
    
    Args:
        model_path: Path to trained model
        image_dir: Directory containing images
        data_dir: Directory with processed data
        patch_size: Size of patches to extract
        stride: Sliding window stride (None = no overlap)
        aggregation_strategy: Strategy for combining patch predictions
        save_visualizations: Whether to save visualizations
    
    Returns:
        List of results with detailed patch information
    """
    model, idx_to_class = load_model_and_mapping(model_path, data_dir)
    
    image_dir = Path(image_dir)
    image_files = list(image_dir.glob("*.png")) + list(image_dir.glob("*.jpg"))
    
    print(f"Found {len(image_files)} images")
    print(f"Using region-based detection with patch_size={patch_size}, strategy={aggregation_strategy.value}\n")
    
    results = []
    for image_file in image_files:
        try:
            # Optionally save visualization
            vis_path = None
            if save_visualizations:
                vis_path = image_file.parent / f"{image_file.stem}_detection.png"
            
            class_name, confidence, detection_result = predict_image_region_based(
                model,
                idx_to_class,
                image_file,
                patch_size=patch_size,
                stride=stride,
                aggregation_strategy=aggregation_strategy,
                save_visualization_path=vis_path
            )
            
            print(f"{image_file.name}: {class_name} ({confidence:.2%})")
            print(f"  Patches analyzed: {detection_result.num_patches_analyzed}, "
                  f"Early stopped: {detection_result.early_stopped}")
            
            results.append({
                'file': image_file.name,
                'predicted_class': class_name,
                'confidence': confidence,
                'num_patches': detection_result.num_patches_analyzed,
                'early_stopped': detection_result.early_stopped,
                'aggregation_strategy': detection_result.aggregation_strategy,
                'patch_predictions': detection_result.patch_predictions
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
    
    # Region-based single image prediction
    region_parser = subparsers.add_parser('region', help='Predict with region-based detection (robust)')
    region_parser.add_argument("--image", required=True, help="Path to image file")
    region_parser.add_argument("--patch-size", type=int, nargs=2, default=[128, 128], help="Patch size (height width)")
    region_parser.add_argument("--stride", type=int, nargs=2, default=None, help="Stride for sliding window")
    region_parser.add_argument("--max-patches", type=int, default=16, help="Maximum patches to analyze")
    region_parser.add_argument("--aggregation", type=str, default="max", 
                                choices=['max', 'mean', 'weighted_mean', 'voting', 'any_positive'],
                                help="Aggregation strategy")
    region_parser.add_argument("--visualize", action='store_true', help="Save visualization")
    
    # Evaluate
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate on test set')
    
    # Batch prediction
    batch_parser = subparsers.add_parser('batch', help='Predict batch of images')
    batch_parser.add_argument("--image-dir", required=True, help="Directory with images")
    
    # Region-based batch prediction
    region_batch_parser = subparsers.add_parser('region-batch', help='Predict batch with region-based detection')
    region_batch_parser.add_argument("--image-dir", required=True, help="Directory with images")
    region_batch_parser.add_argument("--patch-size", type=int, nargs=2, default=[128, 128], help="Patch size")
    region_batch_parser.add_argument("--visualize", action='store_true', help="Save visualizations")
    
    args = parser.parse_args()
    
    if args.mode == 'single':
        model, idx_to_class = load_model_and_mapping(args.model, args.data_dir)
        class_name, confidence, probability = predict_image(model, idx_to_class, args.image, args.threshold)
        print(f"\nPredicted: {class_name}")
        print(f"Confidence: {confidence:.2%}")
        print(f"Raw probability: {probability:.4f}")
    
    elif args.mode == 'region':
        model, idx_to_class = load_model_and_mapping(args.model, args.data_dir)
        
        # Parse aggregation strategy
        strategy_map = {
            'max': AggregationStrategy.MAX,
            'mean': AggregationStrategy.MEAN,
            'weighted_mean': AggregationStrategy.WEIGHTED_MEAN,
            'voting': AggregationStrategy.VOTING,
            'any_positive': AggregationStrategy.ANY_POSITIVE
        }
        aggregation_strategy = strategy_map[args.aggregation]
        
        # Visualization path
        vis_path = None
        if args.visualize:
            img_path = Path(args.image)
            vis_path = img_path.parent / f"{img_path.stem}_detection.png"
        
        class_name, confidence, detection_result = predict_image_region_based(
            model,
            idx_to_class,
            args.image,
            patch_size=tuple(args.patch_size),
            stride=tuple(args.stride) if args.stride else None,
            max_patches=args.max_patches,
            aggregation_strategy=aggregation_strategy,
            threshold=args.threshold,
            save_visualization_path=vis_path
        )
        
        print(f"\n=== Region-Based Detection Results ===")
        print(f"Predicted: {class_name}")
        print(f"Confidence: {confidence:.2%}")
        print(f"Patches analyzed: {detection_result.num_patches_analyzed}")
        print(f"Early stopped: {detection_result.early_stopped}")
        print(f"Aggregation strategy: {detection_result.aggregation_strategy}")
        
        if vis_path:
            print(f"\nVisualization saved to: {vis_path}")
    
    elif args.mode == 'evaluate':
        evaluate_model(args.model, args.data_dir)
    
    elif args.mode == 'batch':
        predict_batch(args.model, args.image_dir, args.data_dir)
    
    elif args.mode == 'region-batch':
        predict_batch_region_based(
            args.model,
            args.image_dir,
            data_dir=args.data_dir,
            patch_size=tuple(args.patch_size),
            save_visualizations=args.visualize
        )
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
