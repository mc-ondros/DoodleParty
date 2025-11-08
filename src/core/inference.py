"""
Comprehensive inference and evaluation system for DoodleHunter binary classifier.

This module provides a complete suite of utilities for loading trained models,
making predictions on individual images or batches, evaluating model performance,
and implementing robust region-based detection for enhanced security against
content dilution attacks.

Key Capabilities:
1. Model Loading: Loads trained models and associated class mappings from disk
2. Single Image Prediction: Classifies individual doodle images with confidence scores
3. Batch Processing: Efficiently processes multiple images in directory-based workflows
4. Model Evaluation: Comprehensive performance metrics including confusion matrices,
   classification reports, and AUC scores on test datasets
5. Region-Based Detection: Advanced sliding window approach that analyzes image
   patches independently to detect inappropriate content even when mixed with
   innocent content (robust against evasion attempts)

Security Considerations:
- Region-based detection provides defense against content dilution attacks where
  malicious content is embedded within larger innocent images
- Multiple aggregation strategies (max, mean, voting, etc.) allow flexible detection
  sensitivity based on security requirements
- Early stopping optimization reduces inference time while maintaining detection accuracy

Related Components:
- src/core/models.py: Model architectures and factory functions
- scripts/train.py: Training pipeline that produces the models used here
- src/data/augmentation.py: Input normalization that must be consistent between
  training and inference to maintain model performance
- src/core/patch_extraction.py: Sliding window detection implementation for
  region-based analysis

Exports:
- load_model_and_mapping: Utility to load trained models and class mappings
- predict_image: Single image classification with confidence scoring
- evaluate_model: Comprehensive model evaluation on test datasets
- predict_batch: Batch processing for directory-based prediction workflows
- predict_image_region_based: Robust region-based detection for security-critical applications
- predict_batch_region_based: Batch processing with region-based detection
"""

import argparse
import numpy as np
import pickle
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

import tensorflow as tf
# Use standalone Keras for compatibility
import keras
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# Import normalization from data augmentation module
from src.data.augmentation import normalize_image, normalize_batch

# Import region-based detection
from src.core.patch_extraction import (
    SlidingWindowDetector,
    AggregationStrategy,
    DetectionResult,
    visualize_detections
)


def load_model_and_mapping(model_path, data_dir = 'data/processed'):
    """
    Load a trained model and its associated class mapping from disk.

    This utility function handles the loading of both the Keras model file and the
    pickled class mapping that was created during the training process. The class
    mapping is essential for interpreting model predictions correctly.

    Important Implementation Details:
    - The model is loaded using keras.models.load_model() which handles the
      complete model architecture, weights, and compilation state
    - The class mapping is stored as a pickle file during training and maps
      class names (strings) to integer indices used by the model
    - An inverse mapping (idx_to_class) is created to convert model output
      indices back to human-readable class names
    - For binary classification, the mapping typically contains:
        {'negative': 0, 'positive': 1} with inverse {0: 'negative', 1: 'positive'}

    Args:
        model_path: String or Path object pointing to the saved Keras model file
                   (typically with .h5 or .keras extension)
        data_dir: String or Path object specifying the directory containing
                 the class_mapping.pkl file. Default is 'data/processed'
                 which matches the standard training output directory.

    Returns:
        tuple: (model, idx_to_class) where:
            - model: Loaded Keras model ready for inference
            - idx_to_class: Dictionary mapping integer indices to class names
                           (e.g., {0: 'negative', 1: 'positive'})

    Raises:
        FileNotFoundError: If either the model file or class mapping file doesn't exist
        pickle.UnpicklingError: If the class mapping file is corrupted
        ValueError: If the model file is invalid or incompatible

    Note: This function assumes the standard training pipeline structure where
    class_mapping.pkl is saved alongside the processed training data.
    """
    model = keras.models.load_model(model_path)
    
    with open(Path(data_dir) / 'class_mapping.pkl', 'rb') as f:
        class_mapping = pickle.load(f)
    
    idx_to_class = {v: k for k, v in class_mapping.items()}
    
    return model, idx_to_class


def predict_image(model, idx_to_class, image_path, threshold=0.5):
    """
    Perform binary classification on a single doodle image.

    This function loads an image file, preprocesses it according to the training
    pipeline specifications, and runs inference using the provided trained model.
    It returns both the predicted class and confidence metrics.

    Critical Preprocessing Steps:
    1. Image Loading: Opens the image file and converts to grayscale (L mode)
       regardless of original format to match training data expectations
    2. Resizing: Resizes to 28x28 pixels to match the input dimensions used during training
    3. Normalization: Applies per-image normalization using the same function
       (normalize_image) that was used during training - this is critical for
       maintaining model performance and must be consistent
    4. Tensor Formatting: Reshapes to (1, 28, 28, 1) format expected by the model

    Prediction Interpretation:
    - The model outputs a single probability value between 0 and 1
    - Values >= threshold are classified as 'positive (in-distribution)'
    - Values < threshold are classified as 'negative (out-of-distribution)'
    - Confidence is calculated as max(probability, 1-probability) to always
      represent the model's certainty in its prediction

    Args:
        model: Trained Keras model for binary classification
        idx_to_class: Dictionary mapping class indices to names (typically
                     {0: 'negative', 1: 'positive'} for binary classification)
        image_path: String or Path object pointing to the image file to classify
                   (supports common formats like PNG, JPG, etc.)
        threshold: Decision threshold for binary classification (default 0.5).
                  Lower values increase sensitivity (more positives), higher values
                  increase specificity (fewer false positives).

    Returns:
        tuple: (class_name, confidence, probability) where:
            - class_name: Human-readable classification result
                         ('positive (in-distribution)' or 'negative (out-of-distribution)')
            - confidence: Model's confidence in its prediction (0.0 to 1.0)
                         Always >= 0.5 since it represents certainty
            - probability: Raw model output probability (0.0 to 1.0)
                         Represents P(positive class)

    Security Note: This function performs standard single-image classification.
    For enhanced security against content dilution attacks, use
    predict_image_region_based() instead.
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


def evaluate_model(model_path, data_dir = 'data/processed'):
    """
    Comprehensive evaluation of binary classification model performance on test set.

    This function loads a trained model and evaluates it against the held-out test
    dataset, generating a complete set of performance metrics and visualizations
    essential for understanding model behavior and identifying potential issues.

    Evaluation Metrics Generated:
    1. Basic Metrics: Test loss, accuracy, and AUC (Area Under ROC Curve)
    2. Classification Report: Precision, recall, F1-score, and support for both classes
    3. Confusion Matrix: Visual heatmap showing true vs. predicted classifications
    4. Class Distribution: Counts of positive and negative samples in test set

    Data Loading Assumptions:
    - Test data is stored as numpy arrays in data_dir with filenames:
      * X_test.npy: Test input features (images)
      * y_test.npy: Test labels (0=negative, 1=positive)
    - Class mapping is stored as class_mapping.pkl in the same directory
    - All files follow the standard training pipeline output format

    Visualization Output:
    - Saves a confusion matrix heatmap as 'confusion_matrix.png' in the same
      directory as the model file
    - Uses seaborn for professional-quality visualization with proper labeling
    - Includes both class names ('negative (OOD)', 'positive (ID)') for clarity

    Args:
        model_path: String or Path object pointing to the trained model file
        data_dir: String or Path object specifying directory containing test data
                 and class mapping (default: 'data/processed')

    Side Effects:
    - Prints comprehensive evaluation results to stdout
    - Saves confusion matrix visualization to disk
    - Loads potentially large test datasets into memory

    Use Cases:
    - Model selection and comparison during development
    - Final validation before deployment
    - Monitoring model performance over time
    - Identifying class imbalance or bias issues
    """
    print('Loading model and data...')
    model, idx_to_class = load_model_and_mapping(model_path, data_dir)
    
    # Load test data
    X_test = np.load(Path(data_dir) / 'X_test.npy')
    y_test = np.load(Path(data_dir) / 'y_test.npy')
    
    print(f"Test set size: {len(X_test)}")
    print(f"Positive samples: {(y_test==1).sum()}, Negative samples: {(y_test==0).sum()}")
    
    # Evaluate
    print('\nEvaluating model...')
    loss, accuracy, auc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test AUC: {auc:.4f}")
    
    # Get predictions
    print('Getting predictions...')
    y_pred_proba = model.predict(X_test, verbose=0).flatten()
    y_pred = (y_pred_proba >= 0.5).astype(int)
    
    # Classification report
    print('\nClassification Report:')
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
    
    output_path = Path(model_path).parent / 'confusion_matrix.png'
    plt.savefig(output_path, dpi=100)
    print(f"✓ Confusion matrix saved to {output_path}")


def predict_batch(model_path, image_dir, data_dir = 'data/processed'):
    """
    Perform batch prediction on all images in a specified directory.

    This utility function processes multiple image files in a directory, applying
    the trained model to each image and returning structured results. It's designed
    for production workflows where multiple images need to be classified efficiently.

    Processing Pipeline:
    1. Model Loading: Loads the trained model and class mapping once at the beginning
    2. File Discovery: Finds all PNG and JPG files in the specified directory
    3. Sequential Processing: Processes each image using predict_image() function
    4. Error Handling: Continues processing even if individual images fail
    5. Result Aggregation: Collects results in a structured list for further processing

    Input Requirements:
    - image_dir must contain only image files or the function will skip non-image files
    - Supported formats: PNG (.png) and JPEG (.jpg, .jpeg) files
    - Images can be any size/format as they will be converted to 28x28 grayscale
    - Directory structure is flat (no subdirectory recursion)

    Output Structure:
    Each result dictionary contains:
    - 'file': Original filename (for tracking)
    - 'predicted_class': Classification result with descriptive labels
    - 'confidence': Model confidence score (0.0 to 1.0)

    Args:
        model_path: String or Path object pointing to the trained model file
        image_dir: String or Path object specifying the directory containing images
        data_dir: String or Path object specifying directory with class mapping
                 (default: 'data/processed')

    Returns:
        list: List of dictionaries containing prediction results for each image.
              Images that fail processing are skipped and not included in results.

    Error Handling:
    - Individual image processing errors are caught and logged to stdout
    - Processing continues with the next image even if one fails
    - Common failure causes: corrupted images, unsupported formats, permission issues

    Performance Considerations:
    - Loads model only once for efficiency
    - Processes images sequentially (not parallelized)
    - Memory usage scales with number of images (one image loaded at a time)
    """
    model, idx_to_class = load_model_and_mapping(model_path, data_dir)
    
    image_dir = Path(image_dir)
    image_files = list(image_dir.glob('*.png')) + list(image_dir.glob('*.jpg'))
    
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
    Perform robust region-based detection using sliding window analysis.

    This advanced detection method provides enhanced security against content
    dilution attacks by analyzing image patches independently rather than
    processing the entire image as a single unit. It's specifically designed
    to detect inappropriate content even when it's embedded within larger
    innocent images or mixed with benign content.

    Security Defense Mechanism:
    - Content Dilution Attack Resistance: Malicious content hidden in small
      regions of large images will be detected when those regions are analyzed
      as individual patches
    - Spatial Independence: Each patch is classified independently, preventing
      dilution of malicious signals by surrounding innocent content
    - Multiple Aggregation Strategies: Flexible combination of patch results
      allows tuning detection sensitivity based on security requirements

    Sliding Window Configuration:
    - patch_size: Defines the analysis window dimensions (default 128x128 pixels)
      Larger patches capture more context but may miss small malicious regions
      Smaller patches increase sensitivity but require more computation
    - stride: Controls overlap between patches. None means no overlap (patch_size stride)
      Smaller strides provide more comprehensive coverage but increase processing time
    - min_content_ratio: Filters out patches with insufficient content (e.g., mostly empty)
      Prevents false positives from analyzing blank or low-information regions
    - max_patches: Limits total patches analyzed to control computational cost
      Prevents excessive processing on very large images

    Detection Strategies:
    - AggregationStrategy.MAX: Most sensitive - uses highest patch probability
    - AggregationStrategy.MEAN: Balanced - averages all patch probabilities
    - AggregationStrategy.VOTING: Conservative - requires majority of patches to be positive
    - AggregationStrategy.ANY_POSITIVE: Ultra-sensitive - triggers on any positive patch
    - AggregationStrategy.WEIGHTED_MEAN: Context-aware - weights patches by content density

    Early Stopping Optimization:
    - When enabled, stops analysis immediately upon detecting a patch with
      confidence above early_stop_threshold (default 0.9)
    - Dramatically reduces inference time for clearly positive images
    - Maintains full analysis for ambiguous or negative cases

    Args:
        model: Trained Keras model for binary classification
        idx_to_class: Class mapping dictionary (typically {0: 'negative', 1: 'positive'})
        image_path: Path to input image file (any size/format supported by PIL)
        patch_size: Tuple (height, width) specifying patch dimensions in pixels
        stride: Tuple (vertical, horizontal) specifying sliding window step size
               If None, uses patch_size (no overlap between patches)
        min_content_ratio: Minimum ratio of non-zero pixels required to analyze a patch
                          (0.05 = 5% non-zero pixels minimum)
        max_patches: Maximum number of patches to analyze (prevents excessive computation)
        early_stopping: Boolean enabling early termination on high-confidence detections
        aggregation_strategy: AggregationStrategy enum specifying how to combine patch results
        threshold: Final decision threshold applied to aggregated result (default 0.5)
        visualize: Boolean flag to generate detection visualization
        save_visualization_path: Path to save visualization image (if visualize=True)

    Returns:
        tuple: (class_name, confidence, detection_result) where:
            - class_name: Final classification ('positive (in-distribution)' or 'negative (out-of-distribution)')
            - confidence: Model confidence in final decision (0.0 to 1.0)
            - detection_result: DetectionResult object containing detailed analysis including:
                * patch_predictions: Individual patch classification results
                * num_patches_analyzed: Total patches processed
                * early_stopped: Whether early stopping was triggered
                * aggregation_strategy: Strategy used for combining results

    Security Recommendation:
    For production deployment in security-critical applications, use this function
    instead of predict_image() to provide robust defense against evasion attempts.
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
    data_dir = 'data/processed',
    patch_size=(128, 128),
    stride=None,
    aggregation_strategy=AggregationStrategy.MAX,
    save_visualizations=False
):
    """
    Perform batch region-based detection on all images in a directory.

    This function extends the security-enhanced region-based detection to batch
    processing, enabling efficient analysis of multiple images with robust defense
    against content dilution attacks. It's ideal for production deployment where
    both security and throughput are critical requirements.

    Batch Processing Advantages:
    - Single Model Load: Loads the model once and reuses it for all images
    - Consistent Configuration: Applies identical detection parameters to all images
    - Structured Output: Returns detailed results including patch-level information
    - Visualization Support: Optionally generates detection visualizations for each image
    - Error Resilience: Continues processing even if individual images fail

    Security Configuration:
    - Uses the same robust sliding window detection as predict_image_region_based()
    - Supports all aggregation strategies for flexible security sensitivity tuning
    - Maintains early stopping optimization for performance on clearly positive images
    - Provides detailed patch analysis results for forensic investigation

    Visualization Output:
    - When save_visualizations=True, saves detection heatmap overlays as
      "{original_filename}_detection.png" in the same directory
    - Visualizations show detected regions with color-coded confidence levels
    - Useful for debugging, validation, and demonstrating detection capabilities

    Args:
        model_path: Path to trained model file (loaded once for all images)
        image_dir: Directory containing images to analyze (PNG/JPG files)
        data_dir: Directory containing class mapping (default: 'data/processed')
        patch_size: Tuple (height, width) for sliding window patch dimensions
        stride: Tuple (vertical, horizontal) for sliding window step size
               None uses patch_size (no overlap between patches)
        aggregation_strategy: AggregationStrategy enum for combining patch results
                             (default: MAX for maximum sensitivity)
        save_visualizations: Boolean flag to generate and save detection visualizations

    Returns:
        list: List of dictionaries containing comprehensive results for each image:
            - 'file': Original filename
            - 'predicted_class': Final classification result
            - 'confidence': Overall confidence score
            - 'num_patches': Number of patches analyzed
            - 'early_stopped': Whether early stopping was triggered
            - 'aggregation_strategy': Strategy used for patch aggregation
            - 'patch_predictions': Detailed per-patch classification results

    Performance Characteristics:
    - Processing time scales with number of images and their dimensions
    - Memory usage is optimized (one image loaded at a time)
    - Early stopping provides significant speedup for positive detections
    - Visualization generation adds computational overhead but provides valuable insights

    Production Recommendation:
    Use this function for security-critical batch processing workflows where
    robust detection against evasion attempts is required. The detailed patch
    information enables both automated decision-making and manual review when needed.
    """
    model, idx_to_class = load_model_and_mapping(model_path, data_dir)
    
    image_dir = Path(image_dir)
    image_files = list(image_dir.glob('*.png')) + list(image_dir.glob('*.jpg'))
    
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
    """
    Command-line interface for DoodleHunter binary classification system.

    This main function provides a comprehensive CLI for all inference and evaluation
    capabilities of the DoodleHunter system. It supports multiple prediction modes
    ranging from simple single-image classification to advanced security-focused
    region-based detection.

    CLI Command Structure:
    The interface uses subcommands to organize different functionality:

    1. single: Basic single-image prediction
       Usage: python inference.py --model MODEL_PATH single --image IMAGE_PATH
       - Fast, simple classification for non-security-critical use cases
       - Uses standard whole-image processing

    2. region: Advanced region-based detection (security-focused)
       Usage: python inference.py --model MODEL_PATH region --image IMAGE_PATH [options]
       - Robust against content dilution attacks
       - Supports configurable patch analysis and aggregation strategies
       - Optional visualization for result verification

    3. evaluate: Comprehensive model evaluation
       Usage: python inference.py --model MODEL_PATH evaluate
       - Generates complete performance metrics on test dataset
       - Produces confusion matrix visualization
       - Essential for model validation and comparison

    4. batch: Efficient batch processing
       Usage: python inference.py --model MODEL_PATH batch --image-dir DIRECTORY
       - Processes all PNG/JPG files in specified directory
       - Optimized for throughput with single model load

    5. region-batch: Security-focused batch processing
       Usage: python inference.py --model MODEL_PATH region-batch --image-dir DIRECTORY [options]
       - Combines batch efficiency with region-based security
       - Optional visualizations for all processed images

    Global Arguments:
    --model: Required path to trained model file (supports .h5, .keras formats)
    --data-dir: Directory containing class mapping and test data (default: 'data/processed')
    --threshold: Classification decision threshold (default: 0.5, range: 0.0-1.0)

    Security Best Practices:
    - For production deployment in security-critical applications, always use
      'region' or 'region-batch' modes instead of 'single' or 'batch'
    - Consider lowering the threshold (e.g., 0.3) for higher sensitivity in
      security applications, accepting more false positives for better recall
    - Use MAX or ANY_POSITIVE aggregation strategies for maximum attack detection

    Example Usage:
    # Basic single image prediction
    python src/core/inference.py --model models/efficientnet_model.h5 single --image test.png

    # Security-focused region-based detection with visualization
    python src/core/inference.py --model models/efficientnet_model.h5 region --image test.png --visualize

    # Batch processing with region-based detection
    python src/core/inference.py --model models/efficientnet_model.h5 region-batch --image-dir ./uploads --visualize

    # Model evaluation
    python src/core/inference.py --model models/efficientnet_model.h5 evaluate
    """
    parser = argparse.ArgumentParser(description = 'Binary classifier for doodle detection')
    parser.add_argument("--model", required=True, help = 'Path to trained model')
    parser.add_argument("--data-dir", default = 'data/processed', help = 'Directory with processed data')
    parser.add_argument("--threshold", type=float, default=0.5, help = 'Classification threshold')

    # Prediction modes
    subparsers = parser.add_subparsers(dest='mode', help='Prediction mode')

    # Single image prediction
    single_parser = subparsers.add_parser('single', help='Predict single image')
    single_parser.add_argument("--image", required=True, help = 'Path to image file')

    # Region-based single image prediction
    region_parser = subparsers.add_parser('region', help='Predict with region-based detection (robust)')
    region_parser.add_argument("--image", required=True, help = 'Path to image file')
    region_parser.add_argument("--patch-size", type=int, nargs=2, default=[128, 128], help = 'Patch size (height width)')
    region_parser.add_argument("--stride", type=int, nargs=2, default=None, help = 'Stride for sliding window')
    region_parser.add_argument("--max-patches", type=int, default=16, help = 'Maximum patches to analyze')
    region_parser.add_argument("--aggregation", type=str, default = 'max',
                                choices=['max', 'mean', 'weighted_mean', 'voting', 'any_positive'],
                                help = 'Aggregation strategy')
    region_parser.add_argument("--visualize", action='store_true', help = 'Save visualization')

    # Evaluate
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate on test set')

    # Batch prediction
    batch_parser = subparsers.add_parser('batch', help='Predict batch of images')
    batch_parser.add_argument("--image-dir", required=True, help = 'Directory with images')

    # Region-based batch prediction
    region_batch_parser = subparsers.add_parser('region-batch', help='Predict batch with region-based detection')
    region_batch_parser.add_argument("--image-dir", required=True, help = 'Directory with images')
    region_batch_parser.add_argument("--patch-size", type=int, nargs=2, default=[128, 128], help = 'Patch size')
    region_batch_parser.add_argument("--visualize", action='store_true', help = 'Save visualizations')

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
        
        print(f"\nRegion-Based Detection Results")
        print(f'Predicted: {class_name}')
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


if __name__ == '__main__':
    main()
