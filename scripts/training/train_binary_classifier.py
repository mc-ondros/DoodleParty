#!/usr/bin/env python3
"""
Train binary classifier for penis detection using QuickDraw 28x28 data.

This script trains a CNN to distinguish between offensive (penis) and safe drawings.
Uses the native 28x28 .npy format from QuickDraw dataset.
"""

import argparse
import os
import sys
import numpy as np
import pickle
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))

from src_py.core.models import build_custom_cnn
from src_py.core.training import create_callbacks, train_model
from src_py.data.augmentation import DataAugmentation


def load_category(data_dir: str, category: str, max_samples: int = None) -> np.ndarray:
    """Load a single category from .npy file."""
    file_path = os.path.join(data_dir, f"{category}.npy")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Category file not found: {file_path}")
    
    print(f"Loading {category}...", end=' ')
    data = np.load(file_path)
    
    if max_samples:
        data = data[:max_samples]
    
    # Normalize to [0, 1] and reshape to (N, 28, 28, 1)
    data = data.astype(np.float32) / 255.0
    data = data.reshape(-1, 28, 28, 1)
    
    print(f"✓ {len(data)} samples")
    return data


def prepare_binary_dataset(
    data_dir: str,
    positive_category: str = "penis",
    negative_categories: list = None,
    max_samples_per_category: int = 10000,
    train_split: float = 0.8
):
    """
    Prepare binary classification dataset.
    
    Args:
        data_dir: Directory containing .npy files
        positive_category: Positive class (offensive)
        negative_categories: List of negative classes (safe)
        max_samples_per_category: Max samples to load per category
        train_split: Train/val split ratio
    
    Returns:
        (train_images, train_labels), (val_images, val_labels)
    """
    if negative_categories is None:
        negative_categories = [
            "circle", "line", "square", "triangle", "star",
            "rectangle", "diamond", "heart", "cloud", "moon"
        ]
    
    print("=" * 70)
    print("Loading Binary Classification Dataset (28x28)")
    print("=" * 70)
    print(f"\nPositive class: {positive_category}")
    print(f"Negative classes: {len(negative_categories)}")
    print(f"Max samples per category: {max_samples_per_category}")
    print()
    
    # Load positive samples (label = 1)
    positive_data = load_category(data_dir, positive_category, max_samples_per_category)
    positive_labels = np.ones(len(positive_data), dtype=np.float32)
    
    # Load negative samples (label = 0)
    negative_data_list = []
    for category in negative_categories:
        try:
            data = load_category(data_dir, category, max_samples_per_category)
            negative_data_list.append(data)
        except FileNotFoundError as e:
            print(f"⚠ Warning: {e}")
            continue
    
    if not negative_data_list:
        raise ValueError("No negative samples loaded!")
    
    negative_data = np.concatenate(negative_data_list, axis=0)
    negative_labels = np.zeros(len(negative_data), dtype=np.float32)
    
    # Combine and shuffle
    print(f"\nCombining datasets...")
    print(f"Positive samples: {len(positive_data):,}")
    print(f"Negative samples: {len(negative_data):,}")
    
    all_data = np.concatenate([positive_data, negative_data], axis=0)
    all_labels = np.concatenate([positive_labels, negative_labels], axis=0)
    
    # Shuffle
    indices = np.random.permutation(len(all_data))
    all_data = all_data[indices]
    all_labels = all_labels[indices]
    
    # Split train/val
    split_idx = int(len(all_data) * train_split)
    train_images = all_data[:split_idx]
    train_labels = all_labels[:split_idx]
    val_images = all_data[split_idx:]
    val_labels = all_labels[split_idx:]
    
    print(f"\nDataset split:")
    print(f"Training: {len(train_images):,} samples")
    print(f"Validation: {len(val_images):,} samples")
    print(f"Class balance (train): {train_labels.mean():.1%} positive")
    print(f"Class balance (val): {val_labels.mean():.1%} positive")
    
    return (train_images, train_labels), (val_images, val_labels)


def main():
    parser = argparse.ArgumentParser(
        description='Train binary classifier for penis detection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with default settings
  python train_binary_classifier.py --data-dir data/raw
  
  # Train with custom parameters
  python train_binary_classifier.py --data-dir data/raw --epochs 50 --batch-size 64
  
  # Train with more negative categories
  python train_binary_classifier.py --data-dir data/raw --max-samples 20000
        """
    )
    
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data/raw',
        help='Directory containing QuickDraw .npy files'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='models',
        help='Output directory for trained model'
    )
    
    parser.add_argument(
        '--model-name',
        type=str,
        default='quickdraw_binary_28x28',
        help='Model name (without extension)'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=30,
        help='Number of training epochs'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for training'
    )
    
    parser.add_argument(
        '--max-samples',
        type=int,
        default=10000,
        help='Max samples per category'
    )
    
    parser.add_argument(
        '--train-split',
        type=float,
        default=0.8,
        help='Train/validation split ratio'
    )
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load dataset
    (train_images, train_labels), (val_images, val_labels) = prepare_binary_dataset(
        args.data_dir,
        max_samples_per_category=args.max_samples,
        train_split=args.train_split
    )
    
    # Build model
    print("\n" + "=" * 70)
    print("Building Model")
    print("=" * 70)
    model = build_custom_cnn(input_shape=(28, 28, 1), num_classes=1)
    
    print("\nModel Summary:")
    model.summary()
    
    total_params = model.count_params()
    print(f"\nTotal parameters: {total_params:,}")
    
    # Create callbacks
    callbacks = create_callbacks(args.model_name, checkpoint_dir=args.output_dir + '/')
    
    # Train model
    print("\n" + "=" * 70)
    print(f"Training for {args.epochs} epochs")
    print("=" * 70)
    
    history = train_model(
        model,
        (train_images, train_labels),
        (val_images, val_labels),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks
    )
    
    # Save final model
    final_model_path = os.path.join(args.output_dir, f"{args.model_name}.h5")
    model.save(final_model_path)
    
    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)
    print(f"\nModel saved to: {final_model_path}")
    print(f"Best model saved to: {args.output_dir}/{args.model_name}_best.h5")
    
    # Final metrics
    final_loss = history.history['loss'][-1]
    final_acc = history.history['accuracy'][-1]
    val_loss = history.history['val_loss'][-1]
    val_acc = history.history['val_accuracy'][-1]
    
    print(f"\nFinal Training Metrics:")
    print(f"  Loss: {final_loss:.4f}")
    print(f"  Accuracy: {final_acc:.4f}")
    print(f"\nFinal Validation Metrics:")
    print(f"  Loss: {val_loss:.4f}")
    print(f"  Accuracy: {val_acc:.4f}")
    
    # Check if model meets targets
    print(f"\nTarget Check:")
    if val_acc >= 0.90:
        print(f"  ✓ Accuracy target met: {val_acc:.1%} >= 90%")
    else:
        print(f"  ✗ Accuracy target not met: {val_acc:.1%} < 90%")
    
    print("\nNext steps:")
    print(f"  1. Evaluate model: python scripts/evaluation/evaluate.py --model {final_model_path}")
    print(f"  2. Convert to TFLite: python scripts/optimization/convert_to_tflite.py --model {final_model_path}")
    print("=" * 70)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
