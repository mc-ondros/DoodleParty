#!/usr/bin/env python3
"""
CLI for model training - Binary classification for content moderation.
"""

import argparse
import sys
import os
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# Import from src-py (note: directory has hyphen, not underscore)
import importlib.util
src_py_path = os.path.join(os.path.dirname(__file__), '../../src-py')
sys.path.insert(0, src_py_path)

from core.models import build_custom_cnn
from core.training import create_callbacks, train_model
from data.loaders import QuickDrawLoader


def main():
    parser = argparse.ArgumentParser(description='Train DoodleParty binary classification model')
    parser.add_argument('--data-dir', type=str, default='data/raw', help='Directory with training data')
    parser.add_argument('--output', type=str, default='models/model.h5', help='Output model path')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--model', type=str, default='custom_cnn', help='Model architecture')
    parser.add_argument('--positive-categories', type=str, nargs='+', 
                        default=['penis'], 
                        help='Inappropriate content categories from Quickdraw-appendix')
    parser.add_argument('--negative-categories', type=str, nargs='+',
                        default=['dog', 'cat', 'house', 'tree', 'car', 'flower', 'sun', 'star'],
                        help='Appropriate content categories from Google QuickDraw')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("DoodleParty Binary Classification Training")
    print("=" * 70)
    print(f"\nModel: {args.model}")
    print(f"Data directory: {args.data_dir}")
    print(f"Output: {args.output}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"\nPositive (inappropriate) categories: {args.positive_categories}")
    print(f"Negative (appropriate) categories: {args.negative_categories}")
    
    # Check if data directory exists and has files
    if not os.path.exists(args.data_dir):
        print(f"\nError: Data directory '{args.data_dir}' does not exist!")
        print("Please download QuickDraw data first using:")
        print(f"  python scripts/data_processing/download_quickdraw_npy.py")
        return 1
    
    # Build model for binary classification
    print("\n" + "=" * 70)
    print("Building model...")
    if args.model == 'custom_cnn':
        model = build_custom_cnn(input_shape=(128, 128, 1), num_classes=2)
    else:
        print(f"Unknown model: {args.model}")
        return 1
    
    print("Model built successfully!")
    model.summary()
    
    # Load data
    print("\n" + "=" * 70)
    print("Loading training data...")
    all_categories = args.positive_categories + args.negative_categories
    
    try:
        (train_images, train_labels), (val_images, val_labels) = QuickDrawLoader.load_quickdraw_split(
            args.data_dir, 
            all_categories,
            train_split=0.8
        )
        
        # Convert multi-class labels to binary (0 = appropriate, 1 = inappropriate)
        # First len(positive_categories) indices are inappropriate
        num_positive = len(args.positive_categories)
        train_labels_binary = (train_labels < num_positive).astype(np.float32)
        val_labels_binary = (val_labels < num_positive).astype(np.float32)
        
        # One-hot encode for categorical crossentropy
        from tensorflow.keras.utils import to_categorical
        train_labels_binary = to_categorical(train_labels_binary, 2)
        val_labels_binary = to_categorical(val_labels_binary, 2)
        
        # Reshape images for CNN (add channel dimension)
        train_images = train_images.reshape(train_images.shape[0], 128, 128, 1)
        val_images = val_images.reshape(val_images.shape[0], 128, 128, 1)
        
        print(f"Training samples: {len(train_images)}")
        print(f"Validation samples: {len(val_images)}")
        print(f"Image shape: {train_images.shape[1:]}")
        
    except Exception as e:
        print(f"\nError loading data: {e}")
        print("\nMake sure you have downloaded the QuickDraw .npy files for these categories:")
        for cat in all_categories:
            expected_file = os.path.join(args.data_dir, f"{cat}.npy")
            print(f"  - {expected_file}")
        return 1
    
    # Create output directory
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    # Create callbacks
    print("\n" + "=" * 70)
    print("Setting up training callbacks...")
    callbacks = create_callbacks('doodleparty', checkpoint_dir=os.path.dirname(args.output) + '/')
    
    # Train model
    print("\n" + "=" * 70)
    print(f"Starting training for {args.epochs} epochs...")
    print("=" * 70)
    
    history = train_model(
        model,
        (train_images, train_labels_binary),
        (val_images, val_labels_binary),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks
    )
    
    # Save final model
    print("\n" + "=" * 70)
    print(f"Saving model to {args.output}...")
    model.save(args.output)
    print("Training complete!")
    print("=" * 70)
    
    # Print final metrics
    final_train_acc = history.history['accuracy'][-1]
    final_val_acc = history.history['val_accuracy'][-1]
    print(f"\nFinal training accuracy: {final_train_acc:.4f}")
    print(f"Final validation accuracy: {final_val_acc:.4f}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
