#!/usr/bin/env python3
"""
Train binary classification model for content moderation.
Uses preprocessed dataset created by create_demo_dataset.py
"""

import argparse
import sys
import os
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau


def build_binary_cnn(input_shape=(128, 128, 1)):
    """Build custom CNN for binary classification (appropriate vs inappropriate)."""
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        
        # First conv block
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Second conv block
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Third conv block
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Dense layers
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        
        # Output layer - sigmoid for binary classification
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
    )
    
    return model


def main():
    parser = argparse.ArgumentParser(
        description='Train binary classification model for content moderation'
    )
    parser.add_argument('--data-dir', type=str, default='data/processed',
                        help='Directory with preprocessed dataset')
    parser.add_argument('--output', type=str, default='models/content_moderator.h5',
                        help='Output model path')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for training')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("Content Moderation Model Training")
    print("=" * 70)
    print(f"\nData directory: {args.data_dir}")
    print(f"Output model: {args.output}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    
    # Load preprocessed data
    print("\n" + "=" * 70)
    print("Loading preprocessed dataset...")
    
    try:
        train_images = np.load(os.path.join(args.data_dir, 'train_images.npy'))
        train_labels = np.load(os.path.join(args.data_dir, 'train_labels.npy'))
        val_images = np.load(os.path.join(args.data_dir, 'val_images.npy'))
        val_labels = np.load(os.path.join(args.data_dir, 'val_labels.npy'))
    except FileNotFoundError as e:
        print(f"\n✗ Error: Preprocessed dataset not found!")
        print(f"  {e}")
        print(f"\nPlease create the dataset first:")
        print(f"  python scripts/training/create_demo_dataset.py")
        return 1
    
    # Normalize and reshape
    train_images = train_images.astype(np.float32) / 255.0
    val_images = val_images.astype(np.float32) / 255.0
    
    train_images = train_images.reshape(-1, 128, 128, 1)
    val_images = val_images.reshape(-1, 128, 128, 1)
    
    print(f"✓ Training samples: {len(train_images)}")
    print(f"✓ Validation samples: {len(val_images)}")
    print(f"✓ Image shape: {train_images.shape[1:]}")
    print(f"✓ Inappropriate content in training: {np.sum(train_labels)} ({100*np.mean(train_labels):.1f}%)")
    
    # Build model
    print("\n" + "=" * 70)
    print("Building model...")
    model = build_binary_cnn()
    print("✓ Model built successfully!")
    model.summary()
    
    # Create callbacks
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            args.output,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    # Train
    print("\n" + "=" * 70)
    print(f"Starting training for {args.epochs} epochs...")
    print("=" * 70 + "\n")
    
    history = model.fit(
        train_images,
        train_labels,
        validation_data=(val_images, val_labels),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    # Final evaluation
    print("\n" + "=" * 70)
    print("Training complete!")
    print("=" * 70)
    
    final_train_loss = history.history['loss'][-1]
    final_train_acc = history.history['accuracy'][-1]
    final_val_loss = history.history['val_loss'][-1]
    final_val_acc = history.history['val_accuracy'][-1]
    
    print(f"\nFinal Training Metrics:")
    print(f"  Loss: {final_train_loss:.4f}")
    print(f"  Accuracy: {final_train_acc:.4f}")
    
    print(f"\nFinal Validation Metrics:")
    print(f"  Loss: {final_val_loss:.4f}")
    print(f"  Accuracy: {final_val_acc:.4f}")
    
    print(f"\n✓ Model saved to: {args.output}")
    
    print("\n" + "=" * 70)
    print("Next Steps:")
    print("=" * 70)
    print("1. Convert to TensorFlow Lite for Raspberry Pi:")
    print("   python scripts/training/convert_to_tflite.py")
    print("2. Evaluate model performance:")
    print("   python scripts/evaluation/evaluate.py")
    print("3. Test inference:")
    print("   python src-py/web/app.py")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
