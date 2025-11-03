"""
Model training script for QuickDraw classification.
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pickle

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Import from models.py
from models import get_model


def build_model(num_classes=2):
    """Build CNN model for binary classification (in-distribution vs out-of-distribution)."""
    model = models.Sequential([
        # First Conv Block
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Second Conv Block
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Third Conv Block
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.25),
        
        # Flatten and Dense layers
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        
        # Output layer - binary classification
        layers.Dense(1, activation='sigmoid')
    ])
    
    return model


def train_model(data_dir, epochs=50, batch_size=32, model_output="models/quickdraw_model.h5", learning_rate=0.001, label_smoothing=0.1, architecture='custom'):
    """
    Train the QuickDraw classifier with data augmentation.
    
    Data augmentation applies random transformations during training:
    - Rotations (±15°)
    - Shifts (±10%)
    - Zoom (±15%)
    
    Label smoothing helps prevent overconfidence:
    - Converts hard labels (0/1) to soft labels (e.g., 0.05/0.95)
    - Reduces overfitting and improves generalization
    
    Args:
        data_dir: Directory containing processed data
        epochs: Number of training epochs
        batch_size: Training batch size
        model_output: Path to save the trained model
        learning_rate: Learning rate for optimizer
        label_smoothing: Label smoothing factor (0-1, typical 0.05-0.2)
        architecture: Model architecture ('custom', 'resnet50', 'mobilenetv3', 'efficientnet')
    """
    data_dir = Path(data_dir)
    
    # Load data
    print("Loading training data...")
    X_train = np.load(data_dir / "X_train.npy")
    y_train = np.load(data_dir / "y_train.npy")
    X_test = np.load(data_dir / "X_test.npy")
    y_test = np.load(data_dir / "y_test.npy")
    
    # Load class mapping
    with open(data_dir / "class_mapping.pkl", 'rb') as f:
        class_mapping = pickle.load(f)
    
    # For binary classification
    if 'negative' in class_mapping and 'positive' in class_mapping:
        print(f"Binary classification task detected")
        print(f"  Positive class (1): in-distribution drawings")
        print(f"  Negative class (0): out-of-distribution noise")
    else:
        num_classes = len(class_mapping)
        class_names = {v: k for k, v in class_mapping.items() if isinstance(v, int)}
        print(f"Multi-class task detected with {num_classes} classes")
    
    # Build model
    print("\nBuilding model...")
    if architecture == 'custom':
        model = build_model()
    else:
        print(f"Using transfer learning architecture: {architecture}")
        model, _ = get_model(architecture, freeze_base=True, summary=False)
    
    # Compile model for binary classification
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.AUC()]
    )
    
    print("\nModel summary:")
    model.summary()
    
    # Create callbacks
    model_dir = Path(model_output).parent
    model_dir.mkdir(parents=True, exist_ok=True)
    
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            str(model_output),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # Data augmentation for better generalization
    print("\nSetting up data augmentation...")
    augmentation = ImageDataGenerator(
        rotation_range=15,           # ±15 degree rotations
        width_shift_range=0.1,       # ±10% horizontal shift
        height_shift_range=0.1,      # ±10% vertical shift
        zoom_range=0.15,             # ±15% zoom
        fill_mode='constant',        # Fill with constant value
        cval=255,                    # White background (for 0-255 range images)
    )
    
    # Scale images to 0-1 range for augmentation
    if X_train.max() > 1.0:
        X_train_aug = X_train * 255.0
    else:
        X_train_aug = X_train
    
    # Create validation split manually
    val_split = 0.2
    val_size = int(len(X_train_aug) * val_split)
    train_size = len(X_train_aug) - val_size
    
    X_train_split = X_train_aug[:train_size]
    y_train_split = y_train[:train_size]
    X_val_split = X_train_aug[train_size:]
    y_val_split = y_train[train_size:]
    
    # Apply label smoothing to reduce overconfidence
    # Convert hard labels (0/1) to soft labels
    if label_smoothing > 0:
        print(f"\nApplying label smoothing (factor={label_smoothing})...")
        y_train_split = y_train_split.astype(float)
        y_train_split = y_train_split * (1 - label_smoothing) + label_smoothing / 2
        y_train_split = y_train_split.astype(np.float32)
        print(f"  Label range: [{y_train_split.min():.3f}, {y_train_split.max():.3f}]")
    
    # Ensure validation data is in 0-1 range for model evaluation
    if X_val_split.max() > 1.0:
        X_val_split_normalized = X_val_split / 255.0
    else:
        X_val_split_normalized = X_val_split
    
    # Ensure test data is in 0-1 range
    X_test_normalized = X_test
    if X_test_normalized.max() > 1.0:
        X_test_normalized = X_test_normalized / 255.0
    
    print(f"  Training samples: {len(X_train_split)}")
    print(f"  Validation samples: {len(X_val_split)}")
    
    # Train model with augmentation
    print("\nTraining model with data augmentation...")
    
    # Create a training data generator that repeats indefinitely
    # The generator will automatically cycle through data
    train_generator = augmentation.flow(
        X_train_split, 
        y_train_split, 
        batch_size=batch_size,
        shuffle=True,
        seed=42
    )
    
    # Calculate steps per epoch with proper rounding
    steps_per_epoch = int(np.ceil(len(X_train_split) / batch_size))
    
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=(X_val_split_normalized, y_val_split),
        callbacks=callbacks,
        steps_per_epoch=steps_per_epoch,
        verbose=1
    )
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    results = model.evaluate(X_test_normalized, y_test, verbose=0)
    if len(results) == 3:
        test_loss, test_accuracy, test_auc = results
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print(f"Test AUC: {test_auc:.4f}")
    else:
        test_loss, test_accuracy = results
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f}")
    
    # Save training history
    history_file = Path(model_output).parent / "training_history.pkl"
    with open(history_file, 'wb') as f:
        pickle.dump(history.history, f)
    
    # Plot training history
    plot_training_history(history, Path(model_output).parent / "training_history.png")
    
    print(f"\n✓ Model saved to {model_output}")


def plot_training_history(history, output_path):
    """Plot and save training history."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Accuracy
    ax1.plot(history.history['accuracy'], label='Train Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Val Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.legend()
    ax1.grid(True)
    
    # Loss
    ax2.plot(history.history['loss'], label='Train Loss')
    ax2.plot(history.history['val_loss'], label='Val Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_title('Model Loss')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=100)
    print(f"✓ Training history plot saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Train QuickDraw classifier")
    parser.add_argument("--data-dir", default="data/processed", help="Directory with processed data")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--label-smoothing", type=float, default=0.1, 
                       help="Label smoothing factor (0-1, default 0.1)")
    parser.add_argument("--architecture", default="custom", 
                       choices=["custom", "resnet50", "mobilenetv3", "efficientnet"],
                       help="Model architecture to use")
    parser.add_argument("--model-output", default="models/quickdraw_model.h5", help="Path to save model")
    
    args = parser.parse_args()
    
    train_model(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        model_output=args.model_output,
        learning_rate=args.learning_rate,
        label_smoothing=args.label_smoothing,
        architecture=args.architecture
    )


if __name__ == "__main__":
    main()
