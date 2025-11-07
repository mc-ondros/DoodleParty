"""
Model training script for QuickDraw classification.

Handles data loading, augmentation, model compilation, and training
with support for multiple architectures and hyperparameters.

Related:
- src/core/models.py (model architectures)
- src/data/augmentation.py (data preprocessing)
- scripts/evaluate.py (model evaluation)

Exports:
- build_model, train_model, plot_training_history
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pickle

import tensorflow as tf
from tensorflow import keras
from keras import layers, models
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.src.legacy.preprocessing.image import ImageDataGenerator

# Import from models.py and data_pipeline.py
from src.core.models import get_model
from src.data.augmentation import prepare_test_data


def build_model(num_classes=2, enhanced=False):
    """
    Build CNN model for binary classification.
    
    Args:
        num_classes: Number of classes (default 2 for binary)
        enhanced: If True, use larger model with more capacity (slower but more accurate)
    """
    if enhanced:
        # ENHANCED MODEL - More capacity for better real-world accuracy (128x128 input)
        model = models.Sequential([
            # First Conv Block - 64 filters
            layers.Conv2D(64, (3, 3), activation='relu', input_shape=(128, 128, 1)),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Second Conv Block - 128 filters
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.3),
            
            # Third Conv Block - 256 filters
            layers.Conv2D(256, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            # Dense layers with more neurons
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            
            # Output layer
            layers.Dense(1, activation='sigmoid')
        ])
        print('✓ Using ENHANCED model (larger capacity, ~2.5M params)')
    else:
        # STANDARD MODEL - Original architecture (128x128 input)
        model = models.Sequential([
            # First Conv Block
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 1)),
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
        print('✓ Using STANDARD model (~423K params)')
    
    return model


def train_model(data_dir, epochs=50, batch_size=32, model_output = 'models/quickdraw_model.h5',
                learning_rate=0.001, label_smoothing=0.1, architecture='custom',
                enhanced=False, aggressive_aug=False, use_class_weighting=False):
    """
    Train the QuickDraw classifier with data augmentation.

    Data augmentation applies random transformations during training:
    - Standard: Rotations (±15°), Shifts (±10%), Zoom (±15%)
    - Aggressive: Rotations (±30°), Shifts (±20%), Zoom (±25%), Brightness, Contrast

    Label smoothing helps prevent overconfidence:
    - Converts hard labels (0/1) to soft labels (e.g., 0.05/0.95)
    - Reduces overfitting and improves generalization

    Class weighting addresses imbalanced datasets:
    - Automatically calculates inverse class frequencies
    - Applies higher loss weights to minority class
    - Improves recall for underrepresented classes

    Args:
        data_dir: Directory containing processed data
        epochs: Number of training epochs
        batch_size: Training batch size
        model_output: Path to save the trained model
        learning_rate: Learning rate for optimizer
        label_smoothing: Label smoothing factor (0-1, typical 0.05-0.2)
        architecture: Model architecture ('custom', 'resnet50', 'mobilenetv3', 'efficientnet')
        enhanced: Use larger model with more capacity (slower, more accurate)
        aggressive_aug: Use more aggressive data augmentation
        use_class_weighting: Apply class weighting for imbalanced data
    """
    data_dir = Path(data_dir)
    
    # Load data
    print('Loading training data...')
    X_train = np.load(data_dir / 'X_train.npy')
    y_train = np.load(data_dir / 'y_train.npy')
    X_test = np.load(data_dir / 'X_test.npy')
    y_test = np.load(data_dir / 'y_test.npy')
    
    # Load class mapping
    with open(data_dir / 'class_mapping.pkl', 'rb') as f:
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

    # Calculate class distribution
    print(f"\nClass distribution in training data:")
    unique, counts = np.unique(y_train, return_counts=True)
    for cls, count in zip(unique, counts):
        percentage = 100 * count / len(y_train)
        print(f"  Class {int(cls)}: {count} samples ({percentage:.1f}%)")

    # Calculate class weights for imbalanced data
    # Using inverse frequency weighting: n_samples / (n_classes * np.bincount(y))
    class_weights = None
    if use_class_weighting:
        # Calculate balanced class weights
        total_samples = len(y_train)
        n_classes = len(unique)

        # Count samples per class
        class_counts = np.bincount(y_train.astype(int))

        # Calculate weights: total_samples / (n_classes * count_per_class)
        # This gives higher weight to minority classes
        computed_weights = {}
        for i in range(n_classes):
            computed_weights[i] = total_samples / (n_classes * class_counts[i])

        # Normalize weights to have mean ≈ 1.0 (optional but recommended)
        mean_weight = np.mean(list(computed_weights.values()))
        for key in computed_weights:
            computed_weights[key] = computed_weights[key] / mean_weight

        class_weights = computed_weights

        print(f"\n✓ Using class weighting for imbalanced data:")
        for cls, weight in class_weights.items():
            print(f"  Class {int(cls)}: {weight:.3f}")
        print(f"  (Higher weight = more emphasis during training)")
    else:
        print(f"\n⚪ Class weighting disabled (using equal weights)")

    # Build model
    print('\nBuilding model...')
    if architecture == 'custom':
        model = build_model(enhanced=enhanced)
    else:
        print(f"Using transfer learning architecture: {architecture}")
        model, _ = get_model(architecture, freeze_base=True, summary=False)
    
    # Compile model for binary classification
    # Add gradient clipping to prevent exploding gradients that cause val_acc crashes
    optimizer = keras.optimizers.Adam(
        learning_rate=learning_rate,
        clipnorm=1.0  # Clip gradients to max norm of 1.0
    )
    
    # Use BinaryCrossentropy with label smoothing for better calibration
    if label_smoothing > 0:
        loss_fn = keras.losses.BinaryCrossentropy(label_smoothing=label_smoothing)
        print(f"\n✓ Using label smoothing: {label_smoothing}")
        print(f"✓ Using gradient clipping (clipnorm=1.0)")
    else:
        loss_fn = 'binary_crossentropy'
        print(f"✓ Using gradient clipping (clipnorm=1.0)")
    
    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=['accuracy', keras.metrics.AUC()]
    )
    
    print('\nModel summary:')
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
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,              # Reduce LR by 50%
            patience=5,              # Wait 5 epochs
            min_lr=1e-6,
            verbose=1
        )
    ]
    
    # Create validation split FIRST (before any augmentation)
    # CRITICAL: Use random split with shuffle to avoid class clustering
    from sklearn.model_selection import train_test_split as split
    
    print('\nCreating train/validation split (stratified and shuffled)...')
    val_split = 0.2
    X_train_split, X_val_split, y_train_split, y_val_split = split(
        X_train, y_train,
        test_size=val_split,
        random_state=456,  # Different seed than test split
        stratify=y_train,  # Maintain class balance
        shuffle=True
    )
    
    print(f"  Training samples: {len(X_train_split)}")
    print(f"    - Positive: {(y_train_split == 1).sum()} ({100*(y_train_split==1).sum()/len(y_train_split):.1f}%)")
    print(f"    - Negative: {(y_train_split == 0).sum()} ({100*(y_train_split==0).sum()/len(y_train_split):.1f}%)")
    print(f"  Validation samples: {len(X_val_split)}")
    print(f"    - Positive: {(y_val_split == 1).sum()} ({100*(y_val_split==1).sum()/len(y_val_split):.1f}%)")
    print(f"    - Negative: {(y_val_split == 0).sum()} ({100*(y_val_split==0).sum()/len(y_val_split):.1f}%)")
    
    # DO NOT manually apply label smoothing - it's already in the loss function!
    # Double smoothing causes training/validation mismatch
    if label_smoothing > 0:
        print(f"\n✓ Label smoothing {label_smoothing} applied via loss function")
        print(f"  (No manual label modification needed)")
    
    # Check if data is already normalized (from preprocessing)
    if X_train.max() <= 1.0:
        print('\n✓ Data already normalized - skipping double normalization')
        X_train_norm = X_train_split
        X_val_norm = X_val_split
        X_test_norm = X_test
    else:
        print('\nNormalizing data with per-image normalization...')
        X_train_norm = prepare_test_data(X_train_split)
        X_val_norm = prepare_test_data(X_val_split)
        X_test_norm = prepare_test_data(X_test)
    
    # Determine proper background fill value from data
    # Sample corner pixels to estimate background
    sample_corners = []
    for i in np.random.choice(len(X_train_norm), min(100, len(X_train_norm)), replace=False):
        img = X_train_norm[i].squeeze()
        corners = [
            img[0:2, 0:2].mean(),
            img[0:2, -2:].mean(),
            img[-2:, 0:2].mean(),
            img[-2:, -2:].mean()
        ]
        sample_corners.extend(corners)
    background_value = np.median(sample_corners)
    print(f"\n✓ Detected background value: {background_value:.3f}")
    
    # Create augmentation generator with CORRECT background fill
    if aggressive_aug:
        print('Setting up AGGRESSIVE data augmentation...')
        augmentation = ImageDataGenerator(
            rotation_range=25,          # ±25 degrees (vs 15)
            width_shift_range=0.15,     # ±15% (vs 10%)
            height_shift_range=0.15,    # ±15% (vs 10%)
            zoom_range=0.2,             # ±20% (vs 15%)
            # NOTE: NO brightness_range! Data is already normalized to remove brightness bias
            fill_mode='constant',
            cval=background_value,      # Use actual background value, not 0.5!
        )
        print(f"  ✓ Using aggressive augmentation with background fill={background_value:.3f}")
    else:
        print('Setting up standard data augmentation...')
        augmentation = ImageDataGenerator(
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.15,
            fill_mode='constant',
            cval=background_value,      # Use actual background value, not 0.5!
        )
        print(f"  ✓ Using standard augmentation with background fill={background_value:.3f}")
    
    # Create generator from the SPLIT training data (not all data)
    train_generator = augmentation.flow(
        X_train_norm, y_train_split,
        batch_size=batch_size,
        shuffle=True,
        seed=42
    )
    
    # Calculate steps per epoch
    steps_per_epoch = int(np.ceil(len(X_train_norm) / batch_size))
    
    print(f"\n✓ Training configuration:")
    print(f"  Samples per epoch: {len(X_train_norm)}")
    print(f"  Steps per epoch: {steps_per_epoch}")
    print(f"  Batch size: {batch_size}")
    
    # Train model
    print('\nStarting training...')

    # Prepare fit arguments
    fit_kwargs = {
        'epochs': epochs,
        'validation_data': (X_val_norm, y_val_split),
        'callbacks': callbacks,
        'steps_per_epoch': steps_per_epoch,
        'verbose': 1
    }

    # Add class weights if specified
    if class_weights is not None:
        fit_kwargs['class_weight'] = class_weights
        print(f"✓ Applying class weights during training")

    history = model.fit(
        train_generator,
        **fit_kwargs
    )
    
    # Evaluate on test set
    print('\nEvaluating on test set...')
    results = model.evaluate(X_test_norm, y_test, verbose=0)
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
    history_file = Path(model_output).parent / 'training_history.pkl'
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
    parser = argparse.ArgumentParser(description = 'Train QuickDraw classifier')
    parser.add_argument("--data-dir", default = 'data/processed', help = 'Directory with processed data')
    parser.add_argument("--epochs", type=int, default=50, help = 'Number of epochs')
    parser.add_argument("--batch-size", type=int, default=32, help = 'Batch size')
    parser.add_argument("--learning-rate", type=float, default=0.001, help = 'Learning rate')
    parser.add_argument("--label-smoothing", type=float, default=0.1, 
                       help = 'Label smoothing factor (0-1, default 0.1)')
    parser.add_argument("--architecture", default = 'custom', 
                       choices=["custom", "resnet50", "mobilenetv3", "efficientnet"],
                       help = 'Model architecture to use')
    parser.add_argument("--model-output", default = 'models/quickdraw_model.h5', help = 'Path to save model')
    parser.add_argument("--enhanced", action = 'store_true', 
                       help = 'Use enhanced model with more capacity (slower, more accurate)')
    parser.add_argument("--aggressive-aug", action = 'store_true',
                       help = 'Use aggressive data augmentation (rotation ±30°, shift ±20%, zoom ±25%)')
    parser.add_argument("--use-class-weighting", action = 'store_true',
                       help = 'Apply class weighting to handle imbalanced data')

    args = parser.parse_args()
    
    train_model(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        model_output=args.model_output,
        learning_rate=args.learning_rate,
        label_smoothing=args.label_smoothing,
        architecture=args.architecture,
        enhanced=args.enhanced,
        aggressive_aug=args.aggressive_aug,
        use_class_weighting=args.use_class_weighting
    )


if __name__ == '__main__':
    main()
