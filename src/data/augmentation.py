"""
Consistent data pipeline for preprocessing and normalization.

Ensures training and inference use the same preprocessing with
per-image z-score normalization to remove brightness bias.

Related:
- scripts/train.py (training pipeline)
- src/core/inference.py (inference preprocessing)
- src/data/loaders.py (data loading)

Exports:
- normalize_image, normalize_batch, get_augmentation_generator, prepare_test_data
"""

import numpy as np
from keras.src.legacy.preprocessing.image import ImageDataGenerator


def normalize_image(img):
    """
    Normalize image to have consistent brightness distribution.
    
    Apply per-image z-score normalization to remove brightness bias
    that causes the model to memorize "dense drawings are positive".
    
    Args:
        img: Image array of shape (H, W) or (H, W, C)
    
    Returns:
        Normalized image in range [0, 1]
    """
    img_flat = img.flatten()
    if img_flat.std() > 0.01:  # Avoid division by near-zero
        img_norm = (img - img_flat.mean()) / (img_flat.std() + 1e-7)
        img_norm = (img_norm + 2) / 4  # Rescale to ~0-1 range
        img_norm = np.clip(img_norm, 0, 1)
        return img_norm
    return img


def normalize_batch(X):
    """
    Apply per-image normalization to entire batch.
    
    Args:
        X: Batch of images (N, H, W, C) or (N, H, W)
    
    Returns:
        Normalized batch
    """
    X_norm = np.zeros_like(X, dtype=np.float32)
    for i in range(len(X)):
        X_norm[i] = normalize_image(X[i])
    return X_norm


def get_augmentation_generator(X_train, y_train, batch_size=32, 
                               rotation_range=15, width_shift=0.1, 
                               height_shift=0.1, zoom_range=0.15):
    """
    Create a data generator with consistent preprocessing.
    
    Args:
        X_train: Training images
        y_train: Training labels
        batch_size: Batch size
        rotation_range: Rotation degrees
        width_shift: Width shift fraction
        height_shift: Height shift fraction
        zoom_range: Zoom range fraction
    
    Returns:
        Fitted ImageDataGenerator and generator
    """
    # Normalize all training data first
    X_train_norm = normalize_batch(X_train)
    
    # Create augmentation generator
    augmentation = ImageDataGenerator(
        rotation_range=rotation_range,
        width_shift_range=width_shift,
        height_shift_range=height_shift,
        zoom_range=zoom_range,
        fill_mode='constant',
        cval=0.5,  # Use middle gray for padding
    )
    
    # Create batch generator
    generator = augmentation.flow(
        X_train_norm, y_train,
        batch_size=batch_size,
        shuffle=True,
        seed=42
    )
    
    return X_train_norm, generator, augmentation


def prepare_test_data(X_test):
    """
    Prepare test data with same normalization as training.
    
    Args:
        X_test: Test images
    
    Returns:
        Normalized test images
    """
    return normalize_batch(X_test)
