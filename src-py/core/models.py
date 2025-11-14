"""
ML model architectures for DoodleParty.
Supports: Custom CNN, ResNet50, MobileNetV3, EfficientNet
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def build_custom_cnn(input_shape=(28, 28, 1), num_classes=1):
    """
    Build a custom CNN architecture for sketch classification.
    
    Optimized for 28x28 QuickDraw format and binary classification.
    Target: ~423K parameters for efficient RPi4 inference.
    
    Args:
        input_shape: Input image shape (height, width, channels) - default 28x28x1
        num_classes: Number of output classes (1 for binary, >1 for multi-class)
    
    Returns:
        Compiled Keras model
    """
    # Binary vs multi-class classification
    is_binary = num_classes == 1
    
    model = keras.Sequential([
        # First conv block: 28x28x1 -> 14x14x32
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape, padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Second conv block: 14x14x32 -> 7x7x64
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Third conv block: 7x7x64 -> 7x7x128
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Dropout(0.25),
        
        # Dense layers
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        
        # Output layer
        layers.Dense(num_classes, activation='sigmoid' if is_binary else 'softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy' if is_binary else 'categorical_crossentropy',
        metrics=['accuracy', 'precision', 'recall']
    )
    
    return model


def build_mobilenet_v3(input_shape=(128, 128, 3), num_classes=345):
    """Build MobileNetV3 model for efficient inference."""
    base_model = keras.applications.MobileNetV3Large(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False
    
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model


def build_efficientnet_b0(input_shape=(128, 128, 3), num_classes=345):
    """Build EfficientNetB0 model."""
    base_model = keras.applications.EfficientNetB0(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False
    
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model
