"""
Training loops and callbacks for model training.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
    TensorBoard
)


def create_callbacks(model_name: str, checkpoint_dir: str = 'models/'):
    """
    Create training callbacks.
    
    Args:
        model_name: Name of the model for checkpoint saving
        checkpoint_dir: Directory to save checkpoints
    
    Returns:
        List of callbacks
    """
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        ),
        ModelCheckpoint(
            f'{checkpoint_dir}{model_name}_best.h5',
            monitor='val_accuracy',
            save_best_only=True
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7
        ),
        TensorBoard(
            log_dir=f'logs/{model_name}',
            histogram_freq=1
        )
    ]
    return callbacks


def train_model(
    model,
    train_data,
    val_data,
    epochs=50,
    batch_size=32,
    callbacks=None
):
    """
    Train a model.
    
    Args:
        model: Keras model to train
        train_data: Training data (X, y) tuple
        val_data: Validation data (X, y) tuple
        epochs: Number of training epochs
        batch_size: Batch size for training
        callbacks: Training callbacks
    
    Returns:
        Training history
    """
    history = model.fit(
        train_data[0],
        train_data[1],
        validation_data=val_data,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks or [],
        verbose=1
    )
    return history
