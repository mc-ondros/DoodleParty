"""
Training loops and callbacks for model training.
"""

import tensorflow as tf
try:
    # TensorFlow 2.16+ uses Keras 3
    import keras
    from keras.callbacks import (
        EarlyStopping,
        ModelCheckpoint,
        ReduceLROnPlateau,
        TensorBoard,
        Callback
    )
except ImportError:
    # Fallback for older TensorFlow versions
    from tensorflow import keras
    from tensorflow.keras.callbacks import (
        EarlyStopping,
        ModelCheckpoint,
        ReduceLROnPlateau,
        TensorBoard,
        Callback
    )
from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.table import Table
from rich import box

console = Console()


class RichProgressCallback(Callback):
    """
    Custom Keras callback with Rich TUI progress display.
    """
    
    def __init__(self, epochs):
        super().__init__()
        self.epochs = epochs
        self.progress = None
        self.task = None
        self.current_epoch = 0
        
    def on_train_begin(self, logs=None):
        console.print("[bold cyan]Starting training...[/bold cyan]\n")
        
    def on_epoch_begin(self, epoch, logs=None):
        self.current_epoch = epoch + 1
        
    def on_epoch_end(self, epoch, logs=None):
        # Create metrics display
        metrics_text = f"[bold cyan]Epoch {self.current_epoch}/{self.epochs}[/bold cyan] | "
        
        if logs:
            if 'loss' in logs:
                metrics_text += f"[yellow]loss:[/yellow] {logs['loss']:.4f} | "
            if 'accuracy' in logs:
                metrics_text += f"[yellow]acc:[/yellow] {logs['accuracy']:.4f} | "
            if 'val_loss' in logs:
                metrics_text += f"[green]val_loss:[/green] {logs['val_loss']:.4f} | "
            if 'val_accuracy' in logs:
                metrics_text += f"[green]val_acc:[/green] {logs['val_accuracy']:.4f}"
        
        console.print(metrics_text)
        
    def on_train_end(self, logs=None):
        console.print("\n[bold green]✓[/bold green] Training session completed!")


def create_callbacks(model_name: str, checkpoint_dir: str = 'models/', epochs: int = 50):
    """
    Create training callbacks.
    
    Args:
        model_name: Name of the model for checkpoint saving
        checkpoint_dir: Directory to save checkpoints
        epochs: Total number of epochs (for progress display)
    
    Returns:
        List of callbacks
    """
    callbacks = [
        RichProgressCallback(epochs=epochs),
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        ),
        ModelCheckpoint(
            f'{checkpoint_dir}{model_name}_best.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=0
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=0
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
        verbose=2  # One line per epoch for cleaner output with Rich
    )
    return history
