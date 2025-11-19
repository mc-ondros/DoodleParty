"""
Apply weight pruning to reduce model size.

This script applies magnitude-based weight pruning to a Keras model to reduce
model size and potentially improve inference speed. Pruning removes the least
important weights while maintaining accuracy through fine-tuning.

Usage:
    # Prune with default settings (50% sparsity)
    python scripts/convert/prune_model.py --model models/quickdraw_model.h5

    # Custom sparsity and fine-tuning
    python scripts/convert/prune_model.py --model models/quickdraw_model.h5 --sparsity 0.7 --epochs 10

    # Structured pruning for better hardware acceleration
    python scripts/convert/prune_model.py --model models/quickdraw_model.h5 --structured

Features:
    - Magnitude-based weight pruning
    - Progressive pruning schedule
    - Fine-tuning after pruning
    - Accuracy preservation
    - Size reduction analysis

Note:
    Requires training data for fine-tuning. The script will load data from
    data/processed/ directory.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmot


def parse_args():
    parser = argparse.ArgumentParser(
        description='Apply weight pruning to Keras model'
    )
    parser.add_argument(
        '--model', '-m',
        required=True,
        help='Path to Keras model (.h5 or .keras)'
    )
    parser.add_argument(
        '--output', '-o',
        default=None,
        help='Path to output pruned model (default: {model}_pruned.keras)'
    )
    parser.add_argument(
        '--sparsity', '-s',
        type=float,
        default=0.5,
        help='Target sparsity (0.0-1.0, default: 0.5 = 50%% weights removed)'
    )
    parser.add_argument(
        '--structured',
        action='store_true',
        help='Use structured pruning (better for hardware acceleration)'
    )
    parser.add_argument(
        '--epochs', '-e',
        type=int,
        default=5,
        help='Fine-tuning epochs after pruning (default: 5)'
    )
    parser.add_argument(
        '--batch-size', '-b',
        type=int,
        default=32,
        help='Batch size for fine-tuning (default: 32)'
    )
    parser.add_argument(
        '--data-dir', '-d',
        default='data/processed',
        help='Directory containing training data (default: data/processed)'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output'
    )
    return parser.parse_args()


def load_training_data(data_dir: Path):
    """Load training data for fine-tuning."""
    print(f"\nLoading training data from: {data_dir}")
    
    X_train_path = data_dir / 'X_train.npy'
    y_train_path = data_dir / 'y_train.npy'
    X_val_path = data_dir / 'X_test.npy'
    y_val_path = data_dir / 'y_test.npy'
    
    if not all([X_train_path.exists(), y_train_path.exists(), 
                X_val_path.exists(), y_val_path.exists()]):
        print('✗ Error: Training data not found')
        print('  Please run data processing first')
        sys.exit(1)
    
    X_train = np.load(X_train_path)
    y_train = np.load(y_train_path)
    X_val = np.load(X_val_path)
    y_val = np.load(y_val_path)
    
    print(f"✓ Loaded training data:")
    print(f"  Train: {X_train.shape}, {y_train.shape}")
    print(f"  Val:   {X_val.shape}, {y_val.shape}")
    
    return X_train, y_train, X_val, y_val


def create_pruning_schedule(sparsity: float, num_samples: int, batch_size: int, epochs: int):
    """Create a polynomial decay pruning schedule."""
    # Calculate steps
    steps_per_epoch = num_samples // batch_size
    total_steps = steps_per_epoch * epochs
    
    # Start pruning after 20% of training, end at 80%
    begin_step = int(0.2 * total_steps)
    end_step = int(0.8 * total_steps)
    
    pruning_schedule = tfmot.sparsity.keras.PolynomialDecay(
        initial_sparsity=0.0,
        final_sparsity=sparsity,
        begin_step=begin_step,
        end_step=end_step,
        frequency=steps_per_epoch
    )
    
    return pruning_schedule


def apply_pruning(model, sparsity: float, num_samples: int, batch_size: int, 
                  epochs: int, structured: bool = False):
    """Apply pruning to model."""
    print('\nApplying pruning...')
    print(f"  Target sparsity: {sparsity * 100:.0f}%")
    print(f"  Pruning type: {'Structured' if structured else 'Unstructured'}")
    
    # Create pruning schedule
    pruning_schedule = create_pruning_schedule(sparsity, num_samples, batch_size, epochs)
    
    # Pruning parameters
    pruning_params = {
        'pruning_schedule': pruning_schedule,
        'block_size': (1, 1) if not structured else (4, 4),
        'block_pooling_type': 'AVG'
    }
    
    # Apply pruning to model
    pruned_model = tfmot.sparsity.keras.prune_low_magnitude(
        model,
        **pruning_params
    )
    
    print('✓ Pruning applied to model')
    return pruned_model


def fine_tune_model(pruned_model, X_train, y_train, X_val, y_val, 
                    epochs: int, batch_size: int, verbose: bool):
    """Fine-tune pruned model."""
    print(f"\nFine-tuning for {epochs} epochs...")
    
    # Compile model
    pruned_model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy', 'precision', 'recall']
    )
    
    # Callbacks
    callbacks = [
        tfmot.sparsity.keras.UpdatePruningStep(),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=3,
            restore_best_weights=True
        )
    ]
    
    # Train
    history = pruned_model.fit(
        X_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1 if verbose else 2
    )
    
    # Final evaluation
    val_loss, val_acc, val_prec, val_rec = pruned_model.evaluate(
        X_val, y_val, verbose=0
    )
    
    print(f"\n✓ Fine-tuning complete")
    print(f"  Validation accuracy:  {val_acc:.4f}")
    print(f"  Validation precision: {val_prec:.4f}")
    print(f"  Validation recall:    {val_rec:.4f}")
    
    return pruned_model, val_acc


def strip_pruning_and_save(pruned_model, output_path: Path):
    """Remove pruning wrappers and save final model."""
    print(f"\nStripping pruning wrappers...")
    
    # Strip pruning wrappers
    final_model = tfmot.sparsity.keras.strip_pruning(pruned_model)
    
    # Save model
    print(f"Saving pruned model to: {output_path}")
    final_model.save(output_path)
    
    print('✓ Model saved')
    return final_model


def analyze_sparsity(model):
    """Analyze actual sparsity in model weights."""
    print('\nAnalyzing model sparsity...')
    
    total_weights = 0
    zero_weights = 0
    
    for layer in model.layers:
        weights = layer.get_weights()
        for w in weights:
            if len(w.shape) > 1:  # Only consider weight matrices (not biases)
                total_weights += w.size
                zero_weights += np.sum(w == 0)
    
    actual_sparsity = (zero_weights / total_weights) if total_weights > 0 else 0
    
    print(f"  Total weights: {total_weights:,}")
    print(f"  Zero weights:  {zero_weights:,}")
    print(f"  Actual sparsity: {actual_sparsity * 100:.2f}%")
    
    return actual_sparsity


def compare_models(original_path: Path, pruned_path: Path, 
                   original_acc: float, pruned_acc: float, actual_sparsity: float):
    """Compare original and pruned models."""
    original_size_mb = original_path.stat().st_size / (1024 * 1024)
    pruned_size_mb = pruned_path.stat().st_size / (1024 * 1024)
    
    # Note: File size may not decrease much until converted to TFLite
    size_reduction = ((original_size_mb - pruned_size_mb) / original_size_mb) * 100
    acc_diff = pruned_acc - original_acc
    
    print("\n" + "=" * 70)
    print('Pruning Results')
    print("=" * 70)
    print(f"{'Metric':<25} {'Original':<20} {'Pruned':<20} {'Change':<15}")
    print("-" * 70)
    print(f"{'Model size (MB)':<25} {original_size_mb:<20.2f} {pruned_size_mb:<20.2f} {size_reduction:<14.1f}%")
    print(f"{'Validation accuracy':<25} {original_acc:<20.4f} {pruned_acc:<20.4f} {acc_diff:<+14.4f}")
    print(f"{'Sparsity':<25} {0.0:<20.1f}% {actual_sparsity * 100:<19.1f}%")
    print("=" * 70)
    
    print('\nNotes:')
    print('  - Keras model size may not decrease significantly')
    print('  - Convert to TFLite for actual size reduction benefits')
    print('  - Pruned models work best with hardware acceleration')
    
    if abs(acc_diff) < 0.01:
        print('\n✓ Excellent: Minimal accuracy loss')
    elif abs(acc_diff) < 0.02:
        print('\n✓ Good: Acceptable accuracy loss')
    else:
        print('\n⚠ Warning: Significant accuracy change')


def main():
    args = parse_args()
    
    # Validate input
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Error: Model file not found: {model_path}")
        sys.exit(1)
    
    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = model_path.parent / f"{model_path.stem}_pruned.keras"
    
    data_dir = Path(args.data_dir)
    
    print("=" * 70)
    print('Weight Pruning')
    print("=" * 70)
    print(f"Input model:    {model_path}")
    print(f"Output model:   {output_path}")
    print(f"Target sparsity: {args.sparsity * 100:.0f}%")
    print(f"Fine-tune epochs: {args.epochs}")
    print("=" * 70)
    
    # Load original model
    print(f"\nLoading original model...")
    original_model = tf.keras.models.load_model(model_path)
    print('✓ Model loaded')
    
    # Load training data
    X_train, y_train, X_val, y_val = load_training_data(data_dir)
    
    # Evaluate original model
    print('\nEvaluating original model...')
    orig_loss, orig_acc, orig_prec, orig_rec = original_model.evaluate(
        X_val, y_val, verbose=0
    )
    print(f"  Accuracy:  {orig_acc:.4f}")
    print(f"  Precision: {orig_prec:.4f}")
    print(f"  Recall:    {orig_rec:.4f}")
    
    # Apply pruning
    pruned_model = apply_pruning(
        original_model,
        args.sparsity,
        len(X_train),
        args.batch_size,
        args.epochs,
        args.structured
    )
    
    # Fine-tune
    pruned_model, pruned_acc = fine_tune_model(
        pruned_model,
        X_train, y_train,
        X_val, y_val,
        args.epochs,
        args.batch_size,
        args.verbose
    )
    
    # Strip pruning wrappers and save
    final_model = strip_pruning_and_save(pruned_model, output_path)
    
    # Analyze sparsity
    actual_sparsity = analyze_sparsity(final_model)
    
    # Compare models
    compare_models(model_path, output_path, orig_acc, pruned_acc, actual_sparsity)
    
    print('\n✓ Pruning complete')
    print('\nNext steps:')
    print('  1. Convert to TFLite to see size reduction')
    print('  2. Benchmark inference speed')
    print('  3. Test on full validation set')


if __name__ == '__main__':
    main()
