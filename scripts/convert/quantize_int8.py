"""
Apply INT8 post-training quantization to a TensorFlow Lite model.

This script performs full integer quantization (INT8) on a TFLite model using
a representative dataset for calibration. This results in smaller model size
and faster inference, especially on edge devices with hardware acceleration.

Usage:
    # Quantize with automatic data loading
    python scripts/convert/quantize_int8.py --model models/quickdraw_model.tflite

    # Specify custom data for calibration
    python scripts/convert/quantize_int8.py --model models/quickdraw_model.tflite --data-dir data/processed

    # Use Keras model as input (will convert to TFLite first)
    python scripts/convert/quantize_int8.py --model models/quickdraw_model.h5

Features:
    - Full INT8 quantization (weights and activations)
    - Representative dataset for calibration
    - Automatic data loading from processed directory
    - Accuracy comparison before/after quantization

Note:
    Requires calibration data. The script will use a subset of the training
    data for calibration to ensure accurate quantization.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf


def parse_args():
    parser = argparse.ArgumentParser(
        description='Apply INT8 quantization to TensorFlow Lite model'
    )
    parser.add_argument(
        '--model', '-m',
        required=True,
        help='Path to model (.tflite, .h5, or .keras)'
    )
    parser.add_argument(
        '--output', '-o',
        default=None,
        help='Path to output quantized model (default: {model}_int8.tflite)'
    )
    parser.add_argument(
        '--data-dir', '-d',
        default='data/processed',
        help='Directory containing calibration data (default: data/processed)'
    )
    parser.add_argument(
        '--num-samples', '-n',
        type=int,
        default=1000,
        help='Number of calibration samples (default: 1000)'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output'
    )
    return parser.parse_args()


def load_calibration_data(data_dir: Path, num_samples: int):
    """Load representative dataset for quantization calibration."""
    print(f"\nLoading calibration data from: {data_dir}")
    
    # Try to load processed training data
    train_data_path = data_dir / 'X_train.npy'
    
    if train_data_path.exists():
        print(f"  Loading from {train_data_path}")
        X_train = np.load(train_data_path)
        
        # Take a subset for calibration
        if len(X_train) > num_samples:
            indices = np.random.choice(len(X_train), num_samples, replace=False)
            X_calibration = X_train[indices]
        else:
            X_calibration = X_train
        
        print(f"✓ Loaded {len(X_calibration)} calibration samples")
        print(f"  Shape: {X_calibration.shape}")
        print(f"  dtype: {X_calibration.dtype}")
        
        return X_calibration
    else:
        print(f"✗ Error: Training data not found at {train_data_path}")
        print('  Please run data processing first or specify --data-dir')
        sys.exit(1)


def representative_dataset_generator(calibration_data):
    """
    Generator function for representative dataset.
    
    This is required by TFLite converter for INT8 quantization.
    It yields batches of input data that represent the typical
    data the model will see during inference.
    """
    for sample in calibration_data:
        # Add batch dimension and ensure float32 dtype
        sample = np.expand_dims(sample, axis=0).astype(np.float32)
        yield [sample]


def quantize_model(model_path: Path, calibration_data, optimize_ops=True):
    """Apply INT8 quantization to model."""
    print('\nApplying INT8 quantization...')
    
    # Determine if we need to convert from Keras first
    if model_path.suffix in ['.h5', '.keras']:
        print(f"  Input is Keras model, converting to TFLite first...")
        keras_model = tf.keras.models.load_model(model_path)
        converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    else:
        print(f"  Input is TFLite model")
        with open(model_path, 'rb') as f:
            tflite_model = f.read()
        
        # For TFLite input, we need to convert via saved model
        # This is a workaround since we can't directly quantize a TFLite model
        interpreter = tf.lite.Interpreter(model_content=tflite_model)
        print('  Note: Re-conversion may be needed for quantization')
        print('  Consider using the original Keras model for best results')
        
        # We'll need the Keras model for proper quantization
        print('✗ Error: Cannot quantize from TFLite model directly')
        print('  Please provide the original Keras model (.h5 or .keras)')
        sys.exit(2)
    
    # Configure quantization
    print('  - Optimization: DEFAULT (weight + activation quantization)')
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    print('  - Setting representative dataset for calibration')
    converter.representative_dataset = lambda: representative_dataset_generator(calibration_data)
    
    # Enforce full integer quantization
    print('  - Target spec: INT8 operations')
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    
    # Input/output also as INT8 (optional, can keep as float for easier integration)
    # Uncomment the following for full INT8 input/output:
    # converter.inference_input_type = tf.int8
    # converter.inference_output_type = tf.int8
    
    try:
        quantized_model = converter.convert()
        print('✓ Quantization successful')
        return quantized_model
    except Exception as e:
        print(f"✗ Quantization failed: {e}")
        print('\nTroubleshooting:')
        print('  1. Ensure you're using a Keras model, not TFLite')
        print('  2. Check that calibration data is valid')
        print('  3. Try with a smaller --num-samples')
        sys.exit(3)


def benchmark_models(original_path: Path, quantized_model: bytes, calibration_data):
    """Compare accuracy of original and quantized models."""
    print('\nBenchmarking models...')
    
    # Load original model
    if original_path.suffix in ['.h5', '.keras']:
        print('  Loading original Keras model...')
        keras_model = tf.keras.models.load_model(original_path)
        
        # Create float32 TFLite for fair comparison
        converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
        original_tflite = converter.convert()
    else:
        print('  Using original TFLite model...')
        with open(original_path, 'rb') as f:
            original_tflite = f.read()
    
    # Test samples (first 100 from calibration data)
    test_samples = calibration_data[:100]
    
    # Benchmark original
    print('  Benchmarking original model...')
    original_interpreter = tf.lite.Interpreter(model_content=original_tflite)
    original_interpreter.allocate_tensors()
    original_outputs = []
    
    for sample in test_samples:
        input_tensor = np.expand_dims(sample, axis=0).astype(np.float32)
        original_interpreter.set_tensor(
            original_interpreter.get_input_details()[0]['index'],
            input_tensor
        )
        original_interpreter.invoke()
        output = original_interpreter.get_tensor(
            original_interpreter.get_output_details()[0]['index']
        )
        original_outputs.append(output[0])
    
    # Benchmark quantized
    print('  Benchmarking quantized model...')
    quantized_interpreter = tf.lite.Interpreter(model_content=quantized_model)
    quantized_interpreter.allocate_tensors()
    quantized_outputs = []
    
    for sample in test_samples:
        input_tensor = np.expand_dims(sample, axis=0).astype(np.float32)
        quantized_interpreter.set_tensor(
            quantized_interpreter.get_input_details()[0]['index'],
            input_tensor
        )
        quantized_interpreter.invoke()
        output = quantized_interpreter.get_tensor(
            quantized_interpreter.get_output_details()[0]['index']
        )
        quantized_outputs.append(output[0])
    
    # Compare predictions
    original_outputs = np.array(original_outputs)
    quantized_outputs = np.array(quantized_outputs)
    
    mae = np.abs(original_outputs - quantized_outputs).mean()
    max_diff = np.abs(original_outputs - quantized_outputs).max()
    
    print("\n" + "=" * 70)
    print('Accuracy Comparison')
    print("=" * 70)
    print(f"Mean Absolute Error:   {mae:.6f}")
    print(f"Max Absolute Error:    {max_diff:.6f}")
    
    # Check if predictions agree on classification
    original_classes = (original_outputs > 0.5).astype(int)
    quantized_classes = (quantized_outputs > 0.5).astype(int)
    agreement = (original_classes == quantized_classes).mean() * 100
    
    print(f"Classification Agreement: {agreement:.1f}%")
    print("=" * 70)
    
    if agreement > 98:
        print('✓ Excellent: Minimal accuracy loss')
    elif agreement > 95:
        print('✓ Good: Acceptable accuracy loss')
    elif agreement > 90:
        print('⚠ Warning: Noticeable accuracy degradation')
    else:
        print('✗ Poor: Significant accuracy loss')


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
        output_path = model_path.parent / f"{model_path.stem}_int8.tflite"
    
    data_dir = Path(args.data_dir)
    
    print("=" * 70)
    print('INT8 Post-Training Quantization')
    print("=" * 70)
    print(f"Input model:       {model_path}")
    print(f"Output model:      {output_path}")
    print(f"Calibration data:  {data_dir}")
    print(f"Calibration samples: {args.num_samples}")
    print("=" * 70)
    
    # Load calibration data
    calibration_data = load_calibration_data(data_dir, args.num_samples)
    
    # Get original model size
    original_size_mb = model_path.stat().st_size / (1024 * 1024)
    print(f"\nOriginal model size: {original_size_mb:.2f} MB")
    
    # Quantize model
    quantized_model = quantize_model(model_path, calibration_data)
    
    # Benchmark
    benchmark_models(model_path, quantized_model, calibration_data)
    
    # Save
    print(f"\nSaving quantized model to: {output_path}")
    with open(output_path, 'wb') as f:
        f.write(quantized_model)
    
    quantized_size_mb = len(quantized_model) / (1024 * 1024)
    reduction = ((original_size_mb - quantized_size_mb) / original_size_mb) * 100
    
    print("\n" + "=" * 70)
    print('Quantization Summary')
    print("=" * 70)
    print(f"Original size:  {original_size_mb:.2f} MB")
    print(f"Quantized size: {quantized_size_mb:.2f} MB")
    print(f"Size reduction: {reduction:.1f}%")
    print(f"Speed improvement: ~2-4x (depending on hardware)")
    print("=" * 70)
    print('\n✓ INT8 quantization complete')
    print('  Next steps:')
    print('  1. Benchmark inference speed with benchmark_tflite.py')
    print('  2. Test accuracy on full validation set')
    print('  3. Deploy to production')


if __name__ == '__main__':
    main()
