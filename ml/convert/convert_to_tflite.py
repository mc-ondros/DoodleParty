"""
Convert a Keras model to TensorFlow Lite format.

This script converts a saved Keras model (.h5 or .keras) to TensorFlow Lite (.tflite)
format for optimized inference on mobile and edge devices.

Usage:
    # Basic conversion (float32)
    python scripts/convert/convert_to_tflite.py --model models/quickdraw_model.h5

    # With custom output path
    python scripts/convert/convert_to_tflite.py --model models/quickdraw_model.h5 --output models/model.tflite

    # Optimize for size (dynamic range quantization)
    python scripts/convert/convert_to_tflite.py --model models/quickdraw_model.h5 --optimize-size

Features:
    - Converts Keras models to TFLite format
    - Optional dynamic range quantization for size reduction
    - Model validation after conversion
    - Size comparison reporting

Note:
    For INT8 quantization with representative dataset, use quantize_int8.py instead.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert Keras model to TensorFlow Lite format'
    )
    parser.add_argument(
        '--model', '-m',
        required=True,
        help='Path to Keras model (.h5 or .keras)'
    )
    parser.add_argument(
        '--output', '-o',
        default=None,
        help='Path to output TFLite model (default: same name with .tflite extension)'
    )
    parser.add_argument(
        '--optimize-size',
        action='store_true',
        help='Apply dynamic range quantization to reduce model size'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output'
    )
    return parser.parse_args()


def load_keras_model(model_path: Path):
    """Load a Keras model from file."""
    print(f"Loading Keras model from: {model_path}")
    try:
        model = tf.keras.models.load_model(model_path)
        print(f"✓ Model loaded successfully")
        return model
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        sys.exit(1)


def convert_to_tflite(model, optimize_size: bool = False):
    """Convert Keras model to TensorFlow Lite format."""
    print('\nConverting to TensorFlow Lite...')
    
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    if optimize_size:
        print('  - Applying dynamic range quantization (weights only)')
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    try:
        tflite_model = converter.convert()
        print('✓ Conversion successful')
        return tflite_model
    except Exception as e:
        print(f"✗ Conversion failed: {e}")
        sys.exit(2)


def validate_tflite_model(tflite_model: bytes, original_model):
    """Validate the TFLite model by running a test inference."""
    print('\nValidating TFLite model...')
    
    try:
        # Create interpreter
        interpreter = tf.lite.Interpreter(model_content=tflite_model)
        interpreter.allocate_tensors()
        
        # Get input/output details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        print(f"  Input shape: {input_details[0]['shape']}")
        print(f"  Input dtype: {input_details[0]['dtype']}")
        print(f"  Output shape: {output_details[0]['shape']}")
        print(f"  Output dtype: {output_details[0]['dtype']}")
        
        # Run a test inference with dummy data
        input_shape = input_details[0]['shape']
        test_input = np.random.rand(*input_shape).astype(input_details[0]['dtype'])
        
        # TFLite inference
        interpreter.set_tensor(input_details[0]['index'], test_input)
        interpreter.invoke()
        tflite_output = interpreter.get_tensor(output_details[0]['index'])
        
        # Keras inference for comparison
        keras_output = original_model.predict(test_input, verbose=0)
        
        # Compare outputs
        diff = np.abs(tflite_output - keras_output).mean()
        print(f"  Mean absolute difference: {diff:.6f}")
        
        if diff < 1e-5:
            print('✓ Validation passed (outputs match)')
        elif diff < 1e-3:
            print('✓ Validation passed (small difference, acceptable)')
        else:
            print(f"⚠ Warning: Large difference between models ({diff:.6f})")
        
        return True
        
    except Exception as e:
        print(f"✗ Validation failed: {e}")
        return False


def save_tflite_model(tflite_model: bytes, output_path: Path):
    """Save TFLite model to file."""
    print(f"\nSaving TFLite model to: {output_path}")
    try:
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
        
        size_mb = len(tflite_model) / (1024 * 1024)
        print(f"✓ Model saved successfully ({size_mb:.2f} MB)")
        return True
    except Exception as e:
        print(f"✗ Error saving model: {e}")
        return False


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
        output_path = model_path.with_suffix('.tflite')
    
    print("=" * 70)
    print('TensorFlow Lite Conversion')
    print("=" * 70)
    print(f"Input model:  {model_path}")
    print(f"Output model: {output_path}")
    print(f"Optimization: {'Dynamic range quantization' if args.optimize_size else 'None (float32)'}")
    print("=" * 70)
    
    # Load model
    keras_model = load_keras_model(model_path)
    
    # Get original model size
    original_size_mb = model_path.stat().st_size / (1024 * 1024)
    print(f"Original model size: {original_size_mb:.2f} MB")
    
    # Convert to TFLite
    tflite_model = convert_to_tflite(keras_model, args.optimize_size)
    
    # Validate
    validate_tflite_model(tflite_model, keras_model)
    
    # Save
    if save_tflite_model(tflite_model, output_path):
        tflite_size_mb = len(tflite_model) / (1024 * 1024)
        reduction = ((original_size_mb - tflite_size_mb) / original_size_mb) * 100
        
        print("\n" + "=" * 70)
        print('Conversion Summary')
        print("=" * 70)
        print(f"Original size:  {original_size_mb:.2f} MB")
        print(f"TFLite size:    {tflite_size_mb:.2f} MB")
        print(f"Size reduction: {reduction:.1f}%")
        print("=" * 70)
        
        if args.optimize_size:
            print('\n✓ Model converted with dynamic range quantization')
            print('  Note: Weights are quantized, but activations are still float32')
            print('  For full INT8 quantization, use quantize_int8.py')
        else:
            print('\n✓ Model converted to TFLite (float32)')
            print('  For smaller size, re-run with --optimize-size')
            print('  For INT8 quantization, use quantize_int8.py')
    else:
        sys.exit(3)


if __name__ == '__main__':
    main()
