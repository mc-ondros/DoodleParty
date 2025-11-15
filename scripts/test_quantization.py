#!/usr/bin/env python3
"""
Test TFLite INT8 Quantization Handling

Quick test to verify dequantization is working correctly.
"""

import sys
import numpy as np
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

import tensorflow as tf

# Find model - try different locations
possible_paths = [
    Path(__file__).parent.parent / 'models' / 'quickdraw_model_int8.tflite',
    Path('/home/mcvaj/DoodleParty/DoodleParty/models/quickdraw_model_int8.tflite'),
]

model_path = None
for p in possible_paths:
    if p.exists():
        model_path = p
        break

print("="*70)
print("  TFLITE INT8 QUANTIZATION TEST")
print("="*70)
print(f"\nModel: {model_path}")
print(f"Exists: {model_path.exists() if model_path else False}")
print()

if not model_path or not model_path.exists():
    print("❌ Model file not found!")
    print("\nSearched:")
    for p in possible_paths:
        print(f"  - {p}")
    sys.exit(1)

# Load interpreter
interpreter = tf.lite.Interpreter(model_path=str(model_path))
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]

print("INPUT DETAILS:")
print(f"  Shape: {input_details['shape']}")
print(f"  Dtype: {input_details['dtype']}")
print(f"  Quantization: {input_details['quantization']}")
print()

print("OUTPUT DETAILS:")
print(f"  Shape: {output_details['shape']}")
print(f"  Dtype: {output_details['dtype']}")
print(f"  Quantization: {output_details['quantization']}")
print()

# Create test input (black background, white "stroke" in center)
test_input = np.zeros((1, 128, 128, 1), dtype=np.float32)
test_input[0, 50:78, 50:78, 0] = 1.0  # White square in center

print("TEST INPUT:")
print(f"  Shape: {test_input.shape}")
print(f"  Dtype: {test_input.dtype}")
print(f"  Range: [{test_input.min():.3f}, {test_input.max():.3f}]")
print(f"  Mean: {test_input.mean():.3f}")
print()

# Quantize input if needed
input_to_use = test_input
if input_details['dtype'] == np.uint8:
    input_scale, input_zero_point = input_details['quantization']
    if input_scale > 0:
        input_to_use = (test_input / input_scale + input_zero_point).astype(np.uint8)
        print("INPUT QUANTIZED:")
        print(f"  Scale: {input_scale}")
        print(f"  Zero point: {input_zero_point}")
        print(f"  Range: [{input_to_use.min()}, {input_to_use.max()}]")
        print()

# Run inference
interpreter.set_tensor(input_details['index'], input_to_use)
interpreter.invoke()

# Get raw output
raw_output = interpreter.get_tensor(output_details['index'])

print("RAW OUTPUT:")
print(f"  Value: {raw_output[0][0]}")
print(f"  Dtype: {raw_output.dtype}")
print()

# Dequantize if needed
final_output = raw_output
if output_details['dtype'] in [np.uint8, np.int8]:
    output_scale, output_zero_point = output_details['quantization']
    if output_scale > 0:
        final_output = (raw_output.astype(np.float32) - output_zero_point) * output_scale
        print("OUTPUT DEQUANTIZED:")
        print(f"  Scale: {output_scale}")
        print(f"  Zero point: {output_zero_point}")
        print(f"  Raw value: {raw_output[0][0]}")
        print(f"  Dequantized value: {final_output[0][0]:.6f}")
        print()

print("="*70)
print(f"FINAL PREDICTION: {final_output[0][0]:.6f}")
print(f"Probability: {final_output[0][0] * 100:.2f}%")
print("="*70)

if final_output[0][0] == 0.0:
    print("\n⚠️  WARNING: Model returned 0.0!")
    print("This likely means:")
    print("  1. Quantization/dequantization is incorrect")
    print("  2. Input format doesn't match training data")
    print("  3. Model needs retraining or different preprocessing")
