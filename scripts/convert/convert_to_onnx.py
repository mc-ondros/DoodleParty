"""
Convert a Keras (.h5/.keras) model to ONNX using tf2onnx.

Usage:
    python scripts/convert/convert_to_onnx.py --model models/quickdraw_model.h5 --output models/quickdraw_model.onnx --opset 15

Notes:
- Requires `tf2onnx`. Install in your venv with:
    pip install tf2onnx
- If your model uses custom layers, pass them via `--custom-module` or modify the script to import them.
- For best compatibility with NPUs, try opset 13-15.
"""

import argparse
import sys
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description='Convert Keras model to ONNX (tf2onnx)')
    p.add_argument('--model', '-m', required=True, help='Path to Keras model (.h5 or .keras)')
    p.add_argument('--output', '-o', default=None, help='Path to write ONNX model (default: same dir, .onnx)')
    p.add_argument('--opset', '-s', type=int, default=15, help='Target ONNX opset version (default: 15)')
    p.add_argument('--dynamic', action='store_true', help='Export with dynamic batch size')
    p.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    return p.parse_args()


def main():
    args = parse_args()
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Error: model file not found: {model_path}")
        sys.exit(2)

    out_path = Path(args.output) if args.output else model_path.with_suffix('.onnx')

    # Lazy import so we can print nice error if missing
    try:
        import tensorflow as tf
        import tf2onnx
        from tf2onnx import convert
    except Exception as e:
        print('Missing required package: tf2onnx (or tensorflow).')
        print('Install with: pip install tf2onnx')
        print("Error details:", e)
        sys.exit(3)

    # Load Keras model
    print(f"Loading Keras model from: {model_path}")
    try:
        model = tf.keras.models.load_model(str(model_path))
    except Exception as e:
        print(f"Failed to load Keras model: {e}")
        sys.exit(4)

    # Prepare input signature
    # Model expects shape (None, 128, 128, 1) per README; use batch dimension dynamic or fixed
    input_shape = None
    try:
        # Try to infer input shape from model
        if hasattr(model, 'inputs') and model.inputs:
            # model.inputs is a list of tensors with shapes
            shape = model.inputs[0].shape.as_list()  # [None, H, W, C]
            input_shape = shape
    except Exception:
        input_shape = None

    if args.verbose:
        print("Inferred model input shape:", input_shape)

    # Build spec
    # tf2onnx.convert.from_keras accepts an opset and output_path
    try:
        print(f"Converting to ONNX (opset {args.opset}) -> {out_path}")
        # dynamic batch if requested
        if args.dynamic:
            # Use None for batch in input signature
            spec = (tf.TensorSpec((None, ) + tuple(input_shape[1:]), tf.float32, name='input'),)
            model_proto, external_tensor_storage = convert.from_keras(model, input_signature=spec, opset=args.opset)
            convert.save_model(model_proto, out_path)
        else:
            # Simple conversion, tf2onnx will infer shapes; pass opset
            model_proto, external_tensor_storage = convert.from_keras(model, opset=args.opset)
            convert.save_model(model_proto, out_path)

        print(f"ONNX model saved to: {out_path}")
    except Exception as e:
        print("Conversion failed:", e)
        # Provide troubleshooting tips
        print('Tips:')
        print(' - Try a different opset (13,14,15)')
        print(' - Install a matching tf2onnx version: pip install tf2onnx')
        print(' - If model uses custom layers, import them before loading the model')
        sys.exit(5)

if __name__ == '__main__':
    main()
