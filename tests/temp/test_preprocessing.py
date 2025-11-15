#!/usr/bin/env python3
"""
Test script to verify preprocessing is correct for the QuickDraw model.
"""
import numpy as np
import cv2
from PIL import Image
import sys
sys.path.insert(0, '.')

from src.web.app import tflite_interpreter, is_tflite

def test_model_with_simple_shapes():
    """Test the model with simple test images."""
    
    print("Testing model with simple shapes...")
    print(f"Using TFLite: {is_tflite}")
    
    if not tflite_interpreter:
        print("ERROR: No model loaded!")
        return
    
    input_details = tflite_interpreter.get_input_details()
    output_details = tflite_interpreter.get_output_details()
    
    print(f"Model expects: {input_details[0]['shape']}")
    
    # Test 1: All white (should get low probability - no penis)
    print("\n=== Test 1: All white canvas (empty) ===")
    white_canvas = np.ones((128, 128, 1), dtype=np.float32)
    white_canvas = white_canvas.reshape(1, 128, 128, 1)
    tflite_interpreter.set_tensor(input_details[0]['index'], white_canvas)
    tflite_interpreter.invoke()
    output = tflite_interpreter.get_tensor(output_details[0]['index'])
    print(f"Confidence: {output[0][0]:.6f}")
    
    # Test 2: All black (should get low probability - no penis)
    print("\n=== Test 2: All black canvas ===")
    black_canvas = np.zeros((128, 128, 1), dtype=np.float32)
    black_canvas = black_canvas.reshape(1, 128, 128, 1)
    tflite_interpreter.set_tensor(input_details[0]['index'], black_canvas)
    tflite_interpreter.invoke()
    output = tflite_interpreter.get_tensor(output_details[0]['index'])
    print(f"Confidence: {output[0][0]:.6f}")
    
    # Test 3: Simple vertical line (white on black - like QuickDraw)
    print("\n=== Test 3: Vertical line (white on black) ===")
    line_canvas = np.zeros((128, 128), dtype=np.uint8)
    # Draw a thick vertical line
    cv2.line(line_canvas, (64, 20), (64, 108), 255, 10)
    line_canvas_float = line_canvas.astype(np.float32) / 255.0
    line_canvas_float = line_canvas_float.reshape(1, 128, 128, 1)
    tflite_interpreter.set_tensor(input_details[0]['index'], line_canvas_float)
    tflite_interpreter.invoke()
    output = tflite_interpreter.get_tensor(output_details[0]['index'])
    print(f"Confidence: {output[0][0]:.6f}")
    
    # Test 4: Simple circle (white on black)
    print("\n=== Test 4: Circle (white on black) ===")
    circle_canvas = np.zeros((128, 128), dtype=np.uint8)
    cv2.circle(circle_canvas, (64, 64), 30, 255, 5)
    circle_canvas_float = circle_canvas.astype(np.float32) / 255.0
    circle_canvas_float = circle_canvas_float.reshape(1, 128, 128, 1)
    tflite_interpreter.set_tensor(input_details[0]['index'], circle_canvas_float)
    tflite_interpreter.invoke()
    output = tflite_interpreter.get_tensor(output_details[0]['index'])
    print(f"Confidence: {output[0][0]:.6f}")
    
    # Test 5: Penis-like shape (shaft + two circles)
    print("\n=== Test 5: Penis-like shape (white on black) ===")
    penis_canvas = np.zeros((128, 128), dtype=np.uint8)
    # Shaft
    cv2.rectangle(penis_canvas, (50, 40), (70, 100), 255, -1)
    # Two balls
    cv2.circle(penis_canvas, (45, 105), 12, 255, -1)
    cv2.circle(penis_canvas, (75, 105), 12, 255, -1)
    # Tip
    cv2.circle(penis_canvas, (60, 35), 12, 255, -1)
    
    penis_canvas_float = penis_canvas.astype(np.float32) / 255.0
    penis_canvas_float = penis_canvas_float.reshape(1, 128, 128, 1)
    tflite_interpreter.set_tensor(input_details[0]['index'], penis_canvas_float)
    tflite_interpreter.invoke()
    output = tflite_interpreter.get_tensor(output_details[0]['index'])
    print(f"Confidence: {output[0][0]:.6f}")
    
    print("\n" + "="*60)
    print("If all confidences are near 0.0, the model might not be properly")
    print("calibrated or there's a preprocessing mismatch.")
    print("If you see varied outputs (some high, some low), preprocessing is OK.")

if __name__ == "__main__":
    test_model_with_simple_shapes()
