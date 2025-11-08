#!/usr/bin/env python3
"""
End-to-end test to verify preprocessing pipeline matches training.
This simulates what happens when a user draws on the canvas.
"""
import numpy as np
import cv2
import sys
import base64
from io import BytesIO
from PIL import Image

sys.path.insert(0, '.')

from src.core.shape_detection import ShapeDetector
from src.web.app import tflite_interpreter, is_tflite, model

def create_test_drawing():
    """Create a simple test drawing (white strokes on black background)."""
    # This simulates what the model expects after all preprocessing
    canvas = np.zeros((512, 512), dtype=np.uint8)
    
    # Draw a simple penis-like shape (shaft + 2 circles)
    # White on black (like QuickDraw dataset)
    cv2.rectangle(canvas, (200, 150), (250, 350), 255, -1)  # Shaft
    cv2.circle(canvas, (190, 365), 25, 255, -1)  # Left ball
    cv2.circle(canvas, (260, 365), 25, 255, -1)  # Right ball
    cv2.circle(canvas, (225, 140), 30, 255, -1)  # Tip
    
    return canvas

def simulate_canvas_drawing():
    """
    Simulate what the canvas sends: black strokes on white background.
    This is what needs to be inverted before processing.
    """
    # Start with white background
    canvas = np.ones((512, 512), dtype=np.uint8) * 255
    
    # Draw black strokes (what user draws)
    cv2.rectangle(canvas, (200, 150), (250, 350), 0, -1)  # Shaft
    cv2.circle(canvas, (190, 365), 25, 0, -1)  # Left ball
    cv2.circle(canvas, (260, 365), 25, 0, -1)  # Right ball
    cv2.circle(canvas, (225, 140), 30, 0, -1)  # Tip
    
    # Convert to base64 (what the web app receives)
    pil_img = Image.fromarray(canvas)
    buffer = BytesIO()
    pil_img.save(buffer, format='PNG')
    img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    data_url = f"data:image/png;base64,{img_base64}"
    
    return canvas, data_url

def test_preprocessing_pipeline():
    """Test the complete preprocessing pipeline."""
    print("="*70)
    print("END-TO-END PREPROCESSING TEST")
    print("="*70)
    
    # Test 1: Direct white-on-black (what model expects)
    print("\n=== Test 1: Direct white-on-black canvas ===")
    canvas_white_on_black = create_test_drawing()
    
    detector = ShapeDetector(
        model=model if model else None,
        tflite_interpreter=tflite_interpreter if tflite_interpreter else None,
        is_tflite=is_tflite,
        classification_threshold=0.5,
    )
    
    result = detector.detect(canvas_white_on_black)
    print(f"Confidence: {result.confidence:.4f}")
    print(f"Is Positive: {result.is_positive}")
    print(f"Shapes Analyzed: {result.num_shapes_analyzed}")
    
    # Test 2: Simulated canvas input (black-on-white, needs inversion)
    print("\n=== Test 2: Simulated canvas drawing (black-on-white) ===")
    canvas_black_on_white, data_url = simulate_canvas_drawing()
    
    # Decode and process like app.py does
    image_data_clean = data_url.split(',')[1]
    decoded = base64.b64decode(image_data_clean)
    pil_img = Image.open(BytesIO(decoded)).convert('L')
    img_array = np.array(pil_img, dtype=np.uint8)
    
    print(f"Before inversion: min={img_array.min()}, max={img_array.max()}, mean={img_array.mean():.1f}")
    
    # IMPORTANT: Invert (like we fixed in app.py)
    img_array = 255 - img_array
    
    print(f"After inversion: min={img_array.min()}, max={img_array.max()}, mean={img_array.mean():.1f}")
    
    if img_array.shape != (512, 512):
        img_array = cv2.resize(img_array, (512, 512), interpolation=cv2.INTER_AREA)
    
    result2 = detector.detect(img_array)
    print(f"Confidence: {result2.confidence:.4f}")
    print(f"Is Positive: {result2.is_positive}")
    print(f"Shapes Analyzed: {result2.num_shapes_analyzed}")
    
    # Test 3: Empty canvas
    print("\n=== Test 3: Empty canvas ===")
    empty_canvas = np.zeros((512, 512), dtype=np.uint8)
    result3 = detector.detect(empty_canvas)
    print(f"Confidence: {result3.confidence:.4f}")
    print(f"Is Positive: {result3.is_positive}")
    
    # Test 4: Simple circle
    print("\n=== Test 4: Simple circle (should be low) ===")
    circle_canvas = np.zeros((512, 512), dtype=np.uint8)
    cv2.circle(circle_canvas, (256, 256), 100, 255, 10)
    result4 = detector.detect(circle_canvas)
    print(f"Confidence: {result4.confidence:.4f}")
    print(f"Is Positive: {result4.is_positive}")
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Test 1 (Direct white-on-black): {result.confidence:.4f}")
    print(f"Test 2 (Canvas simulation):     {result2.confidence:.4f}")
    print(f"Test 3 (Empty canvas):          {result3.confidence:.4f}")
    print(f"Test 4 (Circle):                {result4.confidence:.4f}")
    print()
    
    if result.confidence > 0.01 or result2.confidence > 0.01:
        print("✅ Preprocessing appears to be working! Model is producing varied outputs.")
    else:
        print("⚠️  Still getting very low confidences. Possible issues:")
        print("   - Model may not be properly trained")
        print("   - Model file might be corrupted")
        print("   - There might be another preprocessing mismatch")
    
    print("\nExpected behavior:")
    print("- Penis-like shape: >0.3 confidence (ideally >0.5)")
    print("- Empty canvas: <0.1 confidence")
    print("- Circle: <0.2 confidence")

if __name__ == "__main__":
    test_preprocessing_pipeline()
