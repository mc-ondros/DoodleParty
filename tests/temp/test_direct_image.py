#!/usr/bin/env python3
"""
Direct test using the provided penis drawing image.
"""
import numpy as np
import cv2
import sys
from pathlib import Path

sys.path.insert(0, '.')

from src.core.shape_detection import ShapeDetector
from src.web.app import tflite_interpreter, is_tflite, model

def load_and_test_image(image_path):
    """Load the provided image and test it."""
    
    # Read the image
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"ERROR: Could not load image from {image_path}")
        return None
    
    print(f"Loaded image: {img.shape}, range=[{img.min()}, {img.max()}], mean={img.mean():.1f}")
    
    # The image is black on light gray, need to invert to white on black
    # First, check if it needs inversion
    if img.mean() > 127:  # Light background
        print("Image has light background, inverting...")
        img = 255 - img
        print(f"After inversion: range=[{img.min()}, {img.max()}], mean={img.mean():.1f}")
    
    # Resize to 512x512 if needed
    if img.shape != (512, 512):
        print(f"Resizing from {img.shape} to (512, 512)...")
        img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA)
    
    # Create detector
    detector = ShapeDetector(
        model=model if model else None,
        tflite_interpreter=tflite_interpreter if tflite_interpreter else None,
        is_tflite=is_tflite,
        classification_threshold=0.5,
        min_shape_area=100,
        padding_color=0,
    )
    
    print("\n" + "="*70)
    print("TESTING DIRECT IMAGE")
    print("="*70)
    
    # Run detection
    result = detector.detect(img)
    
    print(f"\nResults:")
    print(f"  Confidence: {result.confidence:.4f} ({result.confidence*100:.2f}%)")
    print(f"  Is Positive: {result.is_positive}")
    print(f"  Shapes Analyzed: {result.num_shapes_analyzed}")
    print(f"  Canvas Dimensions: {result.canvas_dimensions}")
    
    if result.shape_predictions:
        print(f"\n  Individual Shape Confidences:")
        for i, shape in enumerate(result.shape_predictions):
            print(f"    Shape {i}: {shape.confidence:.4f} (bbox: {shape.bounding_box})")
    
    if result.grouped_boxes:
        print(f"\n  Grouped Detections:")
        for i, (box, score) in enumerate(zip(result.grouped_boxes, result.grouped_scores)):
            print(f"    Group {i}: {score:.4f} (bbox: {box})")
    
    print("\n" + "="*70)
    
    if result.confidence >= 0.90:
        print(f"✅ SUCCESS! Confidence {result.confidence:.4f} >= 0.90")
    elif result.confidence >= 0.50:
        print(f"⚠️  PARTIAL: Confidence {result.confidence:.4f} is positive but < 0.90")
        print(f"   Need to improve by {(0.90 - result.confidence)*100:.1f}%")
    else:
        print(f"❌ FAILED: Confidence {result.confidence:.4f} < 0.50")
        print(f"   This is a clear penis drawing, should be much higher!")
    
    return result

if __name__ == "__main__":
    # Check if image file was provided
    if len(sys.argv) > 1:
        image_path = Path(sys.argv[1])
    else:
        # Try common locations
        possible_paths = [
            "/tmp/penis_test.png",
            "/tmp/test_image.png",
            "test_image.png",
        ]
        image_path = None
        for p in possible_paths:
            if Path(p).exists():
                image_path = Path(p)
                break
        
        if not image_path:
            print("ERROR: No image file found!")
            print("Usage: python test_direct_image.py <image_path>")
            print("Or save image to /tmp/penis_test.png")
            sys.exit(1)
    
    if not image_path.exists():
        print(f"ERROR: Image file not found: {image_path}")
        sys.exit(1)
    
    load_and_test_image(image_path)
