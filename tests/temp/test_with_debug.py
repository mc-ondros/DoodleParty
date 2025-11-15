#!/usr/bin/env python3
"""
Test with full debug output and image saving.
"""
import numpy as np
import cv2
import sys
from pathlib import Path

sys.path.insert(0, '.')

from src.core.shape_detection import ShapeDetector
from src.core.shape_normalization import preprocess_for_model
from src.web.app import tflite_interpreter, is_tflite, model

def test_with_debug(image_path):
    """Test with full debugging."""
    
    # Read the image
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    print(f"1. Loaded image: {img.shape}, range=[{img.min()}, {img.max()}], mean={img.mean():.1f}")
    cv2.imwrite('/tmp/debug_01_original.png', img)
    
    # Invert if needed (light background)
    if img.mean() > 127:
        print("2. Inverting (light background detected)...")
        img = 255 - img
        print(f"   After inversion: range=[{img.min()}, {img.max()}], mean={img.mean():.1f}")
        cv2.imwrite('/tmp/debug_02_inverted.png', img)
    
    # Resize to 512x512
    if img.shape != (512, 512):
        print(f"3. Resizing from {img.shape} to (512, 512)...")
        img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA)
        cv2.imwrite('/tmp/debug_03_resized.png', img)
    
    # Extract the shape (find bounding box)
    print("4. Extracting shape bounding box...")
    coords = np.column_stack(np.where(img > 50))
    if len(coords) == 0:
        print("   ERROR: No pixels > 50!")
        return
    
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    print(f"   Bounding box: ({x_min}, {y_min}) to ({x_max}, {y_max})")
    print(f"   Shape size: {x_max - x_min} x {y_max - y_min}")
    
    # Extract the ROI
    shape_img = img[y_min:y_max+1, x_min:x_max+1].copy()
    print(f"   ROI shape: {shape_img.shape}, range=[{shape_img.min()}, {shape_img.max()}]")
    cv2.imwrite('/tmp/debug_04_roi.png', shape_img)
    
    # Preprocess for model (this applies padding, resizing, z-score normalization)
    print("5. Preprocessing for model (128x128, z-score normalized)...")
    preprocessed = preprocess_for_model(shape_img, target_size=128)
    print(f"   Preprocessed shape: {preprocessed.shape}, dtype: {preprocessed.dtype}")
    print(f"   Preprocessed range: [{preprocessed.min():.4f}, {preprocessed.max():.4f}]")
    print(f"   Preprocessed mean: {preprocessed.mean():.4f}, std: {preprocessed.std():.4f}")
    
    # Save preprocessed as viewable image (scale to 0-255 for visualization)
    preprocessed_vis = preprocessed.squeeze()
    preprocessed_vis = np.clip(preprocessed_vis * 255, 0, 255).astype(np.uint8)
    cv2.imwrite('/tmp/debug_05_preprocessed.png', preprocessed_vis)
    
    # Run model inference directly
    print("6. Running model inference...")
    if tflite_interpreter:
        tflite_interpreter.set_tensor(
            tflite_interpreter.get_input_details()[0]['index'],
            preprocessed
        )
        tflite_interpreter.invoke()
        output = tflite_interpreter.get_tensor(
            tflite_interpreter.get_output_details()[0]['index']
        )
        confidence = float(output[0, 0])
        print(f"   Raw model output: {output}")
        print(f"   Confidence: {confidence:.6f} ({confidence*100:.2f}%)")
    
    print("\nDebug images saved to /tmp/debug_*.png")
    print("Examine them to see what's happening at each step!")
    
    return confidence

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_with_debug.py <image_path>")
        sys.exit(1)
    
    image_path = Path(sys.argv[1])
    if not image_path.exists():
        print(f"ERROR: Image not found: {image_path}")
        sys.exit(1)
    
    confidence = test_with_debug(image_path)
    
    print("\n" + "="*70)
    if confidence >= 0.90:
        print(f"✅ SUCCESS! {confidence:.4f} >= 0.90")
    elif confidence >= 0.50:
        print(f"⚠️  PARTIAL: {confidence:.4f} (need {(0.90-confidence)*100:.1f}% more)")
    else:
        print(f"❌ FAILED: {confidence:.4f} (need {(0.90-confidence)*100:.1f}% more)")
