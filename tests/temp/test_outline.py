#!/usr/bin/env python3
"""
Test with outline extraction and skeletonization to match QuickDraw style.
"""
import numpy as np
import cv2
import sys
from pathlib import Path

sys.path.insert(0, '.')

from src.core.shape_normalization import preprocess_for_model
from src.web.app import tflite_interpreter

def process_variants(image_path):
    """Test multiple processing variants."""
    
    # Load
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img.mean() > 127:
        img = 255 - img
    if img.shape != (512, 512):
        img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA)
    
    # Extract ROI
    coords = np.column_stack(np.where(img > 50))
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    roi = img[y_min:y_max+1, x_min:x_max+1].copy()
    
    variants = []
    
    # Variant 1: Original (filled shape)
    variants.append(("Original filled", roi.copy()))
    
    # Variant 2: Outline only (Canny edges)
    edges = cv2.Canny(roi, 100, 200)
    variants.append(("Canny edges", edges))
    
    # Variant 3: Contour outline
    _, binary = cv2.threshold(roi, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_img = np.zeros_like(roi)
    cv2.drawContours(contour_img, contours, -1, 255, 2)
    variants.append(("Contour outline", contour_img))
    
    # Variant 4: Skeleton (thin centerline)
    skeleton_img = cv2.ximgproc.thinning(binary, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
    variants.append(("Skeleton", skeleton_img))
    
    # Variant 5: Dilated skeleton (slightly thicker)
    kernel = np.ones((3,3), np.uint8)
    skeleton_dilated = cv2.dilate(skeleton_img, kernel, iterations=1)
    variants.append(("Skeleton dilated", skeleton_dilated))
    
    # Variant 6: Thinner filled version
    eroded = cv2.erode(roi, kernel, iterations=2)
    variants.append(("Eroded filled", eroded))
    
    # Test each variant
    results = []
    for name, variant_img in variants:
        preprocessed = preprocess_for_model(variant_img, target_size=128)
        
        tflite_interpreter.set_tensor(
            tflite_interpreter.get_input_details()[0]['index'],
            preprocessed
        )
        tflite_interpreter.invoke()
        output = tflite_interpreter.get_tensor(
            tflite_interpreter.get_output_details()[0]['index']
        )
        confidence = float(output[0, 0])
        
        results.append((name, confidence, variant_img))
        print(f"{name:25s}: {confidence:.6f} ({confidence*100:.2f}%)")
        
        # Save debug image
        safe_name = name.replace(' ', '_').lower()
        cv2.imwrite(f'/tmp/variant_{safe_name}.png', variant_img)
    
    # Find best
    results.sort(key=lambda x: x[1], reverse=True)
    best_name, best_conf, best_img = results[0]
    
    print(f"\n{'='*70}")
    print(f"Best variant: {best_name} with {best_conf:.6f} ({best_conf*100:.2f}%)")
    
    if best_conf >= 0.90:
        print("✅ SUCCESS! Achieved >90% confidence!")
    elif best_conf >= 0.50:
        print(f"⚠️  PARTIAL: Need {(0.90-best_conf)*100:.1f}% more")
    else:
        print(f"❌ FAILED: Need {(0.90-best_conf)*100:.1f}% more")
    
    return results

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_outline.py <image_path>")
        sys.exit(1)
    
    image_path = Path(sys.argv[1])
    if not image_path.exists():
        print(f"ERROR: Image not found: {image_path}")
        sys.exit(1)
    
    process_variants(image_path)
