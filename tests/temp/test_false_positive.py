#!/usr/bin/env python3
"""
Test false positive shapes.
"""
import cv2
import numpy as np
import sys

sys.path.insert(0, '.')

from src.core.shape_normalization import normalize_shape, preprocess_for_model
from src.web.app import tflite_interpreter

def test_image(path, name):
    print(f"\n{'='*70}")
    print(f"Testing: {name}")
    print('='*70)
    
    # Load and preprocess
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img.mean() > 127:
        img = 255 - img
    if img.shape != (512, 512):
        img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA)
    
    # Find the shape
    coords = np.column_stack(np.where(img > 50))
    if len(coords) == 0:
        print("No shape found!")
        return
    
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    bbox = (x_min, y_min, x_max - x_min, y_max - y_min)
    
    print(f"Bbox: {bbox}")
    
    # Extract ROI
    roi = img[y_min:y_max+1, x_min:x_max+1]
    cv2.imwrite(f'/tmp/fp_{name}_roi.png', roi)
    
    # Normalize (with padding)
    normalized = normalize_shape(img, bbox, target_size=128, padding_color=0)
    cv2.imwrite(f'/tmp/fp_{name}_normalized.png', normalized)
    print(f"Normalized: range=[{normalized.min()}, {normalized.max()}], bright_ratio={(normalized > 127).sum() / normalized.size:.3f}")
    
    # Preprocess (includes skeletonization)
    preprocessed = preprocess_for_model(normalized, target_size=128, skeletonize=True)
    vis = (preprocessed[0, :, :, 0] * 255).astype(np.uint8)
    cv2.imwrite(f'/tmp/fp_{name}_preprocessed.png', vis)
    
    # Infer
    tflite_interpreter.set_tensor(
        tflite_interpreter.get_input_details()[0]['index'],
        preprocessed
    )
    tflite_interpreter.invoke()
    output = tflite_interpreter.get_tensor(
        tflite_interpreter.get_output_details()[0]['index']
    )
    confidence = float(output[0, 0])
    
    print(f"Confidence: {confidence:.4f} ({confidence*100:.2f}%)")
    
    # Try without skeletonization
    preprocessed_no_skel = preprocess_for_model(normalized, target_size=128, skeletonize=False)
    vis_no_skel = (preprocessed_no_skel[0, :, :, 0] * 255).astype(np.uint8)
    cv2.imwrite(f'/tmp/fp_{name}_no_skel.png', vis_no_skel)
    
    tflite_interpreter.set_tensor(
        tflite_interpreter.get_input_details()[0]['index'],
        preprocessed_no_skel
    )
    tflite_interpreter.invoke()
    output_no_skel = tflite_interpreter.get_tensor(
        tflite_interpreter.get_output_details()[0]['index']
    )
    confidence_no_skel = float(output_no_skel[0, 0])
    
    print(f"Confidence (no skel): {confidence_no_skel:.4f} ({confidence_no_skel*100:.2f}%)")

if __name__ == "__main__":
    test_image("/home/diatom/20251108_08h54m17s_grim.png", "circle")
    test_image("/home/diatom/20251108_08h53m57s_grim.png", "square")
