#!/usr/bin/env python3
"""
Debug binarization in shape extraction.
"""
import cv2
import numpy as np
import sys
from pathlib import Path

def test_binary_debug(image_path):
    """Debug binary processing."""
    
    # Load and preprocess
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img.mean() > 127:
        img = 255 - img
    if img.shape != (512, 512):
        img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA)
    
    print(f"Input: range=[{img.min()}, {img.max()}], mean={img.mean():.1f}")
    cv2.imwrite('/tmp/step1_input.png', img)
    
    # Binarize with Otsu (like extract_shapes does)
    img_blur = cv2.GaussianBlur(img, (3, 3), 0)
    _, binary = cv2.threshold(img_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    print(f"After Otsu: range=[{binary.min()}, {binary.max()}], mean={binary.mean():.1f}")
    cv2.imwrite('/tmp/step2_otsu.png', binary)
    
    # Check the problematic inversion condition
    if np.mean(binary) < 127:
        print("INVERTING because mean < 127!")
        binary_inverted = cv2.bitwise_not(binary)
        cv2.imwrite('/tmp/step3_inverted.png', binary_inverted)
        print(f"After inversion: range=[{binary_inverted.min()}, {binary_inverted.max()}], mean={binary_inverted.mean():.1f}")
        
        # Find contours on inverted
        contours, _ = cv2.findContours(binary_inverted, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        print(f"Found {len(contours)} contours on inverted image")
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            bbox = cv2.boundingRect(contour)
            print(f"  Contour {i}: area={area:.1f}, bbox={bbox}")
    else:
        print("NOT inverting (mean >= 127)")
        # Find contours on original binary
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        print(f"Found {len(contours)} contours on original binary image")
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            bbox = cv2.boundingRect(contour)
            print(f"  Contour {i}: area={area:.1f}, bbox={bbox}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_binary_debug.py <image_path>")
        sys.exit(1)
    
    test_binary_debug(Path(sys.argv[1]))