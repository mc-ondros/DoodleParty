#!/usr/bin/env python3
"""
Debug shape extraction to see what shapes are found.
"""
import numpy as np
import cv2
import sys
from pathlib import Path

sys.path.insert(0, '.')

from src.core.shape_extraction import propose_shapes

def test_shape_extraction(image_path):
    """Test shape extraction."""
    
    # Load and preprocess like the app does
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    print(f"Original: {img.shape}, range=[{img.min()}, {img.max()}], mean={img.mean():.1f}")
    
    # Invert if light background
    if img.mean() > 127:
        img = 255 - img
        print(f"After inversion: range=[{img.min()}, {img.max()}], mean={img.mean():.1f}")
    
    # Resize to 512x512
    if img.shape != (512, 512):
        img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA)
        print(f"After resize: {img.shape}")
    
    cv2.imwrite('/tmp/test_canvas.png', img)
    print(f"\nCanvas saved to /tmp/test_canvas.png")
    
    # Try to extract shapes (no stroke history)
    print(f"\nAttempting shape extraction (no stroke history)...")
    shapes = propose_shapes(gray=img, stroke_history=None, min_shape_area=100)
    
    print(f"\nFound {len(shapes)} shapes:")
    for i, (contour, bbox) in enumerate(shapes):
        x, y, w, h = bbox
        print(f"  Shape {i}: bbox=({x}, {y}, {w}, {h}), area={w*h}")
        
        # Extract and save ROI
        roi = img[y:y+h, x:x+w]
        cv2.imwrite(f'/tmp/test_shape_{i}_roi.png', roi)
        print(f"    Saved ROI to /tmp/test_shape_{i}_roi.png")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_shape_extraction.py <image_path>")
        sys.exit(1)
    
    image_path = Path(sys.argv[1])
    if not image_path.exists():
        print(f"ERROR: Image not found: {image_path}")
        sys.exit(1)
    
    test_shape_extraction(image_path)
