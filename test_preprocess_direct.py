#!/usr/bin/env python3
"""
Test preprocessing with skeletonization directly.
"""
import cv2
import numpy as np
import sys

sys.path.insert(0, '.')

from src.core.shape_normalization import preprocess_for_model
from src.web.app import tflite_interpreter

# Load the ROI
img = cv2.imread('/tmp/test_shape_0_roi.png', cv2.IMREAD_GRAYSCALE)
print(f"Input ROI: shape={img.shape}, range=[{img.min()}, {img.max()}], mean={img.mean():.1f}")

# Preprocess (should include skeletonization)
preprocessed = preprocess_for_model(img, target_size=128, skeletonize=True)
print(f"Preprocessed: shape={preprocessed.shape}, dtype={preprocessed.dtype}")
print(f"  range=[{preprocessed.min():.4f}, {preprocessed.max():.4f}]")
print(f"  mean={preprocessed.mean():.4f}, std={preprocessed.std():.4f}")

# Visualize
vis = preprocessed[0, :, :, 0]
vis_uint8 = (np.clip(vis, 0, 1) * 255).astype(np.uint8)
cv2.imwrite('/tmp/test_preprocessed_with_skel.png', vis_uint8)
print(f"\nSaved visualization to /tmp/test_preprocessed_with_skel.png")

# Run inference
tflite_interpreter.set_tensor(
    tflite_interpreter.get_input_details()[0]['index'],
    preprocessed
)
tflite_interpreter.invoke()
output = tflite_interpreter.get_tensor(
    tflite_interpreter.get_output_details()[0]['index']
)
confidence = float(output[0, 0])

print(f"\nModel confidence: {confidence:.6f} ({confidence*100:.2f}%)")

if confidence >= 0.90:
    print("✅ SUCCESS!")
elif confidence >= 0.50:
    print(f"⚠️  PARTIAL (need {(0.90-confidence)*100:.1f}% more)")
else:
    print(f"❌ FAILED (need {(0.90-confidence)*100:.1f}% more)")
