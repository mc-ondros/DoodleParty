#!/usr/bin/env python3
"""Quick test script for region-based detection."""

import numpy as np
import sys
sys.path.insert(0, '/home/diatom/Documents/DoodleHunter')

from src.core.patch_extraction import SlidingWindowDetector, AggregationStrategy

# Create a dummy 128x128 image
img = np.random.rand(128, 128, 1).astype(np.float32)

print(f"Image shape: {img.shape}")

# Create a dummy model
class DummyModel:
    def predict(self, x, verbose=0):
        """Return random predictions."""
        batch_size = x.shape[0]
        return np.random.rand(batch_size, 1).astype(np.float32)

model = DummyModel()

# Test detector
try:
    detector = SlidingWindowDetector(
        model=model,
        patch_size=(128, 128),
        stride=(128, 128),
        min_content_ratio=0.05,
        max_patches=16,
        early_stopping=True,
        early_stop_threshold=0.9,
        aggregation_strategy=AggregationStrategy.MAX,
        classification_threshold=0.5
    )
    
    print("Detector created successfully")
    
    result = detector.detect_batch(img)
    
    print(f"Detection result: {result}")
    print(f"Is positive: {result.is_positive}")
    print(f"Confidence: {result.confidence}")
    print(f"Patches analyzed: {result.num_patches_analyzed}")
    
except Exception as e:
    import traceback
    print(f"Error: {e}")
    traceback.print_exc()
