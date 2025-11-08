#!/usr/bin/env python3
"""
Simple script to test confidence level debugging.

Run this to generate debug logs that will help identify why confidence levels are constant.
"""

import sys
import logging
import base64
import numpy as np
from PIL import Image
from io import BytesIO

# Add the project root to sys.path so imports like src.core.* work when run directly
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.shape_detection import ShapeDetector
from src.web.app import preprocess_image, load_model_and_mapping

def create_test_image(size=512, pattern='circle'):
    """Create a simple test image with a specific pattern."""
    img = np.ones((size, size), dtype=np.uint8) * 255  # White background
    
    if pattern == 'circle':
        # Draw a black circle
        center = size // 2
        for y in range(size):
            for x in range(size):
                if (x - center) ** 2 + (y - center) ** 2 <= (size // 4) ** 2:
                    img[y, x] = 0  # Black pixel
                    
    elif pattern == 'rectangle':
        # Draw a black rectangle
        margin = size // 4
        img[margin:-margin, margin:-margin] = 0
        
    elif pattern == 'line':
        # Draw a black line
        center = size // 2
        img[center, :] = 0
        
    # Convert to PIL and then to base64
    pil_img = Image.fromarray(img, 'L')
    buffered = BytesIO()
    pil_img.save(buffered, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buffered.getvalue()).decode('utf-8')}"

def test_model_inference():
    """Test model inference with multiple different images."""
    print("=== CONFIDENCE DEBUG: Starting Model Inference Test ===")
    print("This will help identify why you get the same confidence for any drawing.")
    print()
    
    # Set up logging to see our debug messages
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger('test')
    
    # Test images with different patterns
    test_images = {
        'circle': create_test_image(512, 'circle'),
        'rectangle': create_test_image(512, 'rectangle'), 
        'line': create_test_image(512, 'line'),
        'empty': create_test_image(512, 'rectangle')  # Just background
    }
    
    print("Created 4 test images:")
    for name in test_images.keys():
        print(f"  - {name}")
    print()
    
    # Load the model
    try:
        print("Loading model...")
        load_model_and_mapping()
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        return
    
    print("\nTesting preprocessing pipeline...")
    
    # Test preprocessing
    for name, image_data in test_images.items():
        print(f"\n--- Testing image: {name} ---")
        try:
            preprocessed = preprocess_image(image_data)
            print(f"Preprocessed shape: {preprocessed.shape}")
            print(f"Preprocessed range: [{preprocessed.min():.6f}, {preprocessed.max():.6f}]")
            print(f"Unique values: {len(np.unique(preprocessed))}")
            if len(np.unique(preprocessed)) <= 10:
                print(f"UNIQUE VALUES: {np.unique(preprocessed)}")
        except Exception as e:
            print(f"✗ Preprocessing failed: {e}")
    
    print("\n=== Test complete! ===")
    print("Check the logs above for any issues.")
    print("If all images produce the same preprocessed output, that's the problem.")
    print("If the model produces the same output for different inputs, that's the problem.")
    
    print("\nNext steps:")
    print("1. Run the web interface and draw different things")
    print("2. Check the logs in logs/doodlehunter.log for detailed output")
    print("3. Share the relevant log sections to help diagnose the issue")

if __name__ == '__main__':
    test_model_inference()