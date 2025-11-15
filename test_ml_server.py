#!/usr/bin/env python3
"""
Test script for ML server
Creates a simple test image and sends it to the ML server for classification
"""

import requests
from PIL import Image, ImageDraw
import io
import sys

ML_SERVER_URL = 'http://localhost:5000'

def create_test_image(size=64):
    """Create a simple test image with a circle"""
    img = Image.new('RGB', (size, size), color='white')
    draw = ImageDraw.Draw(img)
    
    # Draw a circle
    margin = size // 4
    draw.ellipse([margin, margin, size-margin, size-margin], fill='black', outline='black')
    
    return img

def test_health_check():
    """Test the /health endpoint"""
    print("Testing health check...")
    try:
        response = requests.get(f'{ML_SERVER_URL}/health')
        response.raise_for_status()
        data = response.json()
        print(f"✓ Health check passed")
        print(f"  Status: {data['status']}")
        print(f"  Model loaded: {data['model_loaded']}")
        print(f"  Model path: {data['model_path']}")
        return True
    except Exception as e:
        print(f"✗ Health check failed: {e}")
        return False

def test_classification():
    """Test the /classify endpoint"""
    print("\nTesting classification...")
    try:
        # Create test image
        img = create_test_image()
        
        # Convert to bytes
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        
        # Send to server
        files = {'image': ('test.png', img_bytes, 'image/png')}
        data = {
            'sessionId': 'test-session',
            'bbox': '{"minX": 0, "maxX": 64, "minY": 0, "maxY": 64}'
        }
        
        response = requests.post(f'{ML_SERVER_URL}/classify', files=files, data=data)
        response.raise_for_status()
        result = response.json()
        
        print(f"✓ Classification successful")
        print(f"  Is inappropriate: {result['is_inappropriate']}")
        print(f"  Confidence: {result['confidence']:.3f}")
        print(f"  Class: {result['class_name']}")
        if 'mock' in result:
            print(f"  Mock mode: {result['mock']}")
        
        return True
        
    except Exception as e:
        print(f"✗ Classification failed: {e}")
        return False

def test_config():
    """Test the /config endpoint"""
    print("\nTesting config...")
    try:
        response = requests.get(f'{ML_SERVER_URL}/config')
        response.raise_for_status()
        data = response.json()
        print(f"✓ Config retrieved")
        print(f"  Threshold: {data['threshold']}")
        print(f"  Model path: {data['model_path']}")
        print(f"  Model loaded: {data['model_loaded']}")
        return True
    except Exception as e:
        print(f"✗ Config failed: {e}")
        return False

def main():
    print("=" * 50)
    print("ML Server Test Suite")
    print("=" * 50)
    print()
    
    # Run tests
    tests = [
        test_health_check,
        test_classification,
        test_config
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    # Summary
    print()
    print("=" * 50)
    passed = sum(results)
    total = len(results)
    print(f"Results: {passed}/{total} tests passed")
    print("=" * 50)
    
    return 0 if all(results) else 1

if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nInterrupted")
        sys.exit(1)
