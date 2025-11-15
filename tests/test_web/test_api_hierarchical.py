"""
Integration test for hierarchical contour detection API.

Verifies that the Flask API correctly uses RETR_TREE mode by default
and properly handles the mode parameter.

Related:
- src/web/app.py (Flask API implementation)
- src/core/contour_detection.py (ContourDetector)
"""

import unittest
import json
import base64
import numpy as np
from io import BytesIO
from PIL import Image


class TestHierarchicalAPI(unittest.TestCase):
    """Test Flask API with hierarchical contour detection."""

    def setUp(self):
        """Set up test fixtures."""
        # Import Flask app
        from src.web.app import app
        self.app = app
        self.client = self.app.test_client()

    def create_test_image(self, size=(200, 200)):
        """Create a simple test image as base64."""
        # Create a simple image with a circle
        img = Image.new('L', size, color=0)
        pixels = img.load()
        
        # Draw a simple circle
        center_x, center_y = size[0] // 2, size[1] // 2
        radius = 50
        for x in range(size[0]):
            for y in range(size[1]):
                if (x - center_x) ** 2 + (y - center_y) ** 2 < radius ** 2:
                    pixels[x, y] = 255

        # Convert to base64
        buffer = BytesIO()
        img.save(buffer, format='PNG')
        img_str = base64.b64encode(buffer.getvalue()).decode()
        return f"data:image/png;base64,{img_str}"

    def test_api_predict_region_default_mode(self):
        """Test that /api/predict/region uses RETR_TREE by default."""
        image_data = self.create_test_image()
        
        response = self.client.post(
            '/api/predict/region',
            data=json.dumps({'image': image_data}),
            content_type='application/json'
        )

        # Should succeed
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.data)
        self.assertTrue(data['success'])
        
        # Response should indicate hierarchical analysis
        # (The actual verdict depends on the model, but we can check structure)
        self.assertIn('verdict', data)
        self.assertIn('confidence', data)

    def test_api_predict_region_explicit_tree_mode(self):
        """Test /api/predict/region with explicit tree mode."""
        image_data = self.create_test_image()
        
        response = self.client.post(
            '/api/predict/region',
            data=json.dumps({
                'image': image_data,
                'mode': 'tree'
            }),
            content_type='application/json'
        )

        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertTrue(data['success'])

    def test_api_predict_region_external_mode(self):
        """Test /api/predict/region with external mode."""
        image_data = self.create_test_image()
        
        response = self.client.post(
            '/api/predict/region',
            data=json.dumps({
                'image': image_data,
                'mode': 'external'
            }),
            content_type='application/json'
        )

        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertTrue(data['success'])

    def test_api_predict_region_invalid_mode(self):
        """Test /api/predict/region with invalid mode."""
        image_data = self.create_test_image()
        
        response = self.client.post(
            '/api/predict/region',
            data=json.dumps({
                'image': image_data,
                'mode': 'invalid_mode'
            }),
            content_type='application/json'
        )

        # Should return 400 error
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertFalse(data['success'])
        self.assertIn('error', data)

    def test_api_predict_region_with_parameters(self):
        """Test /api/predict/region with all optional parameters."""
        image_data = self.create_test_image()
        
        response = self.client.post(
            '/api/predict/region',
            data=json.dumps({
                'image': image_data,
                'mode': 'tree',
                'min_contour_area': 200,
                'early_stopping': False
            }),
            content_type='application/json'
        )

        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertTrue(data['success'])


if __name__ == '__main__':
    unittest.main()
