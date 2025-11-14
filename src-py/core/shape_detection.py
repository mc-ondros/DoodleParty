"""
Shape-based detection with stroke awareness.
"""

import numpy as np
from typing import List, Tuple
import cv2


class ShapeDetector:
    """Detect geometric shapes in drawings."""
    
    @staticmethod
    def detect_lines(image: np.ndarray, threshold: float = 0.5) -> List[Tuple[int, int, int, int]]:
        """
        Detect lines in image using Hough transform.
        
        Args:
            image: Input image
            threshold: Detection threshold
        
        Returns:
            List of line coordinates (x1, y1, x2, y2)
        """
        edges = cv2.Canny(image, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=30, maxLineGap=10)
        
        if lines is None:
            return []
        
        return [tuple(line[0]) for line in lines]
    
    @staticmethod
    def detect_circles(image: np.ndarray) -> List[Tuple[int, int, int]]:
        """
        Detect circles in image.
        
        Args:
            image: Input image
        
        Returns:
            List of circle coordinates (x, y, radius)
        """
        circles = cv2.HoughCircles(
            image,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=50,
            param1=50,
            param2=30,
            minRadius=5,
            maxRadius=100
        )
        
        if circles is None:
            return []
        
        circles = np.uint16(np.around(circles))
        return [(circle[0], circle[1], circle[2]) for circle in circles[0, :]]
    
    @staticmethod
    def detect_contours(image: np.ndarray) -> List[np.ndarray]:
        """
        Detect contours in image.
        
        Args:
            image: Input image
        
        Returns:
            List of contours
        """
        contours, _ = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        return contours
    
    @staticmethod
    def stroke_width_variation(image: np.ndarray) -> float:
        """
        Calculate stroke width variation.
        
        Args:
            image: Input image
        
        Returns:
            Stroke width variation coefficient
        """
        # Calculate distance transform
        dist_transform = cv2.distanceTransform(image, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        
        # Get non-zero distances
        distances = dist_transform[dist_transform > 0]
        
        if len(distances) == 0:
            return 0.0
        
        # Calculate coefficient of variation
        mean_width = np.mean(distances) * 2
        std_width = np.std(distances) * 2
        
        if mean_width == 0:
            return 0.0
        
        return std_width / mean_width
