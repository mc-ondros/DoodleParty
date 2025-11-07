"""
Shape-based detection for DoodleHunter.

Detects individual shapes/objects on canvas, normalizes them to 128x128,
and runs ML inference on each shape independently. This solves the tile
fragmentation problem by analyzing complete objects.

Why shape detection:
- Analyzes complete objects rather than arbitrary tile fragments
- Normalizes shapes to 128x128 (model's training size)
- Handles non-square canvases properly
- Provides precise localization of offensive content

Related:
- src/core/contour_detection.py (contour extraction)
- src/core/tile_detection.py (tile-based approach)
- src/web/app.py (Flask API endpoints)

Exports:
- ShapeDetector: Main class for shape-based detection
- ShapeInfo: Dataclass for detected shape information
- ShapeDetectionResult: Dataclass for detection results
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Any
from dataclasses import dataclass


@dataclass
class ShapeInfo:
    """Information about a detected shape."""
    contour: np.ndarray  # Original contour points
    bounding_box: Tuple[int, int, int, int]  # (x, y, width, height) on original canvas
    normalized_image: np.ndarray  # Shape normalized to 128x128
    confidence: float  # ML confidence score
    is_positive: bool  # Whether shape is flagged as offensive
    area: int  # Contour area in pixels
    shape_id: int  # Unique identifier


@dataclass
class ShapeDetectionResult:
    """Result of shape-based detection."""
    is_positive: bool  # Overall verdict
    confidence: float  # Highest confidence among shapes
    shape_predictions: List[ShapeInfo]  # Individual shape results
    num_shapes_analyzed: int
    canvas_dimensions: Tuple[int, int]  # (width, height)


class ShapeDetector:
    """
    Shape-based detection system.
    
    Extracts individual shapes from canvas, normalizes each to 128x128,
    and runs ML inference. Handles non-square canvases and provides
    precise localization.
    
    Example:
        detector = ShapeDetector(model=model, threshold=0.5)
        result = detector.detect(canvas_image)
        
        for shape in result.shape_predictions:
            if shape.is_positive:
                print(f"Offensive shape at {shape.bounding_box}")
    """
    
    def __init__(
        self,
        model: Optional[Any] = None,
        tflite_interpreter: Optional[Any] = None,
        is_tflite: bool = False,
        classification_threshold: float = 0.5,
        min_shape_area: int = 100,
        padding_color: int = 243  # Background gray
    ):
        """
        Initialize shape detector.
        
        Args:
            model: Keras model (if using Keras)
            tflite_interpreter: TFLite interpreter (if using TFLite)
            is_tflite: Whether using TFLite
            classification_threshold: Confidence threshold for positive detection
            min_shape_area: Minimum contour area to consider (filters noise)
            padding_color: Color to use for padding when normalizing shapes
        """
        self.model = model
        self.tflite_interpreter = tflite_interpreter
        self.is_tflite = is_tflite
        self.classification_threshold = classification_threshold
        self.min_shape_area = min_shape_area
        self.padding_color = padding_color
        
        # Detect model's expected input size
        self.target_size = self._detect_model_input_size()
    
    def _detect_model_input_size(self) -> int:
        """Detect the model's expected input size."""
        if self.is_tflite and self.tflite_interpreter:
            input_details = self.tflite_interpreter.get_input_details()
            return input_details[0]['shape'][1]  # Assuming square input
        elif self.model:
            try:
                input_shape = self.model.input_shape
                return input_shape[1]  # Assuming square input
            except:
                return 128  # Default
        return 128  # Default
    
    def extract_shapes_from_strokes(
        self,
        stroke_history: List[dict]
    ) -> List[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
        """
        Extract shapes from stroke history instead of contours.
        
        Groups connected strokes into shapes based on proximity and timing.
        
        Args:
            stroke_history: List of stroke objects with 'points' array
        
        Returns:
            List of (points, bounding_box) tuples for each shape
        """
        if not stroke_history:
            return []
        
        shapes = []
        
        # For now, treat all strokes as one shape (single continuous drawing)
        # Future: implement stroke grouping based on proximity/timing
        all_points = []
        
        for stroke in stroke_history:
            if 'points' in stroke and stroke['points']:
                for point in stroke['points']:
                    all_points.append([point['x'], point['y']])
        
        if not all_points:
            return []
        
        # Convert to numpy array
        points_array = np.array(all_points, dtype=np.int32)
        
        # Get bounding box from all points
        x_min = int(np.min(points_array[:, 0]))
        y_min = int(np.min(points_array[:, 1]))
        x_max = int(np.max(points_array[:, 0]))
        y_max = int(np.max(points_array[:, 1]))
        
        w = x_max - x_min
        h = y_max - y_min
        
        shapes.append((points_array, (x_min, y_min, w, h)))
        
        return shapes
    
    def extract_shapes(self, image: np.ndarray) -> List[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
        """
        Extract individual shapes from image using contour detection.
        
        Args:
            image: Grayscale image (H, W)
        
        Returns:
            List of (contour, bounding_box) tuples
        """
        # Threshold to binary
        _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, hierarchy = cv2.findContours(
            binary,
            cv2.RETR_EXTERNAL,  # Only external contours (separate objects)
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        shapes = []
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filter out noise
            if area < self.min_shape_area:
                continue
            
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            shapes.append((contour, (x, y, w, h)))
        
        return shapes
    
    def normalize_shape(
        self,
        image: np.ndarray,
        bounding_box: Tuple[int, int, int, int],
        target_size: Optional[int] = None
    ) -> np.ndarray:
        """
        Extract shape and normalize with padding.
        
        Maintains aspect ratio and adds margin around shape to preserve context.
        
        Args:
            image: Source image (H, W)
            bounding_box: (x, y, width, height)
            target_size: Target dimension (uses model's expected size if None)
        
        Returns:
            Normalized shape array
        """
        if target_size is None:
            target_size = self.target_size
        
        x, y, w, h = bounding_box
        
        # Add margin around shape (10% on each side)
        margin = max(int(w * 0.1), int(h * 0.1), 5)
        
        # Expand bounding box with margin
        x_start = max(0, x - margin)
        y_start = max(0, y - margin)
        x_end = min(image.shape[1], x + w + margin)
        y_end = min(image.shape[0], y + h + margin)
        
        # Extract shape region with margin
        shape_region = image[y_start:y_end, x_start:x_end].copy()
        
        # Handle empty regions
        if shape_region.size == 0 or shape_region.shape[0] == 0 or shape_region.shape[1] == 0:
            return np.full((target_size, target_size), self.padding_color, dtype=np.uint8)
        
        region_h, region_w = shape_region.shape
        
        # Calculate scaling to fit in 128x128 while maintaining aspect ratio
        # Leave some padding (use 90% of target size)
        max_dim = int(target_size * 0.9)
        scale = min(max_dim / region_w, max_dim / region_h)
        
        new_w = int(region_w * scale)
        new_h = int(region_h * scale)
        
        # Resize shape
        if new_w > 0 and new_h > 0:
            resized = cv2.resize(shape_region, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            resized = shape_region
        
        # Create padded canvas
        normalized = np.full((target_size, target_size), self.padding_color, dtype=np.uint8)
        
        # Center the shape
        offset_x = (target_size - new_w) // 2
        offset_y = (target_size - new_h) // 2
        
        normalized[offset_y:offset_y+new_h, offset_x:offset_x+new_w] = resized
        
        return normalized
    
    def preprocess_for_model(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess normalized shape for model input.
        
        Detects model's expected input size and resizes accordingly.
        
        Args:
            image: Normalized shape (128, 128)
        
        Returns:
            Preprocessed array ready for model
        """
        # Detect expected input size from model
        if self.is_tflite and self.tflite_interpreter:
            input_details = self.tflite_interpreter.get_input_details()
            expected_shape = input_details[0]['shape']
            target_h, target_w = expected_shape[1], expected_shape[2]
        elif self.model:
            # Try to get input shape from Keras model
            try:
                input_shape = self.model.input_shape
                target_h, target_w = input_shape[1], input_shape[2]
            except:
                # Default to 128x128
                target_h, target_w = 128, 128
        else:
            target_h, target_w = 128, 128
        
        # Resize if needed
        if image.shape != (target_h, target_w):
            import cv2
            image = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_AREA)
        
        # Normalize to [0, 1]
        img_array = image.astype(np.float32) / 255.0
        
        # Add batch and channel dimensions
        img_array = img_array.reshape(1, target_h, target_w, 1)
        
        return img_array
    
    def predict_shape(self, normalized_shape: np.ndarray) -> float:
        """
        Run ML inference on normalized shape.
        
        Args:
            normalized_shape: Shape normalized to 128x128
        
        Returns:
            Confidence score (0.0-1.0)
        """
        # Preprocess
        input_array = self.preprocess_for_model(normalized_shape)
        
        # Run inference
        if self.is_tflite and self.tflite_interpreter:
            # TFLite inference
            input_details = self.tflite_interpreter.get_input_details()
            output_details = self.tflite_interpreter.get_output_details()
            
            self.tflite_interpreter.set_tensor(input_details[0]['index'], input_array)
            self.tflite_interpreter.invoke()
            
            output = self.tflite_interpreter.get_tensor(output_details[0]['index'])
            confidence = float(output[0][0])
        elif self.model:
            # Keras inference
            prediction = self.model.predict(input_array, verbose=0)
            confidence = float(prediction[0][0])
        else:
            # No model available
            confidence = 0.0
        
        return confidence
    
    def detect(self, image: np.ndarray) -> ShapeDetectionResult:
        """
        Detect and classify all shapes in image.
        
        Args:
            image: Canvas image (H, W) grayscale
        
        Returns:
            ShapeDetectionResult with all shape predictions
        """
        canvas_h, canvas_w = image.shape[:2]
        
        # Extract shapes
        shapes = self.extract_shapes(image)
        
        # Analyze each shape
        shape_predictions = []
        max_confidence = 0.0
        overall_positive = False
        
        for idx, (contour, bbox) in enumerate(shapes):
            # Normalize shape to 128x128
            normalized = self.normalize_shape(image, bbox)
            
            # Run inference
            confidence = self.predict_shape(normalized)
            is_positive = confidence >= self.classification_threshold
            
            # Track maximum confidence
            if confidence > max_confidence:
                max_confidence = confidence
            
            if is_positive:
                overall_positive = True
            
            # Create shape info
            shape_info = ShapeInfo(
                contour=contour,
                bounding_box=bbox,
                normalized_image=normalized,
                confidence=confidence,
                is_positive=is_positive,
                area=int(cv2.contourArea(contour)),
                shape_id=idx
            )
            
            shape_predictions.append(shape_info)
        
        return ShapeDetectionResult(
            is_positive=overall_positive,
            confidence=max_confidence,
            shape_predictions=shape_predictions,
            num_shapes_analyzed=len(shapes),
            canvas_dimensions=(canvas_w, canvas_h)
        )
