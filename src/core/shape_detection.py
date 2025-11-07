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
    is_positive: bool  # Overall verdict (after grouping)
    confidence: float  # Highest confidence among groups
    shape_predictions: List[ShapeInfo]  # Individual shape results (raw)
    num_shapes_analyzed: int
    canvas_dimensions: Tuple[int, int]  # (width, height)

    # Grouped-level view (e.g. merge nearby positive shapes into one penis-like object)
    grouped_boxes: List[Tuple[int, int, int, int]] = None  # (x, y, w, h) merged boxes
    grouped_scores: List[float] = None  # aggregated confidence per group


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
        stroke_history: List[dict],
        spatial_eps: float = 32.0,
        time_eps_ms: int = 900
    ) -> List[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
        """
        Extract multiple shapes from stroke history using spatial + temporal structure.

        Heuristic clustering:
        - Group strokes whose endpoints are spatially close (<= spatial_eps)
          OR whose time gap is small (<= time_eps_ms).
        - Each cluster becomes a candidate shape with its own bounding box.

        This addresses:
        - "only one big shape" by splitting logically separate doodles.
        - low/unstable scores by avoiding merged heterogeneous content.
        - missed details by isolating smaller but coherent stroke groups.

        Args:
            stroke_history: List of stroke dicts:
                {
                    "points": [{"x": int, "y": int, "t": int}, ...],
                    "timestamp": int (optional)
                }
            spatial_eps: Max pixel distance between stroke endpoints to be
                considered connected.
            time_eps_ms: Max time gap (ms) between strokes to auto-connect.

        Returns:
            List of (points, bounding_box) for each clustered shape.
        """
        if not stroke_history:
            return []

        # Pre-normalize strokes to a safer structure
        norm_strokes = []
        for s in stroke_history:
            pts = s.get("points") or []
            if not pts:
                continue
            xs = [int(p.get("x", 0)) for p in pts]
            ys = [int(p.get("y", 0)) for p in pts]
            ts = [int(p.get("t", 0)) for p in pts if "t" in p]

            start = (xs[0], ys[0])
            end = (xs[-1], ys[-1])
            # Prefer explicit stroke timestamp; fallback to last point time
            stroke_ts = int(s.get("timestamp", ts[-1] if ts else 0))

            norm_strokes.append(
                {
                    "points": np.column_stack([xs, ys]).astype(np.int32),
                    "start": start,
                    "end": end,
                    "t": stroke_ts,
                }
            )

        if not norm_strokes:
            return []

        # Simple union-find style clustering based on spatial + temporal proximity
        n = len(norm_strokes)
        parent = list(range(n))

        def find(i: int) -> int:
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        def union(i: int, j: int) -> None:
            ri, rj = find(i), find(j)
            if ri != rj:
                parent[rj] = ri

        def dist(a: Tuple[int, int], b: Tuple[int, int]) -> float:
            dx = a[0] - b[0]
            dy = a[1] - b[1]
            return (dx * dx + dy * dy) ** 0.5

        for i in range(n):
            for j in range(i + 1, n):
                si, sj = norm_strokes[i], norm_strokes[j]
                # Temporal proximity (if timestamps exist)
                dt = abs(si["t"] - sj["t"])
                # Spatial proximity: connect if any endpoint is close
                spatial_close = (
                    dist(si["end"], sj["start"]) <= spatial_eps
                    or dist(si["start"], sj["end"]) <= spatial_eps
                    or dist(si["start"], sj["start"]) <= spatial_eps
                    or dist(si["end"], sj["end"]) <= spatial_eps
                )
                if spatial_close or (dt and dt <= time_eps_ms):
                    union(i, j)

        clusters = {}
        for idx, s in enumerate(norm_strokes):
            root = find(idx)
            clusters.setdefault(root, []).append(s)

        shapes: List[Tuple[np.ndarray, Tuple[int, int, int, int]]] = []
        for cluster_strokes in clusters.values():
            # Concatenate all points in cluster
            pts = np.concatenate([cs["points"] for cs in cluster_strokes], axis=0)
            if pts.size == 0:
                continue

            x_min = int(np.min(pts[:, 0]))
            y_min = int(np.min(pts[:, 1]))
            x_max = int(np.max(pts[:, 0]))
            y_max = int(np.max(pts[:, 1]))

            w = max(1, x_max - x_min)
            h = max(1, y_max - y_min)

            # Area-based noise filter (reuse min_shape_area)
            if w * h < self.min_shape_area:
                # Very tiny doodles/noise: ignore to reduce false positives
                continue

            shapes.append((pts, (x_min, y_min, w, h)))

        return shapes
    
    def extract_shapes(
        self,
        image: np.ndarray,
        use_adaptive_threshold: bool = True
    ) -> List[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
        """
        Extract individual shapes from image using robust contour detection.

        Improvements:
        - Adaptive/Otsu thresholding for stability across stroke intensities.
        - Optional morphological closing to connect fragmented strokes.
        - Filters tiny/degenerate regions via area and aspect ratio.
        - Keeps multiple shapes instead of merging everything into one.

        Args:
            image: Canvas image (H, W) grayscale or RGB.
            use_adaptive_threshold: If True, use adaptive/Otsu-based thresholding.

        Returns:
            List of (contour, bounding_box) tuples.
        """
        # Ensure grayscale
        if image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Binarize
        if use_adaptive_threshold:
            # Slight blur for robustness
            img_blur = cv2.GaussianBlur(image, (3, 3), 0)
            # Otsu threshold (data-driven)
            _, binary = cv2.threshold(
                img_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
        else:
            _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

        # Ensure background is white, strokes are dark; invert if mostly dark
        if np.mean(binary) < 127:
            binary = cv2.bitwise_not(binary)

        # Light morphological close to connect broken strokes, avoid tiny gaps
        kernel = np.ones((3, 3), np.uint8)
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)

        # Find external contours as candidate shapes
        contours, _ = cv2.findContours(
            closed,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        shapes: List[Tuple[np.ndarray, Tuple[int, int, int, int]]] = []
        h_img, w_img = image.shape[:2]
        canvas_area = float(w_img * h_img)

        for contour in contours:
            if contour is None or contour.size == 0:
                continue

            area = cv2.contourArea(contour)
            if area <= 0:
                continue

            # Aggressively drop tiny blobs to reduce scribble false positives
            if area < self.min_shape_area:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            if w <= 1 or h <= 1:
                continue

            # Reject extremely thin, tiny shapes likely due to edges/artifacts
            aspect_ratio = max(w / max(h, 1), h / max(w, 1))
            rel_area = area / max(canvas_area, 1.0)

            if aspect_ratio > 20.0 and rel_area < 0.001:
                # Very long, very thin, extremely small relative to canvas
                continue

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
        Run ML inference on normalized shape and return a robust offensive score.

        Assumptions:
        - For binary models, output in [0,1] where higher = more offensive.
        - For multi-class outputs, we conservatively take the max probability/logit
          as the offensive-ness proxy unless upstream provides a dedicated index.

        This reduces calibration weirdness that previously caused:
        - low/unstable scores,
        - sensitivity to exact output tensor shape.
        """
        input_array = self.preprocess_for_model(normalized_shape)

        def _to_scalar(output: np.ndarray) -> float:
            if output is None:
                return 0.0
            arr = np.asarray(output, dtype=np.float32)
            if arr.size == 0:
                return 0.0

            # Common cases:
            # - (1, 1): direct binary prob
            # - (1, C): class probs/logits → take max
            # - any other: flatten and take max of first batch
            if arr.ndim >= 2:
                # Flatten per-batch; use first batch
                flat = arr.reshape(arr.shape[0], -1)
                val = float(flat[0].max())
            else:
                val = float(arr.max())

            # If model already outputs probabilities, this is in [0,1].
            # If logits are used, upstream should apply sigmoid; we still clamp.
            return float(max(0.0, min(1.0, val)))

        if self.is_tflite and self.tflite_interpreter:
            input_details = self.tflite_interpreter.get_input_details()
            output_details = self.tflite_interpreter.get_output_details()

            self.tflite_interpreter.set_tensor(input_details[0]["index"], input_array)
            self.tflite_interpreter.invoke()

            output = self.tflite_interpreter.get_tensor(output_details[0]["index"])
            confidence = _to_scalar(output)
        elif self.model:
            prediction = self.model.predict(input_array, verbose=0)
            confidence = _to_scalar(prediction)
        else:
            confidence = 0.0

        return confidence
    
    def _merge_positive_shapes(
        self,
        shapes: List[ShapeInfo],
        distance_thresh: float = 80.0,
        iou_thresh: float = 0.05,
    ) -> Tuple[List[Tuple[int, int, int, int]], List[float], bool, float]:
        """
        Merge nearby positive shapes into groups (e.g., shaft + 2 balls → one penis object).

        Strategy:
        - Consider shapes flagged positive OR near-positive by the caller.
        - Build undirected graph: edges between boxes that are close or overlapping:
          - if center distance <= distance_thresh OR IoU >= iou_thresh → same group.
        - Each connected component is a candidate merged object.
        - Group score = max member confidence in group.

        NOTE:
        - Caller controls which shapes are eligible by setting is_positive on them.
        - This function is intentionally permissive spatially to allow merging multi-part drawings.
        """
        positive_shapes = [s for s in shapes if s.is_positive]
        if not positive_shapes:
            return [], [], False, 0.0

        n = len(positive_shapes)
        parent = list(range(n))

        def find(i: int) -> int:
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        def union(i: int, j: int) -> None:
            ri, rj = find(i), find(j)
            if ri != rj:
                parent[rj] = ri

        def bbox_distance(b1, b2) -> float:
            x1, y1, w1, h1 = b1
            x2, y2, w2, h2 = b2
            cx1, cy1 = x1 + w1 / 2.0, y1 + h1 / 2.0
            cx2, cy2 = x2 + w2 / 2.0, y2 + h2 / 2.0
            dx = cx1 - cx2
            dy = cy1 - cy2
            return float((dx * dx + dy * dy) ** 0.5)

        def bbox_iou(b1, b2) -> float:
            x1, y1, w1, h1 = b1
            x2, y2, w2, h2 = b2
            xa = max(x1, x2)
            ya = max(y1, y2)
            xb = min(x1 + w1, x2 + w2)
            yb = min(y1 + h1, y2 + h2)
            inter_w = max(0.0, xb - xa)
            inter_h = max(0.0, yb - ya)
            inter = inter_w * inter_h
            if inter <= 0:
                return 0.0
            area1 = w1 * h1
            area2 = w2 * h2
            union_area = area1 + area2 - inter
            if union_area <= 0:
                return 0.0
            return float(inter / union_area)

        # Link shapes that are spatially related
        for i in range(n):
            bi = positive_shapes[i].bounding_box
            for j in range(i + 1, n):
                bj = positive_shapes[j].bounding_box
                if bbox_distance(bi, bj) <= distance_thresh or bbox_iou(bi, bj) >= iou_thresh:
                    union(i, j)

        # Build groups
        groups = {}
        for idx in range(n):
            root = find(idx)
            groups.setdefault(root, []).append(positive_shapes[idx])

        grouped_boxes: List[Tuple[int, int, int, int]] = []
        grouped_scores: List[float] = []
        max_group_conf = 0.0

        for members in groups.values():
            xs = []
            ys = []
            x2s = []
            y2s = []
            group_conf = 0.0
            for s in members:
                x, y, w, h = s.bounding_box
                xs.append(x)
                ys.append(y)
                x2s.append(x + w)
                y2s.append(y + h)
                if s.confidence > group_conf:
                    group_conf = s.confidence

            gx = int(min(xs))
            gy = int(min(ys))
            gw = int(max(x2s) - gx)
            gh = int(max(y2s) - gy)

            grouped_boxes.append((gx, gy, gw, gh))
            grouped_scores.append(group_conf)
            if group_conf > max_group_conf:
                max_group_conf = group_conf

        any_group_positive = len(grouped_boxes) > 0
        return grouped_boxes, grouped_scores, any_group_positive, max_group_conf

    def detect(
        self,
        image: np.ndarray,
        stroke_history: Optional[List[dict]] = None
    ) -> ShapeDetectionResult:
        """
        Detect and classify all shapes in image using all available signals.

        Uses:
        - Stroke history (order, timing, coordinates) for better grouping.
        - Contour-based shapes as fallback.
        - Grouping + heuristic to strongly upweight coherent multi-part patterns
          (e.g., classic penis: shaft + two balls).

        Pipeline:
        - Propose shapes from strokes (if provided) or contours.
        - Classify each shape independently.
        - Merge nearby positive shapes into groups (e.g. multi-part penis).
        - Overall verdict is based on merged groups (any group positive => PENIS).
        """
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        canvas_h, canvas_w = gray.shape[:2]
        canvas_area = float(canvas_w * canvas_h)

        # 1) Propose shapes
        # Prefer stroke-based clustering (captures intent), fallback to contours
        if stroke_history:
            stroke_shapes = self.extract_shapes_from_strokes(stroke_history)
        else:
            stroke_shapes = []

        if stroke_shapes:
            shapes_raw = stroke_shapes
        else:
            shapes_raw = self.extract_shapes(gray)

        shape_predictions: List[ShapeInfo] = []

        for idx, (contour_or_points, bbox) in enumerate(shapes_raw):
            x, y, w, h = bbox

            # Guard: allow large shapes but keep note; no special-case filtering here.
            # Very large boxes will still be normalized & classified.
            # Normalize shape
            normalized = self.normalize_shape(gray, bbox)

            # Model inference
            confidence = self.predict_shape(normalized)
            is_positive = confidence >= self.classification_threshold

            # Derive area: for clustered strokes, contourArea may fail, fallback to bbox
            area = int(w * h)
            try:
                if isinstance(contour_or_points, np.ndarray) and contour_or_points.ndim >= 2:
                    ca = int(cv2.contourArea(contour_or_points))
                    if ca > 0:
                        area = ca
            except Exception:
                pass

            shape_predictions.append(
                ShapeInfo(
                    contour=contour_or_points,
                    bounding_box=bbox,
                    normalized_image=normalized,
                    confidence=confidence,
                    is_positive=is_positive,
                    area=area,
                    shape_id=idx,
                )
            )

        # 2) Merge high-confidence shapes into strict groups
        grouped_boxes, grouped_scores, any_group_positive, max_group_conf = self._merge_positive_shapes(
            shape_predictions
        )

        # 3) Compute overall verdict.
        overall_positive = any_group_positive
        overall_confidence = max_group_conf

        # 4) Strong clustered near-positive heuristic (penis-aware):
        # If no strict positive group, but there is a tight cluster of 3+ shapes
        # with high-but-subthreshold scores, promote them as one high-confidence object.
        if not overall_positive and shape_predictions:
            max_shape_conf = max(s.confidence for s in shape_predictions)
            overall_confidence = max_shape_conf

            # Treat ~0.45+ as strong signal when clustered (for default 0.5 threshold).
            near_pos_thresh = max(0.45, self.classification_threshold * 0.9)
            near_shapes = [s for s in shape_predictions if s.confidence >= near_pos_thresh]

            if len(near_shapes) >= 3:
                # Build temp shapes using near_pos_thresh as positive flag
                tmp_shapes: List[ShapeInfo] = []
                for s in shape_predictions:
                    tmp_shapes.append(
                        ShapeInfo(
                            contour=s.contour,
                            bounding_box=s.bounding_box,
                            normalized_image=s.normalized_image,
                            confidence=s.confidence,
                            is_positive=(s.confidence >= near_pos_thresh),
                            area=s.area,
                            shape_id=s.shape_id,
                        )
                    )

                np_boxes, np_scores, np_any, np_max = self._merge_positive_shapes(tmp_shapes)

                # Require they actually merge (single cluster) to avoid random noise
                if np_any and len(np_boxes) == 1:
                    # Strong promotion: combined evidence from multiple near-positives
                    overall_positive = True

                    # Use a high confidence to reflect certainty about the pattern:
                    # - base: max per-shape or group score
                    # - boost: +0.4, capped at 0.99 → 0.49 → ~0.89
                    base = max(max_shape_conf, np_max)
                    combined_conf = min(0.99, base + 0.4)
                    overall_confidence = max(overall_confidence, combined_conf)

                    grouped_boxes = np_boxes
                    grouped_scores = [overall_confidence]

        return ShapeDetectionResult(
            is_positive=overall_positive,
            confidence=overall_confidence,
            shape_predictions=shape_predictions,
            num_shapes_analyzed=len(shape_predictions),
            canvas_dimensions=(canvas_w, canvas_h),
            grouped_boxes=grouped_boxes,
            grouped_scores=grouped_scores,
        )
