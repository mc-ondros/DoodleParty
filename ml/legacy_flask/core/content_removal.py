"""
Content removal system for DoodleHunter.

This module implements various strategies for removing or obscuring offensive
content detected in drawings. Provides precise localization and multiple
removal methods with user feedback mechanisms.

Why content removal:
- Allows users to correct false positives
- Provides visual feedback on what was flagged
- Enables selective content moderation
- Supports undo functionality for better UX

Related:
- src/core/contour_detection.py (contour-based localization)
- src/core/tile_detection.py (tile-based localization)
- src/web/app.py (Flask API endpoints)

Exports:
- RemovalStrategy: Enum of available removal strategies
- ContentRemover: Main class for content removal operations
- LocalizationResult: Dataclass for flagged regions
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum
from PIL import Image, ImageDraw, ImageFilter, ImageFont


class RemovalStrategy(Enum):
    """Available content removal strategies."""
    BLUR = 'blur'  # Gaussian blur overlay
    PLACEHOLDER = 'placeholder'  # "Content Hidden" message
    ERASE = 'erase'  # Clear flagged regions
    HIGHLIGHT = 'highlight'  # Red overlay (for preview before removal)


@dataclass
class FlaggedRegion:
    """Information about a flagged region."""
    bounding_box: Tuple[int, int, int, int]  # (x, y, width, height)
    confidence: float  # Detection confidence
    region_type: str  # 'tile' or 'contour'
    region_id: int  # Unique identifier
    mask: Optional[np.ndarray] = None  # Optional pixel-level mask


@dataclass
class LocalizationResult:
    """Result of content localization."""
    flagged_regions: List[FlaggedRegion]
    overall_confidence: float
    detection_method: str  # 'tile', 'contour', or 'simple'
    canvas_dimensions: Tuple[int, int]  # (width, height)


@dataclass
class RemovalResult:
    """Result of content removal operation."""
    success: bool
    modified_image: Optional[np.ndarray]
    regions_removed: int
    strategy_used: str
    can_undo: bool
    error: Optional[str] = None


class ContentRemover:
    """
    Content removal system with multiple strategies.
    
    Supports precise localization of offensive content and various removal
    methods including blur, placeholder, and selective erase. Provides
    undo functionality and visual feedback.
    
    Example:
        remover = ContentRemover()
        
        # Localize flagged regions from detection results
        localization = remover.localize_from_tiles(tile_result)
        
        # Apply removal strategy
        result = remover.remove_content(
            image,
            localization,
            strategy=RemovalStrategy.BLUR
        )
        
        # Undo if needed
        original = remover.undo()
    """
    
    def __init__(self, blur_kernel_size: int = 25):
        """
        Initialize content remover.
        
        Args:
            blur_kernel_size: Kernel size for Gaussian blur (must be odd)
        """
        self.blur_kernel_size = blur_kernel_size if blur_kernel_size % 2 == 1 else blur_kernel_size + 1
        self.undo_stack: List[np.ndarray] = []
        self.max_undo_depth = 10
    
    def localize_from_tiles(
        self,
        tile_result,
        threshold: float = 0.5
    ) -> LocalizationResult:
        """
        Create localization result from tile detection output.
        
        Args:
            tile_result: TileDetectionResult object
            threshold: Confidence threshold for flagging tiles
        
        Returns:
            LocalizationResult with flagged tile regions
        """
        flagged_regions = []
        
        for idx, tile_info in enumerate(tile_result.tile_predictions):
            if tile_info.is_positive and tile_info.confidence >= threshold:
                region = FlaggedRegion(
                    bounding_box=tile_info.bounding_box,
                    confidence=tile_info.confidence,
                    region_type='tile',
                    region_id=idx
                )
                flagged_regions.append(region)
        
        return LocalizationResult(
            flagged_regions=flagged_regions,
            overall_confidence=tile_result.confidence,
            detection_method='tile',
            canvas_dimensions=tile_result.canvas_dimensions
        )
    
    def localize_from_contours(
        self,
        contour_result,
        threshold: float = 0.5
    ) -> LocalizationResult:
        """
        Create localization result from contour detection output.
        
        Args:
            contour_result: ContourDetectionResult object
            threshold: Confidence threshold for flagging contours
        
        Returns:
            LocalizationResult with flagged contour regions
        """
        flagged_regions = []
        
        for idx, contour_info in enumerate(contour_result.contour_predictions):
            if contour_info.is_positive and contour_info.confidence >= threshold:
                region = FlaggedRegion(
                    bounding_box=contour_info.bounding_box,
                    confidence=contour_info.confidence,
                    region_type='contour',
                    region_id=idx
                )
                flagged_regions.append(region)
        
        # Get canvas dimensions from first contour's image
        canvas_dimensions = (512, 512)  # Default, should be passed in
        
        return LocalizationResult(
            flagged_regions=flagged_regions,
            overall_confidence=contour_result.confidence,
            detection_method='contour',
            canvas_dimensions=canvas_dimensions
        )
    
    def highlight_regions(
        self,
        image: np.ndarray,
        localization: LocalizationResult,
        color: Tuple[int, int, int] = (255, 0, 0),
        alpha: float = 0.3
    ) -> np.ndarray:
        """
        Highlight flagged regions with colored overlay.
        
        Args:
            image: Source image (H, W) or (H, W, C)
            localization: Localization result with flagged regions
            color: RGB color for overlay
            alpha: Transparency (0.0 = transparent, 1.0 = opaque)
        
        Returns:
            Image with highlighted regions
        """
        # Convert to RGB if grayscale
        if len(image.shape) == 2:
            img_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 1:
            img_rgb = cv2.cvtColor(image[:, :, 0], cv2.COLOR_GRAY2RGB)
        else:
            img_rgb = image.copy()
        
        # Create overlay
        overlay = img_rgb.copy()
        
        for region in localization.flagged_regions:
            x, y, w, h = region.bounding_box
            
            # Draw filled rectangle
            cv2.rectangle(overlay, (x, y), (x + w, y + h), color, -1)
            
            # Draw border
            cv2.rectangle(overlay, (x, y), (x + w, y + h), color, 2)
            
            # Add confidence text
            text = f"{region.confidence:.2f}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1
            
            # Get text size for background
            (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
            
            # Draw text background
            cv2.rectangle(
                overlay,
                (x + 5, y + 5),
                (x + 5 + text_w + 4, y + 5 + text_h + 4),
                (255, 255, 255),
                -1
            )
            
            # Draw text
            cv2.putText(
                overlay,
                text,
                (x + 7, y + 5 + text_h),
                font,
                font_scale,
                color,
                thickness
            )
        
        # Blend overlay with original
        result = cv2.addWeighted(img_rgb, 1 - alpha, overlay, alpha, 0)
        
        return result
    
    def remove_content(
        self,
        image: np.ndarray,
        localization: LocalizationResult,
        strategy: RemovalStrategy = RemovalStrategy.BLUR,
        save_for_undo: bool = True
    ) -> RemovalResult:
        """
        Remove or obscure flagged content using specified strategy.
        
        Args:
            image: Source image (H, W) or (H, W, C)
            localization: Localization result with flagged regions
            strategy: Removal strategy to use
            save_for_undo: Whether to save original for undo
        
        Returns:
            RemovalResult with modified image
        """
        if not localization.flagged_regions:
            return RemovalResult(
                success=True,
                modified_image=image.copy(),
                regions_removed=0,
                strategy_used=strategy.value,
                can_undo=False,
                error="No flagged regions to remove"
            )
        
        # Save for undo
        if save_for_undo:
            self._push_undo(image.copy())
        
        # Apply strategy
        try:
            if strategy == RemovalStrategy.BLUR:
                modified = self._apply_blur(image, localization)
            elif strategy == RemovalStrategy.PLACEHOLDER:
                modified = self._apply_placeholder(image, localization)
            elif strategy == RemovalStrategy.ERASE:
                modified = self._apply_erase(image, localization)
            elif strategy == RemovalStrategy.HIGHLIGHT:
                modified = self.highlight_regions(image, localization)
            else:
                return RemovalResult(
                    success=False,
                    modified_image=None,
                    regions_removed=0,
                    strategy_used=strategy.value,
                    can_undo=False,
                    error=f"Unknown strategy: {strategy.value}"
                )
            
            return RemovalResult(
                success=True,
                modified_image=modified,
                regions_removed=len(localization.flagged_regions),
                strategy_used=strategy.value,
                can_undo=len(self.undo_stack) > 0
            )
        
        except Exception as e:
            return RemovalResult(
                success=False,
                modified_image=None,
                regions_removed=0,
                strategy_used=strategy.value,
                can_undo=False,
                error=str(e)
            )
    
    def _apply_blur(
        self,
        image: np.ndarray,
        localization: LocalizationResult
    ) -> np.ndarray:
        """Apply Gaussian blur to flagged regions."""
        result = image.copy()
        
        for region in localization.flagged_regions:
            x, y, w, h = region.bounding_box
            
            # Extract region
            if len(image.shape) == 2:
                roi = result[y:y+h, x:x+w]
            else:
                roi = result[y:y+h, x:x+w, :]
            
            # Apply blur
            blurred = cv2.GaussianBlur(roi, (self.blur_kernel_size, self.blur_kernel_size), 0)
            
            # Replace region
            if len(image.shape) == 2:
                result[y:y+h, x:x+w] = blurred
            else:
                result[y:y+h, x:x+w, :] = blurred
        
        return result
    
    def _apply_placeholder(
        self,
        image: np.ndarray,
        localization: LocalizationResult
    ) -> np.ndarray:
        """Apply placeholder overlay to flagged regions."""
        # Convert to RGB for drawing
        if len(image.shape) == 2:
            result = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 1:
            result = cv2.cvtColor(image[:, :, 0], cv2.COLOR_GRAY2RGB)
        else:
            result = image.copy()
        
        for region in localization.flagged_regions:
            x, y, w, h = region.bounding_box
            
            # Draw semi-transparent gray rectangle
            overlay = result.copy()
            cv2.rectangle(overlay, (x, y), (x + w, y + h), (128, 128, 128), -1)
            result = cv2.addWeighted(result, 0.3, overlay, 0.7, 0)
            
            # Add border
            cv2.rectangle(result, (x, y), (x + w, y + h), (200, 200, 200), 2)
            
            # Add text
            text = "Content Hidden"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.4
            thickness = 1
            
            # Get text size
            (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
            
            # Center text in region
            text_x = x + (w - text_w) // 2
            text_y = y + (h + text_h) // 2
            
            # Draw text
            cv2.putText(
                result,
                text,
                (text_x, text_y),
                font,
                font_scale,
                (255, 255, 255),
                thickness
            )
        
        return result
    
    def _apply_erase(
        self,
        image: np.ndarray,
        localization: LocalizationResult
    ) -> np.ndarray:
        """Erase flagged regions (fill with background color)."""
        result = image.copy()
        
        # Determine background color from image
        # Use most common color in corners
        if len(image.shape) == 2:
            bg_color = 243  # Default gray
        else:
            bg_color = (243, 244, 246)  # Default RGB gray
        
        for region in localization.flagged_regions:
            x, y, w, h = region.bounding_box
            
            # Fill with background color
            if len(image.shape) == 2:
                result[y:y+h, x:x+w] = bg_color
            else:
                result[y:y+h, x:x+w, :] = bg_color
        
        return result
    
    def _push_undo(self, image: np.ndarray):
        """Push image to undo stack."""
        self.undo_stack.append(image.copy())
        
        # Limit stack depth
        if len(self.undo_stack) > self.max_undo_depth:
            self.undo_stack.pop(0)
    
    def undo(self) -> Optional[np.ndarray]:
        """
        Undo last removal operation.
        
        Returns:
            Previous image state, or None if no undo available
        """
        if self.undo_stack:
            return self.undo_stack.pop()
        return None
    
    def can_undo(self) -> bool:
        """Check if undo is available."""
        return len(self.undo_stack) > 0
    
    def clear_undo_stack(self):
        """Clear undo history."""
        self.undo_stack.clear()
