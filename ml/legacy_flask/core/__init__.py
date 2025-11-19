"""Core ML functionality for DoodleHunter."""

from .models import build_custom_cnn, get_model
from .patch_extraction import SlidingWindowDetector, AggregationStrategy

__all__ = [
    'build_custom_cnn',
    'get_model',
    'SlidingWindowDetector',
    'AggregationStrategy',
]
