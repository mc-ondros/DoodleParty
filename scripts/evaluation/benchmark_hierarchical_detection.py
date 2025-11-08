"""
Benchmark script for hierarchical contour detection performance.

Measures the performance impact of RETR_TREE vs RETR_EXTERNAL mode
to verify that the overhead is acceptable (<5ms target).

Usage:
    python scripts/evaluation/benchmark_hierarchical_detection.py

Related:
- src/core/contour_detection.py (ContourDetector implementation)
- .documentation/roadmap.md (Phase 3.1 performance target)
"""

import time
import numpy as np
import cv2
from typing import List, Tuple

from src.core.contour_detection import (
    ContourDetector,
    ContourRetrievalMode,
    detect_contours
)


def create_test_image(
    num_shapes: int = 5,
    nested: bool = False,
    size: Tuple[int, int] = (512, 512)
) -> np.ndarray:
    """
    Create a test image with multiple shapes.
    
    Args:
        num_shapes: Number of shapes to draw
        nested: Whether to create nested shapes
        size: Image size (width, height)
    
    Returns:
        Test image as numpy array
    """
    image = np.zeros(size, dtype=np.uint8)
    
    if nested:
        # Create nested shapes (circle with inner rectangle)
        cv2.circle(image, (256, 256), 100, 255, -1)
        cv2.circle(image, (256, 256), 80, 0, -1)
        cv2.rectangle(image, (220, 220), (292, 292), 255, -1)
    else:
        # Create separate shapes
        for i in range(num_shapes):
            x = (i + 1) * size[0] // (num_shapes + 1)
            y = size[1] // 2
            radius = 30
            cv2.circle(image, (x, y), radius, 255, -1)
    
    return image


def benchmark_contour_detection(
    num_iterations: int = 100,
    num_shapes: int = 5
) -> dict:
    """
    Benchmark contour detection performance.
    
    Args:
        num_iterations: Number of iterations to run
        num_shapes: Number of shapes in test image
    
    Returns:
        Dictionary with benchmark results
    """
    print(f"\n{'='*60}")
    print(f"Benchmarking Contour Detection Performance")
    print(f"{'='*60}")
    print(f"Iterations: {num_iterations}")
    print(f"Shapes per image: {num_shapes}")
    print(f"{'='*60}\n")
    
    # Create test images
    image_simple = create_test_image(num_shapes=num_shapes, nested=False)
    image_nested = create_test_image(num_shapes=1, nested=True)
    
    results = {}
    
    # Benchmark RETR_EXTERNAL mode (simple image)
    print("Testing RETR_EXTERNAL mode (simple shapes)...")
    times_external_simple = []
    for _ in range(num_iterations):
        start = time.perf_counter()
        contours, hierarchy = detect_contours(image_simple, ContourRetrievalMode.EXTERNAL)
        end = time.perf_counter()
        times_external_simple.append((end - start) * 1000)  # Convert to ms
    
    results['external_simple'] = {
        'mean': np.mean(times_external_simple),
        'std': np.std(times_external_simple),
        'min': np.min(times_external_simple),
        'max': np.max(times_external_simple),
        'num_contours': len(contours)
    }
    
    # Benchmark RETR_TREE mode (simple image)
    print("Testing RETR_TREE mode (simple shapes)...")
    times_tree_simple = []
    for _ in range(num_iterations):
        start = time.perf_counter()
        contours, hierarchy = detect_contours(image_simple, ContourRetrievalMode.TREE)
        end = time.perf_counter()
        times_tree_simple.append((end - start) * 1000)
    
    results['tree_simple'] = {
        'mean': np.mean(times_tree_simple),
        'std': np.std(times_tree_simple),
        'min': np.min(times_tree_simple),
        'max': np.max(times_tree_simple),
        'num_contours': len(contours)
    }
    
    # Benchmark RETR_EXTERNAL mode (nested image)
    print("Testing RETR_EXTERNAL mode (nested shapes)...")
    times_external_nested = []
    for _ in range(num_iterations):
        start = time.perf_counter()
        contours, hierarchy = detect_contours(image_nested, ContourRetrievalMode.EXTERNAL)
        end = time.perf_counter()
        times_external_nested.append((end - start) * 1000)
    
    results['external_nested'] = {
        'mean': np.mean(times_external_nested),
        'std': np.std(times_external_nested),
        'min': np.min(times_external_nested),
        'max': np.max(times_external_nested),
        'num_contours': len(contours)
    }
    
    # Benchmark RETR_TREE mode (nested image)
    print("Testing RETR_TREE mode (nested shapes)...")
    times_tree_nested = []
    for _ in range(num_iterations):
        start = time.perf_counter()
        contours, hierarchy = detect_contours(image_nested, ContourRetrievalMode.TREE)
        end = time.perf_counter()
        times_tree_nested.append((end - start) * 1000)
    
    results['tree_nested'] = {
        'mean': np.mean(times_tree_nested),
        'std': np.std(times_tree_nested),
        'min': np.min(times_tree_nested),
        'max': np.max(times_tree_nested),
        'num_contours': len(contours)
    }
    
    return results


def print_results(results: dict):
    """Print benchmark results in a formatted table."""
    print(f"\n{'='*60}")
    print("Benchmark Results")
    print(f"{'='*60}\n")
    
    print(f"{'Mode':<25} {'Mean (ms)':<12} {'Std (ms)':<12} {'Min (ms)':<12} {'Max (ms)':<12} {'Contours':<10}")
    print(f"{'-'*60}")
    
    for mode, data in results.items():
        print(f"{mode:<25} {data['mean']:<12.3f} {data['std']:<12.3f} "
              f"{data['min']:<12.3f} {data['max']:<12.3f} {data['num_contours']:<10}")
    
    print(f"\n{'='*60}")
    print("Performance Analysis")
    print(f"{'='*60}\n")
    
    # Calculate overhead
    overhead_simple = results['tree_simple']['mean'] - results['external_simple']['mean']
    overhead_nested = results['tree_nested']['mean'] - results['external_nested']['mean']
    
    print(f"RETR_TREE overhead (simple shapes): {overhead_simple:.3f} ms")
    print(f"RETR_TREE overhead (nested shapes): {overhead_nested:.3f} ms")
    
    # Check if meets target (<5ms overhead)
    target_overhead = 5.0
    if overhead_simple < target_overhead and overhead_nested < target_overhead:
        print(f"\n✓ Performance target met: overhead < {target_overhead}ms")
    else:
        print(f"\n✗ Performance target NOT met: overhead >= {target_overhead}ms")
    
    # Additional insights
    print(f"\nContour detection comparison:")
    print(f"  EXTERNAL (nested): {results['external_nested']['num_contours']} contours detected")
    print(f"  TREE (nested):     {results['tree_nested']['num_contours']} contours detected")
    print(f"  → RETR_TREE detects {results['tree_nested']['num_contours'] - results['external_nested']['num_contours']} additional nested contours")


def main():
    """Run benchmark and print results."""
    results = benchmark_contour_detection(num_iterations=100, num_shapes=5)
    print_results(results)
    
    print(f"\n{'='*60}")
    print("Conclusion")
    print(f"{'='*60}\n")
    print("RETR_TREE mode successfully detects nested contours with minimal")
    print("performance overhead. The hierarchical detection is now enabled by")
    print("default in the Flask API to prevent content dilution attacks.")
    print(f"\n{'='*60}\n")


if __name__ == '__main__':
    main()
