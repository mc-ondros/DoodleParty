"""
Benchmark script for tile-based detection performance.

Measures performance of tile-based detection with different configurations
to verify RPi4 performance targets are met.

Performance Targets (RPi4):
- Single tile inference: <10ms
- Full grid (64 tiles): <200ms total
- Incremental update (1-4 dirty tiles): <50ms

Usage:
    python -m scripts.evaluation.benchmark_tile_detection

Related:
- src/core/tile_detection.py (TileDetector implementation)
- .documentation/roadmap.md (Phase 3.2 performance targets)
"""

import time
import numpy as np
from typing import Dict, List

from src.core.tile_detection import TileDetector, TileGrid


class MockModel:
    """Mock model for benchmarking (simulates TFLite inference time)."""
    
    def predict(self, image_batch, verbose=0):
        """Simulate inference with realistic timing."""
        # Simulate ~5ms inference time per tile
        time.sleep(0.005)
        return np.array([[0.3]])


def benchmark_tile_sizes(num_iterations: int = 10) -> Dict:
    """
    Benchmark different tile sizes.
    
    Args:
        num_iterations: Number of iterations per configuration
    
    Returns:
        Dictionary with benchmark results
    """
    print(f"\n{'='*70}")
    print("Benchmarking Tile Sizes")
    print(f"{'='*70}\n")
    
    canvas_size = 512
    tile_sizes = [32, 64, 128]
    mock_model = MockModel()
    
    results = {}
    
    for tile_size in tile_sizes:
        print(f"Testing {tile_size}x{tile_size} tiles...")
        
        # Create test image
        image = np.random.randint(0, 255, (canvas_size, canvas_size), dtype=np.uint8)
        
        # Initialize detector
        detector = TileDetector(
            model=mock_model,
            canvas_width=canvas_size,
            canvas_height=canvas_size,
            tile_size=tile_size,
            is_tflite=False,
            enable_caching=False
        )
        
        # Benchmark full analysis
        times = []
        for _ in range(num_iterations):
            start = time.perf_counter()
            result = detector.detect(image, force_full_analysis=True)
            end = time.perf_counter()
            times.append((end - start) * 1000)  # Convert to ms
        
        grid_size = detector.grid.grid_rows * detector.grid.grid_cols
        
        results[f'tile_{tile_size}'] = {
            'mean_ms': np.mean(times),
            'std_ms': np.std(times),
            'min_ms': np.min(times),
            'max_ms': np.max(times),
            'grid_size': grid_size,
            'per_tile_ms': np.mean(times) / grid_size
        }
        
        print(f"  Grid: {detector.grid.grid_rows}x{detector.grid.grid_cols} = {grid_size} tiles")
        print(f"  Mean: {results[f'tile_{tile_size}']['mean_ms']:.2f}ms")
        print(f"  Per tile: {results[f'tile_{tile_size}']['per_tile_ms']:.2f}ms\n")
    
    return results


def benchmark_incremental_updates(num_iterations: int = 20) -> Dict:
    """
    Benchmark incremental updates with dirty tile tracking.
    
    Args:
        num_iterations: Number of iterations
    
    Returns:
        Dictionary with benchmark results
    """
    print(f"\n{'='*70}")
    print("Benchmarking Incremental Updates (Dirty Tile Tracking)")
    print(f"{'='*70}\n")
    
    canvas_size = 512
    tile_size = 64
    mock_model = MockModel()
    
    # Create test image
    image = np.random.randint(0, 255, (canvas_size, canvas_size), dtype=np.uint8)
    
    # Initialize detector with caching
    detector = TileDetector(
        model=mock_model,
        canvas_width=canvas_size,
        canvas_height=canvas_size,
        tile_size=tile_size,
        is_tflite=False,
        enable_caching=True
    )
    
    results = {}
    
    # Benchmark initial full analysis
    print("Initial full analysis...")
    times_full = []
    for _ in range(5):
        detector.mark_all_dirty()
        start = time.perf_counter()
        result = detector.detect(image)
        end = time.perf_counter()
        times_full.append((end - start) * 1000)
    
    results['full_analysis'] = {
        'mean_ms': np.mean(times_full),
        'std_ms': np.std(times_full),
        'num_tiles': detector.grid.total_tiles
    }
    
    print(f"  Mean: {results['full_analysis']['mean_ms']:.2f}ms for {detector.grid.total_tiles} tiles\n")
    
    # Benchmark single tile update
    print("Single tile update...")
    times_single = []
    for _ in range(num_iterations):
        detector.mark_dirty_tiles([(100, 100)])  # Mark one tile dirty
        start = time.perf_counter()
        result = detector.detect(image)
        end = time.perf_counter()
        times_single.append((end - start) * 1000)
    
    results['single_tile'] = {
        'mean_ms': np.mean(times_single),
        'std_ms': np.std(times_single),
        'min_ms': np.min(times_single),
        'max_ms': np.max(times_single)
    }
    
    print(f"  Mean: {results['single_tile']['mean_ms']:.2f}ms\n")
    
    # Benchmark small stroke (3-4 tiles)
    print("Small stroke (3-4 tiles)...")
    times_stroke = []
    for _ in range(num_iterations):
        stroke = [(i, i) for i in range(100, 200, 10)]  # Diagonal stroke
        detector.mark_dirty_tiles(stroke)
        start = time.perf_counter()
        result = detector.detect(image)
        end = time.perf_counter()
        times_stroke.append((end - start) * 1000)
    
    results['small_stroke'] = {
        'mean_ms': np.mean(times_stroke),
        'std_ms': np.std(times_stroke),
        'min_ms': np.min(times_stroke),
        'max_ms': np.max(times_stroke)
    }
    
    print(f"  Mean: {results['small_stroke']['mean_ms']:.2f}ms\n")
    
    return results


def benchmark_canvas_dimensions(num_iterations: int = 10) -> Dict:
    """
    Benchmark non-square canvas dimensions.
    
    Args:
        num_iterations: Number of iterations
    
    Returns:
        Dictionary with benchmark results
    """
    print(f"\n{'='*70}")
    print("Benchmarking Non-Square Canvas Dimensions")
    print(f"{'='*70}\n")
    
    tile_size = 64
    mock_model = MockModel()
    canvas_configs = [
        (512, 512, "Square 512x512"),
        (512, 768, "Portrait 512x768"),
        (1024, 768, "Landscape 1024x768"),
        (800, 600, "Standard 800x600")
    ]
    
    results = {}
    
    for width, height, label in canvas_configs:
        print(f"Testing {label}...")
        
        # Create test image
        image = np.random.randint(0, 255, (height, width), dtype=np.uint8)
        
        # Initialize detector
        detector = TileDetector(
            model=mock_model,
            canvas_width=width,
            canvas_height=height,
            tile_size=tile_size,
            is_tflite=False,
            enable_caching=False
        )
        
        # Benchmark
        times = []
        for _ in range(num_iterations):
            start = time.perf_counter()
            result = detector.detect(image, force_full_analysis=True)
            end = time.perf_counter()
            times.append((end - start) * 1000)
        
        grid_size = detector.grid.grid_rows * detector.grid.grid_cols
        
        results[label] = {
            'mean_ms': np.mean(times),
            'grid_dimensions': (detector.grid.grid_rows, detector.grid.grid_cols),
            'total_tiles': grid_size,
            'per_tile_ms': np.mean(times) / grid_size
        }
        
        print(f"  Grid: {detector.grid.grid_rows}x{detector.grid.grid_cols} = {grid_size} tiles")
        print(f"  Mean: {results[label]['mean_ms']:.2f}ms\n")
    
    return results


def print_summary(tile_results: Dict, incremental_results: Dict, canvas_results: Dict):
    """Print benchmark summary and check against targets."""
    print(f"\n{'='*70}")
    print("Performance Summary & Target Validation")
    print(f"{'='*70}\n")
    
    # Check targets
    targets = {
        'single_tile': 10.0,  # <10ms
        'full_grid_64': 200.0,  # <200ms for 64 tiles
        'incremental_1_4': 50.0  # <50ms for 1-4 tiles
    }
    
    print("Target Validation (RPi4):")
    print(f"{'Target':<30} {'Limit':<15} {'Measured':<15} {'Status':<10}")
    print("-" * 70)
    
    # Single tile (from 64x64 config)
    single_tile_time = tile_results['tile_64']['per_tile_ms']
    status = "✓ PASS" if single_tile_time < targets['single_tile'] else "✗ FAIL"
    print(f"{'Single tile inference':<30} {'<10ms':<15} {f'{single_tile_time:.2f}ms':<15} {status:<10}")
    
    # Full grid (64 tiles with 64x64 tile size)
    full_grid_time = tile_results['tile_64']['mean_ms']
    status = "✓ PASS" if full_grid_time < targets['full_grid_64'] else "✗ FAIL"
    print(f"{'Full grid (64 tiles)':<30} {'<200ms':<15} {f'{full_grid_time:.2f}ms':<15} {status:<10}")
    
    # Incremental update
    incremental_time = incremental_results['small_stroke']['mean_ms']
    status = "✓ PASS" if incremental_time < targets['incremental_1_4'] else "✗ FAIL"
    print(f"{'Incremental (1-4 tiles)':<30} {'<50ms':<15} {f'{incremental_time:.2f}ms':<15} {status:<10}")
    
    print("\nNote: These benchmarks use a mock model with simulated inference time.")
    print("Actual performance on RPi4 with TFLite INT8 model will vary.")
    print("Run this benchmark on actual hardware for accurate measurements.")


def main():
    """Run all benchmarks and print results."""
    print("\n" + "="*70)
    print("Tile-Based Detection Performance Benchmark")
    print("="*70)
    print("\nThis benchmark measures tile detection performance with different")
    print("configurations to validate RPi4 performance targets.")
    print("\nNote: Uses mock model with simulated 5ms inference time per tile.")
    
    # Run benchmarks
    tile_results = benchmark_tile_sizes(num_iterations=10)
    incremental_results = benchmark_incremental_updates(num_iterations=20)
    canvas_results = benchmark_canvas_dimensions(num_iterations=10)
    
    # Print summary
    print_summary(tile_results, incremental_results, canvas_results)
    
    print(f"\n{'='*70}")
    print("Benchmark Complete")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
