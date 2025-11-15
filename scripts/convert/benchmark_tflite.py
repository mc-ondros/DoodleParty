"""
Benchmark TensorFlow Lite model performance.

This script measures inference latency, throughput, and memory usage for TFLite
models. It's designed to compare float32 vs INT8 quantized models and provide
detailed performance metrics.

Usage:
    # Benchmark a single model
    python scripts/convert/benchmark_tflite.py --model models/quickdraw_model.tflite

    # Compare float32 vs INT8
    python scripts/convert/benchmark_tflite.py --model models/quickdraw_model.tflite --quantized models/quickdraw_model_int8.tflite

    # Benchmark with custom settings
    python scripts/convert/benchmark_tflite.py --model models/quickdraw_model.tflite --iterations 1000 --warmup 50

Features:
    - Latency measurement (mean, p50, p95, p99)
    - Throughput calculation (inferences per second)
    - Model size comparison
    - Memory usage profiling
    - Batch inference benchmarking

Output:
    Generates detailed performance report and saves results to models/benchmarks/
"""

import argparse
import sys
import time
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import tensorflow as tf


def parse_args():
    parser = argparse.ArgumentParser(
        description='Benchmark TensorFlow Lite model performance'
    )
    parser.add_argument(
        '--model', '-m',
        required=True,
        help='Path to TFLite model'
    )
    parser.add_argument(
        '--quantized', '-q',
        default=None,
        help='Path to quantized model for comparison'
    )
    parser.add_argument(
        '--iterations', '-i',
        type=int,
        default=500,
        help='Number of benchmark iterations (default: 500)'
    )
    parser.add_argument(
        '--warmup',
        type=int,
        default=20,
        help='Number of warmup iterations (default: 20)'
    )
    parser.add_argument(
        '--batch-sizes',
        type=int,
        nargs='+',
        default=[1, 4, 8, 16],
        help='Batch sizes to test (default: 1 4 8 16)'
    )
    parser.add_argument(
        '--output', '-o',
        default='models/benchmarks',
        help='Directory to save benchmark results (default: models/benchmarks)'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output'
    )
    return parser.parse_args()


def load_tflite_model(model_path: Path):
    """Load TFLite model and return interpreter."""
    print(f"Loading model: {model_path}")
    try:
        with open(model_path, 'rb') as f:
            model_content = f.read()
        
        interpreter = tf.lite.Interpreter(model_content=model_content)
        interpreter.allocate_tensors()
        
        size_mb = len(model_content) / (1024 * 1024)
        print(f"✓ Model loaded ({size_mb:.2f} MB)")
        
        return interpreter, model_content
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        sys.exit(1)


def get_model_info(interpreter):
    """Extract model input/output information."""
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    
    return {
        'input_shape': input_details['shape'].tolist(),
        'input_dtype': str(input_details['dtype']),
        'output_shape': output_details['shape'].tolist(),
        'output_dtype': str(output_details['dtype']),
    }


def generate_test_data(input_shape, batch_size=1):
    """Generate random test data matching input shape."""
    shape = [batch_size] + list(input_shape[1:])
    return np.random.rand(*shape).astype(np.float32)


def benchmark_latency(interpreter, test_data, iterations: int, warmup: int):
    """Benchmark inference latency."""
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    
    # Warmup
    for _ in range(warmup):
        interpreter.set_tensor(input_details['index'], test_data)
        interpreter.invoke()
    
    # Benchmark
    latencies = []
    for _ in range(iterations):
        start = time.perf_counter()
        interpreter.set_tensor(input_details['index'], test_data)
        interpreter.invoke()
        _ = interpreter.get_tensor(output_details['index'])
        end = time.perf_counter()
        
        latencies.append((end - start) * 1000)  # Convert to ms
    
    return np.array(latencies)


def calculate_statistics(latencies: np.ndarray) -> Dict[str, float]:
    """Calculate latency statistics."""
    return {
        'mean': float(np.mean(latencies)),
        'std': float(np.std(latencies)),
        'min': float(np.min(latencies)),
        'max': float(np.max(latencies)),
        'p50': float(np.percentile(latencies, 50)),
        'p95': float(np.percentile(latencies, 95)),
        'p99': float(np.percentile(latencies, 99)),
    }


def print_benchmark_results(model_name: str, stats: Dict[str, float], size_mb: float):
    """Print formatted benchmark results."""
    print("\n" + "=" * 70)
    print(f"Benchmark Results: {model_name}")
    print("=" * 70)
    print(f"Model size:     {size_mb:.2f} MB")
    print(f"Mean latency:   {stats['mean']:.2f} ms")
    print(f"Std dev:        {stats['std']:.2f} ms")
    print(f"Min latency:    {stats['min']:.2f} ms")
    print(f"Max latency:    {stats['max']:.2f} ms")
    print(f"p50 (median):   {stats['p50']:.2f} ms")
    print(f"p95:            {stats['p95']:.2f} ms")
    print(f"p99:            {stats['p99']:.2f} ms")
    print(f"Throughput:     {1000 / stats['mean']:.1f} inferences/sec")
    print("=" * 70)


def compare_models(baseline_stats: Dict, quantized_stats: Dict, 
                   baseline_size: float, quantized_size: float):
    """Print comparison between baseline and quantized models."""
    speedup = baseline_stats['mean'] / quantized_stats['mean']
    size_reduction = ((baseline_size - quantized_size) / baseline_size) * 100
    
    print("\n" + "=" * 70)
    print('Model Comparison')
    print("=" * 70)
    print(f"{'Metric':<20} {'Baseline':<15} {'Quantized':<15} {'Improvement':<15}")
    print("-" * 70)
    print(f"{'Size (MB)':<20} {baseline_size:<15.2f} {quantized_size:<15.2f} {size_reduction:<14.1f}%")
    print(f"{'Mean latency (ms)':<20} {baseline_stats['mean']:<15.2f} {quantized_stats['mean']:<15.2f} {speedup:<14.2f}x")
    print(f"{'p50 latency (ms)':<20} {baseline_stats['p50']:<15.2f} {quantized_stats['p50']:<15.2f} {baseline_stats['p50']/quantized_stats['p50']:<14.2f}x")
    print(f"{'p95 latency (ms)':<20} {baseline_stats['p95']:<15.2f} {quantized_stats['p95']:<15.2f} {baseline_stats['p95']/quantized_stats['p95']:<14.2f}x")
    print(f"{'p99 latency (ms)':<20} {baseline_stats['p99']:<15.2f} {quantized_stats['p99']:<15.2f} {baseline_stats['p99']/quantized_stats['p99']:<14.2f}x")
    
    baseline_throughput = 1000 / baseline_stats['mean']
    quantized_throughput = 1000 / quantized_stats['mean']
    print(f"{'Throughput (inf/s)':<20} {baseline_throughput:<15.1f} {quantized_throughput:<15.1f} {speedup:<14.2f}x")
    print("=" * 70)
    
    # Assessment
    print('\nAssessment:')
    if speedup >= 3.0:
        print(f"✓ Excellent speedup: {speedup:.1f}x faster")
    elif speedup >= 2.0:
        print(f"✓ Good speedup: {speedup:.1f}x faster")
    elif speedup >= 1.5:
        print(f"✓ Moderate speedup: {speedup:.1f}x faster")
    else:
        print(f"⚠ Limited speedup: {speedup:.1f}x faster")
    
    if size_reduction >= 70:
        print(f"✓ Excellent size reduction: {size_reduction:.0f}%")
    elif size_reduction >= 50:
        print(f"✓ Good size reduction: {size_reduction:.0f}%")
    else:
        print(f"⚠ Limited size reduction: {size_reduction:.0f}%")
    
    # Target check
    if quantized_stats['mean'] < 20:
        print(f"✓ Target achieved: <20ms latency ({quantized_stats['mean']:.1f}ms)")
    else:
        print(f"⚠ Target not met: {quantized_stats['mean']:.1f}ms (target: <20ms)")


def benchmark_batch_inference(interpreter, input_shape, batch_sizes: List[int], iterations: int):
    """Benchmark different batch sizes (simulated by running multiple inferences)."""
    print("\n" + "=" * 70)
    print('Batch Inference Benchmark (Simulated)')
    print("=" * 70)
    print('Note: TFLite models have fixed batch size. Testing sequential inference.')
    print(f"{'Batch Size':<15} {'Total (ms)':<15} {'Per-sample (ms)':<18} {'Throughput (inf/s)':<20}")
    print("-" * 70)
    
    # TFLite models typically have batch_size=1, so we simulate batches
    single_input = generate_test_data(input_shape, 1)
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    
    for batch_size in batch_sizes:
        # Simulate batch by running single inference multiple times
        total_times = []
        
        # Warmup
        for _ in range(5):
            for _ in range(batch_size):
                interpreter.set_tensor(input_details['index'], single_input)
                interpreter.invoke()
        
        # Benchmark
        for _ in range(iterations // 4):  # Fewer iterations for batches
            start = time.perf_counter()
            for _ in range(batch_size):
                interpreter.set_tensor(input_details['index'], single_input)
                interpreter.invoke()
                _ = interpreter.get_tensor(output_details['index'])
            end = time.perf_counter()
            total_times.append((end - start) * 1000)
        
        mean_total = np.mean(total_times)
        per_sample = mean_total / batch_size
        throughput = (batch_size * 1000) / mean_total
        
        print(f"{batch_size:<15} {mean_total:<15.2f} {per_sample:<18.2f} {throughput:<20.1f}")
    
    print("=" * 70)
    print('Note: These are sequential inference times. For true batch inference,')
    print('      consider using TensorFlow SavedModel or batched preprocessing.')


def save_results(output_dir: Path, results: Dict[str, Any]):
    """Save benchmark results to JSON file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    import json
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"benchmark_{timestamp}.json"
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to: {output_file}")


def main():
    args = parse_args()
    
    # Validate input
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Error: Model file not found: {model_path}")
        sys.exit(1)
    
    print("=" * 70)
    print('TensorFlow Lite Model Benchmark')
    print("=" * 70)
    print(f"Model:       {model_path}")
    print(f"Iterations:  {args.iterations}")
    print(f"Warmup:      {args.warmup}")
    print("=" * 70)
    
    # Load baseline model
    interpreter, model_content = load_tflite_model(model_path)
    model_info = get_model_info(interpreter)
    baseline_size = len(model_content) / (1024 * 1024)
    
    print(f"\nModel info:")
    print(f"  Input shape:  {model_info['input_shape']}")
    print(f"  Input dtype:  {model_info['input_dtype']}")
    print(f"  Output shape: {model_info['output_shape']}")
    print(f"  Output dtype: {model_info['output_dtype']}")
    
    # Generate test data
    test_data = generate_test_data(model_info['input_shape'])
    
    # Benchmark baseline
    print(f"\nBenchmarking baseline model ({args.iterations} iterations)...")
    baseline_latencies = benchmark_latency(interpreter, test_data, args.iterations, args.warmup)
    baseline_stats = calculate_statistics(baseline_latencies)
    print_benchmark_results(model_path.name, baseline_stats, baseline_size)
    
    results = {
        'baseline': {
            'model': str(model_path),
            'size_mb': baseline_size,
            'stats': baseline_stats,
            'model_info': model_info,
        }
    }
    
    # Benchmark quantized model if provided
    if args.quantized:
        quantized_path = Path(args.quantized)
        if not quantized_path.exists():
            print(f"Warning: Quantized model not found: {quantized_path}")
        else:
            print(f"\nLoading quantized model: {quantized_path}")
            q_interpreter, q_model_content = load_tflite_model(quantized_path)
            quantized_size = len(q_model_content) / (1024 * 1024)
            
            print(f"Benchmarking quantized model ({args.iterations} iterations)...")
            quantized_latencies = benchmark_latency(q_interpreter, test_data, args.iterations, args.warmup)
            quantized_stats = calculate_statistics(quantized_latencies)
            print_benchmark_results(quantized_path.name, quantized_stats, quantized_size)
            
            # Compare models
            compare_models(baseline_stats, quantized_stats, baseline_size, quantized_size)
            
            results['quantized'] = {
                'model': str(quantized_path),
                'size_mb': quantized_size,
                'stats': quantized_stats,
            }
    
    # Batch inference benchmark
    if len(args.batch_sizes) > 1:
        benchmark_batch_inference(interpreter, model_info['input_shape'], args.batch_sizes, args.iterations)
    
    # Save results
    output_dir = Path(args.output)
    save_results(output_dir, results)
    
    print('\n✓ Benchmark complete')


if __name__ == '__main__':
    main()
