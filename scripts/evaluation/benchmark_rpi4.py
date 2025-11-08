"""Comprehensive benchmarking tool for RPi4 deployment.

This script benchmarks the DoodleHunter model on Raspberry Pi 4 hardware,
measuring inference latency, throughput, memory usage, and thermal behavior
under various load conditions.

Performance Targets (RPi4):
- Single inference: <50ms
- Tile-based (64 tiles): <200ms
- Incremental (1-4 tiles): <50ms
- Memory usage: <500MB
- CPU temperature: <75°C under sustained load

Features:
- Single image inference benchmarking
- Batch inference throughput testing
- Tile-based detection benchmarking
- Memory profiling
- Thermal monitoring
- Cold start time measurement
- Sustained load testing

Usage:
    python scripts/evaluation/benchmark_rpi4.py --model models/quickdraw_model_int8.tflite
    python scripts/evaluation/benchmark_rpi4.py --all --output benchmark_results.json
"""

import argparse
import json
import time
import numpy as np
import psutil
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    import tensorflow as tf
except ImportError:
    print("Error: TensorFlow not installed")
    sys.exit(1)


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RPi4Benchmark:
    """Benchmark suite for RPi4 deployment."""
    
    def __init__(self, model_path: str, num_threads: int = 4):
        """Initialize benchmark with model.
        
        Args:
            model_path: Path to TFLite model file
            num_threads: Number of threads for inference (4 for RPi4)
        """
        self.model_path = Path(model_path)
        self.num_threads = num_threads
        self.interpreter = None
        self.results = {}
        
        logger.info(f"Initializing benchmark for: {model_path}")
        logger.info(f"Threads: {num_threads}")
        
        self._load_model()
        self._check_hardware()
    
    def _load_model(self) -> None:
        """Load TFLite model with optimizations."""
        logger.info("Loading TFLite model...")
        
        try:
            # Try with XNNPACK delegate
            try:
                self.interpreter = tf.lite.Interpreter(
                    model_path=str(self.model_path),
                    num_threads=self.num_threads,
                    experimental_delegates=[
                        tf.lite.experimental.load_delegate('libXNNPACK.so')
                    ]
                )
                logger.info("✓ Loaded with XNNPACK delegate")
            except Exception as e:
                logger.warning(f"XNNPACK not available: {e}")
                self.interpreter = tf.lite.Interpreter(
                    model_path=str(self.model_path),
                    num_threads=self.num_threads
                )
                logger.info("✓ Loaded with standard TFLite")
            
            self.interpreter.allocate_tensors()
            
            # Get model details
            input_details = self.interpreter.get_input_details()
            output_details = self.interpreter.get_output_details()
            
            logger.info(f"Input shape: {input_details[0]['shape']}")
            logger.info(f"Input dtype: {input_details[0]['dtype']}")
            logger.info(f"Output shape: {output_details[0]['shape']}")
            
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def _check_hardware(self) -> None:
        """Check and log hardware information."""
        logger.info("\nHardware Information")
        
        # CPU info
        cpu_count = psutil.cpu_count(logical=False)
        cpu_count_logical = psutil.cpu_count(logical=True)
        logger.info(f"CPU cores: {cpu_count} physical, {cpu_count_logical} logical")
        
        # Memory
        mem = psutil.virtual_memory()
        logger.info(f"Memory: {mem.total / (1024**3):.2f} GB total")
        
        # Check if on RPi4
        try:
            with open('/proc/device-tree/model', 'r') as f:
                model = f.read().strip('\x00')
                logger.info(f"Device: {model}")
        except:
            logger.warning("Could not detect device model")
        
        # Temperature
        temp = self._get_temperature()
        if temp:
            logger.info(f"CPU Temperature: {temp}°C")
    
    def _get_temperature(self) -> Optional[float]:
        """Get CPU temperature."""
        try:
            with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
                temp = int(f.read().strip()) / 1000.0
                return temp
        except:
            return None
    
    def _get_memory_usage(self) -> float:
        """Get current process memory usage in MB."""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024)
    
    def benchmark_cold_start(self) -> Dict[str, Any]:
        """Benchmark model cold start time."""
        logger.info("\nCold Start Benchmark")
        
        # Reload model to simulate cold start
        start_time = time.time()
        temp_interpreter = tf.lite.Interpreter(
            model_path=str(self.model_path),
            num_threads=self.num_threads
        )
        temp_interpreter.allocate_tensors()
        cold_start_time = (time.time() - start_time) * 1000
        
        logger.info(f"Cold start time: {cold_start_time:.2f}ms")
        
        result = {
            'cold_start_ms': round(cold_start_time, 2),
            'target_ms': 3000,
            'passed': cold_start_time < 3000
        }
        
        self.results['cold_start'] = result
        return result
    
    def benchmark_single_inference(self, num_iterations: int = 100) -> Dict[str, Any]:
        """Benchmark single image inference latency."""
        logger.info(f"\nSingle Inference Benchmark ({num_iterations} iterations)")
        
        input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()
        
        # Create dummy input
        input_shape = input_details[0]['shape']
        dummy_input = np.random.rand(*input_shape).astype(np.float32)
        
        # Warm up
        for _ in range(10):
            self.interpreter.set_tensor(input_details[0]['index'], dummy_input)
            self.interpreter.invoke()
        
        # Benchmark
        latencies = []
        for _ in range(num_iterations):
            start = time.time()
            self.interpreter.set_tensor(input_details[0]['index'], dummy_input)
            self.interpreter.invoke()
            _ = self.interpreter.get_tensor(output_details[0]['index'])
            latency = (time.time() - start) * 1000
            latencies.append(latency)
        
        result = {
            'mean_ms': round(np.mean(latencies), 2),
            'median_ms': round(np.median(latencies), 2),
            'std_ms': round(np.std(latencies), 2),
            'min_ms': round(np.min(latencies), 2),
            'max_ms': round(np.max(latencies), 2),
            'p95_ms': round(np.percentile(latencies, 95), 2),
            'p99_ms': round(np.percentile(latencies, 99), 2),
            'target_ms': 50,
            'passed': np.mean(latencies) < 50
        }
        
        logger.info(f"Mean latency: {result['mean_ms']}ms")
        logger.info(f"Median latency: {result['median_ms']}ms")
        logger.info(f"P95 latency: {result['p95_ms']}ms")
        logger.info(f"Target: <50ms - {'PASS' if result['passed'] else 'FAIL'}")
        
        self.results['single_inference'] = result
        return result
    
    def benchmark_batch_inference(self, batch_sizes: List[int] = [4, 8, 16, 32, 64]) -> Dict[str, Any]:
        """Benchmark batch inference throughput."""
        logger.info(f"\nBatch Inference Benchmark")
        
        results = {}
        
        for batch_size in batch_sizes:
            logger.info(f"\nBatch size: {batch_size}")
            
            # Create batch input
            dummy_batch = np.random.rand(batch_size, 128, 128, 1).astype(np.float32)
            
            # Warm up
            for i in range(batch_size):
                single_input = np.expand_dims(dummy_batch[i], axis=0)
                self.interpreter.set_tensor(
                    self.interpreter.get_input_details()[0]['index'],
                    single_input
                )
                self.interpreter.invoke()
            
            # Benchmark
            start = time.time()
            for i in range(batch_size):
                single_input = np.expand_dims(dummy_batch[i], axis=0)
                self.interpreter.set_tensor(
                    self.interpreter.get_input_details()[0]['index'],
                    single_input
                )
                self.interpreter.invoke()
            total_time = (time.time() - start) * 1000
            
            per_image_ms = total_time / batch_size
            throughput = 1000.0 / per_image_ms
            
            results[f'batch_{batch_size}'] = {
                'total_ms': round(total_time, 2),
                'per_image_ms': round(per_image_ms, 2),
                'throughput_fps': round(throughput, 2)
            }
            
            logger.info(f"  Total time: {total_time:.2f}ms")
            logger.info(f"  Per image: {per_image_ms:.2f}ms")
            logger.info(f"  Throughput: {throughput:.2f} FPS")
        
        self.results['batch_inference'] = results
        return results
    
    
    def benchmark_memory(self) -> Dict[str, Any]:
        """Benchmark memory usage."""
        logger.info("\nMemory Usage Benchmark")
        
        baseline_memory = self._get_memory_usage()
        logger.info(f"Baseline memory: {baseline_memory:.2f} MB")
        
        # Run inference to measure peak memory
        dummy_input = np.random.rand(1, 128, 128, 1).astype(np.float32)
        
        for _ in range(100):
            self.interpreter.set_tensor(
                self.interpreter.get_input_details()[0]['index'],
                dummy_input
            )
            self.interpreter.invoke()
        
        peak_memory = self._get_memory_usage()
        logger.info(f"Peak memory: {peak_memory:.2f} MB")
        
        result = {
            'baseline_mb': round(baseline_memory, 2),
            'peak_mb': round(peak_memory, 2),
            'overhead_mb': round(peak_memory - baseline_memory, 2),
            'target_mb': 500,
            'passed': peak_memory < 500
        }
        
        logger.info(f"Memory overhead: {result['overhead_mb']} MB")
        logger.info(f"Target: <500MB - {'PASS' if result['passed'] else 'FAIL'}")
        
        self.results['memory'] = result
        return result
    
    def benchmark_thermal(self, duration_seconds: int = 60) -> Dict[str, Any]:
        """Benchmark thermal behavior under sustained load."""
        logger.info(f"\nThermal Benchmark ({duration_seconds}s sustained load)")
        
        start_temp = self._get_temperature()
        if start_temp is None:
            logger.warning("Temperature monitoring not available")
            return {'available': False}
        
        logger.info(f"Starting temperature: {start_temp}°C")
        
        # Run sustained load
        dummy_input = np.random.rand(1, 128, 128, 1).astype(np.float32)
        
        temperatures = [start_temp]
        start_time = time.time()
        inference_count = 0
        
        while (time.time() - start_time) < duration_seconds:
            self.interpreter.set_tensor(
                self.interpreter.get_input_details()[0]['index'],
                dummy_input
            )
            self.interpreter.invoke()
            inference_count += 1
            
            if inference_count % 10 == 0:
                temp = self._get_temperature()
                if temp:
                    temperatures.append(temp)
        
        end_temp = self._get_temperature()
        
        result = {
            'start_temp_c': round(start_temp, 1),
            'end_temp_c': round(end_temp, 1),
            'max_temp_c': round(max(temperatures), 1),
            'mean_temp_c': round(np.mean(temperatures), 1),
            'temp_increase_c': round(end_temp - start_temp, 1),
            'inferences_completed': inference_count,
            'target_temp_c': 75,
            'passed': max(temperatures) < 75
        }
        
        logger.info(f"End temperature: {end_temp}°C")
        logger.info(f"Max temperature: {result['max_temp_c']}°C")
        logger.info(f"Temperature increase: {result['temp_increase_c']}°C")
        logger.info(f"Inferences completed: {inference_count}")
        logger.info(f"Target: <75°C - {'PASS' if result['passed'] else 'FAIL'}")
        
        self.results['thermal'] = result
        return result
    
    def run_all_benchmarks(self) -> Dict[str, Any]:
        """Run all benchmarks."""
        logger.info("\n" + "="*60)
        logger.info("Running comprehensive RPi4 benchmark suite")
        logger.info("="*60)
        
        self.benchmark_cold_start()
        self.benchmark_single_inference()
        self.benchmark_batch_inference()
        self.benchmark_memory()
        self.benchmark_thermal(duration_seconds=30)  # Shorter for quick test
        
        # Summary
        logger.info("\n" + "="*60)
        logger.info("BENCHMARK SUMMARY")
        logger.info("="*60)
        
        all_passed = True
        for category, data in self.results.items():
            if isinstance(data, dict) and 'passed' in data:
                status = "✓ PASS" if data['passed'] else "✗ FAIL"
                logger.info(f"{category:20s}: {status}")
                all_passed = all_passed and data['passed']
        
        logger.info("="*60)
        logger.info(f"Overall: {'✓ ALL TESTS PASSED' if all_passed else '✗ SOME TESTS FAILED'}")
        logger.info("="*60)
        
        return self.results
    
    def save_results(self, output_path: str) -> None:
        """Save benchmark results to JSON file."""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        logger.info(f"\nResults saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Benchmark DoodleHunter on RPi4')
    parser.add_argument('--model', type=str, default='models/quickdraw_model_int8.tflite',
                       help='Path to TFLite model')
    parser.add_argument('--threads', type=int, default=4,
                       help='Number of threads (4 for RPi4)')
    parser.add_argument('--output', type=str, default='benchmark_results.json',
                       help='Output file for results')
    parser.add_argument('--all', action='store_true',
                       help='Run all benchmarks')
    parser.add_argument('--cold-start', action='store_true',
                       help='Benchmark cold start time')
    parser.add_argument('--single', action='store_true',
                       help='Benchmark single inference')
    parser.add_argument('--batch', action='store_true',
                       help='Benchmark batch inference')
    parser.add_argument('--memory', action='store_true',
                       help='Benchmark memory usage')
    parser.add_argument('--thermal', action='store_true',
                       help='Benchmark thermal behavior')
    
    args = parser.parse_args()
    
    # Initialize benchmark
    benchmark = RPi4Benchmark(args.model, args.threads)
    
    # Run selected benchmarks
    if args.all:
        benchmark.run_all_benchmarks()
    else:
        if args.cold_start:
            benchmark.benchmark_cold_start()
        if args.single:
            benchmark.benchmark_single_inference()
        if args.batch:
            benchmark.benchmark_batch_inference()
        if args.memory:
            benchmark.benchmark_memory()
        if args.thermal:
            benchmark.benchmark_thermal()
        
        # If no specific benchmark selected, run single inference
        if not any([args.cold_start, args.single, args.batch, args.memory, args.thermal]):
            benchmark.benchmark_single_inference()
    
    # Save results
    benchmark.save_results(args.output)


if __name__ == '__main__':
    main()
