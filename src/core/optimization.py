"""Performance optimization and data-flywheel system for DoodleHunter.

Provides comprehensive optimization framework including:
- INT8 quantization for TensorFlow Lite models
- Weight pruning to reduce model size
- Knowledge distillation for smaller models
- Graph optimization for TensorFlow models
- Performance metrics collection and monitoring
- Baseline establishment and tracking
- Systematic evaluation of optimization candidates
- Continuous feedback loops for continuous improvement

Related:
- scripts/convert/quantize_int8.py (INT8 quantization script)
- scripts/convert/prune_model.py (pruning script)
- scripts/convert/distill_model.py (distillation script)
- scripts/convert/optimize_graph.py (graph optimization script)
- data_flywheel/ (metrics, baselines, evaluations, feedback)

Exports:
- quantize_model_int8: Apply INT8 quantization
- prune_model: Apply weight pruning
- distill_model: Knowledge distillation
- optimize_graph: TensorFlow graph optimization
- Performance monitoring and evaluation utilities
"""

import numpy as np
import tensorflow as tf
import time
import json
import pickle
from pathlib import Path
from typing import Callable, Optional, Tuple, Dict, List
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import pandas as pd


def quantize_model_int8(
    model_path: str,
    calibration_data: np.ndarray,
    output_path: Optional[str] = None
) -> str:
    """
    Apply INT8 post-training quantization to a model.
    
    Args:
        model_path: Path to model (.h5, .keras, or .tflite)
        calibration_data: Representative dataset for calibration
        output_path: Path to save quantized model (default: {model}_int8.tflite)
    
    Returns:
        Path to quantized model
    """
    model_path = Path(model_path)
    
    if output_path is None:
        output_path = model_path.parent / f"{model_path.stem}_int8.tflite"
    else:
        output_path = Path(output_path)
    
    if model_path.suffix in ['.h5', '.keras']:
        model = tf.keras.models.load_model(model_path)
    else:
        raise ValueError(f"Unsupported model format: {model_path.suffix}")
    
    def representative_dataset():
        for sample in calibration_data:
            yield [np.expand_dims(sample, axis=0).astype(np.float32)]
    
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.float32
    converter.inference_output_type = tf.float32
    
    tflite_model = converter.convert()
    output_path.write_bytes(tflite_model)
    
    return str(output_path)


def prune_model(
    model,
    target_sparsity: float = 0.5,
    epochs: int = 10
):
    """Apply magnitude-based weight pruning to reduce model size."""
    try:
        import tensorflow_model_optimization as tfmot
    except ImportError:
        raise ImportError(
            "tensorflow_model_optimization is required for pruning. "
            "Install with: pip install tensorflow-model-optimization"
        )
    
    pruning_params = {
        'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
            initial_sparsity=0.0,
            final_sparsity=target_sparsity,
            begin_step=0,
            end_step=epochs * 100
        )
    }
    
    return tfmot.sparsity.keras.prune_low_magnitude(model, **pruning_params)


def distill_model(
    teacher_model,
    student_model,
    train_data: Tuple[np.ndarray, np.ndarray],
    temperature: float = 3.0,
    alpha: float = 0.1,
    epochs: int = 10
):
    """Knowledge distillation: train smaller student model from larger teacher."""
    X_train, y_train = train_data
    teacher_predictions = teacher_model.predict(X_train, verbose=0)
    
    def distillation_loss(y_true, y_pred):
        hard_loss = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        soft_loss = tf.keras.losses.kullback_leibler_divergence(
            tf.nn.softmax(teacher_predictions / temperature),
            tf.nn.softmax(y_pred / temperature)
        )
        return alpha * hard_loss + (1 - alpha) * soft_loss * (temperature ** 2)
    
    student_model.compile(optimizer='adam', loss=distillation_loss, metrics=['accuracy'])
    student_model.fit(X_train, y_train, epochs=epochs, validation_split=0.2, verbose=1)
    
    return student_model


def optimize_graph(model_path: str, output_path: Optional[str] = None) -> str:
    """Optimize TensorFlow graph for inference."""
    model_path = Path(model_path)
    
    if output_path is None:
        output_path = model_path.parent / f"{model_path.stem}_optimized{model_path.suffix}"
    else:
        output_path = Path(output_path)
    
    model = tf.keras.models.load_model(model_path)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    tflite_model = converter.convert()
    output_path = output_path.with_suffix('.tflite')
    output_path.write_bytes(tflite_model)
    
    return str(output_path)


# Data-Flywheel Performance Monitoring System
# Implements continuous optimization through production feedback loops

class PerformanceMonitor:
    """Monitor and collect performance metrics for data-flywheel optimization."""

    def __init__(self, data_flywheel_dir: str = 'data_flywheel'):
        self.data_flywheel_dir = Path(data_flywheel_dir)
        self.metrics_dir = self.data_flywheel_dir / 'metrics'
        self.baselines_dir = self.data_flywheel_dir / 'baselines'
        self.evaluations_dir = self.data_flywheel_dir / 'evaluations'
        self.feedback_dir = self.data_flywheel_dir / 'feedback'

        # Create directories if they don't exist
        for dir_path in [self.metrics_dir, self.baselines_dir, self.evaluations_dir, self.feedback_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

    def collect_inference_metrics(
        self,
        model,
        test_data,
        model_name: str = 'current',
        batch_sizes: List[int] = [1, 8, 16, 32, 64]
    ):
        """Collect comprehensive inference performance metrics."""
        X_test, y_test = test_data

        # Collect latency and throughput metrics
        latency_results = {}
        throughput_results = {}
        accuracy_results = {}

        for batch_size in batch_sizes:
            if batch_size > len(X_test):
                continue

            # Extract subset for testing
            test_subset = X_test[:batch_size]
            labels_subset = y_test[:batch_size]

            # Warm up model
            for _ in range(3):
                _ = model.predict(test_subset, verbose=0)

            # Measure latency (milliseconds)
            start_time = time.perf_counter()
            predictions = model.predict(test_subset, verbose=0)
            end_time = time.perf_counter()

            avg_latency = (end_time - start_time) / batch_size * 1000  # ms per inference
            latency_results[f'batch_{batch_size}'] = {
                'avg_latency_ms': avg_latency,
                'total_time_ms': (end_time - start_time) * 1000
            }

            # Calculate throughput (inferences per second)
            throughput = batch_size / (end_time - start_time)
            throughput_results[f'batch_{batch_size}'] = {
                'throughput_inferences_per_sec': throughput,
                'throughput_samples_per_sec': throughput
            }

            # Calculate accuracy for each batch
            y_pred = (predictions > 0.5).astype(int)
            accuracy = accuracy_score(labels_subset, y_pred)
            accuracy_results[f'batch_{batch_size}'] = {
                'accuracy': accuracy,
                'precision': precision_score(labels_subset, y_pred, zero_division=0),
                'recall': recall_score(labels_subset, y_pred, zero_division=0),
                'f1_score': f1_score(labels_subset, y_pred, zero_division=0)
            }

        # Calculate model size and memory usage
        model_size_mb = model.count_params() * 4 / (1024 * 1024)  # Approximate size in MB

        metrics = {
            'model_name': model_name,
            'timestamp': datetime.now().isoformat(),
            'test_samples': len(X_test),
            'model_size_mb': model_size_mb,
            'latency': latency_results,
            'throughput': throughput_results,
            'accuracy': accuracy_results,
            'performance_score': self._calculate_performance_score(latency_results, accuracy_results)
        }

        # Save metrics
        metrics_file = self.metrics_dir / f'metrics_{model_name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)

        return metrics

    def _calculate_performance_score(
        self,
        latency_results: Dict,
        accuracy_results: Dict,
        weight_latency: float = 0.3,
        weight_accuracy: float = 0.7
    ) -> float:
        """Calculate combined performance score (higher is better)."""
        # Use single inference as reference
        single_batch_latency = latency_results.get('batch_1', {}).get('avg_latency_ms', 100)
        single_batch_accuracy = accuracy_results.get('batch_1', {}).get('accuracy', 0.5)

        # Normalize metrics (invert latency so lower is better, then normalize)
        # Assume 50ms is excellent latency, 200ms is poor
        latency_score = max(0, min(1, (200 - single_batch_latency) / 150))
        accuracy_score = single_batch_accuracy

        # Combined score
        performance_score = weight_latency * latency_score + weight_accuracy * accuracy_score

        return performance_score

    def establish_baseline(
        self,
        model,
        test_data,
        baseline_name: str = 'baseline_v1'
    ):
        """Establish performance baseline for comparison."""
        print(f"Establishing baseline: {baseline_name}")

        # Collect metrics
        metrics = self.collect_inference_metrics(model, test_data, baseline_name)

        # Add baseline-specific metadata
        baseline_data = {
            'baseline_name': baseline_name,
            'established_date': datetime.now().isoformat(),
            'model_summary': model.get_config(),
            'metrics': metrics,
            'validation_status': 'established'
        }

        # Save baseline
        baseline_file = self.baselines_dir / f'{baseline_name}.json'
        with open(baseline_file, 'w') as f:
            json.dump(baseline_data, f, indent=2, default=str)

        print(f"✓ Baseline established and saved to {baseline_file}")
        return baseline_data

    def evaluate_optimization_candidate(
        self,
        original_model,
        optimized_model,
        test_data,
        candidate_name: str,
        baseline_name: str = 'baseline_v1'
    ):
        """Evaluate an optimization candidate against baseline."""
        print(f"Evaluating optimization candidate: {candidate_name}")

        # Collect metrics for both models
        print("  Collecting metrics for original model...")
        original_metrics = self.collect_inference_metrics(original_model, test_data, f'{candidate_name}_original')

        print("  Collecting metrics for optimized model...")
        optimized_metrics = self.collect_inference_metrics(optimized_model, test_data, f'{candidate_name}_optimized')

        # Load baseline
        baseline_file = self.baselines_dir / f'{baseline_name}.json'
        if baseline_file.exists():
            with open(baseline_file, 'r') as f:
                baseline_data = json.load(f)
            baseline_metrics = baseline_data['metrics']
        else:
            print("  Warning: No baseline found, using original as baseline")
            baseline_metrics = original_metrics

        # Compare metrics
        evaluation = {
            'candidate_name': candidate_name,
            'evaluation_date': datetime.now().isoformat(),
            'baseline_name': baseline_name,
            'original_metrics': original_metrics,
            'optimized_metrics': optimized_metrics,
            'baseline_metrics': baseline_metrics,
            'comparisons': self._compare_metrics(original_metrics, optimized_metrics, baseline_metrics),
            'recommendation': 'pending'
        }

        # Determine recommendation
        evaluation['recommendation'] = self._determine_recommendation(evaluation['comparisons'])

        # Save evaluation
        eval_file = self.evaluations_dir / f'evaluation_{candidate_name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(eval_file, 'w') as f:
            json.dump(evaluation, f, indent=2, default=str)

        # Print summary
        self._print_evaluation_summary(evaluation)

        return evaluation

    def _compare_metrics(
        self,
        original_metrics,
        optimized_metrics,
        baseline_metrics
    ):
        """Compare metrics between original, optimized, and baseline."""
        comparisons = {}

        # Model size comparison
        orig_size = original_metrics['model_size_mb']
        opt_size = optimized_metrics['model_size_mb']
        base_size = baseline_metrics['model_size_mb']

        comparisons['model_size'] = {
            'original_mb': orig_size,
            'optimized_mb': opt_size,
            'baseline_mb': base_size,
            'size_reduction_vs_original': (orig_size - opt_size) / orig_size,
            'size_reduction_vs_baseline': (base_size - opt_size) / base_size
        }

        # Single inference latency comparison
        orig_latency = original_metrics['latency']['batch_1']['avg_latency_ms']
        opt_latency = optimized_metrics['latency']['batch_1']['avg_latency_ms']
        base_latency = baseline_metrics['latency']['batch_1']['avg_latency_ms']

        comparisons['latency'] = {
            'original_ms': orig_latency,
            'optimized_ms': opt_latency,
            'baseline_ms': base_latency,
            'latency_improvement_vs_original': (orig_latency - opt_latency) / orig_latency,
            'latency_improvement_vs_baseline': (base_latency - opt_latency) / base_latency
        }

        # Accuracy comparison
        orig_accuracy = original_metrics['accuracy']['batch_1']['accuracy']
        opt_accuracy = optimized_metrics['accuracy']['batch_1']['accuracy']
        base_accuracy = baseline_metrics['accuracy']['batch_1']['accuracy']

        comparisons['accuracy'] = {
            'original': orig_accuracy,
            'optimized': opt_accuracy,
            'baseline': base_accuracy,
            'accuracy_change_vs_original': opt_accuracy - orig_accuracy,
            'accuracy_change_vs_baseline': opt_accuracy - base_accuracy
        }

        # Performance score comparison
        orig_score = original_metrics['performance_score']
        opt_score = optimized_metrics['performance_score']
        base_score = baseline_metrics['performance_score']

        comparisons['performance_score'] = {
            'original': orig_score,
            'optimized': opt_score,
            'baseline': base_score,
            'score_improvement_vs_original': (opt_score - orig_score) / orig_score,
            'score_improvement_vs_baseline': (opt_score - base_score) / base_score
        }

        return comparisons

    def _determine_recommendation(self, comparisons):
        """Determine if optimization should be recommended."""
        # Check if accuracy is maintained (within 2% of baseline)
        accuracy_change = comparisons['accuracy']['accuracy_change_vs_baseline']
        if accuracy_change < -0.02:
            return 'reject: significant accuracy loss'

        # Check if performance is improved
        performance_improvement = comparisons['performance_score']['score_improvement_vs_baseline']
        latency_improvement = comparisons['latency']['latency_improvement_vs_baseline']
        size_reduction = comparisons['model_size']['size_reduction_vs_baseline']

        # Must have at least 10% improvement in at least one metric
        if performance_improvement > 0.1 or latency_improvement > 0.1 or size_reduction > 0.1:
            if accuracy_change >= -0.01:  # Within 1% accuracy loss
                return 'accept: significant improvement with maintained accuracy'

        return 'reject: insufficient improvement'

    def _print_evaluation_summary(self, evaluation):
        """Print evaluation summary."""
        print("\n" + "="*60)
        print(f"EVALUATION SUMMARY: {evaluation['candidate_name']}")
        print("="*60)

        comp = evaluation['comparisons']

        print(f"Model Size: {comp['model_size']['original_mb']:.1f}MB → {comp['model_size']['optimized_mb']:.1f}MB "
              f"({comp['model_size']['size_reduction_vs_original']:.1%} reduction)")

        print(f"Latency: {comp['latency']['original_ms']:.1f}ms → {comp['latency']['optimized_ms']:.1f}ms "
              f"({comp['latency']['latency_improvement_vs_original']:.1%} improvement)")

        print(f"Accuracy: {comp['accuracy']['original']:.3f} → {comp['accuracy']['optimized']:.3f} "
              f"({comp['accuracy']['accuracy_change_vs_original']:+.3f} change)")

        print(f"Performance Score: {comp['performance_score']['original']:.3f} → {comp['performance_score']['optimized']:.3f} "
              f"({comp['performance_score']['score_improvement_vs_original']:.1%} improvement)")

        print(f"\nRecommendation: {evaluation['recommendation']}")
        print("="*60)

    def generate_performance_report(
        self,
        time_period_days: int = 30,
        output_file: Optional[str] = None
    ) -> str:
        """Generate comprehensive performance report."""
        # Collect all metrics files from time period
        cutoff_date = datetime.now().timestamp() - (time_period_days * 24 * 60 * 60)
        metrics_files = []

        for metrics_file in self.metrics_dir.glob('metrics_*.json'):
            if metrics_file.stat().st_mtime > cutoff_date:
                metrics_files.append(metrics_file)

        if not metrics_files:
            return "No metrics data found for the specified time period."

        # Load and analyze metrics
        all_metrics = []
        for file in metrics_files:
            with open(file, 'r') as f:
                all_metrics.append(json.load(f))

        # Generate report
        report_lines = [
            "# DoodleHunter Performance Report",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Time Period: Last {time_period_days} days",
            f"Models Analyzed: {len(all_metrics)}",
            "",
            "## Executive Summary",
        ]

        # Calculate trends
        if len(all_metrics) >= 2:
            latest = all_metrics[-1]
            previous = all_metrics[-2]

            latest_perf = latest['performance_score']
            previous_perf = previous['performance_score']
            trend = "improving" if latest_perf > previous_perf else "declining" if latest_perf < previous_perf else "stable"

            report_lines.append(f"- **Performance Trend**: {trend}")
            report_lines.append(f"- **Latest Performance Score**: {latest_perf:.3f}")
            report_lines.append(f"- **Models Evaluated**: {len(all_metrics)}")

        report_lines.extend([
            "",
            "## Detailed Analysis",
            ""
        ])

        # Add model analysis
        for i, metrics in enumerate(all_metrics[-5:], 1):  # Last 5 models
            report_lines.extend([
                f"### Model {i}: {metrics['model_name']}",
                f"- **Test Samples**: {metrics['test_samples']}",
                f"- **Model Size**: {metrics['model_size_mb']:.1f} MB",
                f"- **Single Inference Latency**: {metrics['latency']['batch_1']['avg_latency_ms']:.1f} ms",
                f"- **Accuracy**: {metrics['accuracy']['batch_1']['accuracy']:.3f}",
                f"- **Performance Score**: {metrics['performance_score']:.3f}",
                ""
            ])

        # Add optimization opportunities
        report_lines.extend([
            "## Optimization Opportunities",
            "",
            "Based on the collected metrics, consider the following optimizations:",
            ""
        ])

        # Analyze for optimization opportunities
        if all_metrics:
            avg_latency = np.mean([m['latency']['batch_1']['avg_latency_ms'] for m in all_metrics])
            avg_accuracy = np.mean([m['accuracy']['batch_1']['accuracy'] for m in all_metrics])

            if avg_latency > 50:
                report_lines.append("- **Latency Optimization**: Consider quantization or pruning for faster inference")

            if avg_accuracy < 0.85:
                report_lines.append("- **Accuracy Improvement**: Review model architecture or training data")

            if len(all_metrics) < 10:
                report_lines.append("- **Data Collection**: Continue collecting metrics for better trend analysis")

        report_content = "\n".join(report_lines)

        if output_file:
            with open(output_file, 'w') as f:
                f.write(report_content)
            print(f"Performance report saved to {output_file}")

        return report_content


def create_optimization_feedback_loop(
    model_path: str,
    test_data: Tuple[np.ndarray, np.ndarray],
    data_flywheel_dir: str = 'data_flywheel',
    baseline_name: str = 'baseline_v1'
) -> PerformanceMonitor:
    """Create and initialize optimization feedback loop."""
    monitor = PerformanceMonitor(data_flywheel_dir)

    # Load model
    model = tf.keras.models.load_model(model_path)

    # Establish baseline if it doesn't exist
    baseline_file = monitor.baselines_dir / f'{baseline_name}.json'
    if not baseline_file.exists():
        print("No baseline found. Establishing baseline...")
        monitor.establish_baseline(model, test_data, baseline_name)
    else:
        print(f"Using existing baseline: {baseline_name}")

    return monitor


if __name__ == '__main__':
    print('Model Optimization and Data-Flywheel Performance Monitoring')
    print('Use PerformanceMonitor class for systematic optimization evaluation')
    print('')
    print('Key Features:')
    print('- Performance metrics collection and monitoring')
    print('- Baseline establishment and tracking')
    print('- Systematic evaluation of optimization candidates')
    print('- Continuous feedback loops for improvement')
    print('- Comprehensive performance reporting')
