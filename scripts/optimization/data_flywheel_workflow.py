#!/usr/bin/env python3
"""
Data-Flywheel Optimization Workflow for DoodleHunter

Implements continuous optimization through production feedback loops following
NVIDIA-inspired enterprise patterns for systematic performance improvement.

Key Features:
- Automated baseline establishment and tracking
- Systematic evaluation of optimization candidates
- Data-driven decision making for optimization selection
- Continuous monitoring and feedback collection
- Comprehensive performance reporting

Usage:
    python scripts/optimization/data_flywheel_workflow.py --mode establish-baseline --model models/current_model.h5
    python scripts/optimization/data_flywheel_workflow.py --mode evaluate-candidate --original models/baseline.h5 --candidate models/optimized.h5
    python scripts/optimization/data_flywheel_workflow.py --mode run-optimization-loop
    python scripts/optimization/data_flywheel_workflow.py --mode generate-report --days 30

Related:
- src/core/optimization.py (PerformanceMonitor class)
- data_flywheel/ (metrics, baselines, evaluations, feedback)
- scripts/convert/ (optimization conversion scripts)
"""

import argparse
import sys
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import tensorflow as tf

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core.optimization import (
    PerformanceMonitor,
    create_optimization_feedback_loop,
    quantize_model_int8,
    optimize_graph
)
from src.core.models import create_model


class DataFlywheelOrchestrator:
    """Orchestrates data-flywheel optimization workflow."""

    def __init__(self, data_flywheel_dir: str = 'data_flywheel'):
        self.data_flywheel_dir = Path(data_flywheel_dir)
        self.monitor = PerformanceMonitor(data_flywheel_dir)
        self.models_dir = Path('models')
        self.models_dir.mkdir(exist_ok=True)

    def establish_baseline(
        self,
        model_path: str,
        test_data: Tuple[np.ndarray, np.ndarray],
        baseline_name: str = 'baseline_v1',
        model_description: str = 'Initial baseline model'
    ) -> Dict:
        """Establish performance baseline for future comparisons."""
        print("="*60)
        print("ESTABLISHING PERFORMANCE BASELINE")
        print("="*60)
        print(f"Model: {model_path}")
        print(f"Baseline name: {baseline_name}")
        print(f"Description: {model_description}")
        print()

        # Load model
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        model = tf.keras.models.load_model(model_path)

        # Establish baseline
        baseline_data = self.monitor.establish_baseline(model, test_data, baseline_name)

        # Add metadata
        baseline_data['model_description'] = model_description
        baseline_data['model_path'] = model_path

        # Save updated baseline with metadata
        baseline_file = self.monitor.baselines_dir / f'{baseline_name}.json'
        with open(baseline_file, 'w') as f:
            json.dump(baseline_data, f, indent=2, default=str)

        print(" Baseline establishment complete")
        return baseline_data

    def evaluate_optimization_candidate(
        self,
        original_model_path: str,
        candidate_model_path: str,
        test_data: Tuple[np.ndarray, np.ndarray],
        candidate_name: str,
        baseline_name: str = 'baseline_v1'
    ) -> Dict:
        """Evaluate optimization candidate against baseline."""
        print("="*60)
        print("EVALUATING OPTIMIZATION CANDIDATE")
        print("="*60)
        print(f"Original: {original_model_path}")
        print(f"Candidate: {candidate_model_path}")
        print(f"Candidate name: {candidate_name}")
        print()

        # Load models
        original_model = tf.keras.models.load_model(original_model_path)
        candidate_model = tf.keras.models.load_model(candidate_model_path)

        # Evaluate candidate
        evaluation = self.monitor.evaluate_optimization_candidate(
            original_model,
            candidate_model,
            test_data,
            candidate_name,
            baseline_name
        )

        # Store feedback
        feedback_data = {
            'candidate_name': candidate_name,
            'evaluation_date': evaluation['evaluation_date'],
            'recommendation': evaluation['recommendation'],
            'comparisons': evaluation['comparisons'],
            'status': 'evaluated'
        }

        feedback_file = self.monitor.feedback_dir / f'feedback_{candidate_name}.json'
        with open(feedback_file, 'w') as f:
            json.dump(feedback_data, f, indent=2)

        print(" Evaluation complete")
        return evaluation

    def run_optimization_experiment(
        self,
        baseline_model_path: str,
        test_data: Tuple[np.ndarray, np.ndarray],
        optimization_strategies: List[str] = ['quantization', 'graph_optimization'],
        experiment_name: str = 'optimization_experiment'
    ) -> List[Dict]:
        """Run optimization experiment with multiple strategies."""
        print("="*60)
        print("RUNNING OPTIMIZATION EXPERIMENT")
        print("="*60)
        print(f"Baseline: {baseline_model_path}")
        print(f"Strategies: {optimization_strategies}")
        print(f"Experiment: {experiment_name}")
        print()

        results = []
        baseline_model = tf.keras.models.load_model(baseline_model_path)

        for strategy in optimization_strategies:
            try:
                print(f"Testing strategy: {strategy}")
                candidate_name = f"{experiment_name}_{strategy}"

                # Create optimized model based on strategy
                if strategy == 'quantization':
                    # Use subset of data for calibration
                    X_cal, _ = test_data
                    calibration_data = X_cal[:100]  # Use first 100 samples for calibration

                    candidate_path = self.models_dir / f"{candidate_name}.tflite"
                    quantize_model_int8(baseline_model_path, calibration_data, str(candidate_path))
                    candidate_model = tf.lite.TFLiteConverter.from_frozen_graph(str(candidate_path))

                elif strategy == 'graph_optimization':
                    candidate_path = self.models_dir / f"{candidate_name}_optimized.h5"
                    optimize_graph(baseline_model_path, str(candidate_path))
                    candidate_model = tf.keras.models.load_model(candidate_path)

                else:
                    print(f"Unknown strategy: {strategy}")
                    continue

                # Evaluate candidate
                evaluation = self.evaluate_optimization_candidate(
                    baseline_model_path,
                    str(candidate_path),
                    test_data,
                    candidate_name
                )

                results.append({
                    'strategy': strategy,
                    'candidate_path': str(candidate_path),
                    'evaluation': evaluation
                })

            except Exception as e:
                print(f"Error with strategy {strategy}: {e}")
                continue

        # Summary
        print("\n" + "="*60)
        print("EXPERIMENT SUMMARY")
        print("="*60)

        accepted_candidates = [r for r in results if 'accept' in r['evaluation']['recommendation']]

        if accepted_candidates:
            print(f" {len(accepted_candidates)} optimization(s) recommended:")
            for result in accepted_candidates:
                comp = result['evaluation']['comparisons']
                print(f"  - {result['strategy']}: "
                      f"{comp['model_size']['size_reduction_vs_baseline']:.1%} size reduction, "
                      f"{comp['latency']['latency_improvement_vs_baseline']:.1%} latency improvement")
        else:
            print("No optimizations recommended")

        return results

    def continuous_optimization_loop(
        self,
        model_path: str,
        test_data: Tuple[np.ndarray, np.ndarray],
        monitoring_interval: int = 100,  # inferences between checks
        max_iterations: int = 10
    ) -> Dict:
        """Run continuous optimization loop with monitoring."""
        print("="*60)
        print("CONTINUOUS OPTIMIZATION LOOP")
        print("="*60)
        print(f"Model: {model_path}")
        print(f"Monitoring interval: {monitoring_interval} inferences")
        print(f"Max iterations: {max_iterations}")
        print()

        # Initialize
        current_model_path = model_path
        iteration = 0
        loop_history = []

        while iteration < max_iterations:
            print(f"Iteration {iteration + 1}/{max_iterations}")
            print("-" * 40)

            # Collect current performance metrics
            model = tf.keras.models.load_model(current_model_path)
            current_metrics = self.monitor.collect_inference_metrics(
                model, test_data, f'iteration_{iteration + 1}'
            )

            # Check if optimization is needed (simplified criteria)
            if current_metrics['latency']['batch_1']['avg_latency_ms'] > 100:  # > 100ms latency
                print("  High latency detected, testing optimizations...")

                # Test simple quantization
                try:
                    X_cal, _ = test_data
                    calibration_data = X_cal[:50]

                    optimized_path = self.models_dir / f'iter_{iteration + 1}_optimized.tflite'
                    quantize_model_int8(current_model_path, calibration_data, str(optimized_path))

                    # Evaluate optimization
                    eval_result = self.evaluate_optimization_candidate(
                        current_model_path,
                        str(optimized_path),
                        test_data,
                        f'iteration_{iteration + 1}_opt'
                    )

                    if 'accept' in eval_result['recommendation']:
                        print("   Optimization accepted, updating baseline")
                        current_model_path = str(optimized_path)
                    else:
                        print("   Optimization rejected, keeping current model")

                except Exception as e:
                    print(f"  Error in optimization: {e}")

            else:
                print("  Performance acceptable, no optimization needed")

            # Record iteration results
            loop_history.append({
                'iteration': iteration + 1,
                'model_path': current_model_path,
                'metrics': current_metrics,
                'optimization_applied': current_model_path != model_path
            })

            iteration += 1

        # Final summary
        print("\n" + "="*60)
        print("CONTINUOUS OPTIMIZATION COMPLETE")
        print("="*60)

        final_metrics = loop_history[-1]['metrics']
        print(f"Final model: {current_model_path}")
        print(f"Final latency: {final_metrics['latency']['batch_1']['avg_latency_ms']:.1f}ms")
        print(f"Final accuracy: {final_metrics['accuracy']['batch_1']['accuracy']:.3f}")
        print(f"Performance score: {final_metrics['performance_score']:.3f}")

        return {
            'final_model_path': current_model_path,
            'iterations_completed': iteration,
            'history': loop_history
        }

    def generate_comprehensive_report(
        self,
        days: int = 30,
        output_file: Optional[str] = None
    ) -> str:
        """Generate comprehensive optimization report."""
        print("="*60)
        print("GENERATING COMPREHENSIVE REPORT")
        print("="*60)

        report = self.monitor.generate_performance_report(days)

        # Add optimization-specific analysis
        report += "\n\n## Optimization Analysis\n"

        # Analyze all evaluations
        eval_files = list(self.monitor.evaluations_dir.glob('evaluation_*.json'))
        if eval_files:
            accepted = 0
            rejected = 0
            total_improvements = []

            for eval_file in eval_files:
                with open(eval_file, 'r') as f:
                    evaluation = json.load(f)

                if 'accept' in evaluation['recommendation']:
                    accepted += 1
                    comp = evaluation['comparisons']
                    improvement = comp['performance_score']['score_improvement_vs_baseline']
                    total_improvements.append(improvement)
                else:
                    rejected += 1

            report += f"- **Total Evaluations**: {len(eval_files)}\n"
            report += f"- **Accepted Optimizations**: {accepted}\n"
            report += f"- **Rejected Optimizations**: {rejected}\n"
            report += f"- **Acceptance Rate**: {accepted / len(eval_files) * 100:.1f}%\n"

            if total_improvements:
                avg_improvement = np.mean(total_improvements) * 100
                report += f"- **Average Performance Improvement**: {avg_improvement:.1f}%\n"

        # Add recommendations
        report += "\n## Next Steps\n"
        report += "- Continue monitoring performance metrics\n"
        report += "- Test additional optimization strategies\n"
        report += "- Establish regular review cadence\n"
        report += "- Document optimization decisions and rationale\n"

        if output_file:
            with open(output_file, 'w') as f:
                f.write(report)
            print(f" Report saved to {output_file}")

        return report


def load_test_data(data_dir: str = 'data/processed', max_samples: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
    """Load test data for evaluation."""
    data_dir = Path(data_dir)

    X_test = np.load(data_dir / 'X_test.npy')
    y_test = np.load(data_dir / 'y_test.npy')

    # Limit samples for faster evaluation
    if len(X_test) > max_samples:
        indices = np.random.choice(len(X_test), max_samples, replace=False)
        X_test = X_test[indices]
        y_test = y_test[indices]

    return X_test, y_test


def main():
    """Main workflow orchestrator."""
    parser = argparse.ArgumentParser(
        description='Data-Flywheel Optimization Workflow',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--mode', required=True, choices=[
        'establish-baseline',
        'evaluate-candidate',
        'run-experiment',
        'continuous-loop',
        'generate-report'
    ], help='Workflow mode')

    parser.add_argument('--model', help='Path to model for baseline establishment')
    parser.add_argument('--original', help='Path to original model for evaluation')
    parser.add_argument('--candidate', help='Path to candidate model for evaluation')
    parser.add_argument('--candidate-name', help='Name for evaluation candidate')
    parser.add_argument('--baseline-name', default='baseline_v1', help='Baseline name')
    parser.add_argument('--data-dir', default='data/processed', help='Data directory')
    parser.add_argument('--output-dir', default='data_flywheel', help='Data-flywheel directory')
    parser.add_argument('--days', type=int, default=30, help='Days for report generation')
    parser.add_argument('--iterations', type=int, default=10, help='Max iterations for continuous loop')
    parser.add_argument('--interval', type=int, default=100, help='Monitoring interval for continuous loop')

    args = parser.parse_args()

    try:
        # Initialize orchestrator
        orchestrator = DataFlywheelOrchestrator(args.output_dir)

        # Load test data
        print("Loading test data...")
        X_test, y_test = load_test_data(args.data_dir, max_samples=500)
        test_data = (X_test, y_test)
        print(f"Loaded {len(X_test)} test samples\n")

        if args.mode == 'establish-baseline':
            if not args.model:
                raise ValueError("--model required for establish-baseline mode")

            orchestrator.establish_baseline(
                args.model,
                test_data,
                args.baseline_name,
                "Initial performance baseline"
            )

        elif args.mode == 'evaluate-candidate':
            if not args.original or not args.candidate or not args.candidate_name:
                raise ValueError("--original, --candidate, and --candidate-name required")

            orchestrator.evaluate_optimization_candidate(
                args.original,
                args.candidate,
                test_data,
                args.candidate_name,
                args.baseline_name
            )

        elif args.mode == 'run-experiment':
            if not args.model:
                raise ValueError("--model required for run-experiment mode")

            strategies = ['quantization', 'graph_optimization']
            orchestrator.run_optimization_experiment(
                args.model,
                test_data,
                strategies,
                'experiment_v1'
            )

        elif args.mode == 'continuous-loop':
            if not args.model:
                raise ValueError("--model required for continuous-loop mode")

            orchestrator.continuous_optimization_loop(
                args.model,
                test_data,
                args.interval,
                args.iterations
            )

        elif args.mode == 'generate-report':
            report = orchestrator.generate_comprehensive_report(
                args.days,
                f"performance_report_{args.days}days.md"
            )
            print("\n" + "="*60)
            print("REPORT PREVIEW")
            print("="*60)
            print(report[:1000] + "..." if len(report) > 1000 else report)

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()