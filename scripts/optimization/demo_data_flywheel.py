#!/usr/bin/env python3
"""
Data-Flywheel System Demonstration for DoodleHunter

This script demonstrates the complete data-flywheel optimization workflow:
1. Creates a sample model for demonstration
2. Establishes performance baseline
3. Evaluates optimization candidates
4. Generates comprehensive reports

Usage:
    python scripts/optimization/demo_data_flywheel.py
"""

import argparse
import sys
import numpy as np
import tensorflow as tf
from pathlib import Path
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core.optimization import PerformanceMonitor, create_optimization_feedback_loop
from src.core.models import get_model


def create_demo_model(save_path: str = 'models/demo_model.keras') -> str:
    """Create a simple demo model for testing the data-flywheel system."""
    print("Creating demo model...")

    try:
        # Create simple model using TensorFlow/Keras
        model, base_model = get_model(architecture='custom', summary=False)

        # Compile model
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        # Create models directory
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)

        # Save model in Keras format
        try:
            model.save(save_path)
        except:
            # Fallback to older format
            save_path_h5 = save_path.replace('.keras', '.h5')
            model.save(save_path_h5)
            save_path = save_path_h5

        print(f"✓ Demo model saved to: {save_path}")
        return save_path

    except Exception as e:
        print(f"Note: TensorFlow/Keras version compatibility issue: {e}")
        print("Creating simple model placeholder...")

        # Create a simple placeholder model file
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)

        # Write a simple model config as placeholder
        model_config = {
            "name": "demo_model",
            "layers": [
                {"type": "Conv2D", "filters": 32, "kernel_size": 3},
                {"type": "MaxPooling2D", "pool_size": 2},
                {"type": "Flatten"},
                {"type": "Dense", "units": 64},
                {"type": "Dense", "units": 1, "activation": "sigmoid"}
            ],
            "input_shape": [28, 28, 1],
            "params": 423000,
            "size_mb": 1.6
        }

        import json
        with open(save_path.replace('.keras', '.json').replace('.h5', '.json'), 'w') as f:
            json.dump(model_config, f, indent=2)

        print(f"✓ Demo model config saved to: {save_path}")
        return save_path


def create_demo_test_data(
    num_samples: int = 200,
    save_dir: str = 'data_flywheel/demo_data'
) -> str:
    """Create demo test data for evaluation."""
    print(f"Creating {num_samples} demo test samples...")

    # Create random demo data (simulating 28x28 grayscale images)
    X_demo = np.random.rand(num_samples, 28, 28, 1).astype(np.float32)
    y_demo = np.random.randint(0, 2, num_samples).astype(np.float32)

    # Save data
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    np.save(save_dir / 'X_demo.npy', X_demo)
    np.save(save_dir / 'y_demo.npy', y_demo)

    print(f"✓ Demo data saved to: {save_dir}")
    return str(save_dir)


def demonstrate_data_flywheel_workflow():
    """Demonstrate complete data-flywheel optimization workflow."""
    print("="*70)
    print("DOODLEHUNTER DATA-FLYWHEEL SYSTEM DEMONSTRATION")
    print("="*70)
    print()

    # Step 1: Create demo model and data
    print("Step 1: Setting up demo environment")
    print("-" * 40)

    model_path = create_demo_model()
    data_dir = create_demo_test_data(200, 'data_flywheel/demo_data')

    # Load demo data
    X_demo = np.load(f'{data_dir}/X_demo.npy')
    y_demo = np.load(f'{data_dir}/y_demo.npy')
    test_data = (X_demo, y_demo)

    print(f"Model: {model_path}")
    print(f"Test data: {len(X_demo)} samples")
    print()

    # Step 2: Initialize performance monitor
    print("Step 2: Initializing Performance Monitor")
    print("-" * 40)

    monitor = PerformanceMonitor('data_flywheel/demo')
    print("✓ Performance monitor initialized")
    print(f"Data directory: {monitor.data_flywheel_dir}")
    print()

    # Step 3: Establish baseline
    print("Step 3: Establishing Performance Baseline")
    print("-" * 40)

    model = tf.keras.models.load_model(model_path)
    baseline_data = monitor.establish_baseline(model, test_data, 'demo_baseline_v1')

    print("✓ Baseline established successfully")
    print(f"Baseline metrics:")
    print(f"  - Latency: {baseline_data['metrics']['latency']['batch_1']['avg_latency_ms']:.1f}ms")
    print(f"  - Accuracy: {baseline_data['metrics']['accuracy']['batch_1']['accuracy']:.3f}")
    print(f"  - Model size: {baseline_data['metrics']['model_size_mb']:.1f}MB")
    print()

    # Step 4: Simulate model optimization
    print("Step 4: Simulating Model Optimization")
    print("-" * 40)

    # Create a "simulated" optimized model (using same model for demo)
    optimized_path = 'models/demo_optimized_model.h5'

    # Simulate slight differences by saving again (for demo purposes)
    model.save(optimized_path)
    optimized_model = tf.keras.models.load_model(optimized_path)

    print("✓ Simulated optimized model created")
    print()

    # Step 5: Evaluate optimization candidate
    print("Step 5: Evaluating Optimization Candidate")
    print("-" * 40)

    evaluation = monitor.evaluate_optimization_candidate(
        model,
        optimized_model,
        test_data,
        'demo_optimization_candidate',
        'demo_baseline_v1'
    )

    print()
    print("✓ Evaluation complete")
    print(f"Recommendation: {evaluation['recommendation']}")
    print()

    # Step 6: Generate performance report
    print("Step 6: Generating Performance Report")
    print("-" * 40)

    report_content = monitor.generate_performance_report(
        time_period_days=1,
        output_file='data_flywheel/demo/performance_report.md'
    )

    print("✓ Performance report generated")
    print(f"Report preview:")
    print(report_content[:500] + "..." if len(report_content) > 500 else report_content)
    print()

    # Step 7: Show system status
    print("Step 7: Data-Flywheel System Status")
    print("-" * 40)

    print("Data-Flywheel Directory Structure:")
    data_flywheel_dir = Path('data_flywheel/demo')

    for subdir in ['metrics', 'baselines', 'evaluations', 'feedback']:
        subdir_path = data_flywheel_dir / subdir
        if subdir_path.exists():
            files = list(subdir_path.glob('*'))
            print(f"  {subdir}/: {len(files)} files")
        else:
            print(f"  {subdir}/: directory not created")

    print()
    print("="*70)
    print("DEMONSTRATION COMPLETE")
    print("="*70)
    print()
    print("Key Features Demonstrated:")
    print("✓ Performance metrics collection")
    print("✓ Baseline establishment and tracking")
    print("✓ Systematic optimization evaluation")
    print("✓ Data-driven decision making")
    print("✓ Comprehensive performance reporting")
    print()
    print("Next Steps:")
    print("1. Use establish_baseline.py to create real baselines")
    print("2. Use data_flywheel_workflow.py for optimization experiments")
    print("3. Use performance_monitor.py for real-time monitoring")
    print("4. Review generated reports in data_flywheel/ directory")


def show_usage_examples():
    """Show usage examples for the data-flywheel system."""
    print("="*70)
    print("DATA-FLYWHEEL SYSTEM USAGE EXAMPLES")
    print("="*70)
    print()

    examples = [
        {
            "title": "1. Establish Performance Baseline",
            "command": "python scripts/optimization/establish_baseline.py --model models/current_model.h5",
            "description": "Create initial performance baseline for future comparisons"
        },
        {
            "title": "2. Evaluate Optimization Candidate",
            "command": "python scripts/optimization/data_flywheel_workflow.py --mode evaluate-candidate --original models/baseline.h5 --candidate models/optimized.h5 --candidate-name quantization_v1",
            "description": "Compare optimized model against baseline with data-driven evaluation"
        },
        {
            "title": "3. Run Optimization Experiment",
            "command": "python scripts/optimization/data_flywheel_workflow.py --mode run-experiment --model models/baseline.h5",
            "description": "Test multiple optimization strategies automatically"
        },
        {
            "title": "4. Continuous Monitoring",
            "command": "python scripts/optimization/performance_monitor.py --model models/current.h5 --interval 300",
            "description": "Real-time performance monitoring with alerting"
        },
        {
            "title": "5. Generate Performance Report",
            "command": "python scripts/optimization/data_flywheel_workflow.py --mode generate-report --days 30",
            "description": "Comprehensive performance analysis and trend reporting"
        }
    ]

    for example in examples:
        print(f"{example['title']}")
        print(f"Command: {example['command']}")
        print(f"Description: {example['description']}")
        print()


def main():
    parser = argparse.ArgumentParser(
        description='Data-Flywheel System Demonstration',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--mode',
        choices=['demo', 'examples', 'both'],
        default='both',
        help='Demo mode: run demonstration; Examples: show usage examples; Both: run both'
    )

    args = parser.parse_args()

    if args.mode in ['demo', 'both']:
        demonstrate_data_flywheel_workflow()

    if args.mode in ['examples', 'both']:
        show_usage_examples()


if __name__ == '__main__':
    main()