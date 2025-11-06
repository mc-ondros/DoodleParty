#!/usr/bin/env python3
"""
Establish Performance Baseline for DoodleHunter

Quick script to establish a performance baseline for future optimization comparisons.
Creates comprehensive performance metrics that serve as reference point for all future
optimization evaluations.

Usage:
    python scripts/optimization/establish_baseline.py --model models/current_model.h5
    python scripts/optimization/establish_baseline.py --model models/current_model.h5 --name custom_baseline
"""

import argparse
import sys
import numpy as np
from pathlib import Path
import tensorflow as tf

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core.optimization import create_optimization_feedback_loop


def main():
    parser = argparse.ArgumentParser(description='Establish Performance Baseline')
    parser.add_argument('--model', required=True, help='Path to model for baseline')
    parser.add_argument('--data-dir', default='data/processed', help='Data directory')
    parser.add_argument('--output-dir', default='data_flywheel', help='Data-flywheel directory')
    parser.add_argument('--name', default='baseline_v1', help='Baseline name')
    parser.add_argument('--description', default='Initial performance baseline', help='Baseline description')
    parser.add_argument('--max-samples', type=int, default=1000, help='Max samples for evaluation')

    args = parser.parse_args()

    # Check if model exists
    if not Path(args.model).exists():
        print(f"Error: Model not found: {args.model}")
        sys.exit(1)

    print("="*60)
    print("ESTABLISHING DOODLEHUNTER PERFORMANCE BASELINE")
    print("="*60)
    print(f"Model: {args.model}")
    print(f"Baseline name: {args.name}")
    print(f"Description: {args.description}")
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    print()

    try:
        # Load test data
        print("Loading test data...")
        data_dir = Path(args.data_dir)

        if not (data_dir / 'X_test.npy').exists():
            print(f"Error: Test data not found in {data_dir}")
            print("Please ensure data is processed first.")
            sys.exit(1)

        X_test = np.load(data_dir / 'X_test.npy')
        y_test = np.load(data_dir / 'y_test.npy')

        # Limit samples for faster baseline establishment
        if len(X_test) > args.max_samples:
            print(f"Using {args.max_samples} samples from {len(X_test)} available")
            indices = np.random.choice(len(X_test), args.max_samples, replace=False)
            X_test = X_test[indices]
            y_test = y_test[indices]

        test_data = (X_test, y_test)
        print(f"Loaded {len(X_test)} test samples\n")

        # Create feedback loop and establish baseline
        monitor = create_optimization_feedback_loop(
            args.model,
            test_data,
            args.output_dir,
            args.name
        )

        # Additional baseline metadata
        model = tf.keras.models.load_model(args.model)

        # Update baseline with additional metadata
        baseline_file = Path(args.output_dir) / 'baselines' / f'{args.name}.json'
        if baseline_file.exists():
            import json
            with open(baseline_file, 'r') as f:
                baseline_data = json.load(f)

            baseline_data['model_description'] = args.description
            baseline_data['model_path'] = args.model
            baseline_data['model_params'] = model.count_params()
            baseline_data['test_samples_used'] = len(X_test)

            with open(baseline_file, 'w') as f:
                json.dump(baseline_data, f, indent=2, default=str)

        print("="*60)
        print("✓ BASELINE ESTABLISHMENT COMPLETE")
        print("="*60)
        print(f"Baseline saved to: {baseline_file}")
        print("\nNext steps:")
        print("1. Use this baseline for future optimization evaluations")
        print("2. Run optimization candidates with evaluate-candidate mode")
        print("3. Monitor performance trends with generate-report mode")

    except Exception as e:
        print(f"Error establishing baseline: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()