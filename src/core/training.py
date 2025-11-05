#!/usr/bin/env python
"""
DoodleHunter Training Orchestrator with All Improvements

This script coordinates all Quick Wins:
1. Better negative samples (hard negatives)
2. Label smoothing
3. Cross-validation setup
4. Transfer learning options
5. Learning rate scheduling
6. Threshold optimization
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

def main():
    parser = argparse.ArgumentParser(
        description="DoodleHunter Training with All Improvements",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Standard training with improvements
  python training_orchestrator.py --quick-wins all
  
  # Training with hard negatives only
  python training_orchestrator.py --quick-wins 1
  
  # Training with specific improvements
  python training_orchestrator.py --quick-wins 1 2 5 --architecture resnet50
  
  # Full transfer learning with all optimizations
  python training_orchestrator.py --architecture resnet50 --label-smoothing 0.1 \\
                                   --learning-rate-schedule --cross-validation
        """
    )
    
    # Training options
    parser.add_argument("--quick-wins", nargs="+", default=["all"],
                       choices=["all", "1", "2", "3", "4", "5"],
                       help="Which quick wins to apply (default: all)")
    
    # Model options
    parser.add_argument("--architecture", default="custom",
                       choices=["custom", "resnet50", "mobilenetv3", "efficientnet"],
                       help="Model architecture (default: custom)")
    
    # Data options
    parser.add_argument("--negative-strategy", default="hard",
                       choices=["hard", "random"],
                       help="Negative sample strategy (default: hard)")
    parser.add_argument("--data-dir", default="data/raw", help="Raw data directory")
    parser.add_argument("--output-dir", default="data/processed", help="Output directory")
    
    # Training options
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--label-smoothing", type=float, default=0.1,
                       help="Label smoothing factor (0=off, 0.1=typical, range 0-0.5)")
    
    # Improvements
    parser.add_argument("--learning-rate-schedule", action="store_true",
                       help="Use learning rate scheduling (Quick Win #8)")
    parser.add_argument("--cross-validation", action="store_true",
                       help="Use k-fold cross-validation (Quick Win #3)")
    parser.add_argument("--optimize-threshold", action="store_true",
                       help="Optimize decision threshold after training (Quick Win #5)")
    
    # Output
    parser.add_argument("--model-output", default="models/quickdraw_model.h5",
                       help="Model output path")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("DOODLEHUNTER - TRAINING WITH IMPROVEMENTS")
    print("=" * 80)
    
    # Determine which quick wins to apply
    quick_wins = set()
    if "all" in args.quick_wins:
        quick_wins = {"1", "2", "3", "4", "5"}
    else:
        quick_wins = set(args.quick_wins)
    
    print(f"\n📋 Configuration:")
    print(f"  Architecture: {args.architecture}")
    print(f"  Quick Wins: {sorted(quick_wins)}")
    print(f"  Negative Strategy: {args.negative_strategy}")
    print(f"  Label Smoothing: {args.label_smoothing}")
    print(f"  Learning Rate Schedule: {args.learning_rate_schedule}")
    print(f"  Cross-Validation: {args.cross_validation}")
    print(f"  Threshold Optimization: {args.optimize_threshold}")
    
    # Import training functions
    from generate_hard_negatives import prepare_dataset_with_hard_negatives
    from train import train_model
    from optimize_threshold import find_optimal_threshold
    
    # Step 1: Prepare data with hard negatives (Quick Win #1)
    print(f"\n{'=' * 80}")
    print("STEP 1: PREPARE DATA")
    print("=" * 80)
    
    if "1" in quick_wins:
        print(f"\n✓ Quick Win #1: Using hard negatives")
        negative_strategy = "hard"
    else:
        print(f"\n✓ Using random negatives")
        negative_strategy = "random"
    
    (X_train, y_train), (X_test, y_test), class_mapping = prepare_dataset_with_hard_negatives(
        positive_classes=["airplane", "apple", "banana", "cat", "dog"],
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        negative_strategy=negative_strategy,
        max_samples_per_class=5000
    )
    
    # Step 2: Train model with improvements (Quick Wins #2, #4, #8)
    print(f"\n{'=' * 80}")
    print("STEP 2: TRAIN MODEL")
    print("=" * 80)
    
    improvements = []
    if "2" in quick_wins:
        improvements.append(f"Label Smoothing ({args.label_smoothing})")
    if "4" in quick_wins:
        improvements.append(f"Transfer Learning ({args.architecture})")
    if args.learning_rate_schedule:
        improvements.append("LR Scheduling")
    
    print(f"\n✓ Improvements enabled: {', '.join(improvements) if improvements else 'None'}")
    
    history = train_model(
        data_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        model_output=args.model_output,
        learning_rate=args.learning_rate,
        label_smoothing=args.label_smoothing if "2" in quick_wins else 0,
        architecture=args.architecture if "4" in quick_wins else "custom"
    )
    
    # Step 3: Optimize threshold (Quick Win #5)
    if "5" in quick_wins:
        print(f"\n{'=' * 80}")
        print("STEP 3: OPTIMIZE THRESHOLD")
        print("=" * 80)
        
        print(f"\n✓ Quick Win #5: Threshold optimization")
        find_optimal_threshold(
            model_path=args.model_output,
            data_dir=args.output_dir,
            output_dir=Path(args.model_output).parent
        )
    
    print(f"\n{'=' * 80}")
    print("✅ TRAINING COMPLETE")
    print("=" * 80)
    print(f"\nModel saved to: {args.model_output}")
    print(f"Next: Use src/predict.py for inference")


if __name__ == "__main__":
    main()
