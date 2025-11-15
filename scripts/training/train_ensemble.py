"""
Train Multiple Models and Create Ensemble

Script to train multiple model architectures with different configurations
and combine them into an ensemble for improved accuracy.

Usage:
    python scripts/train_ensemble.py --output-dir models/ensemble_test
"""

import argparse
import subprocess
from pathlib import Path
import sys


def run_command(cmd, description):
    """Run command and handle errors."""
    print(f"\n{'='*70}")
    print(f"{description}")
    print(f"{'='*70}")
    print(f"Running: {cmd}")
    
    result = subprocess.run(cmd, shell=True, capture_output=False)
    
    if result.returncode != 0:
        print(f"✗ Error: {description}")
        return False
    else:
        print(f"✓ Success: {description}")
        return True


def main():
    parser = argparse.ArgumentParser(description = 'Train ensemble of models')
    parser.add_argument("--data-dir", default = 'data/processed',
                       help = 'Directory with processed data')
    parser.add_argument("--output-dir", default = 'models/ensemble_models',
                       help = 'Directory to save ensemble models')
    parser.add_argument("--epochs", type=int, default=30,
                       help = 'Number of epochs per model')
    parser.add_argument("--batch-size", type=int, default=32,
                       help = 'Batch size')
    parser.add_argument("--use-class-weighting", action = 'store_true',
                       help = 'Use class weighting')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*70)
    print('ENSEMBLE TRAINING PIPELINE')
    print("="*70)
    print(f"\nOutput directory: {output_dir}")
    print(f"Training {args.epochs} epochs per model")
    print(f"Batch size: {args.batch_size}")
    print(f"Class weighting: {args.use_class_weighting}")
    
    # Model configurations to train
    models = [
        {
            'name': 'custom_standard',
            'architecture': 'custom',
            'enhanced': False,
            'aggressive_aug': False,
            'output': output_dir / 'model_custom_standard.h5'
        },
        {
            'name': 'custom_enhanced',
            'architecture': 'custom',
            'enhanced': True,
            'aggressive_aug': False,
            'output': output_dir / 'model_custom_enhanced.h5'
        },
        {
            'name': 'custom_enhanced_aug',
            'architecture': 'custom',
            'enhanced': True,
            'aggressive_aug': True,
            'output': output_dir / 'model_custom_enhanced_aug.h5'
        }
    ]
    
    # Train each model
    for i, model_config in enumerate(models, 1):
        print(f"\n\nTraining Model {i}/{len(models)}: {model_config['name']}")
        print("-" * 70)
        
        cmd = f"python scripts/train.py"
        cmd += f" --data-dir {args.data_dir}"
        cmd += f" --epochs {args.epochs}"
        cmd += f" --batch-size {args.batch_size}"
        cmd += f" --architecture {model_config['architecture']}"
        cmd += f" --model-output {model_config['output']}"
        
        if model_config['enhanced']:
            cmd += " --enhanced"
        
        if model_config['aggressive_aug']:
            cmd += " --aggressive-aug"
        
        if args.use_class_weighting:
            cmd += " --use-class-weighting"
        
        success = run_command(cmd, f"Train {model_config['name']}")
        
        if not success:
            print(f"✗ Failed to train {model_config['name']}")
            sys.exit(1)
    
    # Create ensemble from trained models
    print(f"\n\n{'='*70}")
    print('CREATING ENSEMBLE')
    print(f"{'='*70}")
    
    model_paths = [str(m['output']) for m in models]
    model_paths_str = ' '.join(model_paths)
    
    cmd = f"python scripts/ensemble_model.py"
    cmd += f" --models {model_paths_str}"
    cmd += f" --data-dir {args.data_dir}"
    cmd += f" --method weighted"
    cmd += f" --output {output_dir}/ensemble_config.pkl"
    
    success = run_command(cmd, "Create ensemble")
    
    if not success:
        print('✗ Failed to create ensemble')
        sys.exit(1)
    
    print("\n" + "="*70)
    print('✓ ENSEMBLE TRAINING COMPLETE!')
    print("="*70)
    print(f"\nModels trained: {len(models)}")
    print(f"Ensemble configuration: {output_dir}/ensemble_config.pkl")
    print('\nTo use the ensemble:')
    print(f"  python scripts/ensemble_model.py --models {model_paths_str}")


if __name__ == '__main__':
    main()
