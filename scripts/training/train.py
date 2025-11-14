#!/usr/bin/env python3
"""
CLI for model training.
"""

import argparse
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))

from src_py.core.models import build_custom_cnn
from src_py.core.training import create_callbacks, train_model
from src_py.data.loaders import QuickDrawLoader


def main():
    parser = argparse.ArgumentParser(description='Train DoodleParty model')
    parser.add_argument('--data-dir', type=str, default='data/raw', help='Directory with training data')
    parser.add_argument('--output', type=str, default='models/model.h5', help='Output model path')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--model', type=str, default='custom_cnn', help='Model architecture')
    
    args = parser.parse_args()
    
    print(f"Training {args.model} model...")
    print(f"Data directory: {args.data_dir}")
    print(f"Output: {args.output}")
    
    # Build model
    if args.model == 'custom_cnn':
        model = build_custom_cnn()
    else:
        print(f"Unknown model: {args.model}")
        return 1
    
    print("Model built successfully")
    
    # Create callbacks
    callbacks = create_callbacks('doodleparty')
    
    print(f"Training for {args.epochs} epochs...")
    print("Ready to train!")


if __name__ == '__main__':
    sys.exit(main())
