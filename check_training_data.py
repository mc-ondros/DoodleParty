"""
Check if training data needs to be regenerated with the fixed penis data.
"""

import numpy as np
import pickle
from pathlib import Path

print("Checking training data status...\n")

# Check if training data exists
X_train_path = Path('data/processed/X_train.npy')
y_train_path = Path('data/processed/y_train.npy')
class_map_path = Path('data/processed/class_mapping.pkl')
penis_data_path = Path('data/processed/penis_raw_X.npy')

if X_train_path.exists():
    X_train = np.load(X_train_path)
    y_train = np.load(y_train_path)
    
    print(f"Existing training data:")
    print(f"  X_train shape: {X_train.shape}")
    print(f"  y_train shape: {y_train.shape}")
    print(f"  Training samples: {len(X_train):,}")
    
    # Check class distribution
    unique, counts = np.unique(y_train, return_counts=True)
    print(f"\nClass distribution in y_train:")
    for cls, count in zip(unique, counts):
        print(f"  Class {cls}: {count:,} samples ({100*count/len(y_train):.1f}%)")
    
    if class_map_path.exists():
        with open(class_map_path, 'rb') as f:
            class_mapping = pickle.load(f)
        print(f"\nClass mapping:")
        for name, idx in class_mapping.items():
            print(f"  {name}: {idx}")

# Check penis data
if penis_data_path.exists():
    penis_data = np.load(penis_data_path)
    print(f"\nFixed penis data:")
    print(f"  Shape: {penis_data.shape}")
    print(f"  Samples: {len(penis_data):,}")

# Check file timestamps
import os
import datetime

print("\n" + "="*60)
print("FILE TIMESTAMPS:")
print("="*60)

files = [
    ('penis_raw_X.npy', penis_data_path),
    ('X_train.npy', X_train_path),
    ('y_train.npy', y_train_path),
]

for name, path in files:
    if path.exists():
        mtime = os.path.getmtime(path)
        dt = datetime.datetime.fromtimestamp(mtime)
        print(f"{name:20s} - {dt.strftime('%Y-%m-%d %H:%M:%S')}")

print("\n" + "="*60)
print("RECOMMENDATION:")
print("="*60)

if X_train_path.exists():
    penis_mtime = os.path.getmtime(penis_data_path)
    train_mtime = os.path.getmtime(X_train_path)
    
    if penis_mtime > train_mtime:
        print("⚠️  Penis data was updated AFTER training data was created!")
        print("    Training data should be regenerated to use fixed penis data.")
        print("\nTo regenerate training data, run:")
        print("  python src/data_pipeline.py")
    else:
        print("✓ Training data is up-to-date with penis data")
        print("  Ready to train!")
else:
    print("⚠️  No training data found. Generate it first:")
    print("  python src/data_pipeline.py")
