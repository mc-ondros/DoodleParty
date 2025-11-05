"""
Dataset loading and preprocessing for QuickDraw data.
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import requests
from tqdm import tqdm
import argparse
import pickle
import gzip
import io


class QuickDrawDataset:
    """Handles QuickDraw dataset loading and preprocessing."""
    
    # GCS public bucket with numpy bitmap format (smaller, pre-processed)
    BASE_URL = 'https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap'
    
    def __init__(self, data_dir = 'data/raw'):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
    def download_class(self, class_name):
        """
        Download a single class from QuickDraw dataset (numpy bitmap format).
        Format: Pre-processed 28x28 grayscale bitmap images
        Size: ~100-200MB per category (much smaller than ndjson)
        """
        url = f"{self.BASE_URL}/{class_name}.npy"
        filepath = self.data_dir / f"{class_name}.npy"
        
        if filepath.exists():
            file_size = filepath.stat().st_size / 1024 / 1024
            print(f"✓ {class_name} already downloaded ({file_size:.1f}MB)")
            return
        
        print(f"⬇ Downloading {class_name} from GCS...")
        print(f"   URL: {url}")
        try:
            response = requests.get(url, stream=True, timeout=60)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            print(f"   File size: {total_size / 1024 / 1024:.1f}MB")
            
            with open(filepath, 'wb') as f:
                with tqdm(total=total_size, unit='B', unit_scale=True, desc=class_name) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
            print(f"✓ {class_name} downloaded successfully")
        except Exception as e:
            print(f"✗ Error downloading {class_name}: {e}")
            if filepath.exists():
                filepath.unlink()  # Clean up partial download
    
    def load_class_data(self, class_name, max_samples=None, normalize=True):
        """Load data for a single class."""
        filepath = self.data_dir / f"{class_name}.npy"
        
        if not filepath.exists():
            raise FileNotFoundError(f"Data file not found: {filepath}")
        
        print(f"Loading {class_name}...")
        data = np.load(filepath)
        
        # Limit samples if specified
        if max_samples is not None:
            data = data[:max_samples]
        
        # Normalize to 0-1 range
        if normalize:
            data = data.astype(np.float32) / 255.0
        
        return data
    
    def prepare_dataset(self, classes, output_dir = 'data/processed', 
                       max_samples_per_class=None, test_split=0.2):
        """
        Prepare binary classification dataset.
        Positive class: doodles from specified classes
        Negative class: random noise/other doodles
        
        Args:
            classes: List of class names for positive examples
            output_dir: Directory to save processed data
            max_samples_per_class: Maximum samples per class (None for all)
            test_split: Fraction of data to use for testing
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        all_images = []
        all_labels = []
        
        print(f"\nPreparing binary dataset with {len(classes)} positive classes...")
        
        # Load positive examples (in-distribution)
        print('\nLoading positive examples (in-distribution)...')
        for class_name in classes:
            try:
                images = self.load_class_data(class_name, max_samples_per_class)
                labels = np.ones(len(images), dtype=np.int32)  # Label 1 for in-distribution
                
                all_images.append(images)
                all_labels.append(labels)
                print(f"  ✓ {class_name}: {len(images)} positive samples")
            except Exception as e:
                print(f"  ✗ Error loading {class_name}: {e}")
        
        # Generate negative examples (out-of-distribution random noise)
        print('\nGenerating negative examples (out-of-distribution)...')
        num_positive = sum(len(img) for img in all_images)
        negative_images = np.random.randint(0, 256, (num_positive, 28, 28), dtype=np.uint8)
        negative_labels = np.zeros(num_positive, dtype=np.int32)  # Label 0 for out-of-distribution
        
        all_images.append(negative_images.astype(np.float32) / 255.0)
        all_labels.append(negative_labels)
        print(f"  ✓ Random noise: {num_positive} negative samples")
        
        # Combine all data
        X = np.concatenate(all_images, axis=0)
        y = np.concatenate(all_labels, axis=0)
        
        # Reshape to add channel dimension if needed
        if X.ndim == 3:
            X = X.reshape(-1, 28, 28, 1)
        
        X = X.astype(np.float32)
        if X.max() > 1.0:
            X = X / 255.0
        
        # Shuffle data
        indices = np.random.permutation(len(X))
        X = X[indices]
        y = y[indices]
        
        # Split into train and test
        total_samples = len(X)
        test_size = int(total_samples * test_split)
        train_size = total_samples - test_size
        
        X_train, y_train = X[:train_size], y[:train_size]
        X_test, y_test = X[train_size:], y[train_size:]
        
        # Save datasets
        np.save(output_dir / "X_train.npy", X_train)
        np.save(output_dir / "y_train.npy", y_train)
        np.save(output_dir / "X_test.npy", X_test)
        np.save(output_dir / "y_test.npy", y_test)
        
        # Save class mapping (binary: 0=negative, 1=positive)
        class_mapping = {'negative': 0, 'positive': 1, 'positive_classes': classes}
        with open(output_dir / "class_mapping.pkl", 'wb') as f:
            pickle.dump(class_mapping, f)
        
        print(f"\n✓ Binary dataset prepared successfully!")
        print(f"  Training samples: {len(X_train)} (positive: {(y_train==1).sum()}, negative: {(y_train==0).sum()})")
        print(f"  Test samples: {len(X_test)} (positive: {(y_test==1).sum()}, negative: {(y_test==0).sum()})")
        print(f"  Positive classes: {classes}")
        
        return (X_train, y_train), (X_test, y_test), class_mapping


def main():
    parser = argparse.ArgumentParser(description = 'Download and prepare QuickDraw dataset')
    parser.add_argument("--download", action = 'store_true', help = 'Download raw data')
    parser.add_argument("--classes", nargs = '+', default=["airplane", "apple", "banana", "cat", "dog"],
                       help = 'Classes to download')
    parser.add_argument("--raw-dir", default = 'data/raw', help = 'Directory for raw data')
    parser.add_argument("--output-dir", default = 'data/processed', help = 'Directory for processed data')
    parser.add_argument("--max-samples", type=int, default=5000, help = 'Max samples per class')
    parser.add_argument("--test-split", type=float, default=0.2, help = 'Test set fraction')
    
    args = parser.parse_args()
    
    dataset = QuickDrawDataset(data_dir=args.raw_dir)
    
    if args.download:
        print(f"Downloading {len(args.classes)} classes...")
        for class_name in args.classes:
            dataset.download_class(class_name)
    
    dataset.prepare_dataset(
        args.classes,
        output_dir=args.output_dir,
        max_samples_per_class=args.max_samples,
        test_split=args.test_split
    )


if __name__ == "__main__":
    main()
