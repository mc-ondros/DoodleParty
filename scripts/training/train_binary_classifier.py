#!/usr/bin/env python3
"""
Train binary classifier for penis detection using QuickDraw 28x28 data.

This script trains a CNN to distinguish between offensive (penis) and safe drawings.
Uses the native 28x28 .npy format from QuickDraw dataset.
"""

import argparse
import os
import sys
import numpy as np
import pickle
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
from rich.table import Table
from rich.layout import Layout
from rich import box
from rich.live import Live
from rich.text import Text

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))

from src_py.core.models import build_custom_cnn
from src_py.core.training import create_callbacks, train_model
from src_py.data.augmentation import DataAugmentation

# Initialize Rich console
console = Console()


def load_category(data_dir: str, category: str, max_samples: int = None) -> np.ndarray:
    """Load a single category from .npy file."""
    file_path = os.path.join(data_dir, f"{category}.npy")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Category file not found: {file_path}")
    
    with console.status(f"[cyan]Loading {category}...", spinner="dots"):
        data = np.load(file_path)
        
        if max_samples:
            data = data[:max_samples]
        
        # Normalize to [0, 1] and reshape to (N, 28, 28, 1)
        data = data.astype(np.float32) / 255.0
        data = data.reshape(-1, 28, 28, 1)
    
    console.print(f"  [green]✓[/green] {category}: [bold]{len(data):,}[/bold] samples loaded")
    return data


def prepare_binary_dataset(
    data_dir: str,
    positive_category: str = "penis",
    negative_categories: list = None,
    max_samples_per_category: int = 10000,
    train_split: float = 0.8
):
    """
    Prepare binary classification dataset.
    
    Args:
        data_dir: Directory containing .npy files
        positive_category: Positive class (offensive)
        negative_categories: List of negative classes (safe)
        max_samples_per_category: Max samples to load per category
        train_split: Train/val split ratio
    
    Returns:
        (train_images, train_labels), (val_images, val_labels)
    """
    if negative_categories is None:
        negative_categories = [
            "circle", "line", "square", "triangle", "star",
            "rectangle", "diamond", "heart", "cloud", "moon"
        ]
    
    # Display header
    console.print()
    console.print(Panel.fit(
        "[bold cyan]Binary Classification Dataset Loader[/bold cyan]\n"
        f"[dim]28x28 QuickDraw Format[/dim]",
        border_style="cyan",
        box=box.DOUBLE
    ))
    
    # Configuration table
    config_table = Table(show_header=False, box=box.SIMPLE, padding=(0, 2))
    config_table.add_row("[cyan]Positive class:[/cyan]", f"[yellow]{positive_category}[/yellow]")
    config_table.add_row("[cyan]Negative classes:[/cyan]", f"[yellow]{len(negative_categories)}[/yellow] categories")
    config_table.add_row("[cyan]Max samples/category:[/cyan]", f"[yellow]{max_samples_per_category:,}[/yellow]")
    config_table.add_row("[cyan]Train/Val split:[/cyan]", f"[yellow]{train_split:.0%} / {(1-train_split):.0%}[/yellow]")
    console.print(config_table)
    console.print()
    
    # Load positive samples (label = 1)
    console.print("[bold cyan]Loading positive samples:[/bold cyan]")
    positive_data = load_category(data_dir, positive_category, max_samples_per_category)
    positive_labels = np.ones(len(positive_data), dtype=np.float32)
    
    # Load negative samples (label = 0)
    console.print("\n[bold cyan]Loading negative samples:[/bold cyan]")
    negative_data_list = []
    for category in negative_categories:
        try:
            data = load_category(data_dir, category, max_samples_per_category)
            negative_data_list.append(data)
        except FileNotFoundError as e:
            console.print(f"  [yellow]⚠[/yellow] Warning: {e}")
            continue
    
    if not negative_data_list:
        raise ValueError("No negative samples loaded!")
    
    negative_data = np.concatenate(negative_data_list, axis=0)
    negative_labels = np.zeros(len(negative_data), dtype=np.float32)
    
    # Combine and shuffle
    console.print("\n[bold cyan]Dataset Summary:[/bold cyan]")
    
    summary_table = Table(box=box.ROUNDED, show_header=True, header_style="bold magenta")
    summary_table.add_column("Class", style="cyan")
    summary_table.add_column("Samples", justify="right", style="yellow")
    summary_table.add_row("Positive (offensive)", f"{len(positive_data):,}")
    summary_table.add_row("Negative (safe)", f"{len(negative_data):,}")
    console.print(summary_table)
    
    with console.status("[cyan]Combining and shuffling datasets...", spinner="dots"):
        all_data = np.concatenate([positive_data, negative_data], axis=0)
        all_labels = np.concatenate([positive_labels, negative_labels], axis=0)
        
        # Shuffle
        indices = np.random.permutation(len(all_data))
        all_data = all_data[indices]
        all_labels = all_labels[indices]
        
        # Split train/val
        split_idx = int(len(all_data) * train_split)
        train_images = all_data[:split_idx]
        train_labels = all_labels[:split_idx]
        val_images = all_data[split_idx:]
        val_labels = all_labels[split_idx:]
    
    console.print("\n[bold cyan]Data Split:[/bold cyan]")
    split_table = Table(box=box.ROUNDED, show_header=True, header_style="bold magenta")
    split_table.add_column("Split", style="cyan")
    split_table.add_column("Samples", justify="right", style="yellow")
    split_table.add_column("Balance", justify="right", style="green")
    split_table.add_row("Training", f"{len(train_images):,}", f"{train_labels.mean():.1%} positive")
    split_table.add_row("Validation", f"{len(val_images):,}", f"{val_labels.mean():.1%} positive")
    split_table.add_row("[bold]Total[/bold]", f"[bold]{len(all_data):,}[/bold]", f"[bold]{all_labels.mean():.1%} positive[/bold]")
    console.print(split_table)
    
    return (train_images, train_labels), (val_images, val_labels)


def main():
    parser = argparse.ArgumentParser(
        description='Train binary classifier for penis detection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with default settings
  python train_binary_classifier.py --data-dir data/raw
  
  # Train with custom parameters
  python train_binary_classifier.py --data-dir data/raw --epochs 50 --batch-size 64
  
  # Train with more negative categories
  python train_binary_classifier.py --data-dir data/raw --max-samples 20000
        """
    )
    
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data/raw',
        help='Directory containing QuickDraw .npy files'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='models',
        help='Output directory for trained model'
    )
    
    parser.add_argument(
        '--model-name',
        type=str,
        default='quickdraw_binary_28x28',
        help='Model name (without extension)'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=30,
        help='Number of training epochs'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for training'
    )
    
    parser.add_argument(
        '--max-samples',
        type=int,
        default=10000,
        help='Max samples per category'
    )
    
    parser.add_argument(
        '--train-split',
        type=float,
        default=0.8,
        help='Train/validation split ratio'
    )
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load dataset
    (train_images, train_labels), (val_images, val_labels) = prepare_binary_dataset(
        args.data_dir,
        max_samples_per_category=args.max_samples,
        train_split=args.train_split
    )
    
    # Build model
    console.print()
    console.print(Panel.fit(
        "[bold cyan]Model Architecture[/bold cyan]\n"
        f"[dim]Input: 28x28x1 | Output: Binary Classification[/dim]",
        border_style="cyan",
        box=box.DOUBLE
    ))
    
    with console.status("[cyan]Building model architecture...", spinner="dots"):
        model = build_custom_cnn(input_shape=(28, 28, 1), num_classes=1)
    
    console.print("\n[bold cyan]Model Summary:[/bold cyan]")
    model.summary()
    
    total_params = model.count_params()
    console.print(f"\n[bold green]✓[/bold green] Total parameters: [bold yellow]{total_params:,}[/bold yellow]")
    
    # Create callbacks
    callbacks = create_callbacks(args.model_name, checkpoint_dir=args.output_dir + '/', epochs=args.epochs)
    
    # Train model
    console.print()
    console.print(Panel.fit(
        f"[bold cyan]Training Session[/bold cyan]\n"
        f"[yellow]Epochs:[/yellow] {args.epochs} | [yellow]Batch Size:[/yellow] {args.batch_size}",
        border_style="green",
        box=box.DOUBLE
    ))
    
    history = train_model(
        model,
        (train_images, train_labels),
        (val_images, val_labels),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks
    )
    
    # Save final model
    final_model_path = os.path.join(args.output_dir, f"{args.model_name}.h5")
    
    with console.status("[cyan]Saving final model...", spinner="dots"):
        model.save(final_model_path)
    
    console.print()
    console.print(Panel.fit(
        "[bold green]✓ Training Complete![/bold green]",
        border_style="green",
        box=box.DOUBLE
    ))
    
    console.print("\n[bold cyan]Model Artifacts:[/bold cyan]")
    artifacts_table = Table(box=box.ROUNDED, show_header=False)
    artifacts_table.add_row("[cyan]Final model:[/cyan]", f"[yellow]{final_model_path}[/yellow]")
    artifacts_table.add_row("[cyan]Best model:[/cyan]", f"[yellow]{args.output_dir}/{args.model_name}_best.h5[/yellow]")
    console.print(artifacts_table)
    
    # Final metrics
    final_loss = history.history['loss'][-1]
    final_acc = history.history['accuracy'][-1]
    val_loss = history.history['val_loss'][-1]
    val_acc = history.history['val_accuracy'][-1]
    
    console.print("\n[bold cyan]Final Metrics:[/bold cyan]")
    metrics_table = Table(box=box.ROUNDED, show_header=True, header_style="bold magenta")
    metrics_table.add_column("Metric", style="cyan")
    metrics_table.add_column("Training", justify="right", style="yellow")
    metrics_table.add_column("Validation", justify="right", style="green")
    metrics_table.add_row("Loss", f"{final_loss:.4f}", f"{val_loss:.4f}")
    metrics_table.add_row("Accuracy", f"{final_acc:.2%}", f"{val_acc:.2%}")
    console.print(metrics_table)
    
    # Check if model meets targets
    console.print("\n[bold cyan]Target Validation:[/bold cyan]")
    if val_acc >= 0.90:
        console.print(f"  [bold green]✓[/bold green] Accuracy target met: [green]{val_acc:.1%}[/green] >= [dim]90%[/dim]")
    else:
        console.print(f"  [bold red]✗[/bold red] Accuracy target not met: [red]{val_acc:.1%}[/red] < [dim]90%[/dim]")
    
    console.print("\n[bold cyan]Next Steps:[/bold cyan]")
    steps_table = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
    steps_table.add_row("[cyan]1.[/cyan]", f"Evaluate model: [dim]python scripts/evaluation/evaluate.py --model {final_model_path}[/dim]")
    steps_table.add_row("[cyan]2.[/cyan]", f"Convert to TFLite: [dim]python scripts/optimization/convert_to_tflite.py --model {final_model_path}[/dim]")
    console.print(steps_table)
    console.print()
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
