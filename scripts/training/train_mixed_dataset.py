#!/usr/bin/env python3
"""
Train binary classifier using mixed dataset from QuickDraw .npy (28x28) and Appendix (128x128).

This script automatically downscales the 128x128 appendix images to 28x28 to match
the model input size, allowing you to leverage both datasets seamlessly.
"""

import argparse
import os
import sys
import numpy as np
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
from src_py.data.loaders import QuickDrawAppendixLoader

# Initialize Rich console
console = Console()


def main():
    parser = argparse.ArgumentParser(
        description='Train binary classifier with mixed QuickDraw + Appendix dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with default settings
  python train_mixed_dataset.py --npy-dir data/raw --appendix-dir data/appendix
  
  # Train with custom parameters
  python train_mixed_dataset.py \
      --npy-dir data/raw \
      --appendix-dir data/appendix \
      --epochs 50 \
      --batch-size 64 \
      --max-npy-samples 15000 \
      --max-appendix-samples 10000
        """
    )
    
    parser.add_argument(
        '--npy-dir',
        type=str,
        default='data/raw',
        help='Directory containing QuickDraw .npy files (28x28)'
    )
    
    parser.add_argument(
        '--appendix-dir',
        type=str,
        default='data/appendix',
        help='Directory containing QuickDraw Appendix images (128x128)'
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
        default='quickdraw_mixed_28x28',
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
        '--max-npy-samples',
        type=int,
        default=10000,
        help='Max samples per category from .npy files'
    )
    
    parser.add_argument(
        '--max-appendix-samples',
        type=int,
        default=5000,
        help='Max samples from appendix dataset'
    )
    
    parser.add_argument(
        '--train-split',
        type=float,
        default=0.8,
        help='Train/validation split ratio'
    )
    
    parser.add_argument(
        '--positive-category',
        type=str,
        default='penis',
        help='Positive class category name'
    )
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Display header
    console.print()
    console.print(Panel.fit(
        "[bold cyan]Mixed Dataset Training[/bold cyan]\n"
        f"[dim]QuickDraw .npy (28x28) + Appendix (128x128→28x28)[/dim]",
        border_style="cyan",
        box=box.DOUBLE
    ))
    
    # Configuration table
    config_table = Table(show_header=False, box=box.SIMPLE, padding=(0, 2))
    config_table.add_row("[cyan]QuickDraw .npy dir:[/cyan]", f"[yellow]{args.npy_dir}[/yellow]")
    config_table.add_row("[cyan]Appendix dir:[/cyan]", f"[yellow]{args.appendix_dir}[/yellow]")
    config_table.add_row("[cyan]Positive category:[/cyan]", f"[yellow]{args.positive_category}[/yellow]")
    config_table.add_row("[cyan]Max .npy samples:[/cyan]", f"[yellow]{args.max_npy_samples:,}[/yellow]")
    config_table.add_row("[cyan]Max appendix samples:[/cyan]", f"[yellow]{args.max_appendix_samples:,}[/yellow]")
    config_table.add_row("[cyan]Train/Val split:[/cyan]", f"[yellow]{args.train_split:.0%} / {(1-args.train_split):.0%}[/yellow]")
    console.print(config_table)
    console.print()
    
    # Load mixed dataset
    console.print("[bold cyan]Loading Mixed Dataset:[/bold cyan]")
    console.print("[dim]Appendix images will be automatically downscaled from 128x128 to 28x28[/dim]\n")
    
    with console.status("[cyan]Loading datasets...", spinner="dots"):
        try:
            (train_images, train_labels), (val_images, val_labels) = \
                QuickDrawAppendixLoader.load_mixed_dataset(
                    npy_data_dir=args.npy_dir,
                    appendix_data_dir=args.appendix_dir,
                    positive_category=args.positive_category,
                    max_npy_samples=args.max_npy_samples,
                    max_appendix_samples=args.max_appendix_samples,
                    train_split=args.train_split
                )
        except Exception as e:
            console.print(f"[red]✗[/red] Error loading datasets: {e}")
            return 1
    
    console.print("[green]✓[/green] Datasets loaded successfully\n")
    
    # Dataset summary
    console.print("[bold cyan]Dataset Summary:[/bold cyan]")
    summary_table = Table(box=box.ROUNDED, show_header=True, header_style="bold magenta")
    summary_table.add_column("Split", style="cyan")
    summary_table.add_column("Samples", justify="right", style="yellow")
    summary_table.add_column("Positive", justify="right", style="green")
    summary_table.add_column("Negative", justify="right", style="blue")
    summary_table.add_column("Balance", justify="right", style="white")
    
    train_pos = np.sum(train_labels)
    train_neg = len(train_labels) - train_pos
    val_pos = np.sum(val_labels)
    val_neg = len(val_labels) - val_pos
    
    summary_table.add_row(
        "Training",
        f"{len(train_images):,}",
        f"{int(train_pos):,}",
        f"{int(train_neg):,}",
        f"{train_labels.mean():.1%}"
    )
    summary_table.add_row(
        "Validation",
        f"{len(val_images):,}",
        f"{int(val_pos):,}",
        f"{int(val_neg):,}",
        f"{val_labels.mean():.1%}"
    )
    summary_table.add_row(
        "[bold]Total[/bold]",
        f"[bold]{len(train_images) + len(val_images):,}[/bold]",
        f"[bold]{int(train_pos + val_pos):,}[/bold]",
        f"[bold]{int(train_neg + val_neg):,}[/bold]",
        f"[bold]{np.concatenate([train_labels, val_labels]).mean():.1%}[/bold]"
    )
    console.print(summary_table)
    
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
    
    # Show data source breakdown
    console.print("\n[bold cyan]Training Data Sources:[/bold cyan]")
    console.print(f"  [green]✓[/green] QuickDraw .npy (28x28 native)")
    console.print(f"  [green]✓[/green] QuickDraw Appendix (128x128 → 28x28 downscaled)")
    
    console.print("\n[bold cyan]Next Steps:[/bold cyan]")
    steps_table = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
    steps_table.add_row("[cyan]1.[/cyan]", f"Evaluate model: [dim]python scripts/evaluation/evaluate.py --model {final_model_path}[/dim]")
    steps_table.add_row("[cyan]2.[/cyan]", f"Convert to TFLite: [dim]python scripts/optimization/convert_to_tflite.py --model {final_model_path}[/dim]")
    steps_table.add_row("[cyan]3.[/cyan]", f"Test inference: [dim]python scripts/inference/test_inference.py --model {final_model_path}[/dim]")
    console.print(steps_table)
    console.print()
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
