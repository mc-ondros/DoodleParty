#!/usr/bin/env python3
"""
CLI for model training.
"""

import argparse
import sys
import os
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))

from src_py.core.models import build_custom_cnn
from src_py.core.training import create_callbacks, train_model
from src_py.data.loaders import QuickDrawLoader

# Initialize Rich console
console = Console()


def main():
    parser = argparse.ArgumentParser(description='Train DoodleParty model')
    parser.add_argument('--data-dir', type=str, default='data/raw', help='Directory with training data')
    parser.add_argument('--output', type=str, default='models/model.h5', help='Output model path')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--model', type=str, default='custom_cnn', help='Model architecture')
    parser.add_argument('--test-split', type=float, default=0.2, help='Test/validation split ratio')
    parser.add_argument('--limit-samples', type=int, default=None, help='Limit samples per category')
    
    args = parser.parse_args()
    
    console.print()
    console.print(Panel.fit(
        "[bold cyan]DoodleParty Model Training[/bold cyan]\n"
        f"[dim]Multi-Class Classifier Training[/dim]",
        border_style="cyan",
        box=box.DOUBLE
    ))
    
    # Configuration table
    config_table = Table(show_header=False, box=box.SIMPLE, padding=(0, 2))
    config_table.add_row("[cyan]Model architecture:[/cyan]", f"[yellow]{args.model}[/yellow]")
    config_table.add_row("[cyan]Data directory:[/cyan]", f"[yellow]{args.data_dir}[/yellow]")
    config_table.add_row("[cyan]Output path:[/cyan]", f"[yellow]{args.output}[/yellow]")
    config_table.add_row("[cyan]Epochs:[/cyan]", f"[yellow]{args.epochs}[/yellow]")
    config_table.add_row("[cyan]Batch size:[/cyan]", f"[yellow]{args.batch_size}[/yellow]")
    console.print(config_table)
    console.print()
    
    # Load training data
    console.print("[cyan]📂 Loading training data...[/cyan]")
    
    # Get available categories
    import glob
    npy_files = glob.glob(os.path.join(args.data_dir, '*.npy'))
    if not npy_files:
        console.print("[red]Error:[/red] No .npy files found in", args.data_dir)
        return 1
    
    categories = [os.path.basename(f)[:-4] for f in npy_files]  # Remove .npy extension
    console.print(f"[green]✓[/green] Found {len(categories)} categories: {', '.join(categories[:5])}{'...' if len(categories) > 5 else ''}")
    console.print()
    
    # Load and split data
    with console.status("[cyan]Loading and preparing data...", spinner="dots"):
        (train_images, train_labels), (val_images, val_labels) = QuickDrawLoader.load_quickdraw_split(
            args.data_dir,
            categories,
            train_split=1.0 - args.test_split
        )
    
    console.print(f"[green]✓[/green] Data loaded")
    console.print(f"  Training samples: {len(train_images)}")
    console.print(f"  Validation samples: {len(val_images)}")
    console.print()
    
    # Build model
    with console.status("[cyan]Building model architecture...", spinner="dots"):
        if args.model == 'custom_cnn':
            model = build_custom_cnn(input_shape=(28, 28, 1), num_classes=len(categories))
        else:
            console.print(f"[red]Error:[/red] Unknown model: {args.model}")
            return 1
    
    console.print(f"[green]✓[/green] Model built: [bold]{args.model}[/bold]")
    
    # Create output directory
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    
    # Create callbacks
    callbacks = create_callbacks('doodleparty', checkpoint_dir=os.path.dirname(args.output) or 'models/', epochs=args.epochs)
    
    console.print()
    console.print(Panel.fit(
        f"[bold cyan]Training Session[/bold cyan]\n"
        f"[yellow]Starting training for {args.epochs} epochs[/yellow]\n"
        f"[dim]Batch size: {args.batch_size}[/dim]",
        border_style="green",
        box=box.DOUBLE
    ))
    console.print()
    
    # Train model
    history = train_model(
        model,
        (train_images, train_labels),
        (val_images, val_labels),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks
    )
    
    # Save model
    console.print()
    console.print("[cyan]💾 Saving model...[/cyan]")
    model.save(args.output)
    console.print(f"[green]✓[/green] Model saved to: [bold]{args.output}[/bold]")
    
    console.print()
    console.print(Panel.fit(
        f"[bold green]✓ Training Complete![/bold green]\n"
        f"[green]Model:[/green] {args.output}\n"
        f"[green]Epochs:[/green] {args.epochs}",
        border_style="green",
        box=box.DOUBLE
    ))


if __name__ == '__main__':
    sys.exit(main())
