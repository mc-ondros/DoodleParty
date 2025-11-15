#!/usr/bin/env python3
"""
Demo script showcasing the Rich TUI features used in DoodleParty.

This script demonstrates all the visual elements without actually running
expensive operations like training or downloading.
"""

import time
import argparse
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, SpinnerColumn
from rich.table import Table
from rich import box
from rich.layout import Layout
from rich.text import Text

console = Console()


def demo_panels():
    """Demonstrate styled panels."""
    console.print("\n[bold yellow]═══ Panel Styles ═══[/bold yellow]\n")
    
    console.print(Panel.fit(
        "[bold cyan]Training Session[/bold cyan]\n"
        "[dim]This is a double-border panel[/dim]",
        border_style="cyan",
        box=box.DOUBLE
    ))
    
    console.print()
    
    console.print(Panel.fit(
        "[bold green]✓ Operation Complete![/bold green]\n"
        "[dim]This is a rounded-border panel[/dim]",
        border_style="green",
        box=box.ROUNDED
    ))


def demo_tables():
    """Demonstrate formatted tables."""
    console.print("\n[bold yellow]═══ Table Styles ═══[/bold yellow]\n")
    
    # Configuration table
    config_table = Table(show_header=False, box=box.SIMPLE, padding=(0, 2))
    config_table.add_row("[cyan]Model:[/cyan]", "[yellow]custom_cnn[/yellow]")
    config_table.add_row("[cyan]Epochs:[/cyan]", "[yellow]30[/yellow]")
    config_table.add_row("[cyan]Batch size:[/cyan]", "[yellow]32[/yellow]")
    console.print(config_table)
    
    console.print()
    
    # Metrics table
    metrics_table = Table(box=box.ROUNDED, show_header=True, header_style="bold magenta")
    metrics_table.add_column("Metric", style="cyan")
    metrics_table.add_column("Training", justify="right", style="yellow")
    metrics_table.add_column("Validation", justify="right", style="green")
    metrics_table.add_row("Loss", "0.1234", "0.1456")
    metrics_table.add_row("Accuracy", "95.67%", "94.23%")
    console.print(metrics_table)


def demo_progress():
    """Demonstrate progress bars."""
    console.print("\n[bold yellow]═══ Progress Bars ═══[/bold yellow]\n")
    
    with Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=40),
        "[progress.percentage]{task.percentage:>3.0f}%",
        TimeElapsedColumn(),
        console=console
    ) as progress:
        
        # Simulate multiple tasks
        task1 = progress.add_task("[cyan]Loading dataset...", total=100)
        task2 = progress.add_task("[cyan]Processing images...", total=100)
        task3 = progress.add_task("[cyan]Building model...", total=100)
        
        for i in range(100):
            time.sleep(0.02)
            progress.update(task1, advance=1)
            if i > 30:
                progress.update(task2, advance=1)
            if i > 60:
                progress.update(task3, advance=1)


def demo_spinners():
    """Demonstrate spinner animations."""
    console.print("\n[bold yellow]═══ Spinners ═══[/bold yellow]\n")
    
    tasks = [
        ("Loading configuration...", 0.8),
        ("Downloading weights...", 1.2),
        ("Initializing model...", 0.6),
        ("Compiling...", 0.9)
    ]
    
    for task_name, duration in tasks:
        with console.status(f"[cyan]{task_name}", spinner="dots"):
            time.sleep(duration)
        console.print(f"  [green]✓[/green] {task_name.replace('...', '')} [dim]complete[/dim]")


def demo_training_output():
    """Simulate training epoch output."""
    console.print("\n[bold yellow]═══ Training Progress ═══[/bold yellow]\n")
    
    console.print(Panel.fit(
        "[bold cyan]Training Session Started[/bold cyan]\n"
        "[yellow]Epochs:[/yellow] 5 | [yellow]Batch Size:[/yellow] 32",
        border_style="green",
        box=box.DOUBLE
    ))
    
    console.print()
    
    import random
    for epoch in range(1, 6):
        loss = 0.5 - (epoch * 0.08) + random.uniform(-0.02, 0.02)
        acc = 0.75 + (epoch * 0.04) + random.uniform(-0.01, 0.01)
        val_loss = loss + random.uniform(0.01, 0.05)
        val_acc = acc - random.uniform(0.01, 0.03)
        
        metrics_text = (
            f"[bold cyan]Epoch {epoch}/5[/bold cyan] | "
            f"[yellow]loss:[/yellow] {loss:.4f} | "
            f"[yellow]acc:[/yellow] {acc:.4f} | "
            f"[green]val_loss:[/green] {val_loss:.4f} | "
            f"[green]val_acc:[/green] {val_acc:.4f}"
        )
        console.print(metrics_text)
        time.sleep(0.3)
    
    console.print()
    console.print("[bold green]✓[/bold green] Training session completed!")


def demo_summary():
    """Show a complete summary."""
    console.print("\n[bold yellow]═══ Complete Summary ═══[/bold yellow]\n")
    
    console.print(Panel.fit(
        "[bold green]✓ All Operations Complete![/bold green]\n"
        "[green]Success:[/green] 100% | [red]Failed:[/red] 0%",
        border_style="green",
        box=box.DOUBLE
    ))
    
    console.print("\n[bold cyan]Next Steps:[/bold cyan]")
    steps_table = Table(show_header=False, box=box.SIMPLE, padding=(0, 1))
    steps_table.add_row("[cyan]1.[/cyan]", "Evaluate model: [dim]python scripts/evaluation/evaluate.py[/dim]")
    steps_table.add_row("[cyan]2.[/cyan]", "Optimize threshold: [dim]python scripts/evaluation/optimize_threshold.py[/dim]")
    steps_table.add_row("[cyan]3.[/cyan]", "Convert to TFLite: [dim]python scripts/optimization/convert_to_tflite.py[/dim]")
    console.print(steps_table)


def main():
    parser = argparse.ArgumentParser(
        description='Demo of DoodleParty TUI features',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all demos
  python scripts/demo_tui.py
  
  # Run specific demo
  python scripts/demo_tui.py --demo panels
  python scripts/demo_tui.py --demo tables
  python scripts/demo_tui.py --demo progress
        """
    )
    
    parser.add_argument(
        '--demo',
        type=str,
        choices=['all', 'panels', 'tables', 'progress', 'spinners', 'training', 'summary'],
        default='all',
        help='Which demo to run'
    )
    
    args = parser.parse_args()
    
    # Header
    console.print()
    console.print(Panel.fit(
        "[bold cyan]DoodleParty TUI Features Demo[/bold cyan]\n"
        "[dim]Showcasing Rich Terminal UI Elements[/dim]",
        border_style="magenta",
        box=box.DOUBLE
    ))
    
    # Run selected demos
    if args.demo in ['all', 'panels']:
        demo_panels()
    
    if args.demo in ['all', 'tables']:
        demo_tables()
    
    if args.demo in ['all', 'progress']:
        demo_progress()
    
    if args.demo in ['all', 'spinners']:
        demo_spinners()
    
    if args.demo in ['all', 'training']:
        demo_training_output()
    
    if args.demo in ['all', 'summary']:
        demo_summary()
    
    console.print()
    console.print("[bold green]✓[/bold green] Demo complete! Check out the scripts in [cyan]scripts/[/cyan] to see these in action.")
    console.print()


if __name__ == '__main__':
    main()
