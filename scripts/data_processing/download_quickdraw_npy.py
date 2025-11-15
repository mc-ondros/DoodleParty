#!/usr/bin/env python3
"""
Download QuickDraw dataset as numpy files.

Downloads from Google's QuickDraw dataset in NumPy bitmap format (28x28 pre-processed).
The official numpy format is hosted at: https://quickdraw.withgoogle.com/data

This script downloads and organizes the dataset for training the binary classifier
(penis vs safe drawings).
"""

import argparse
import os
import sys
import urllib.request
import urllib.error
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, DownloadColumn, BarColumn, TextColumn, TransferSpeedColumn, TimeRemainingColumn
from rich.table import Table
from rich import box

# Initialize Rich console
console = Console()


# Official Google QuickDraw dataset URL (hosted on Google Cloud Storage)
QUICKDRAW_BASE_URL = "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap"

# Quickdraw Appendix (explicit content) - raw NDJSON format
QUICKDRAW_APPENDIX_BASE_URL = "https://raw.githubusercontent.com/studiomoniker/Quickdraw-appendix/master"

# Categories for binary classification
# Positive class: explicit/inappropriate content from Quickdraw Appendix
POSITIVE_CATEGORIES = [
    "penis",  # From Quickdraw Appendix
]

# Negative categories: shapes and objects that might look similar (to reduce false positives)
# Include elongated/organic shapes that could be confused with explicit content
NEGATIVE_CATEGORIES = [
    "banana",       # Elongated organic shape
    "carrot",       # Similar shape profile  
    "pencil",       # Elongated object
    "candle",       # Cylindrical with top
    "mushroom",     # Could have similar silhouette
    "lollipop",     # Round top with stick
    "circle",       # Basic shape
    "triangle",     # Basic shape
    "line",         # Basic element
]


def download_category(category: str, output_dir: str, progress: Progress = None, task_id = None) -> bool:
    """
    Download a single QuickDraw category.
    
    Args:
        category: Category name (e.g., 'penis', 'circle')
        output_dir: Directory to save the .npy file
        progress: Rich Progress object
        task_id: Progress task ID
    
    Returns:
        True if successful, False otherwise
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Construct download URL
    url = f"{QUICKDRAW_BASE_URL}/{category}.npy"
    output_path = os.path.join(output_dir, f"{category}.npy")
    
    # Skip if already downloaded
    if os.path.exists(output_path):
        file_size = os.path.getsize(output_path)
        if progress and task_id is not None:
            progress.update(task_id, description=f"[cyan]{category}[/cyan]", completed=file_size, total=file_size)
        return True
    
    try:
        # Start download with timeout
        response = urllib.request.urlopen(url, timeout=30)
        total_size = int(response.headers.get('content-length', 0))
        
        if progress and task_id is not None:
            progress.update(task_id, total=total_size, description=f"[cyan]{category}[/cyan]")
        
        # Download in chunks
        downloaded = 0
        chunk_size = 8192
        
        with open(output_path, 'wb') as f:
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                
                if progress and task_id is not None:
                    progress.update(task_id, completed=downloaded)
        
        return True
    
    except urllib.error.HTTPError as e:
        console.print(f"  [red]✗[/red] Error downloading {category}: HTTP {e.code}")
        if os.path.exists(output_path):
            os.remove(output_path)
        return False
    
    except urllib.error.URLError as e:
        console.print(f"  [red]✗[/red] Network error downloading {category}: {str(e)}")
        if os.path.exists(output_path):
            os.remove(output_path)
        return False
    
    except Exception as e:
        console.print(f"  [red]✗[/red] Error downloading {category}: {str(e)}")
        if os.path.exists(output_path):
            os.remove(output_path)
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Download QuickDraw dataset for DoodleParty',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download all categories (1 positive + 21 negative)
  python download_quickdraw_npy.py --output-dir data/raw
  
  # Download only positive category
  python download_quickdraw_npy.py --output-dir data/raw --positive-only
  
  # Download specific categories
  python download_quickdraw_npy.py --output-dir data/raw --categories penis circle star
  
  # Show what would be downloaded without downloading
  python download_quickdraw_npy.py --output-dir data/raw --dry-run
        """
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/raw',
        help='Output directory for downloaded .npy files (default: data/raw)'
    )
    
    parser.add_argument(
        '--categories',
        type=str,
        nargs='+',
        help='Specific categories to download (overrides default selection)'
    )
    
    parser.add_argument(
        '--positive-only',
        action='store_true',
        help='Download only positive (offensive) categories'
    )
    
    parser.add_argument(
        '--negative-only',
        action='store_true',
        help='Download only negative (safe) categories'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be downloaded without actually downloading'
    )
    
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress progress messages'
    )
    
    args = parser.parse_args()
    
    # Determine categories to download
    if args.categories:
        categories = args.categories
    elif args.positive_only:
        categories = POSITIVE_CATEGORIES
    elif args.negative_only:
        categories = NEGATIVE_CATEGORIES
    else:
        categories = POSITIVE_CATEGORIES + NEGATIVE_CATEGORIES
    
    verbose = not args.quiet
    
    if verbose:
        console.print()
        console.print(Panel.fit(
            "[bold cyan]QuickDraw Dataset Downloader[/bold cyan]\n"
            f"[dim]DoodleParty Training Data Preparation[/dim]",
            border_style="cyan",
            box=box.DOUBLE
        ))
        
        # Configuration table
        config_table = Table(show_header=False, box=box.SIMPLE, padding=(0, 2))
        config_table.add_row("[cyan]Categories:[/cyan]", f"[yellow]{len(categories)}[/yellow]")
        config_table.add_row("[cyan]Output directory:[/cyan]", f"[yellow]{args.output_dir}[/yellow]")
        config_table.add_row("[cyan]Estimated size:[/cyan]", f"[yellow]~{len(categories) * 25}MB[/yellow]")
        console.print(config_table)
        console.print()
    
    if args.dry_run:
        if verbose:
            console.print("[bold yellow]DRY RUN[/bold yellow] - Categories to download:")
            cat_table = Table(show_header=False, box=box.SIMPLE)
            for i, cat in enumerate(categories, 1):
                cat_table.add_row(f"[dim]{i}.[/dim]", f"[cyan]{cat}[/cyan]")
            console.print(cat_table)
        return 0
    
    # Download categories with progress bar
    successful = 0
    failed = 0
    
    if verbose:
        with Progress(
            TextColumn("{task.description}"),
            BarColumn(bar_width=40),
            "[progress.percentage]{task.percentage:>3.0f}%",
            TransferSpeedColumn(),
            TimeRemainingColumn(),
            console=console
        ) as progress:
            for category in categories:
                task = progress.add_task(f"[cyan]{category}[/cyan]", total=100)
                if download_category(category, args.output_dir, progress, task):
                    successful += 1
                    progress.update(task, description=f"[green]✓[/green] [cyan]{category}[/cyan]")
                else:
                    failed += 1
                    progress.update(task, description=f"[red]✗[/red] [cyan]{category}[/cyan]")
    else:
        for category in categories:
            if download_category(category, args.output_dir):
                successful += 1
            else:
                failed += 1
    
    if verbose:
        console.print()
        console.print(Panel.fit(
            f"[bold green]✓ Download Complete![/bold green]\n"
            f"[green]Successful:[/green] {successful} | [red]Failed:[/red] {failed}",
            border_style="green" if failed == 0 else "yellow",
            box=box.DOUBLE
        ))
        
        if successful > 0:
            console.print("\n[bold cyan]Dataset Location:[/bold cyan]")
            console.print(f"  [yellow]{os.path.abspath(args.output_dir)}[/yellow]")
            
            # List files
            npy_files = sorted([f for f in os.listdir(args.output_dir) if f.endswith('.npy')])
            console.print(f"\n[bold cyan]Files Downloaded:[/bold cyan] [yellow]{len(npy_files)}[/yellow]")
            
            file_table = Table(show_header=False, box=box.SIMPLE, padding=(0, 1))
            for f in npy_files[:5]:
                file_size = os.path.getsize(os.path.join(args.output_dir, f))
                file_table.add_row("[dim]•[/dim]", f"[cyan]{f}[/cyan]", f"[dim]({file_size / 1024 / 1024:.1f} MB)[/dim]")
            
            if len(npy_files) > 5:
                file_table.add_row("[dim]•[/dim]", f"[dim]... and {len(npy_files) - 5} more[/dim]", "")
            
            console.print(file_table)
        
        console.print()
    
    return 0 if failed == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
