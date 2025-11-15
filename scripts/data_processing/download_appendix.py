#!/usr/bin/env python3
"""
Download and convert Quickdraw Appendix dataset (explicit content).

The Appendix contains categories not included in Google's official dataset.
Downloads NDJSON format and converts to NumPy format compatible with the training pipeline.
"""

import argparse
import json
import os
import sys
import urllib.request
import numpy as np
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, BarColumn, TextColumn
from rich.table import Table
from rich import box

console = Console()

# Quickdraw Appendix GitHub repo
APPENDIX_BASE_URL = "https://raw.githubusercontent.com/studiomoniker/Quickdraw-appendix/master"

# Available categories in the appendix
APPENDIX_CATEGORIES = {
    "penis": "penis-simplified.ndjson",  # Use simplified version (cleaner strokes)
}


def ndjson_to_numpy(ndjson_path: str, max_samples: int = None) -> np.ndarray:
    """
    Convert NDJSON drawing format to NumPy array (28x28 images).
    
    Args:
        ndjson_path: Path to NDJSON file
        max_samples: Maximum number of samples to convert (None for all)
    
    Returns:
        NumPy array of shape (N, 28, 28) with pixel values 0-255
    """
    images = []
    
    with open(ndjson_path, 'r') as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            
            try:
                drawing = json.loads(line)
                
                # Create blank 28x28 image
                img = np.zeros((28, 28), dtype=np.uint8)
                
                # Draw strokes
                for stroke in drawing['drawing']:
                    x_coords = stroke[0]
                    y_coords = stroke[1]
                    
                    # Normalize coordinates to 0-27 range
                    x_coords = [int(x * 27 / 255) for x in x_coords]
                    y_coords = [int(y * 27 / 255) for y in y_coords]
                    
                    # Draw lines between consecutive points
                    for j in range(len(x_coords) - 1):
                        # Simple line drawing (could use cv2 for better results)
                        x1, y1 = x_coords[j], y_coords[j]
                        x2, y2 = x_coords[j + 1], y_coords[j + 1]
                        
                        # Clip to valid range
                        x1, y1 = max(0, min(27, x1)), max(0, min(27, y1))
                        x2, y2 = max(0, min(27, x2)), max(0, min(27, y2))
                        
                        # Draw pixel
                        img[y1, x1] = 255
                        img[y2, x2] = 255
                
                images.append(img)
                
            except (json.JSONDecodeError, KeyError, IndexError) as e:
                console.print(f"[yellow]Warning:[/yellow] Skipping invalid drawing at line {i+1}: {e}")
                continue
    
    return np.array(images, dtype=np.uint8)


def download_category(category: str, output_dir: str, max_samples: int = None) -> bool:
    """
    Download and convert an Appendix category.
    
    Args:
        category: Category name (e.g., 'penis')
        output_dir: Directory to save the .npy file
        max_samples: Maximum samples to include in output
    
    Returns:
        True if successful, False otherwise
    """
    if category not in APPENDIX_CATEGORIES:
        console.print(f"[red]Error:[/red] Category '{category}' not available in Appendix")
        console.print(f"Available categories: {', '.join(APPENDIX_CATEGORIES.keys())}")
        return False
    
    os.makedirs(output_dir, exist_ok=True)
    
    ndjson_filename = APPENDIX_CATEGORIES[category]
    url = f"{APPENDIX_BASE_URL}/{ndjson_filename}"
    temp_ndjson = os.path.join(output_dir, f"{category}_temp.ndjson")
    output_npy = os.path.join(output_dir, f"{category}.npy")
    
    # Skip if already exists
    if os.path.exists(output_npy):
        console.print(f"[green]✓[/green] {category}.npy already exists")
        return True
    
    try:
        # Download NDJSON file
        console.print(f"[cyan]⬇[/cyan]  Downloading {category} from Appendix...")
        urllib.request.urlretrieve(url, temp_ndjson, 
                                  reporthook=lambda b, bs, ts: None)
        
        file_size = os.path.getsize(temp_ndjson) / 1024 / 1024
        console.print(f"[green]✓[/green] Downloaded {file_size:.1f} MB")
        
        # Convert to NumPy format
        console.print(f"[cyan]🔄[/cyan] Converting to NumPy format...")
        images = ndjson_to_numpy(temp_ndjson, max_samples)
        
        # Save as .npy
        np.save(output_npy, images)
        output_size = os.path.getsize(output_npy) / 1024 / 1024
        console.print(f"[green]✓[/green] Saved {len(images)} images ({output_size:.1f} MB)")
        
        # Clean up temp file
        os.remove(temp_ndjson)
        
        return True
        
    except Exception as e:
        console.print(f"[red]✗[/red] Error: {str(e)}")
        # Clean up on error
        if os.path.exists(temp_ndjson):
            os.remove(temp_ndjson)
        if os.path.exists(output_npy):
            os.remove(output_npy)
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Download Quickdraw Appendix dataset (explicit content)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download penis category (default: all samples)
  python download_appendix.py --output-dir data/raw
  
  # Limit to 10,000 samples
  python download_appendix.py --output-dir data/raw --max-samples 10000
        """
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/raw',
        help='Output directory for .npy files (default: data/raw)'
    )
    
    parser.add_argument(
        '--category',
        type=str,
        default='penis',
        choices=list(APPENDIX_CATEGORIES.keys()),
        help='Category to download (default: penis)'
    )
    
    parser.add_argument(
        '--max-samples',
        type=int,
        default=None,
        help='Maximum number of samples to include (default: all)'
    )
    
    args = parser.parse_args()
    
    console.print()
    console.print(Panel.fit(
        "[bold cyan]Quickdraw Appendix Downloader[/bold cyan]\n"
        f"[dim]Explicit Content Dataset[/dim]",
        border_style="cyan",
        box=box.DOUBLE
    ))
    
    config_table = Table(show_header=False, box=box.SIMPLE, padding=(0, 2))
    config_table.add_row("[cyan]Category:[/cyan]", f"[yellow]{args.category}[/yellow]")
    config_table.add_row("[cyan]Output directory:[/cyan]", f"[yellow]{args.output_dir}[/yellow]")
    if args.max_samples:
        config_table.add_row("[cyan]Max samples:[/cyan]", f"[yellow]{args.max_samples:,}[/yellow]")
    else:
        config_table.add_row("[cyan]Max samples:[/cyan]", f"[yellow]All (~25,000)[/yellow]")
    console.print(config_table)
    console.print()
    
    success = download_category(args.category, args.output_dir, args.max_samples)
    
    console.print()
    if success:
        console.print(Panel.fit(
            f"[bold green]✓ Download Complete![/bold green]\n"
            f"[green]File saved:[/green] {os.path.join(args.output_dir, args.category + '.npy')}",
            border_style="green",
            box=box.DOUBLE
        ))
        return 0
    else:
        console.print(Panel.fit(
            f"[bold red]✗ Download Failed[/bold red]",
            border_style="red",
            box=box.DOUBLE
        ))
        return 1


if __name__ == '__main__':
    sys.exit(main())
