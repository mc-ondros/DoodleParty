"""
Download QuickDraw data in raw NDJSON format.

Downloads stroke data instead of pre-rendered bitmaps, allowing
rendering at 128x128 for crisp, high-quality images.

Related:
- scripts/data_processing/process_all_data_128x128.py (bitmap rendering)
- src/data/appendix_loader.py (NDJSON parsing)

Exports:
- download_quickdraw_ndjson
"""

import requests
from pathlib import Path
from tqdm import tqdm

def download_quickdraw_ndjson(category, output_dir, max_samples=5000):
    """
    Download QuickDraw data in raw NDJSON format.
    
    Args:
        category: Class name (e.g., 'cat', 'airplane')
        output_dir: Directory to save the file
        max_samples: Maximum samples to download
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # QuickDraw raw data is available at:
    # https://storage.googleapis.com/quickdraw_dataset/full/raw/{category}.ndjson
    base_url = 'https://storage.googleapis.com/quickdraw_dataset/full/raw'
    url = f"{base_url}/{category}.ndjson"
    
    output_file = output_dir / f"{category}-raw.ndjson"
    
    if output_file.exists():
        print(f"✓ {category}: Already downloaded")
        return output_file
    
    print(f"⬇ Downloading {category} (raw NDJSON)...")
    print(f"   URL: {url}")
    
    try:
        response = requests.get(url, stream=True, timeout=120)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        print(f"   File size: {total_size / 1024 / 1024:.1f}MB")
        
        # Download and save, optionally limiting lines
        with open(output_file, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=category) as pbar:
                lines_written = 0
                buffer = b''
                
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        buffer += chunk
                        pbar.update(len(chunk))
                        
                        # Process complete lines if limiting samples
                        if max_samples:
                            while b'\n' in buffer:
                                line, buffer = buffer.split(b'\n', 1)
                                f.write(line + b'\n')
                                lines_written += 1
                                if lines_written >= max_samples:
                                    print(f"   ✓ Downloaded {lines_written} samples (limit reached)")
                                    return output_file
                        else:
                            f.write(chunk)
                
                # Write any remaining buffer
                if buffer and (not max_samples or lines_written < max_samples):
                    f.write(buffer)
                    
        print(f"   ✓ {category}: Downloaded successfully")
        return output_file
        
    except Exception as e:
        print(f"   ✗ Error downloading {category}: {e}")
        if output_file.exists():
            output_file.unlink()
        return None


def main():
    """Download all QuickDraw negative classes in raw NDJSON format."""
    print('\nDOWNLOADING QUICKDRAW DATA (RAW NDJSON FORMAT)\n')
    
    project_root = Path(__file__).parent.parent.parent
    ndjson_dir = project_root / 'data' / 'raw_ndjson'
    
    negative_classes = [
        'airplane', 'apple', 'arm', 'banana', 'bird', 'boomerang',
        'cat', 'circle', 'cloud', 'dog', 'drill', 'fish', 'flower',
        'house', 'moon', 'pencil', 'square', 'star', 'sun', 'tree', 'triangle'
    ]
    
    print(f"\nDownloading {len(negative_classes)} classes...")
    print(f"Output directory: {ndjson_dir}")
    print(f"Limiting to 5000 samples per class for faster downloads\n")
    
    success_count = 0
    failed = []
    
    for cls in negative_classes:
        result = download_quickdraw_ndjson(cls, ndjson_dir, max_samples=5000)
        if result:
            success_count += 1
        else:
            failed.append(cls)
    
    print('\nSUMMARY\n')
    print(f"✓ Successfully downloaded: {success_count}/{len(negative_classes)} classes")
    
    if failed:
        print(f"✗ Failed: {', '.join(failed)}")
    
    print(f"\nNext step: Process NDJSON to 128x128 bitmaps")
    print('  python scripts/data_processing/process_all_data_128x128.py')


if __name__ == '__main__':
    main()
