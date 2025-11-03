"""
Reprocess penis data with the fixed normalization and centering.
"""

import sys
sys.path.append('src')

from appendix_loader import parse_ndjson_file
import numpy as np
from pathlib import Path

# Load and process with fixed code
print("Reprocessing penis data with fixed normalization...")
filepath = 'quickdraw_appendix/penis-raw.ndjson'

# Process the data
images = parse_ndjson_file(filepath, max_samples=None, format_type='raw')

# Save processed data
output_path = Path('data/processed/penis_raw_X.npy')
np.save(output_path, images)

print(f"\n✓ Saved {len(images)} processed images to {output_path}")
print(f"  Shape: {images.shape}")
print(f"  Data type: {images.dtype}")
print(f"  Value range: {images.min()} - {images.max()}")
