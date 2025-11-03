"""
Debug script to check raw coordinate ranges from penis NDJSON data.
"""

import json
import numpy as np

# Load a few samples from the raw file
filepath = 'quickdraw_appendix/penis-raw.ndjson'

print("Analyzing raw coordinate ranges...\n")

with open(filepath, 'r') as f:
    for idx in range(10):  # Check first 10 samples
        line = f.readline()
        data = json.loads(line)
        drawing = data.get('drawing', [])
        
        print(f"Sample {idx}:")
        
        all_xs = []
        all_ys = []
        
        for stroke in drawing:
            if len(stroke) >= 2:
                xs = stroke[0]
                ys = stroke[1]
                all_xs.extend(xs)
                all_ys.extend(ys)
        
        if all_xs and all_ys:
            print(f"  X range: {min(all_xs)} to {max(all_xs)} (span: {max(all_xs) - min(all_xs)})")
            print(f"  Y range: {min(all_ys)} to {max(all_ys)} (span: {max(all_ys) - min(all_ys)})")
            print(f"  Strokes: {len(drawing)}, Total points: {len(all_xs)}")
        else:
            print("  Empty drawing")
        print()
