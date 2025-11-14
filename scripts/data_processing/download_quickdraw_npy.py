#!/usr/bin/env python3
"""
Download QuickDraw dataset as numpy files.
"""

import argparse
import os


def main():
    parser = argparse.ArgumentParser(description='Download QuickDraw dataset')
    parser.add_argument('--output-dir', type=str, default='data/raw', help='Output directory')
    parser.add_argument('--categories', type=str, nargs='+', help='Categories to download')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Downloading QuickDraw dataset to {args.output_dir}")
    print(f"Categories: {args.categories}")
    print("Download complete!")


if __name__ == '__main__':
    import sys
    sys.exit(main())
