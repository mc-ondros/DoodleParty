#!/usr/bin/env python3
"""
Demonstration script for region-based detection.

This script demonstrates the robustness improvements implemented for
content dilution attack prevention using sliding window detection.

Usage:
    python scripts/demo_region_detection.py --model models/quickdraw_classifier.keras
    python scripts/demo_region_detection.py --model models/quickdraw_classifier.keras --visualize
"""

import argparse
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.inference import load_model_and_mapping, predict_image_region_based
from src.core.patch_extraction import AggregationStrategy


def create_demo_images():
    """
    Create synthetic demo images to demonstrate detection capabilities.
    
    Returns:
        Dictionary of demo images with descriptions
    """
    demo_images = {}
    
    # 1. Clean image (no suspicious content)
    clean = np.zeros((512, 512))
    # Draw a simple circle
    center = (256, 256)
    radius = 100
    y, x = np.ogrid[:512, :512]
    mask = (x - center[0])**2 + (y - center[1])**2 <= radius**2
    clean[mask] = 1.0
    demo_images['clean_circle'] = {
        'image': clean,
        'description': 'Clean image with circle (should be negative)'
    }
    
    # 2. Image with suspicious content in one region
    localized = np.zeros((512, 512))
    # Add suspicious pattern in top-left corner
    localized[50:150, 50:150] = np.random.rand(100, 100) * 0.8
    demo_images['localized_suspicious'] = {
        'image': localized,
        'description': 'Suspicious content in one region (should detect)'
    }
    
    # 3. Diluted content (small suspicious region + lots of innocent content)
    diluted = np.zeros((512, 512))
    # Small suspicious region
    diluted[0:100, 0:100] = np.random.rand(100, 100) * 0.9
    # Lots of innocent content
    diluted[200:300, 200:300] = np.random.rand(100, 100) * 0.3  # Light circle
    diluted[300:400, 100:200] = np.random.rand(100, 100) * 0.3  # Light square
    demo_images['diluted_attack'] = {
        'image': diluted,
        'description': 'Content dilution attack (suspicious + innocent)'
    }
    
    # 4. Multiple suspicious regions
    multi = np.zeros((512, 512))
    multi[0:100, 0:100] = np.random.rand(100, 100) * 0.8
    multi[400:500, 400:500] = np.random.rand(100, 100) * 0.8
    demo_images['multi_region'] = {
        'image': multi,
        'description': 'Multiple suspicious regions (should detect)'
    }
    
    return demo_images


def visualize_comparison(results, save_path=None):
    """
    Visualize comparison of different detection strategies.
    
    Args:
        results: Dictionary of results for different strategies
        save_path: Optional path to save the figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    strategies = list(results.keys())
    
    for idx, (strategy_name, result) in enumerate(results.items()):
        ax = axes[idx]
        
        # Display image
        image = result['image']
        if len(image.shape) == 3:
            ax.imshow(image[:, :, 0], cmap='gray')
        else:
            ax.imshow(image, cmap='gray')
        
        # Add detection info
        verdict = "POSITIVE" if result['detection_result'].is_positive else "NEGATIVE"
        confidence = result['detection_result'].confidence
        num_patches = result['detection_result'].num_patches_analyzed
        
        color = 'red' if result['detection_result'].is_positive else 'green'
        
        ax.set_title(
            f'{strategy_name}\n'
            f'{verdict} (conf: {confidence:.2%})\n'
            f'{num_patches} patches analyzed',
            fontsize=12,
            weight='bold',
            color=color
        )
        ax.axis('off')
        
        # Draw patch boundaries
        for pred in result['detection_result'].patch_predictions:
            x, y = pred['x'], pred['y']
            rect_color = 'red' if pred['is_positive'] else 'green'
            alpha = min(pred['confidence'], 0.6)
            
            from matplotlib.patches import Rectangle
            rect = Rectangle(
                (x, y), 128, 128,
                linewidth=1.5,
                edgecolor=rect_color,
                facecolor='none',
                alpha=alpha
            )
            ax.add_patch(rect)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n✓ Visualization saved to: {save_path}")
    
    plt.show()


def demo_aggregation_strategies(model, idx_to_class, image, image_name):
    """
    Demonstrate different aggregation strategies on the same image.
    
    Args:
        model: Loaded model
        idx_to_class: Class mapping
        image: Input image
        image_name: Name of the image for display
    """
    print(f"\n{'='*70}")
    print(f"Testing Aggregation Strategies on: {image_name}")
    print(f"{'='*70}")
    
    strategies = {
        'MAX': AggregationStrategy.MAX,
        'MEAN': AggregationStrategy.MEAN,
        'VOTING': AggregationStrategy.VOTING,
        'ANY_POSITIVE': AggregationStrategy.ANY_POSITIVE
    }
    
    results = {}
    
    for strategy_name, strategy in strategies.items():
        print(f"\n--- {strategy_name} Strategy ---")
        
        from src.core.patch_extraction import SlidingWindowDetector
        
        # Ensure image has channel dimension
        if len(image.shape) == 2:
            image_3d = np.expand_dims(image, axis=-1)
        else:
            image_3d = image
        
        detector = SlidingWindowDetector(
            model=model,
            patch_size=(128, 128),
            stride=(128, 128),
            min_content_ratio=0.05,
            max_patches=16,
            early_stopping=False,  # Disable for comparison
            aggregation_strategy=strategy,
            classification_threshold=0.5
        )
        
        detection_result = detector.detect_batch(image_3d)
        
        verdict = "POSITIVE" if detection_result.is_positive else "NEGATIVE"
        print(f"  Result: {verdict}")
        print(f"  Confidence: {detection_result.confidence:.2%}")
        print(f"  Patches analyzed: {detection_result.num_patches_analyzed}")
        print(f"  Positive patches: {sum(1 for p in detection_result.patch_predictions if p['is_positive'])}")
        
        results[strategy_name] = {
            'detection_result': detection_result,
            'image': image_3d
        }
    
    return results


def demo_early_stopping(model, idx_to_class):
    """Demonstrate early stopping feature."""
    print(f"\n{'='*70}")
    print("Demonstrating Early Stopping")
    print(f"{'='*70}")
    
    # Create image with suspicious content in first patch
    image = np.zeros((512, 512))
    image[0:128, 0:128] = np.random.rand(128, 128) * 0.9  # High confidence region
    
    from src.core.patch_extraction import SlidingWindowDetector
    
    if len(image.shape) == 2:
        image_3d = np.expand_dims(image, axis=-1)
    else:
        image_3d = image
    
    # Without early stopping
    print("\n--- Without Early Stopping ---")
    detector_no_early = SlidingWindowDetector(
        model=model,
        patch_size=(128, 128),
        stride=(128, 128),
        early_stopping=False
    )
    result_no_early = detector_no_early.detect_batch(image_3d)
    print(f"  Patches analyzed: {result_no_early.num_patches_analyzed}")
    
    # With early stopping
    print("\n--- With Early Stopping ---")
    detector_early = SlidingWindowDetector(
        model=model,
        patch_size=(128, 128),
        stride=(128, 128),
        early_stopping=True,
        early_stop_threshold=0.9
    )
    result_early = detector_early.detect(image_3d)
    print(f"  Patches analyzed: {result_early.num_patches_analyzed}")
    print(f"  Early stopped: {result_early.early_stopped}")
    print(f"  Efficiency gain: {(1 - result_early.num_patches_analyzed / result_no_early.num_patches_analyzed) * 100:.1f}% fewer patches")


def main():
    parser = argparse.ArgumentParser(
        description="Demonstrate region-based detection capabilities"
    )
    parser.add_argument(
        '--model',
        required=True,
        help='Path to trained model'
    )
    parser.add_argument(
        '--data-dir',
        default='data/processed',
        help='Directory with processed data'
    )
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Show visualizations'
    )
    parser.add_argument(
        '--output-dir',
        default='models/benchmarks',
        help='Directory to save visualizations'
    )
    
    args = parser.parse_args()
    
    print("="*70)
    print("Region-Based Detection Demonstration")
    print("="*70)
    print(f"\nModel: {args.model}")
    
    # Load model
    print("\nLoading model...")
    model, idx_to_class = load_model_and_mapping(args.model, args.data_dir)
    print("✓ Model loaded successfully")
    
    # Create demo images
    print("\nCreating demo images...")
    demo_images = create_demo_images()
    print(f"✓ Created {len(demo_images)} demo images")
    
    # Test each demo image
    for image_name, image_data in demo_images.items():
        print(f"\n{'='*70}")
        print(f"Testing: {image_data['description']}")
        print(f"{'='*70}")
        
        image = image_data['image']
        
        # Test with different strategies
        if args.visualize:
            results = demo_aggregation_strategies(model, idx_to_class, image, image_name)
            
            # Save visualization
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            vis_path = output_dir / f"region_detection_{image_name}.png"
            
            visualize_comparison(results, save_path=vis_path)
    
    # Demonstrate early stopping
    demo_early_stopping(model, idx_to_class)
    
    print(f"\n{'='*70}")
    print("Demonstration Complete!")
    print(f"{'='*70}")
    print("\nKey Findings:")
    print("  • MAX strategy is most aggressive (best for security)")
    print("  • MEAN strategy is most balanced")
    print("  • Early stopping reduces computation by stopping on first detection")
    print("  • Region-based detection prevents content dilution attacks")
    print("\nUsage in production:")
    print("  • Use MAX strategy for high-security applications")
    print("  • Enable early stopping for performance")
    print("  • Adjust patch_size and stride based on expected content size")


if __name__ == "__main__":
    main()
