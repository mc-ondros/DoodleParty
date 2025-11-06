"""
Generate diverse negative samples for training the binary classifier.

Creates realistic "out-of-distribution" samples:
1. Pure random noise (current approach)
2. Gaussian blur of random noise (smoother patterns)
3. Random lines/scribbles (drawing-like but unstructured)
4. Partially filled patterns (grid, circles, strokes at odd angles)
5. Inverted positives (flipped/rotated versions)
"""

import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt


def pure_random_noise(size=28):
    """Pure random pixels - very easy to distinguish."""
    return np.random.randint(0, 256, (size, size), dtype=np.uint8)


def gaussian_noise(size=28, sigma=30):
    """Gaussian blur of noise - smoother, more drawing-like."""
    img = Image.new('L', (size, size), color=128)
    rng = np.random.default_rng()
    for _ in range(50):
        x = np.random.randint(0, size)
        y = np.random.randint(0, size)
        val = int(rng.normal(128, sigma))
        val = max(0, min(255, val))
        img.putpixel((x, y), val)
    
    # Blur to smooth
    img = img.filter(Image.BLUR)
    return np.array(img, dtype=np.uint8)


def random_scribbles(size=28):
    """Random lines/scribbles - looks somewhat like drawings but unstructured."""
    img = Image.new('L', (size, size), color=255)  # White background
    draw = ImageDraw.Draw(img)
    
    # Draw 3-8 random scribbles
    num_scribbles = np.random.randint(3, 9)
    for _ in range(num_scribbles):
        # Random start/end points
        x1, y1 = np.random.randint(0, size, 2)
        x2, y2 = np.random.randint(0, size, 2)
        
        # Random thickness and color
        width = np.random.randint(1, 4)
        color = np.random.randint(0, 200)
        
        draw.line([(x1, y1), (x2, y2)], fill=color, width=width)
    
    return np.array(img, dtype=np.uint8)


def geometric_patterns(size=28):
    """Geometric patterns - grid, circles, diamonds (non-drawing-like)."""
    img = Image.new('L', (size, size), color=255)
    draw = ImageDraw.Draw(img)
    
    pattern_type = np.random.choice(['grid', 'circles', 'diamonds', 'lines'])
    
    if pattern_type == 'grid':
        step = np.random.randint(4, 10)
        for x in range(0, size, step):
            draw.line([(x, 0), (x, size)], fill=100, width=1)
        for y in range(0, size, step):
            draw.line([(0, y), (size, y)], fill=100, width=1)
    
    elif pattern_type == 'circles':
        num_circles = np.random.randint(2, 5)
        for _ in range(num_circles):
            x, y = np.random.randint(0, size, 2)
            r = np.random.randint(2, 10)
            draw.ellipse([(x-r, y-r), (x+r, y+r)], outline=100, width=1)
    
    elif pattern_type == 'diamonds':
        step = np.random.randint(6, 12)
        for x in range(0, size, step):
            for y in range(0, size, step):
                r = 3
                draw.polygon([(x, y-r), (x+r, y), (x, y+r), (x-r, y)], outline=100, width=1)
    
    else:  # lines
        num_lines = np.random.randint(3, 8)
        for _ in range(num_lines):
            y = np.random.randint(0, size)
            draw.line([(0, y), (size, y)], fill=100, width=1)
    
    return np.array(img, dtype=np.uint8)


def inverted_positive(positive_image):
    """Invert a positive sample - flipped/transformed version."""
    # Random transformation
    transform = np.random.choice(['invert', 'flip_h', 'flip_v', 'rotate'])
    
    if transform == 'invert':
        return 255 - positive_image
    elif transform == 'flip_h':
        return np.fliplr(positive_image)
    elif transform == 'flip_v':
        return np.flipud(positive_image)
    else:  # rotate
        k = np.random.randint(1, 4)
        return np.rot90(positive_image, k)


def sparse_pixels(size=28):
    """Very sparse pixel patterns - hard to draw with."""
    img = Image.new('L', (size, size), color=255)
    
    # Place 10-30 random black pixels
    num_pixels = np.random.randint(10, 30)
    for _ in range(num_pixels):
        x, y = np.random.randint(0, size, 2)
        img.putpixel((x, y), 0)
    
    return np.array(img, dtype=np.uint8)


def connected_scribbles(size=28):
    """Connected scribbles that look somewhat like real strokes but aren't."""
    img = Image.new('L', (size, size), color=255)
    draw = ImageDraw.Draw(img)
    
    # Start with a random point
    x, y = np.random.randint(2, size-2, 2)
    
    # Draw a continuous path with random turns
    num_segments = np.random.randint(5, 15)
    for _ in range(num_segments):
        # Random angle change
        angle = np.random.uniform(-np.pi/2, np.pi/2)
        
        # Random length
        length = np.random.randint(2, 8)
        
        # Calculate endpoint
        x2 = int(x + length * np.cos(angle))
        y2 = int(y + length * np.sin(angle))
        
        # Clamp to image bounds
        x2 = max(0, min(size-1, x2))
        y2 = max(0, min(size-1, y2))
        
        # Draw line
        width = np.random.randint(1, 3)
        color = np.random.randint(50, 200)
        draw.line([(x, y), (x2, y2)], fill=color, width=width)
        
        x, y = x2, y2
    
    return np.array(img, dtype=np.uint8)


def concentric_circles(size=28):
    """Concentric circles - structured but not a drawing."""
    img = Image.new('L', (size, size), color=255)
    draw = ImageDraw.Draw(img)
    
    center = size // 2
    num_circles = np.random.randint(2, 6)
    
    for i in range(num_circles):
        radius = (i + 1) * (size // (num_circles + 1))
        color = np.random.randint(50, 200)
        draw.ellipse(
            [(center - radius, center - radius), (center + radius, center + radius)],
            outline=color,
            width=1
        )
    
    return np.array(img, dtype=np.uint8)


def crossed_lines(size=28):
    """Crossed/hatched lines."""
    img = Image.new('L', (size, size), color=255)
    draw = ImageDraw.Draw(img)
    
    num_lines = np.random.randint(3, 8)
    angle = np.random.uniform(0, np.pi)
    
    for i in range(num_lines):
        offset = (i - num_lines / 2) * (size / num_lines)
        x1 = int(offset * np.cos(angle))
        y1 = int(offset * np.sin(angle))
        x2 = int(x1 + size * np.cos(angle))
        y2 = int(y1 + size * np.sin(angle))
        
        color = np.random.randint(50, 200)
        draw.line([(x1, y1), (x2, y2)], fill=color, width=1)
    
    return np.array(img, dtype=np.uint8)


def wave_pattern(size=28):
    """Wave/sine pattern."""
    img = Image.new('L', (size, size), color=255)
    draw = ImageDraw.Draw(img)
    
    amplitude = np.random.randint(2, 8)
    frequency = np.random.uniform(0.5, 3)
    
    points = []
    for x in range(size):
        y = int(size / 2 + amplitude * np.sin(2 * np.pi * frequency * x / size))
        y = max(0, min(size - 1, y))
        points.append((x, y))
    
    color = np.random.randint(50, 200)
    draw.line(points, fill=color, width=2)
    
    return np.array(img, dtype=np.uint8)


def generate_diverse_negatives(num_samples, positive_images=None):
    """
    Generate diverse negative samples.
    
    Args:
        num_samples: Number of negative samples to generate
        positive_images: Optional array of positive samples for inversion augmentation
    
    Returns:
        Array of negative images (num_samples, 28, 28)
    """
    strategies = [
        pure_random_noise,
        gaussian_noise,
        random_scribbles,
        geometric_patterns,
        sparse_pixels,
        connected_scribbles,
        concentric_circles,
        crossed_lines,
        wave_pattern,
    ]
    
    if positive_images is not None:
        strategies.append(
            lambda: inverted_positive(
                positive_images[np.random.randint(0, len(positive_images))]
            )
        )
    
    negatives = []
    strategy_counts = {s.__name__: 0 for s in strategies}
    
    print(f"Generating {num_samples} diverse negative samples...")
    
    for i in range(num_samples):
        # Cycle through strategies or sample randomly
        strategy = strategies[i % len(strategies)]
        
        try:
            negative = strategy()
            negatives.append(negative)
            strategy_counts[strategy.__name__] += 1
        except Exception as e:
            print(f"Error in {strategy.__name__}: {e}")
            negatives.append(pure_random_noise())
        
        if (i + 1) % 5000 == 0:
            print(f"  Generated {i + 1}/{num_samples} samples...")
    
    print(f"✓ Generated {num_samples} negative samples")
    print(f"  Strategy breakdown:")
    for name, count in sorted(strategy_counts.items()):
        print(f"    {name}: {count}")
    
    return np.array(negatives, dtype=np.uint8)


if __name__ == '__main__':
    # Example: generate 100 samples and visualize
    print('Generating example negative samples...')
    negatives = generate_diverse_negatives(100)
    
    # Visualize a sample of each type
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    
    types = [
        ("Pure Noise", pure_random_noise()),
        ("Gaussian Noise", gaussian_noise()),
        ("Scribbles", random_scribbles()),
        ("Geometric", geometric_patterns()),
        ("Sparse", sparse_pixels()),
        ("Random Negative", negatives[np.random.randint(0, len(negatives))]),
    ]
    
    for ax, (name, img) in zip(axes.flat, types):
        ax.imshow(img, cmap='gray')
        ax.set_title(name)
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('data/processed/negative_samples_preview.png', dpi=100)
    print('✓ Saved preview to data/processed/negative_samples_preview.png')
