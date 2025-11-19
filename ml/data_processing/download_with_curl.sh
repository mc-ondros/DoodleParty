#!/usr/bin/env bash
# Alternative download script using curl for better NixOS compatibility
# Usage: ./download_with_curl.sh [output_dir]

OUTPUT_DIR="${1:-data/raw}"
BASE_URL="https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Categories to download (verified to exist in dataset)
CATEGORIES=(
    "banana"
    "carrot"
    "pencil"
    "candle"
    "mushroom"
    "lollipop"
    "circle"
    "triangle"
    "line"
)

echo "📥 Downloading QuickDraw dataset to: $OUTPUT_DIR"
echo "Categories: ${#CATEGORIES[@]}"
echo ""

SUCCESS=0
FAILED=0

for category in "${CATEGORIES[@]}"; do
    output_file="$OUTPUT_DIR/${category}.npy"
    
    # Skip if already exists
    if [ -f "$output_file" ]; then
        echo "✓ $category (already exists)"
        ((SUCCESS++))
        continue
    fi
    
    url="$BASE_URL/${category}.npy"
    echo -n "⬇  Downloading $category... "
    
    if curl -L --fail --silent --show-error --max-time 300 "$url" -o "$output_file"; then
        size=$(du -h "$output_file" | cut -f1)
        echo "✓ ($size)"
        ((SUCCESS++))
    else
        echo "✗ Failed"
        rm -f "$output_file"
        ((FAILED++))
    fi
done

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✓ Download complete!"
echo "  Successful: $SUCCESS"
echo "  Failed: $FAILED"
echo "  Location: $(realpath $OUTPUT_DIR)"
