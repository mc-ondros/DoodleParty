#!/usr/bin/env bash
# Download and convert Quickdraw Appendix dataset (explicit content)
# Uses curl for better NixOS compatibility

set -e

OUTPUT_DIR="${1:-data/raw}"
APPENDIX_BASE_URL="https://raw.githubusercontent.com/studiomoniker/Quickdraw-appendix/master"
NDJSON_FILE="penis-simplified.ndjson"

echo "📥 Downloading Quickdraw Appendix - penis category"
echo "Output directory: $OUTPUT_DIR"
echo ""

mkdir -p "$OUTPUT_DIR"

TEMP_NDJSON="$OUTPUT_DIR/penis_temp.ndjson"
OUTPUT_NPY="$OUTPUT_DIR/penis.npy"

# Skip if already exists
if [ -f "$OUTPUT_NPY" ]; then
    echo "✓ penis.npy already exists"
    ls -lh "$OUTPUT_NPY"
    exit 0
fi

# Download NDJSON
echo "⬇  Downloading penis from Appendix..."
URL="$APPENDIX_BASE_URL/$NDJSON_FILE"

if curl -L --fail --progress-bar --max-time 600 "$URL" -o "$TEMP_NDJSON"; then
    FILE_SIZE=$(du -h "$TEMP_NDJSON" | cut -f1)
    echo "✓ Downloaded $FILE_SIZE"
else
    echo "✗ Download failed"
    rm -f "$TEMP_NDJSON"
    exit 1
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✓ Download complete!"
echo "  File: $TEMP_NDJSON"
echo "  Size: $(du -h $TEMP_NDJSON | cut -f1)"
echo ""
echo "Note: NDJSON file downloaded. To convert to NumPy format:"
echo "  python scripts/data_processing/download_appendix.py --input $TEMP_NDJSON --output-dir $OUTPUT_DIR"
