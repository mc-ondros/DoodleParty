#!/bin/bash
# Quick testing script for the trained QuickDraw Appendix detector

MODEL_PATH="/home/mcvaj/ML/models/quickdraw_appendix_detector.h5"
DATA_DIR="/home/mcvaj/ML/data/processed"
VENV="/home/mcvaj/ML/.venv/bin/python"

echo "QuickDraw Appendix Detector - Testing"
echo ""

# Test 1: Evaluate on test set
echo "[1] Evaluating model on test set..."
$VENV /home/mcvaj/ML/src/predict.py \
    --model "$MODEL_PATH" \
    --data-dir "$DATA_DIR" \
    evaluate

echo ""
echo "Testing complete!"
echo ""
echo "To test on your own images:"
echo "  $VENV /home/mcvaj/ML/src/predict.py --model $MODEL_PATH single --image /path/to/image.png"
echo ""
echo "To batch predict a directory:"
echo "  $VENV /home/mcvaj/ML/src/predict.py --model $MODEL_PATH batch --image-dir /path/to/images/"
