#!/bin/bash

# Maximum Accuracy Training Script
# Trains model with ALL improvements for best real-world performance
# WARNING: This will take 3-5x longer than standard training!

echo "🚀 Starting MAXIMUM ACCURACY training"
echo ""
echo "Improvements enabled:"
echo "  ✓ Enhanced model (2.5M params vs 423K)"
echo "  ✓ Aggressive augmentation (±30° rotation, ±20% shift, ±25% zoom)"
echo "  ✓ Label smoothing (0.1)"
echo "  ✓ Learning rate scheduling"
echo "  ✓ 200 epochs with patience=30"
echo ""
echo "Expected training time: 30-60 minutes (CPU)"
echo "Expected improvement: +3-5% accuracy vs baseline"
echo ""

# Kill any existing training
pkill -f "python.*train.py" 2>/dev/null
sleep 2

# Train with all improvements
cd /home/mcvaj/ML

/home/mcvaj/ML/.venv/bin/python src/train.py \
  --data-dir data/processed \
  --epochs 200 \
  --batch-size 32 \
  --model-output models/quickdraw_model_enhanced.h5 \
  --learning-rate 0.0005 \
  --label-smoothing 0.1 \
  --architecture custom \
  --enhanced \
  --aggressive-aug \
  2>&1 | tee training_run_enhanced.log

echo ""
echo "✅ Training complete!"
echo "Model saved to: models/quickdraw_model_enhanced.h5"
echo ""
echo "Next steps:"
echo "  1. Test with TTA: python src/test_time_augmentation.py"
echo "  2. Compare: python src/predict.py --compare-models"
