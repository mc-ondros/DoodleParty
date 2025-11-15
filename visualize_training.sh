#!/bin/bash
# Quick script to visualize training data format

echo "🎨 Training Data Visualizer"
echo "=========================="
echo ""

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    echo "Activating venv..."
    source venv/bin/activate
elif [ -d "express_canvas/venv_ml" ]; then
    echo "Activating ML venv..."
    source express_canvas/venv_ml/bin/activate
fi

# Run the visualization script
python scripts/visualization/quick_training_viz.py "$@"

echo ""
echo "✅ Done! Check logs/training_data_visualization.png"
