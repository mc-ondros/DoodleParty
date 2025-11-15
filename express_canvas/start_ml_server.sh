#!/bin/bash
# Start ML Server for Content Detection

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "========================================"
echo "DoodleParty ML Server Startup"
echo "========================================"

# Configuration
# Default to the quantized QuickDraw TFLite blob provided with the project
ML_MODEL_PATH="${ML_MODEL_PATH:-./models/quickdraw_model_int8.tflite}"
ML_INPUT_SIZE="${ML_INPUT_SIZE:-64}"
ML_CONFIDENCE_THRESHOLD="${ML_CONFIDENCE_THRESHOLD:-0.7}"
ML_SERVER_PORT="${ML_SERVER_PORT:-5000}"

# Export environment variables
export ML_MODEL_PATH
export ML_INPUT_SIZE
export ML_CONFIDENCE_THRESHOLD
export ML_SERVER_PORT

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is not installed"
    echo "Please install Python 3.8 or higher"
    exit 1
fi

echo "Python version: $(python3 --version)"

# Check if virtual environment exists
if [ ! -d "venv_ml" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv_ml
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv_ml/bin/activate

# Install/update dependencies
echo "Installing dependencies..."
pip install -q --upgrade pip
pip install -q -r ml_requirements.txt

# Check if model exists
if [ ! -f "$ML_MODEL_PATH" ]; then
    echo ""
    echo "WARNING: Model file not found at $ML_MODEL_PATH"
    echo "The server will run with mock predictions for testing."
    echo "To use a real model, place your .h5 model file at: $ML_MODEL_PATH"
    echo ""
fi

echo ""
echo "Configuration:"
echo "  Model Path: $ML_MODEL_PATH"
echo "  Input Size: ${ML_INPUT_SIZE}x${ML_INPUT_SIZE}"
echo "  Threshold: $ML_CONFIDENCE_THRESHOLD"
echo "  Port: $ML_SERVER_PORT"
echo ""
echo "Starting ML server..."
echo "========================================"
echo ""

# Start the ML server
python3 ml_server.py

# Deactivate virtual environment on exit
deactivate
