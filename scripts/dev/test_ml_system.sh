#!/bin/bash
# Quick test script to verify ML detection system

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "========================================================================"
echo "🧪 TESTING ML DETECTION SYSTEM"
echo "========================================================================"
echo ""

# Check if services are running
echo "1. Checking if services are running..."

if [ -f "logs/express.pid" ]; then
    EXPRESS_PID=$(cat logs/express.pid)
    if ps -p $EXPRESS_PID > /dev/null 2>&1; then
        echo "  ✓ Express server running (PID: $EXPRESS_PID)"
    else
        echo "  ✗ Express server not running"
        echo "  Start with: ./start_doodleparty.sh"
        exit 1
    fi
else
    echo "  ✗ Express server not running"
    echo "  Start with: ./start_doodleparty.sh"
    exit 1
fi

if [ -f "logs/ml.pid" ]; then
    ML_PID=$(cat logs/ml.pid)
    if ps -p $ML_PID > /dev/null 2>&1; then
        echo "  ✓ ML server running (PID: $ML_PID)"
    else
        echo "  ✗ ML server not running"
        exit 1
    fi
else
    echo "  ✗ ML server not running"
    exit 1
fi

echo ""
echo "2. Checking logs..."

# Check Express log
if [ -f "logs/express.log" ]; then
    EXPRESS_ERRORS=$(grep -i "error" logs/express.log | tail -3 || echo "No errors")
    if [ "$EXPRESS_ERRORS" != "No errors" ]; then
        echo "  ⚠️  Express log has errors:"
        echo "$EXPRESS_ERRORS" | sed 's/^/    /'
    else
        echo "  ✓ Express log clean"
    fi
fi

# Check ML log
if [ -f "logs/ml.log" ]; then
    if grep -q "✓ ML server ready" logs/ml.log; then
        echo "  ✓ ML server connected successfully"
    else
        echo "  ⚠️  ML server connection issues:"
        tail -5 logs/ml.log | sed 's/^/    /'
    fi
fi

echo ""
echo "3. Checking directories..."

if [ -d "data/ml_detections" ]; then
    DETECTION_COUNT=$(find data/ml_detections -name "*.png" 2>/dev/null | wc -l)
    echo "  ✓ Detections directory exists ($DETECTION_COUNT images)"
else
    echo "  ⚠️  Detections directory not created yet"
fi

if [ -d "data/ml_visualizations" ]; then
    VIZ_COUNT=$(find data/ml_visualizations -name "*.png" 2>/dev/null | wc -l)
    echo "  ✓ Visualizations directory exists ($VIZ_COUNT images)"
else
    echo "  ⚠️  Visualizations directory not created yet"
fi

echo ""
echo "4. Testing model..."

source venv/bin/activate 2>/dev/null || true

python3 << 'EOF'
import sys
try:
    import tensorflow as tf
    from pathlib import Path
    
    model_path = Path('models/quickdraw_model_int8.tflite')
    if not model_path.exists():
        print(f"  ✗ Model not found at {model_path}")
        sys.exit(1)
    
    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    print(f"  ✓ Model loaded successfully")
    print(f"    Path: {model_path}")
    print(f"    Size: {model_path.stat().st_size / 1024:.1f} KB")
    
except Exception as e:
    print(f"  ✗ Model loading failed: {e}")
    sys.exit(1)
EOF

echo ""
echo "========================================================================"
echo "✅ SYSTEM CHECK COMPLETE"
echo "========================================================================"
echo ""
echo "Next steps:"
echo "  1. Open browser: http://localhost:3000/doodleparty"
echo "  2. Draw something on the canvas"
echo "  3. Click the '🔍 Detect' button"
echo "  4. Check browser console for results"
echo "  5. Check visualizations: data/ml_visualizations/"
echo ""
echo "Monitoring:"
echo "  Express log: tail -f logs/express.log"
echo "  ML log:      tail -f logs/ml.log"
echo ""
