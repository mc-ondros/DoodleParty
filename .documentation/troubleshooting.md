# DoodleHunter Troubleshooting Guide

**Purpose:** Solutions for common issues.

## Training Issues

### Out of Memory During Training

**Symptoms:** Python crashes with OOM error

**Solutions:**
```bash
# Reduce batch size
python src/train.py --batch-size 16

# Reduce image size
python src/train.py --image-size 64

# Use data generators (already implemented in train.py)
```

### Low Model Accuracy

**Symptoms:** Validation accuracy <80%

**Solutions:**
1. Increase training epochs: `--epochs 100`
2. Add more data augmentation in `src/data/augmentation.py`
3. Try different model architecture in `src/core/models.py`
4. Check data quality and class balance
5. Verify labels are correct

### Training Extremely Slow

**Symptoms:** Training takes >1 hour per epoch

**Solutions:**
```bash
# Use GPU if available
export CUDA_VISIBLE_DEVICES=0

# Reduce dataset size
python scripts/train.py --max-samples 5000

# Enable mixed precision
export TF_ENABLE_AUTO_MIXED_PRECISION=1
```

## Web Interface Issues

### Flask App Won't Start

**Symptoms:** `Address already in use` error

**Solutions:**
```bash
# Check what's using port 5000
lsof -i :5000

# Kill the process
kill -9 <PID>

# Or use different port
FLASK_PORT=5001 python app.py
```

### Model Not Loading

**Symptoms:** `FileNotFoundError` for model file

**Solutions:**
```bash
# Verify model exists
ls -lh models/

# Check path in src/web/app.py
# Update MODEL_PATH if needed

# Retrain model if missing
./scripts/train_max_accuracy.sh
```

### Canvas Not Responding

**Symptoms:** Drawing doesn't work in browser

**Solutions:**
1. Clear browser cache (Ctrl+Shift+R)
2. Try different browser (Chrome recommended)
3. Check browser console for JavaScript errors
4. Verify `src/web/static/js/canvas.js` exists

## Data Issues

### QuickDraw Download Fails

**Symptoms:** Download script errors or hangs

**Solutions:**
```bash
# Retry with specific categories
python scripts/data_processing/download_quickdraw_ndjson.py --categories penis circle square

# Check internet connection
ping storage.googleapis.com

# Manual download from:
# https://console.cloud.google.com/storage/browser/quickdraw_dataset
```

### Class Imbalance

**Symptoms:** Model always predicts one class

**Solutions:**
```bash
# Use class weighting flag when training
python scripts/train.py --use-class-weighting --epochs 50
```

## Performance Issues

### Slow Inference

**Symptoms:** Prediction takes >1 second

**Solutions:**
```python
# Convert to TensorFlow Lite
import tensorflow as tf

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open('model.tflite', 'wb') as f:
    f.write(tflite_model)
```

### High Memory Usage

**Symptoms:** System runs out of RAM

**Solutions:**
1. Close other applications
2. Use smaller batch size
3. Clear Keras session: `tf.keras.backend.clear_session()`
4. Restart Python process

## Common Error Messages

### `ModuleNotFoundError: No module named 'tensorflow'`

```bash
pip install tensorflow>=2.13.0
```

### `ValueError: Input arrays should have the same number of samples`

Check data loading in `src/data/loaders.py`:
```python
# Verify X and y have same length
print(f"X shape: {X.shape}, y shape: {y.shape}")
```

### `ResourceExhaustedError: OOM when allocating tensor`

Reduce batch size or image size as shown above.

## Getting Help

If issues persist:
1. Check [GitHub Issues](https://github.com/yourusername/doodlehunter/issues)
2. Review [Architecture](architecture.md) for system design
3. See [API Reference](api.md) for correct usage

*Troubleshooting guide for DoodleHunter v1.0*
