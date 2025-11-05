# Model Conversion & Optimization Scripts

This directory contains scripts for converting and optimizing the trained QuickDraw classifier for production deployment.

## Overview

The scripts provide a complete optimization pipeline:
1. **TFLite Conversion** - Convert Keras to TensorFlow Lite
2. **INT8 Quantization** - Apply post-training quantization
3. **Benchmarking** - Measure performance improvements
4. **Pruning** - Remove unnecessary weights
5. **Distillation** - Train smaller student models
6. **Graph Optimization** - Apply TensorFlow graph optimizations
7. **ONNX Conversion** - Export to ONNX format

## Quick Start

### Complete Optimization Workflow

```bash
# 1. Convert to TFLite (float32)
python scripts/convert/convert_to_tflite.py --model models/quickdraw_model.h5

# 2. Apply INT8 quantization
python scripts/convert/quantize_int8.py --model models/quickdraw_model.h5

# 3. Benchmark performance
python scripts/convert/benchmark_tflite.py \
  --model models/quickdraw_model.tflite \
  --quantized models/quickdraw_model_int8.tflite
```

## Available Scripts

### 1. convert_to_tflite.py
Converts Keras models to TensorFlow Lite format.

**Usage:**
```bash
python scripts/convert/convert_to_tflite.py --model models/quickdraw_model.h5
```

### 2. quantize_int8.py
Applies full INT8 post-training quantization.

**Usage:**
```bash
python scripts/convert/quantize_int8.py --model models/quickdraw_model.h5
```

### 3. benchmark_tflite.py
Benchmarks TFLite model performance.

**Usage:**
```bash
python scripts/convert/benchmark_tflite.py --model models/quickdraw_model.tflite
```

### 4. prune_model.py
Applies weight pruning to reduce model size.

**Usage:**
```bash
python scripts/convert/prune_model.py --model models/quickdraw_model.h5 --sparsity 0.5
```

### 5. distill_model.py
Trains a smaller student model using knowledge distillation.

**Usage:**
```bash
python scripts/convert/distill_model.py --teacher models/quickdraw_model.h5
```

### 6. optimize_graph.py
Applies TensorFlow graph-level optimizations.

**Usage:**
```bash
python scripts/convert/optimize_graph.py --model models/quickdraw_model.h5
```

### 7. convert_to_onnx.py
Converts Keras models to ONNX format.

**Usage:**
```bash
python scripts/convert/convert_to_onnx.py --model models/quickdraw_model.h5
```

## Performance Targets

| Optimization | Target Latency | Target Size | Expected Speedup |
|--------------|----------------|-------------|------------------|
| Float32 TFLite | ~70-80ms | ~20-30 MB | 1.0x (baseline) |
| INT8 Quant | **<20ms** | **<5 MB** | **2-4x** |
| Distilled + INT8 | <10ms | <2 MB | 6-8x |

## Docker Support

Build the image:
```bash
cd scripts/convert
docker build -t dhl-convert:tf2onnx .
```

Run conversions:
```bash
docker run --rm -v $(pwd)/../../models:/models dhl-convert:tf2onnx \
  python convert_to_onnx.py --model /models/quickdraw_model.h5
```

For detailed usage of each script, run with `--help` flag.

See [Project Documentation](../../.documentation/) for more details.
