# DoodleHunter Development Roadmap

**Goal:** Build a binary classification system for drawing content moderation using TensorFlow and QuickDraw dataset.

**Status: Experimental** - This project is under active development. Features marked as complete may still need verification.

## Table of Contents

- [Legend](#legend)
- [Phase 1: Foundation & Core Features](#phase-1-foundation--core-features)
- [Phase 2: Enhancement & Optimization](#phase-2-enhancement--optimization)
- [Phase 3: Production Ready](#phase-3-production-ready)
- [Future Enhancements](#future-enhancements)

## Legend

- `[x]` Complete
- `[-]` In Progress
- `[~]` Marked as complete but unverified
- `[ ]` Todo
- `[!]` Blocked
- `[E]` Experimental/Unstable

## Phase 1: Foundation & Core Features

### 1.1 Data Pipeline

- `[x]` **QuickDraw Dataset Integration**
  - `[x]` Download QuickDraw NPY files
  - `[x]` Implement data loading from NPY format
  - `[x]` Create category selection system
  - `[x]` Implement train/validation split

- `[x]` **Data Preprocessing**
  - `[x]` Image handling (28x28 NumPy bitmap format)
  - `[x]` Normalization to [0, 1] range + z-score normalization
  - `[x]` Grayscale conversion
  - `[x]` Class mapping creation

- `[x]` **Data Augmentation**
  - `[x]` Random rotation (±15°)
  - `[x]` Random translation (±10%)
  - `[x]` Random zoom (90-110%)
  - `[x]` Horizontal flip

### 1.2 Model Development

- `[x]` **CNN Architecture**
  - `[~]` Multiple CNN architectures (Custom, ResNet50, MobileNetV3, EfficientNet)
  - `[x]` Implement Conv2D + MaxPooling layers
  - `[x]` Add Dense layers with dropout
  - `[x]` Binary classification output (sigmoid)

- `[x]` **Training Pipeline**
  - `[x]` Model compilation (Adam optimizer)
  - `[x]` Binary crossentropy loss
  - `[x]` Training loop implementation
  - `[x]` Validation during training
  - `[x]` Model checkpointing
  - `[x]` Early stopping

- `[x]` **Model Evaluation**
  - `[x]` Accuracy, precision, recall metrics
  - `[x]` Confusion matrix generation
  - `[x]` ROC curve and AUC
  - `[x]` Sample predictions visualization

### 1.3 Web Interface

- `[x]` **Flask Application**
  - `[x]` Flask server setup
  - `[x]` Route definitions (/, /predict, /health)
  - `[x]` Model loading on startup
  - `[x]` Error handling

- `[x]` **Drawing Interface**
  - `[x]` HTML5 Canvas implementation
  - `[x]` Drawing functionality (mouse/touch)
  - `[x]` Clear canvas button
  - `[x]` Submit for classification

- `[x]` **Inference Pipeline**
  - `[x]` Base64 image decoding
  - `[x]` Image preprocessing for inference
  - `[x]` Model prediction
  - `[x]` Result formatting and display

### 1.4 Documentation

- `[x]` **Core Documentation**
  - `[x]` README.md with quick start
  - `[x]` STYLE_GUIDE.md for code standards
  - `[x]` requirements.txt for dependencies
  - `[x]` .gitignore configuration

- `[x]` **Technical Documentation**
  - `[x]` API reference (api.md)
  - `[x]` Architecture documentation (architecture.md)
  - `[x]` Project structure (structure.md)
  - `[x]` Development roadmap (this file)

## Phase 2: Enhancement & Optimization

### 2.1 Model Improvements

- `[x]` **Advanced Training**
  - `[x]` Implement learning rate scheduling
  - `[x]` Add class weighting for imbalanced data
  - `[x]` Experiment with different architectures
  - `[x]` Ensemble multiple models

- `[x]` **Threshold Optimization**
  - `[x]` Implement threshold optimization script
  - `[x]` ROC curve analysis for optimal threshold
  - `[x]` Precision-recall trade-off analysis
  - `[x]` Per-class threshold tuning

- `[~]` **Test-Time Augmentation**
  - `[~]` TTA implementation script
  - `[~]` Multiple augmentation strategies
  - `[~]` Prediction averaging
  - `[~]` Confidence score calibration

### 2.2 Web Interface Enhancements

- `[~]` **UI Improvements**
  - `[~]` Responsive design for mobile
  - `[~]` Better visual feedback
  - `[~]` Confidence score visualization
  - `[~]` Drawing history

- `[~]` **Performance Optimization**
  - `[~]` Model caching
  - `[~]` Request batching
  - `[~]` Async inference
  - `[~]` Response compression

### 2.3 Data Quality

- `[~]` **Hard Negative Mining**
  - `[~]` Hard negative generation script
  - `[~]` Identify misclassified samples
  - `[~]` Retrain with hard negatives
  - `[~]` Iterative improvement

- `[~]` **Dataset Expansion**
  - `[~]` Add more QuickDraw categories
  - `[~]` Balance class distribution
  - `[~]` Collect edge cases
  - `[~]` Validate data quality

## Phase 3: Production Ready

### 3.1 Deployment

- `[~]` **Production Setup**
  - `[~]` Gunicorn for multi-worker deployment
  - `[~]` Nginx reverse proxy
  - `[~]` SSL/TLS configuration
  - `[~]` Environment configuration

- `[ ]` **Model Optimization (RPi4 Target)**
  - `[x]` Convert to TensorFlow Lite
  - `[x]` Model quantization (INT8)
  - `[x]` Benchmark quantized model performance (target: <20ms per inference)
  - `[x]` Reduce model size (<25MB for TFLite)
  - `[ ]` Knowledge distillation (train smaller student model from teacher)
  - `[ ]` Model pruning (target 30-50% sparsity)
  - `[ ]` Architecture simplification for edge deployment
  - `[ ]` Representative dataset for INT8 calibration
  - `[ ]` Validate accuracy retention (>88% post-quantization)

- `[~]` **Robustness Improvements (Deprecated - Replaced by Tile-Based Detection)**
  - `[~]` Region-based detection (sliding window) - **Failed: didn't isolate individual shapes**
  - `[~]` Implement canvas patch extraction - **Replaced by contour-based detection**
  - `[~]` Batch inference for multiple patches (single forward pass)
  - `[~]` Adaptive patch selection (skip empty regions)
  - `[~]` Early stopping (flag on first positive detection)
  - `[~]` Aggregation strategy for patch predictions
  - `[~]` Prevent content dilution attacks - **Partial: needs tile-based approach**

- `[x]` **Current Implementation: Contour-Based Detection**
  - `[x]` Contour detection using OpenCV (RETR_EXTERNAL)
  - `[x]` Individual contour extraction and classification
  - `[x]` Contour-based vs. simple detection toggle
  - `[x]` Gradual stroke erasing (all strokes when positive)
  - `[x]` Real-time analysis with debouncing (500ms)
  - `[x]` Auto-erase toggle
  - `[ ]` **Limitation**: Cannot map contours to individual strokes for selective removal
  - `[ ]` **Next**: Implement tile-based detection for better granularity

- `[ ]` **Advanced Detection: Tile-Based Segmentation**
  - `[ ]` **Fixed Tile Grid (64x64 Recommended)**
    - `[ ]` Divide canvas into 8x8 grid (64 tiles, ~1.2% coverage each)
    - `[ ]` Update only tiles affected by new strokes (dirty tracking)
    - `[ ]` Run inference on changed tiles only
    - `[ ]` Flag individual tiles containing offensive content
    - `[ ]` Batch inference for all dirty tiles in single forward pass
  - `[ ]` **Alternative Tile Grids**
    - `[ ]` 32x32 High Precision (16x16 grid, 256 tiles) - better for fine details
    - `[ ]` 128x128 Low Budget (4x4 grid, 16 tiles) - minimal inference load
  - `[ ]` **Tile Optimization**
    - `[ ]` Implement overlapping tiles (offset by 32px) to reduce boundary artifacts
    - `[ ]` Maintain dirty tile tracking for incremental updates
    - `[ ]` Cache results for unchanged regions
  - `[ ]` **Two-Stage Detection Pipeline**
    - `[ ]` Stage 1: Fast classifier flags suspicious tiles
    - `[ ]` Stage 2: Object detection/segmentation localizes offensive parts
    - `[ ]` Generate bounding boxes or segmentation masks
    - `[ ]` Remove/blur only precise offensive regions
  - `[ ]` **Content Removal Strategies**
    - `[ ]` Stroke-level removal (calculate intersection with offensive regions)
    - `[ ]` Overlay masking (blur or placeholder)
    - `[ ]` Live warning UI (highlight and allow user modification)
    - `[ ]` Selective stroke removal (only offensive strokes, not all)
  - `[ ]` **Prevention Mechanisms**
    - `[ ]` Run classifier after first 3-5 strokes (early warning)
    - `[ ]` Shape prototype detection (match against offensive patterns)
    - `[ ]` Inline prompts when similarity detected
  - `[ ]` **Performance Targets**
    - `[ ]` Single tile inference: <10ms
    - `[ ]` Full 64-tile grid: <200ms (batched)
    - `[ ]` Incremental update (1-4 tiles): <50ms
    - `[ ]` UI responsiveness: <16ms frame time maintained
    - `[ ]` Memory overhead: <100MB additional

- `[-]` **Inference Optimization (RPi4 ARM)**
  - `[x]` Batch inference API (process multiple patches together)
  - `[x]` TFLite runtime with XNNPACK delegate (ARM NEON SIMD)
  - `[x]` Configure 4-thread inference (all RPi4 cores)
  - `[x]` Model memory mapping (mmap for faster loading)
  - `[x]` Model warm-up on startup
  - `[ ]` TensorFlow graph optimization
  - `[ ]` ONNX Runtime evaluation (alternative to TFLite)
  - `[ ]` Thread pool for parallel preprocessing
  - `[ ]` Async inference pipeline
  - `[x]` Result caching for identical inputs (tile-based caching)

- `[x]` **RPi4 System Optimization**
  - `[x]` CPU governor set to 'performance' mode
  - `[x]` Active cooling setup (heatsink + fan)
  - `[x]` Thermal monitoring and throttling detection
  - `[x]` Disable swap or minimize swappiness
  - `[x]` Process priority configuration
  - `[ ]` Minimal OS (Raspberry Pi OS Lite)
  - `[ ]` Garbage collection tuning
  - `[x]` Memory usage profiling (<500MB target)

- `[ ]` **Infrastructure Optimization**
  - `[ ]` Gunicorn with multiple workers (limited on RPi4)
  - `[ ]` Load balancing across workers
  - `[ ]` Redis for prediction caching
  - `[ ]` CDN for static assets
  - `[ ]` WebSocket for progressive results
  - `[ ]` Client-side preprocessing (reduce payload)

- `[ ]` **Monitoring**
  - `[ ]` Logging system
  - `[ ]` Performance metrics (latency, throughput)
  - `[ ]` Error tracking
  - `[ ]` Usage analytics

### 3.2 Testing

- `[~]` **Unit Tests**
  - `[~]` Data pipeline tests
  - `[~]` Model architecture tests
  - `[~]` Preprocessing tests
  - `[~]` API endpoint tests

- `[~]` **Integration Tests**
  - `[~]` End-to-end workflow tests
  - `[~]` Flask application tests
  - `[~]` Model inference tests
  - `[~]` Error handling tests

- `[ ]` **Performance Tests (RPi4 Hardware)**
  - `[ ]` Single inference latency on RPi4 (<50ms target)
  - `[ ]` Multi-region inference latency (<200ms for 9-16 patches)
  - `[ ]` Batch inference throughput (16 patches in ~2x single time)
  - `[ ]` Memory usage profiling (<500MB target)
  - `[ ]` Thermal throttling tests (sustained load)
  - `[ ]` CPU temperature monitoring (<75°C target)
  - `[ ]` Accuracy validation post-quantization (>88%)
  - `[ ]` Cold start time (<3s model loading)
  - `[ ]` Concurrent request handling (limited on RPi4)
  - `[ ]` Load testing (realistic RPi4 throughput)

### 3.3 Documentation

- `[ ]` **User Documentation**
  - `[ ]` Installation guide
  - `[ ]` Usage tutorial
  - `[ ]` Troubleshooting guide
  - `[ ]` FAQ

- `[ ]` **Developer Documentation**
  - `[ ]` Contributing guidelines
  - `[ ]` Code architecture deep dive
  - `[ ]` Testing guide
  - `[ ]` Deployment guide

## Future Enhancements

### Advanced Features

- `[ ]` **Multi-Class Classification**
  - `[ ]` Extend to multiple categories
  - `[ ]` Hierarchical classification
  - `[ ]` Fine-grained categorization

- `[ ]` **Real-Time Features**
  - `[ ]` WebSocket for live classification
  - `[ ]` Progressive drawing analysis
  - `[ ]` Confidence updates during drawing

- `[ ]` **User Features**
  - `[ ]` User accounts and history
  - `[ ]` Save/load drawings
  - `[ ]` Gallery of classified drawings
  - `[ ]` Batch upload and classification

### Technical Improvements

- `[ ]` **Model Enhancements**
  - `[ ]` Transfer learning from larger models
  - `[ ]` Attention mechanisms
  - `[ ]` Multi-modal input (strokes + raster)
  - `[ ]` Explainable AI (CAM, Grad-CAM)

- `[ ]` **Infrastructure**
  - `[ ]` Docker containerization
  - `[ ]` Kubernetes deployment
  - `[ ]` Cloud deployment (AWS/GCP)
  - `[ ]` CDN for static assets

- `[ ]` **Analytics**
  - `[ ]` Classification statistics dashboard
  - `[ ]` Model performance tracking
  - `[ ]` A/B testing framework
  - `[ ]` User behavior analytics

## Success Metrics

### Phase 1 (Completed)
- `[x]` Model accuracy >80% on validation set
- `[x]` Flask app serves predictions successfully
- `[x]` Drawing interface functional
- `[x]` Core documentation complete

### Phase 2 (Completed)
- `[x]` Model accuracy >90% on validation set
- `[x]` Inference time <100ms
- `[x]` Web interface responsive on mobile
- `[x]` Hard negative mining improves accuracy

### Phase 3 (RPi4 Deployment)
- `[ ]` Model accuracy >88% post-quantization (RPi4)
- `[ ]` Single inference <50ms on RPi4 (INT8 TFLite + XNNPACK)
- `[ ]` Multi-region inference <200ms (9-16 patches batched)
- `[ ]` Model size <5MB (TFLite INT8)
- `[ ]` Memory usage <500MB on RPi4
- `[ ]` Robust against content dilution attacks
- `[ ]` RPi4 deployment successful with active cooling
- `[ ]` Comprehensive test coverage (>80%)
- `[ ]` Complete user and developer documentation

## Timeline

**Phase 1:** In Progress
- Core functionality implemented
- Basic web interface working
- Initial model trained

**Phase 2:** In Progress
- Model improvements partially implemented
- UI enhancements partially implemented
- Data quality improvements partially implemented

**Phase 3:** Planned
- RPi4 deployment optimization
- Comprehensive testing on target hardware
- Full documentation

## Dependencies

**Required:**
- Python 3.9+
- TensorFlow 2.13+
- Flask 2.0+
- NumPy, Pandas, Matplotlib
- QuickDraw dataset

**Optional:**
- Gunicorn (production)
- Redis (caching)
- Docker (containerization)

**Model Optimization:**
- TensorFlow Lite (included in TensorFlow 2.13+)
- ONNX Runtime (alternative inference engine)
- Representative dataset for INT8 quantization calibration

**Performance Tools:**
- TensorFlow Model Optimization Toolkit
- TensorFlow Profiler
- Locust or Apache Bench (load testing)
- cProfile (Python profiling)

## Risk Mitigation

**Technical Risks:**
- **Model Accuracy:** Continuous evaluation and retraining
- **Inference Speed:** Multi-stage optimization (quantization, batching, caching)
- **Data Quality:** Validation and hard negative mining
- **Quantization Accuracy Loss:** Benchmark INT8 model against float32 baseline, use representative calibration dataset
- **Content Dilution Attacks:** Region-based detection to prevent circumvention by mixing innocent and inappropriate content
- **Batch Processing Complexity:** Careful implementation of batch inference with proper error handling
- **Memory Constraints:** Monitor memory usage with multiple patches, implement streaming if needed

**Operational Risks:**
- **Deployment Issues:** Thorough testing and staging environment
- **Scalability:** Load testing and optimization
- **Maintenance:** Comprehensive documentation

## Related Documentation

- [Architecture](architecture.md) - System design
- [API Reference](api.md) - API documentation
- [Project Structure](structure.md) - File organization
- [README](../README.md) - Project overview

*Development roadmap for DoodleHunter v1.0*
