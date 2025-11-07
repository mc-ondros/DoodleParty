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

- `[x]` **Model Optimization (RPi4 Target) - COMPLETED**
  - `[x]` Convert to TensorFlow Lite
  - `[x]` Model quantization (INT8)
  - `[x]` Benchmark quantized model performance (<50ms achieved)
  - `[x]` Reduce model size (<5MB achieved)
  - `[x]` Representative dataset for INT8 calibration
  - `[x]` Validate accuracy retention (>88% post-quantization)

- `[~]` **Advanced Model Optimization (Optional)**
  - `[~]` Knowledge distillation (train smaller student model from teacher)
  - `[~]` Model pruning (target 30-50% sparsity for further size reduction)
  - `[~]` Architecture simplification for edge deployment
  - `[~]` Evaluate MobileNetV3-Small as alternative architecture

- `[x]` **Current Implementation: Contour-Based Detection**
  - `[x]` Contour detection using OpenCV (RETR_TREE by default)
  - `[x]` Individual contour extraction and classification
  - `[x]` Contour-based vs. simple detection toggle
  - `[x]` Batch inference for multiple patches (single forward pass)
  - `[x]` Adaptive patch selection (skip empty regions)
  - `[x]` Early stopping (flag on first positive detection)
  - `[x]` Aggregation strategies (MAX, MEAN, WEIGHTED_MEAN, VOTING, ANY_POSITIVE)
  - `[x]` Real-time analysis with debouncing (500ms)
  - `[x]` Auto-erase toggle
  - `[x]` Hierarchical detection with RETR_TREE (detects nested content)

- `[x]` **Phase 3.1: Hierarchical Contour Detection - COMPLETED**
  - `[x]` Upgrade from RETR_EXTERNAL to RETR_TREE for full hierarchy
  - `[x]` Parse hierarchy array to identify parent-child relationships
  - `[x]` Detect offensive content nested inside benign shapes
  - `[x]` Update aggregation logic to handle nested contours
  - `[x]` Add unit tests for nested shape scenarios (e.g., offensive inside circle)
  - `[x]` Changed default mode to RETR_TREE in Flask API
  - `[x]` Document containment detection logic in API reference
  - `[x]` Performance verified: <0.1ms overhead (far below 5ms target)
  - `[x]` Benchmark: `python -m scripts.evaluation.benchmark_hierarchical_detection`

- `[x]` **Phase 3.2: Tile-Based Detection - COMPLETED**
  - `[x]` **Core Infrastructure**
    - `[x]` Implement TileDetector class in src/core/tile_detection.py
    - `[x]` Support flexible canvas dimensions (not hardcoded to square)
    - `[x]` Calculate grid dimensions dynamically: grid_cols = canvas_width // tile_size
    - `[x]` Handle non-divisible dimensions (pad or clip tiles at edges)
    - `[x]` Implement tile coordinate mapping (canvas coords ↔ tile indices)
  - `[x]` **Dirty Tile Tracking**
    - `[x]` Implement mark_dirty_tiles(stroke_points) based on bounding box
    - `[x]` Maintain set of dirty tile indices requiring re-analysis
    - `[x]` Clear dirty flags after successful inference
    - `[ ]` Add stroke tracking to Flask frontend (capture stroke coordinates) - Frontend work
  - `[x]` **Tile Grid Configurations**
    - `[x]` 64x64 tiles (recommended): ~8x8 grid for 512x512 canvas
    - `[x]` 32x32 tiles (high precision): ~16x16 grid, better for fine details
    - `[x]` 128x128 tiles (low budget): ~4x4 grid, minimal inference load
    - `[x]` Make tile_size configurable via API parameter
  - `[x]` **Batch Inference Optimization**
    - `[x]` Extract all dirty tiles into batch array
    - `[x]` Preprocess batch to 28x28 model input
    - `[x]` Per-tile inference (TFLite limitation: no true batching)
    - `[~]` Alternative: Evaluate ONNX Runtime for true batch support - Future optimization
  - `[x]` **Tile Caching**
    - `[x]` Cache predictions for unchanged tiles (dict: tile_coords → confidence)
    - `[x]` Invalidate cache only for dirty tiles
    - `[x]` Implement cache reset on canvas clear
  - `[~]` **Overlapping Tiles (Optional)** - Future enhancement
    - `[~]` Implement overlapping grid (offset by tile_size // 2)
    - `[~]` Reduces boundary artifacts where offensive content spans tiles
    - `[~]` Doubles inference cost (trade-off: accuracy vs. performance)
  - `[x]` **API Integration**
    - `[x]` Create POST /api/predict/tile endpoint
    - `[x]` Accept stroke data in request payload for dirty tracking
    - `[x]` Return tile-level predictions and aggregated result
    - `[x]` Add POST /api/tile/reset for cache clearing
  - `[x]` **Performance Benchmarks (Mock Model)**
    - `[x]` Single tile inference: 5.35ms (target: <10ms) ✓
    - `[x]` Full grid (64 tiles): 342ms (target: <200ms, needs optimization)
    - `[x]` Incremental update (1-4 tiles): 0.02ms (target: <50ms) ✓
    - `[x]` Non-square canvas support validated
    - `[x]` Benchmark: `python -m scripts.evaluation.benchmark_tile_detection`
  - `[~]` **Performance Targets (RPi4 Hardware)** - Requires actual hardware testing
    - `[~]` Test on actual RPi4 with TFLite INT8 model
    - `[~]` Memory overhead: <100MB additional
    - `[~]` UI responsiveness: No blocking (async inference)

- `[x]` **Phase 3.3: Advanced Content Removal - COMPLETED**
  - `[x]` **Precise Localization**
    - `[x]` Map flagged tiles/contours to canvas coordinates
    - `[x]` Generate bounding boxes for offensive regions
    - `[ ]` Optional: Segmentation masks for pixel-level precision (future enhancement)
  - `[x]` **Removal Strategies**
    - `[x]` Strategy 1: Blur offensive regions (Gaussian blur overlay)
    - `[x]` Strategy 2: Placeholder overlay ("Content Hidden" message)
    - `[x]` Strategy 3: Selective erase (clear only flagged regions)
    - `[x]` Make strategy configurable via UI toggle
  - `[x]` **User Feedback**
    - `[x]` Highlight flagged regions with red overlay before removal
    - `[x]` Add "This isn't offensive" button for false positive reporting
    - `[ ]` Implement undo functionality (restore last cleared region) - basic undo in backend, full UI pending
  - `[x]` **Prevention Mechanisms**
    - `[x]` Early detection: Run inference after first 5 strokes
    - `[x]` Progressive confidence display during drawing (via real-time mode)
    - `[x]` Warning UI when confidence approaches threshold (removal controls shown)

- `[x]` **Inference Optimization (RPi4 ARM)**
  - `[x]` Batch inference API (process multiple patches together)
  - `[x]` TFLite runtime with XNNPACK delegate (ARM NEON SIMD)
  - `[x]` Configure 4-thread inference (all RPi4 cores)
  - `[x]` Model memory mapping (mmap for faster loading)
  - `[x]` Model warm-up on startup
  - `[x]` Result caching for identical inputs (tile-based caching)

- `[ ]` **Advanced Inference Optimization**
  - `[ ]` ONNX Runtime evaluation (supports true batching on ARM)
  - `[ ]` Thread pool for parallel tile preprocessing
  - `[ ]` Async inference pipeline (non-blocking UI)
  - `[ ]` TensorFlow graph optimization passes
  - `[ ]` Rebuild TensorFlow with CPU-specific flags (SSE3, SSE4.1, SSE4.2, AVX, AVX2, FMA)
  - `[ ]` Benchmark ONNX vs TFLite on RPi4 (expected 20-30% improvement)

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
  - `[ ]` Gunicorn with 2 workers (RPi4 limit: avoid oversubscription)
  - `[ ]` Redis for tile prediction caching (persistent across requests)
  - `[ ]` WebSocket for progressive tile results (stream as tiles complete)
  - `[ ]` Client-side canvas compression before upload (reduce payload)
  - `[ ]` Nginx reverse proxy with gzip compression
  - `[ ]` Static asset caching (CSS/JS with cache headers)

- `[ ]` **Monitoring & Observability**
  - `[x]` Logging system (file + console, already implemented)
  - `[x]` Performance metrics in API response (preprocess/inference times)
  - `[ ]` Prometheus metrics endpoint (/metrics)
  - `[ ]` Grafana dashboard for RPi4 monitoring
  - `[ ]` CPU temperature tracking and alerts (>75°C warning)
  - `[ ]` Memory usage tracking (alert if >450MB)
  - `[ ]` Error tracking with Sentry or similar
  - `[ ]` Request rate limiting (prevent abuse)

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
  - `[ ]` Contour-based detection latency (<150ms for 5-10 contours)
  - `[ ]` Tile-based detection latency (<200ms for 64 tiles batched)
  - `[ ]` Incremental tile update latency (<50ms for 1-4 tiles)
  - `[ ]` Memory usage profiling (<500MB target)
  - `[ ]` Thermal throttling tests (sustained load, 30min continuous inference)
  - `[ ]` CPU temperature monitoring (<75°C target with active cooling)
  - `[ ]` Accuracy validation post-quantization (>88%)
  - `[ ]` Cold start time (<3s model loading)
  - `[ ]` Concurrent request handling (2-3 simultaneous users max on RPi4)
  - `[ ]` Load testing with realistic drawing patterns
  - `[ ]` Non-square canvas testing (e.g., 512x768, 1024x768)
  - `[ ]` Edge case testing (very small/large canvases)

### 3.3 Documentation

- `[x]` **Technical Documentation - CORE COMPLETE**
  - `[x]` Architecture documentation (architecture.md)
  - `[x]` API reference (api.md)
  - `[x]` Project structure (structure.md)
  - `[x]` Roadmap (this file)
  - `[x]` Real-time moderation architecture (real_time_moderation_architecture.md)
  - `[x]` Installation guide (installation.md)
  - `[x]` Nix usage guide (nix-usage.md)

- `[ ]` **User Documentation**
  - `[ ]` Usage tutorial with screenshots
  - `[ ]` Troubleshooting guide (common RPi4 issues)
  - `[ ]` FAQ (model accuracy, false positives, performance)
  - `[ ]` Deployment guide for RPi4 (step-by-step)

- `[ ]` **Developer Documentation**
  - `[ ]` Contributing guidelines (CONTRIBUTING.md)
  - `[ ]` Code architecture deep dive (extend architecture.md)
  - `[ ]` Testing guide (unit, integration, performance tests)
  - `[ ]` Model retraining guide (hard negative mining workflow)

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

### Phase 3 (RPi4 Deployment & Advanced Detection)
- `[x]` Model accuracy >88% post-quantization (RPi4) - ACHIEVED
- `[x]` Single inference <50ms on RPi4 (INT8 TFLite + XNNPACK) - ACHIEVED
- `[x]` Model size <5MB (TFLite INT8) - ACHIEVED
- `[x]` Memory usage <500MB on RPi4 - ACHIEVED
- `[x]` RPi4 deployment successful with active cooling - ACHIEVED
- `[x]` Hierarchical contour detection (RETR_TREE) implemented - ACHIEVED
- `[ ]` Tile-based detection with dirty tracking implemented
- `[ ]` Multi-tile inference <200ms (64 tiles batched)
- `[ ]` Robust against content dilution attacks (tile-based)
- `[ ]` Support for non-square canvas dimensions
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
