# Changelog

All notable changes to the DoodleHunter project.

## [2.0.0] - 2025-11-05

### 🚀 Major Upgrade: High-Resolution Model

#### Added
- **128×128 High-Resolution Model**
  - Upgraded from 28×28 to 128×128 by rendering from NDJSON vector strokes
  - Achieved **97.25% test accuracy** on 10,080 samples
  - 25.8M parameters, optimized CNN architecture
  - ~70ms inference time on CPU

- **Comprehensive Documentation**
  - `docs/ML_API_DOCUMENTATION.md` - Complete API reference with preprocessing details
  - Updated README with architecture, performance metrics, and troubleshooting
  - Drawing guidelines for optimal model performance
  - Python and JavaScript usage examples

- **Data Pipeline Overhaul**
  - Download QuickDraw NDJSON files (raw vector strokes, not pre-rendered bitmaps)
  - Render at 256×256 with 12px strokes → downsample to 128×128 with LANCZOS
  - Per-image normalization prevents brightness-based classification shortcuts
  - Consistent 6px effective stroke width across all classes

- **Web Application Enhancements**
  - Updated preprocessing to exactly match training pipeline
  - Increased default brush size from 5px to 24px (matches training data)
  - Improved canvas resolution to 512×512 for smoother drawing
  - Real-time predictions with ~100ms total response time

- **Data Processing Scripts**
  - `scripts/data_processing/download_quickdraw_ndjson.py` - Download raw stroke data
  - `scripts/data_processing/process_all_data_128x128.py` - Render all data at 128×128
  - `scripts/data_processing/regenerate_training_data.py` - Create train/test splits
  - `scripts/visualization/` - Multiple visualization utilities

#### Fixed
- **Data Quality Issues**
  - Fixed color inversion in airplane and apple classes (white bg → black bg)
  - Filtered 7,894 blank/corrupted samples from apple class (78.9% were blank)
  - Ensured consistent black background + white strokes across all classes
  - Verified sharpness parity: penis 0.0174 vs QuickDraw 0.0176 (nearly identical)

- **Model Classification Issues**
  - Fixed web app brush size (too thin at 5px, now 24px default)
  - Web app drawings now have proper stroke width matching training data
  - Preprocessing pipeline matches training exactly (color inversion + per-image norm)

#### Changed
- **Dataset Composition**
  - Positive class: 25,209 penis drawings (from custom NDJSON)
  - Negative class: 25,200 QuickDraw drawings (21 classes × 1,200 samples each)
  - Total training: 40,320 samples | Total testing: 10,080 samples
  - All rendered from vector strokes at native 128×128 resolution

- **Model Training**
  - Input shape: (None, 128, 128, 1) instead of (None, 28, 28, 1)
  - Training time: ~8 hours on CPU (Intel with AVX512) for 50 epochs
  - No data augmentation needed - high-res data generalizes well

- **Preprocessing Pipeline**
  - Render from strokes → 256×256 with 12px width
  - Downsample → 128×128 using LANCZOS filter
  - Normalize → [0, 1] range
  - Per-image normalize → center around mean, scale by std, rescale to [0, 1]
  - Result → Grey background (~0.45), white strokes (~0.9-1.0)

#### Removed
- Low-resolution 28×28 pre-rendered bitmap approach
- Old visualization files and confusion matrix images
- Debug code from Flask application
- Temporary processing scripts

### Performance Improvements
- **Accuracy**: 97.25% (up from baseline)
- **Image Quality**: Sharp 128×128 images vs blurry upscaled 28×28
- **Inference Speed**: 70ms per image (CPU), 4ms batched
- **Model Size**: 296MB (25.8M parameters)

### Technical Details
- Python 3.12, TensorFlow 2.x with Keras
- PIL/Pillow for image processing with LANCZOS resampling
- Flask web app with CORS support
- RESTful JSON API for predictions

---

## [1.0.0] - Initial Release

### Added
- Basic CNN binary classifier
- 28×28 image processing
- QuickDraw dataset integration
- Flask web interface
- Model training scripts
