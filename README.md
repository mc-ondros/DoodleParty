# DoodleHunter 🎨

A machine learning project for training a CNN-based binary classifier that detects whether a doodle belongs to the QuickDraw dataset or not.

## Overview

**DoodleHunter** uses TensorFlow/Keras to train a convolutional neural network (CNN) as a binary classifier. The model learns to distinguish between:
- **Positive class**: Doodles that match the QuickDraw dataset (in-distribution)
- **Negative class**: Random doodles, sketches, or out-of-distribution samples

## Project Structure

```
├── data/                 # Dataset directory
│   ├── raw/             # Raw downloaded data
│   └── processed/       # Processed training data
├── models/              # Trained model files
├── notebooks/           # Jupyter notebooks for exploration
├── src/
│   ├── dataset.py       # Data loading and preprocessing
│   ├── train.py         # Model training script
│   └── predict.py       # Inference and evaluation
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

## Setup

### Prerequisites

- Python 3.8+
- pip

### Installation

1. Clone the repository:
```bash
git clone https://github.com/mc-ondros/DoodleHunter.git
cd DoodleHunter
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Download QuickDraw Dataset

```bash
# Download 21 categories from Google Cloud Storage (optimized numpy format)
python src/download_quickdraw_npy.py
```

### Prepare Data for Training

```bash
python src/dataset.py --classes airplane apple banana cat dog --output-dir data/processed --max-samples 5000
```

### Train the Model

```bash
python src/train.py --data-dir data/processed --epochs 50 --batch-size 32 --model-output models/quickdraw_model.h5
```

### Make Predictions

```bash
python src/predict.py --model models/quickdraw_model.h5 --image path/to/image.png
```

## Model Architecture

**DoodleHunter** uses a CNN with:
- 3 Convolutional layers with ReLU activation
- Batch normalization and dropout for regularization  
- Dense layers for feature extraction
- **Single sigmoid output neuron** for binary classification
  - 0 = Out-of-distribution (not a QuickDraw doodle)
  - 1 = In-distribution (valid QuickDraw doodle)

The model optimizes binary cross-entropy loss to distinguish between valid QuickDraw doodles and out-of-distribution samples.

## Dataset

**DoodleHunter** uses the QuickDraw dataset from Google's Creative Lab:
- Source: https://github.com/googlecreativelab/quickdraw-dataset
- Format: Pre-processed 28x28 grayscale bitmaps
- Categories: 21 different object types (~2.6M total doodles)
- Download: `src/download_quickdraw_npy.py` script handles automated downloads

## Training Results

Results will be displayed after training completes, including:
- Training and validation accuracy
- Training and validation loss curves
- Model summary

## Future Improvements

- [ ] Add data augmentation (rotation, scaling, distortion)
- [ ] Experiment with different CNN architectures (ResNet, MobileNet)
- [ ] Implement multi-class classification for specific object types
- [ ] Add cross-validation and k-fold evaluation
- [ ] Real-time drawing prediction web interface
- [ ] Model quantization for mobile deployment
- [ ] Ensemble methods for improved accuracy

## License

MIT

## References

- QuickDraw Dataset: https://github.com/googlecreativelab/quickdraw-dataset
- TensorFlow: https://www.tensorflow.org/
- Keras: https://keras.io/
