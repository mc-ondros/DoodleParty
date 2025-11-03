# QuickDraw ML Classifier - Binary Detector

A machine learning project for training a binary classifier that detects whether a doodle belongs to a specific dataset or not.

## Overview

This project uses TensorFlow/Keras to train a convolutional neural network (CNN) as a binary classifier. The model learns to distinguish between:
- **Positive class**: Doodles that match the dataset (in-distribution)
- **Negative class**: Random doodles or sketches that don't match (out-of-distribution)

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
git clone <repository-url>
cd ML
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

### Download and Prepare Data

```bash
python src/dataset.py --download --classes airplane apple banana --output-dir data/processed
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

The project uses a CNN with:
- 3 Convolutional layers with ReLU activation
- Batch normalization and dropout for regularization
- Dense layers for feature extraction
- **Single sigmoid output neuron** for binary classification (0 = out-of-distribution, 1 = in-distribution)

## Binary Classification Approach

**Positive Class (Label 1)**: Doodles from the QuickDraw dataset (in-distribution)
**Negative Class (Label 0)**: Random noise/out-of-distribution samples

The model learns to distinguish between valid doodles and random noise by optimizing binary cross-entropy loss.

## Dataset

The QuickDraw dataset is sourced from: https://github.com/googlecreativelab/quickdraw-dataset

Each drawing is a 28x28 grayscale image.

## Training Results

Results will be displayed after training completes, including:
- Training and validation accuracy
- Training and validation loss curves
- Model summary

## Future Improvements

- [ ] Add data augmentation
- [ ] Experiment with different architectures
- [ ] Add cross-validation
- [ ] Implement model ensemble methods
- [ ] Add real-time drawing prediction interface

## License

MIT

## References

- QuickDraw Dataset: https://github.com/googlecreativelab/quickdraw-dataset
- TensorFlow: https://www.tensorflow.org/
- Keras: https://keras.io/
