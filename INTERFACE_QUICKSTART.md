# Quick Start Guide - DoodleHunter Interface

## 🚀 Get Started in 2 Minutes

### Option 1: Using the Startup Script (Recommended)

From the ML project root directory:

```bash
bash run_interface.sh
```

This will:
1. Check for virtual environment
2. Install Flask if needed
3. Start the web server automatically
4. Open at `http://localhost:5000`

### Option 2: Manual Start

```bash
# Activate virtual environment
source .venv/bin/activate

# Install Flask (first time only)
pip install flask flask-cors

# Start the app
cd app
python app.py
```

## 🎨 Using the Interface

1. **Navigate to** `http://localhost:5000` in your browser
2. **Draw** a doodle on the canvas using your mouse or trackpad
3. **Adjust** brush size with the slider
4. **Click "Get Verdict"** to send your drawing to the ML model
5. **View results** including:
   - Model's prediction (IN or OUT of distribution)
   - Confidence score (0-100%)
   - Raw probability (0.0-1.0)
   - Decision threshold

## 📋 What the Model Does

The QuickDraw classifier identifies whether your drawing matches the style of the QuickDraw dataset:

- **✓ IN-DISTRIBUTION**: Drawing looks like it matches QuickDraw style
- **✗ OUT-OF-DISTRIBUTION**: Drawing doesn't match QuickDraw style

## ⚙️ Technical Details

- **Input Size**: 28×28 grayscale pixels
- **Model**: CNN trained on QuickDraw dataset
- **Output**: Binary classification (probability 0-1)
- **Decision Threshold**: 0.5 (configurable)

## 🔧 Configuration

### Change the Prediction Threshold

Edit `app/app.py`, line ~18:

```python
THRESHOLD = 0.5  # Change to 0.4, 0.6, etc.
```

Lower threshold → More predictions classified as IN-DISTRIBUTION
Higher threshold → More predictions classified as OUT-OF-DISTRIBUTION

### Change Port

Edit `app/app.py`, last line:

```python
app.run(debug=True, host='0.0.0.0', port=8000)  # Change from 5000 to 8000
```

## 🐛 Troubleshooting

**Server won't start:**
- Ensure you're in the ML project root directory
- Check that `.venv` virtual environment exists
- Try: `pip install flask flask-cors`

**Model not loading:**
- Check that `models/quickdraw_model.h5` exists
- Check console for detailed error message
- Verify TensorFlow is installed: `pip install tensorflow`

**Drawing not responsive:**
- Try clearing cache (Ctrl+F5)
- Check browser console for JavaScript errors
- Ensure JavaScript is enabled

**Port already in use:**
- Change port in startup script or `app.py`
- Or find and kill the process: `lsof -i :5000`

## 📁 Project Structure

```
ML/
├── app/                          # Web interface (THIS IS NEW)
│   ├── app.py                   # Flask backend
│   ├── requirements.txt         # Dependencies
│   ├── templates/
│   │   └── index.html          # Web interface
│   └── static/
│       ├── style.css           # Styling
│       └── script.js           # Drawing logic
│
├── models/
│   ├── quickdraw_model.h5      # Trained model
│   └── ...
├── src/
│   ├── predict.py              # Original prediction script
│   └── ...
└── run_interface.sh            # Startup script
```

## 🎯 Next Steps

1. Run the interface
2. Test with different drawing styles
3. Adjust threshold if needed
4. Explore model behavior with various inputs

## 📚 More Information

- Full documentation: `app/README.md`
- API reference: `app/README.md#api-endpoints`
- Model info: `../README.md`

---

**Made with ❤️ for DoodleHunter ML Testing**
