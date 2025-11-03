# DoodleHunter ML Testing Interface

A simple web-based drawing board interface to test the DoodleHunter ML model in real-time.

## Features

- **Interactive Drawing Canvas**: Draw directly in your browser with adjustable brush size
- **Real-time Prediction**: Send your drawing to the ML model and get instant feedback
- **Confidence Score**: See how confident the model is about its prediction
- **Clean UI**: Modern, responsive interface that works on desktop and mobile
- **Model Info**: Display raw probability and decision threshold

## Setup

### Prerequisites

- Python 3.8+
- The DoodleHunter ML model trained and saved at `../models/quickdraw_model.h5`
- The class mapping file at `../data/processed/class_mapping.pkl`

### Installation

1. Install Flask and dependencies:
```bash
pip install -r requirements.txt
```

Or use the main project requirements:
```bash
cd ..
pip install -r requirements.txt
pip install flask flask-cors
```

## Usage

### Start the Server

Run the Flask app from the app directory:

```bash
python app.py
```

Or from the project root:
```bash
python -m flask --app app/app run
```

### Access the Interface

Open your browser and navigate to:
```
http://localhost:5000
```

You should see the drawing board interface.

## How to Use

1. **Draw**: Use your mouse (or touch on mobile) to draw on the canvas
2. **Adjust Brush**: Use the brush size slider to change pen thickness
3. **Get Verdict**: Click "Get Verdict" to send your drawing to the ML model
4. **View Results**: See the model's prediction with confidence score and raw probability
5. **Clear**: Click "Clear" to erase the canvas and start over

## API Endpoints

### POST `/api/predict`
Send a drawing for prediction.

**Request Body:**
```json
{
  "image": "<base64_encoded_png_image>"
}
```

**Response:**
```json
{
  "success": true,
  "verdict": "IN-DISTRIBUTION",
  "verdict_text": "Looks like a QuickDraw doodle! ✓",
  "confidence": 0.95,
  "raw_probability": 0.95,
  "threshold": 0.5
}
```

### GET `/api/health`
Check if the server and model are ready.

**Response:**
```json
{
  "status": "ok",
  "model_loaded": true,
  "threshold": 0.5
}
```

## Prediction Details

The model works as follows:

1. **Input**: Your drawing is converted to a 28×28 grayscale image
2. **Processing**: The image is normalized and fed to the CNN
3. **Output**: The model outputs a probability between 0 and 1
4. **Decision**: 
   - Probability ≥ 0.5 → **IN-DISTRIBUTION** (valid QuickDraw style)
   - Probability < 0.5 → **OUT-OF-DISTRIBUTION** (not QuickDraw style)

## Troubleshooting

### Model Not Loading
- Verify the model file exists at `../models/quickdraw_model.h5`
- Check the console output for error messages
- Ensure TensorFlow is installed: `pip install tensorflow>=2.13.0`

### Predictions Not Working
- Make sure you draw something on the canvas before clicking "Get Verdict"
- Check browser console (F12) for JavaScript errors
- Check Flask console for backend errors

### Port Already in Use
If port 5000 is in use, you can change it in `app.py`:
```python
app.run(debug=True, host='0.0.0.0', port=5001)  # Use different port
```

## File Structure

```
app/
├── app.py                    # Flask application
├── requirements.txt          # Python dependencies
├── templates/
│   └── index.html           # Main interface HTML
├── static/
│   ├── style.css            # Styling
│   └── script.js            # Drawing and prediction logic
└── README.md                # This file
```

## Customization

### Adjust Brush Smoothing
Edit the `draw()` function in `script.js` to adjust line smoothness.

### Change Threshold
Edit `THRESHOLD` in `app.py` to adjust the decision boundary:
```python
THRESHOLD = 0.5  # Change this value
```

### Modify Canvas Size
Update the canvas dimensions in `templates/index.html` and `app.py`:
```html
<canvas id="drawingCanvas" width="500" height="500"></canvas>
```

Also update the preprocessing in `app.py` to match (currently 28×28 for model input).

## Notes

- The interface automatically resizes drawings to 28×28 pixels for model input
- Drawings are converted to grayscale as the model expects
- The model uses sigmoid activation, so output is a probability [0, 1]
- Binary classification: 0 = negative (OOD), 1 = positive (in-distribution)

## Future Enhancements

- [ ] Save/export predictions history
- [ ] Batch upload for multiple images
- [ ] Adjustable confidence threshold
- [ ] Model comparison (test multiple models)
- [ ] Visualization of model internals (feature maps)
- [ ] Mobile app version
- [ ] Dark mode theme

## License

MIT - Same as DoodleHunter project
