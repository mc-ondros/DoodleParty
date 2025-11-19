#!/usr/bin/env python3
"""
ML Inference Server for DoodleParty
Receives images from Express server via Socket.IO and runs inference
"""

import os
import sys
import json
import base64
import numpy as np
from io import BytesIO
from pathlib import Path
from datetime import datetime
import logging
from PIL import Image
import socketio
import eventlet
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Setup paths
PROJECT_ROOT = Path(__file__).parent.parent.parent  # Go up from ml/socket_client to repo root
MODEL_PATH = PROJECT_ROOT / 'models' / 'quickdraw_model_int8.tflite'
VIZ_DIR = PROJECT_ROOT / 'data' / 'ml_visualizations'
VIZ_DIR.mkdir(parents=True, exist_ok=True)

# Configuration flags
ENABLE_VISUALIZATIONS = os.environ.get('ENABLE_VISUALIZATIONS', 'false').lower() == 'true'

# Load model
logger.info("Loading TFLite model...")
try:
    import tensorflow as tf
    interpreter = tf.lite.Interpreter(model_path=str(MODEL_PATH))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    logger.info(f"✓ Model loaded: {MODEL_PATH}")
    logger.info(f"  Input shape: {input_details[0]['shape']}")
    logger.info(f"  Output shape: {output_details[0]['shape']}")
    MODEL_LOADED = True
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    logger.warning("Running in demo mode without ML inference")
    interpreter = None
    MODEL_LOADED = False

# Socket.IO client to connect to Express server
sio = socketio.Client(logger=False, engineio_logger=False)

class MLInferenceServer:
    def __init__(self):
        self.inference_count = 0
        self.session_results = {}
        
    def preprocess_image(self, image_data):
        """Preprocess image for model input"""
        # Convert to grayscale and normalize
        if len(image_data.shape) == 3:
            if image_data.shape[2] == 4:  # RGBA
                image_data = image_data[:, :, :3]  # Drop alpha
            image_data = np.mean(image_data, axis=2)  # Convert to grayscale
        
        # Ensure 128x128
        if image_data.shape != (128, 128):
            img = Image.fromarray(image_data.astype('uint8'))
            img = img.resize((128, 128), Image.Resampling.LANCZOS)
            image_data = np.array(img)
        
        # Normalize to [0, 1]
        image_data = image_data.astype(np.float32) / 255.0
        
        # Add channel dimension and batch dimension
        image_data = image_data.reshape(1, 128, 128, 1)
        
        return image_data
    
    def run_inference(self, image_data):
        """Run inference on preprocessed image"""
        if not MODEL_LOADED or interpreter is None:
            # Demo mode - return random prediction
            return {
                'prediction': 0.5,
                'class': 'unknown',
                'confidence': 0.5,
                'demo_mode': True
            }
        
        try:
            # Set input tensor
            interpreter.set_tensor(input_details[0]['index'], image_data)
            
            # Run inference
            interpreter.invoke()
            
            # Get output
            output = interpreter.get_tensor(output_details[0]['index'])
            prediction = float(output[0][0])
            
            # Binary classification: 0 = negative, 1 = positive (penis)
            is_positive = prediction > 0.5
            confidence = prediction if is_positive else (1 - prediction)
            
            return {
                'prediction': prediction,
                'class': 'positive' if is_positive else 'negative',
                'confidence': confidence,
                'demo_mode': False
            }
        except Exception as e:
            logger.error(f"Inference error: {e}")
            return {
                'prediction': 0.0,
                'class': 'error',
                'confidence': 0.0,
                'error': str(e)
            }
    
    def save_input_visualization(self, session_id, objects_data, timestamp):
        """Save visualization of input objects before inference"""
        num_objects = len(objects_data)
        if num_objects == 0:
            return None
        
        # Create figure with grid
        cols = min(4, num_objects)
        rows = (num_objects + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
        if num_objects == 1:
            axes = [axes]
        elif isinstance(axes, np.ndarray):
            axes = axes.flatten()
        else:
            axes = [axes]
        
        fig.suptitle(f'ML Input - Session: {session_id}', 
                     fontsize=14, fontweight='bold')
        
        for idx, obj_data in enumerate(objects_data):
            if idx >= len(axes):
                break
                
            ax = axes[idx]
            
            # Decode image
            img_b64 = obj_data['image'].split(',')[1] if ',' in obj_data['image'] else obj_data['image']
            img_bytes = base64.b64decode(img_b64)
            img = Image.open(BytesIO(img_bytes))
            img_array = np.array(img)
            
            # Display image
            ax.imshow(img_array, cmap='gray')
            ax.set_title(f'Object {idx}\nSize: {img_array.shape[0]}x{img_array.shape[1]}', 
                        fontsize=10)
            
            # Add bounding box info
            bbox = obj_data.get('boundingBox', {})
            bbox_text = (f"Bbox: ({bbox.get('x1', 0)}, {bbox.get('y1', 0)})\n"
                        f"to ({bbox.get('x2', 0)}, {bbox.get('y2', 0)})")
            ax.text(0.02, 0.98, bbox_text, transform=ax.transAxes,
                   fontsize=7, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
            
            ax.axis('off')
        
        # Hide unused subplots
        for idx in range(num_objects, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        
        # Save visualization
        timestamp_str = datetime.fromtimestamp(timestamp/1000).strftime('%Y%m%d_%H%M%S')
        viz_path = VIZ_DIR / f'{session_id}_{timestamp_str}_INPUT.png'
        plt.savefig(viz_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"📥 Input visualization saved: {viz_path}")
        return str(viz_path)
    
    def create_visualization(self, session_id, objects_data, results):
        """Create visualization of detected objects and predictions"""
        num_objects = len(objects_data)
        if num_objects == 0:
            return None
        
        # Create figure with grid
        cols = min(4, num_objects)
        rows = (num_objects + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
        if num_objects == 1:
            axes = [axes]
        elif isinstance(axes, np.ndarray):
            axes = axes.flatten()
        else:
            axes = [axes]
        
        fig.suptitle(f'ML Detection Results - Session: {session_id}', 
                     fontsize=14, fontweight='bold')
        
        for idx, (obj_data, result) in enumerate(zip(objects_data, results)):
            if idx >= len(axes):
                break
                
            ax = axes[idx]
            
            # Decode image
            img_b64 = obj_data['image'].split(',')[1] if ',' in obj_data['image'] else obj_data['image']
            img_bytes = base64.b64decode(img_b64)
            img = Image.open(BytesIO(img_bytes))
            img_array = np.array(img)
            
            # Display image
            ax.imshow(img_array, cmap='gray')
            
            # Add prediction info
            pred_class = result['class']
            confidence = result['confidence']
            prediction_val = result['prediction']
            
            color = 'red' if pred_class == 'positive' else 'green'
            if pred_class == 'error':
                color = 'orange'
            
            title = f"Object {idx}\n{pred_class.upper()}\nConf: {confidence:.2%}\nRaw: {prediction_val:.3f}"
            ax.set_title(title, fontsize=10, color=color, fontweight='bold')
            
            # Add bounding box info
            bbox = obj_data.get('boundingBox', {})
            bbox_text = f"Bbox: ({bbox.get('x1', 0)}, {bbox.get('y1', 0)})"
            ax.text(0.02, 0.98, bbox_text, transform=ax.transAxes,
                   fontsize=7, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
            
            ax.axis('off')
        
        # Hide unused subplots
        for idx in range(num_objects, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        
        # Save visualization
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        viz_path = VIZ_DIR / f'{session_id}_{timestamp}_RESULTS.png'
        plt.savefig(viz_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"📊 Results visualization saved: {viz_path}")
        return str(viz_path)
    
    def process_detection_request(self, payload):
        """Process incoming object detection request"""
        session_id = payload.get('sessionId', 'unknown')
        objects_data = payload.get('objects', [])
        timestamp = payload.get('timestamp', 0)
        
        self.inference_count += 1
        
        logger.info("="*70)
        logger.info(f"🔍 ML DETECTION REQUEST #{self.inference_count}")
        logger.info(f"Session ID: {session_id}")
        logger.info(f"Objects: {len(objects_data)}")
        logger.info(f"Timestamp: {datetime.fromtimestamp(timestamp/1000)}")
        logger.info("="*70)
        
        if len(objects_data) == 0:
            logger.warning("No objects in request")
            return {
                'success': False,
                'error': 'No objects provided',
                'results': []
            }
        
        # Save input visualization first (if enabled)
        input_viz_path = None
        if ENABLE_VISUALIZATIONS:
            input_viz_path = self.save_input_visualization(session_id, objects_data, timestamp)
        
        # Process each object
        results = []
        for idx, obj_data in enumerate(objects_data):
            logger.info(f"\n📦 Processing Object {idx}:")
            
            # Decode image
            try:
                img_b64 = obj_data['image'].split(',')[1] if ',' in obj_data['image'] else obj_data['image']
                img_bytes = base64.b64decode(img_b64)
                img = Image.open(BytesIO(img_bytes))
                img_array = np.array(img)
                
                logger.info(f"  Image shape: {img_array.shape}")
                logger.info(f"  Image range: [{img_array.min()}, {img_array.max()}]")
                
                # Preprocess
                processed = self.preprocess_image(img_array)
                logger.info(f"  Preprocessed shape: {processed.shape}")
                
                # Run inference
                result = self.run_inference(processed)
                results.append(result)
                
                # Log results
                status_icon = "🔴" if result['class'] == 'positive' else "🟢"
                logger.info(f"  {status_icon} Prediction: {result['class'].upper()}")
                logger.info(f"  Confidence: {result['confidence']:.2%}")
                logger.info(f"  Raw score: {result['prediction']:.4f}")
                
            except Exception as e:
                logger.error(f"  ❌ Error processing object {idx}: {e}")
                results.append({
                    'prediction': 0.0,
                    'class': 'error',
                    'confidence': 0.0,
                    'error': str(e)
                })
        
        # Create visualization (if enabled)
        viz_path = None
        if ENABLE_VISUALIZATIONS:
            viz_path = self.create_visualization(session_id, objects_data, results)
        
        # Store results
        if session_id not in self.session_results:
            self.session_results[session_id] = []
        
        self.session_results[session_id].append({
            'timestamp': timestamp,
            'results': results,
            'input_visualization': input_viz_path,
            'results_visualization': viz_path
        })
        
        # Summary
        positive_count = sum(1 for r in results if r['class'] == 'positive')
        negative_count = sum(1 for r in results if r['class'] == 'negative')
        
        logger.info("\n" + "="*70)
        logger.info(f"✅ DETECTION COMPLETE")
        logger.info(f"Summary: {positive_count} positive, {negative_count} negative")
        if ENABLE_VISUALIZATIONS:
            logger.info(f"Input visualization: {input_viz_path}")
            logger.info(f"Results visualization: {viz_path}")
        
        # Log inappropriate content warning
        if positive_count > 0:
            logger.warning(f"🚨 INAPPROPRIATE CONTENT DETECTED!")
            logger.warning(f"   {positive_count} object(s) flagged for removal")
            for idx, (result, obj_data) in enumerate(zip(results, objects_data)):
                if result['class'] == 'positive':
                    bbox = obj_data.get('boundingBox', {})
                    logger.warning(f"   - Object {idx}: confidence={result['confidence']:.2%}, "
                                 f"bbox=({bbox.get('x1', 0)}, {bbox.get('y1', 0)}) to "
                                 f"({bbox.get('x2', 0)}, {bbox.get('y2', 0)})")
            logger.warning(f"   → Sending removal command to clients")
        
        logger.info("="*70 + "\n")
        
        return {
            'success': True,
            'sessionId': session_id,
            'results': results,
            'objectsData': objects_data,  # Include original objects data for removal
            'summary': {
                'total': len(results),
                'positive': positive_count,
                'negative': negative_count
            },
            'inputVisualization': input_viz_path,
            'resultsVisualization': viz_path
        }

# Create server instance
ml_server = MLInferenceServer()

# Socket.IO event handlers
@sio.event
def connect():
    logger.info("✓ Connected to Express server")

@sio.event
def disconnect():
    logger.info("✗ Disconnected from Express server")

@sio.on('ml.detectObjects')
def handle_ml_detection(data):
    """Handle ML detection request from Express server"""
    try:
        result = ml_server.process_detection_request(data)
        
        # Send results back to Express server
        sio.emit('ml.detectionResults', result)
        
    except Exception as e:
        logger.error(f"Error handling detection: {e}")
        sio.emit('ml.detectionResults', {
            'success': False,
            'error': str(e)
        })

def main():
    """Main entry point"""
    EXPRESS_URL = os.environ.get('EXPRESS_URL', 'http://localhost:3000')
    
    logger.info("="*70)
    logger.info("🤖 DOODLEPARTY ML INFERENCE SERVER")
    logger.info("="*70)
    logger.info(f"Model: {MODEL_PATH}")
    logger.info(f"Model loaded: {'✓ Yes' if MODEL_LOADED else '✗ No (demo mode)'}")
    logger.info(f"Visualizations: {'✓ Enabled' if ENABLE_VISUALIZATIONS else '✗ Disabled'}")
    if ENABLE_VISUALIZATIONS:
        logger.info(f"Visualization dir: {VIZ_DIR}")
    logger.info(f"Express server: {EXPRESS_URL}")
    logger.info("="*70 + "\n")
    
    # Wait a bit for Express server to fully start
    import time
    logger.info("Waiting for Express server to be ready...")
    time.sleep(2)
    
    # Connect to Express server
    max_retries = 5
    retry_delay = 2
    
    for attempt in range(1, max_retries + 1):
        try:
            logger.info(f"Connecting to Express server at {EXPRESS_URL}... (attempt {attempt}/{max_retries})")
            sio.connect(EXPRESS_URL, transports=['websocket', 'polling'], wait_timeout=10)
            logger.info("✓ ML server ready and listening for requests\n")
            
            # Keep running
            sio.wait()
            break
            
        except KeyboardInterrupt:
            logger.info("\n⚠️  Shutting down ML server...")
            try:
                sio.disconnect()
            except:
                pass
            break
        except Exception as e:
            logger.warning(f"Connection attempt {attempt} failed: {e}")
            if attempt < max_retries:
                logger.info(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                logger.error(f"Failed to connect after {max_retries} attempts")
                logger.error("Make sure Express server is running on port 3000")
                logger.error(f"Check logs/express.log for details")
                sys.exit(1)

if __name__ == '__main__':
    main()
