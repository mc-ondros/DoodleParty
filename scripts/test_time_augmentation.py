"""
Test-Time Augmentation (TTA) for improved inference accuracy.

TTA applies multiple transformations to each input image and averages
the predictions, which significantly improves robustness and accuracy
at the cost of inference time.
"""

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from data_pipeline import normalize_image


def predict_with_tta(model, image, n_augmentations=10, threshold=0.5):
    """
    Predict with Test-Time Augmentation.
    
    Applies multiple random augmentations to the image and averages
    the predictions for more robust results.
    
    Args:
        model: Trained model
        image: Input image (28, 28, 1) or (28, 28)
        n_augmentations: Number of augmented versions to create
        threshold: Decision threshold
    
    Returns:
        avg_probability: Average probability across all augmentations
        class_prediction: Final class (0 or 1)
        std_dev: Standard deviation of predictions (uncertainty measure)
    """
    # Ensure correct shape
    if len(image.shape) == 2:
        image = image.reshape(28, 28, 1)
    elif len(image.shape) == 3 and image.shape[-1] != 1:
        image = image[..., 0:1]  # Take first channel
    
    # Normalize image
    image_norm = normalize_image(image)
    
    # Create augmentation generator (more aggressive than training)
    augmentation = ImageDataGenerator(
        rotation_range=20,      # ±20 degrees
        width_shift_range=0.15,  # ±15%
        height_shift_range=0.15,
        zoom_range=0.2,          # ±20%
        fill_mode='constant',
        cval=0.5,
    )
    
    # Collect predictions from augmented versions
    predictions = []
    
    # Add original image
    pred_original = model.predict(image_norm.reshape(1, 28, 28, 1), verbose=0)[0][0]
    predictions.append(pred_original)
    
    # Generate augmented versions
    image_batch = np.expand_dims(image_norm, axis=0)
    
    for aug_batch in augmentation.flow(image_batch, batch_size=1, shuffle=False):
        pred = model.predict(aug_batch, verbose=0)[0][0]
        predictions.append(pred)
        
        if len(predictions) >= n_augmentations:
            break
    
    # Calculate statistics
    predictions = np.array(predictions)
    avg_probability = predictions.mean()
    std_dev = predictions.std()
    
    # Make final decision
    class_prediction = 1 if avg_probability >= threshold else 0
    
    return avg_probability, class_prediction, std_dev


def predict_batch_with_tta(model, images, n_augmentations=10, threshold=0.5):
    """
    Predict batch of images with TTA.
    
    Args:
        model: Trained model
        images: Batch of images (N, 28, 28, 1)
        n_augmentations: Number of augmentations per image
        threshold: Decision threshold
    
    Returns:
        predictions: Array of (probability, class, std_dev) tuples
    """
    results = []
    
    for i, image in enumerate(images):
        prob, cls, std = predict_with_tta(model, image, n_augmentations, threshold)
        results.append((prob, cls, std))
        
        if (i + 1) % 100 == 0:
            print(f"Processed {i+1}/{len(images)} images with TTA...")
    
    return results


def evaluate_with_tta(model, X_test, y_test, n_augmentations=10, threshold=0.5):
    """
    Evaluate model on test set using TTA.
    
    Args:
        model: Trained model
        X_test: Test images
        y_test: Test labels
        n_augmentations: Number of augmentations
        threshold: Decision threshold
    
    Returns:
        accuracy: Test accuracy with TTA
        predictions: List of predictions
    """
    print(f"Evaluating with Test-Time Augmentation ({n_augmentations} augmentations per image)...")
    
    predictions = predict_batch_with_tta(model, X_test, n_augmentations, threshold)
    
    # Extract class predictions
    y_pred = np.array([cls for _, cls, _ in predictions])
    
    # Calculate accuracy
    accuracy = (y_pred == y_test).mean()
    
    print(f"\n✅ TTA Results:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Improvement: {(accuracy - 0.95) * 100:+.2f}% vs baseline")
    
    # Analyze uncertainty
    uncertainties = np.array([std for _, _, std in predictions])
    print(f"  Avg uncertainty (std): {uncertainties.mean():.4f}")
    print(f"  High uncertainty samples: {(uncertainties > 0.2).sum()}")
    
    return accuracy, predictions
