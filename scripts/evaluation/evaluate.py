"""
Comprehensive evaluation script with ROC curves, confusion matrices, and metrics.

Generates:
- ROC curve with optimal threshold
- Confusion matrix
- Precision/Recall/F1-score
- Threshold analysis
- Visualization plots
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
import argparse
from sklearn.metrics import (
    roc_curve, auc, confusion_matrix, 
    classification_report, f1_score, precision_score, recall_score
)
import tensorflow as tf


def load_model_and_data(model_path, data_dir):
    """Load trained model and test data."""
    print('Loading model and data...')
    model = tf.keras.models.load_model(model_path)
    
    X_test = np.load(Path(data_dir) / "X_test.npy")
    y_test = np.load(Path(data_dir) / "y_test.npy")
    
    # Normalize to 0-1 if needed
    if X_test.max() > 1.0:
        X_test = X_test / 255.0
    
    return model, X_test, y_test


def evaluate_model(model, X_test, y_test):
    """Get predictions and probabilities."""
    print('Generating predictions...')
    y_prob = model.predict(X_test, verbose=0)
    y_prob = y_prob.flatten()
    
    return y_prob


def find_optimal_threshold(y_true, y_prob):
    """Find optimal threshold from ROC curve."""
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    
    # Find threshold that maximizes F1-score
    best_f1 = 0
    best_threshold = 0.5
    best_idx = 0
    
    for idx, threshold in enumerate(thresholds):
        y_pred = (y_prob >= threshold).astype(int)
        f1 = f1_score(y_true, y_pred)
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
            best_idx = idx
    
    roc_auc = auc(fpr, tpr)
    
    return {
        'fpr': fpr,
        'tpr': tpr,
        'thresholds': thresholds,
        'roc_auc': roc_auc,
        'optimal_threshold': best_threshold,
        'best_f1': best_f1,
        'best_idx': best_idx
    }


def calculate_metrics(y_true, y_prob, threshold=0.5):
    """Calculate all evaluation metrics."""
    y_pred = (y_prob >= threshold).astype(int)
    
    metrics = {
        'threshold': threshold,
        'accuracy': np.mean(y_pred == y_true),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred),
    }
    
    return metrics, y_pred


def plot_roc_curve(fpr, tpr, roc_auc, optimal_threshold, output_path):
    """Plot ROC curve with optimal threshold."""
    plt.figure(figsize=(10, 8))
    
    # ROC curve
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random classifier')
    
    # Mark optimal threshold
    optimal_idx = np.argmin(np.abs(optimal_threshold - np.linspace(0, 1, len(fpr))))
    plt.plot(fpr[optimal_idx], tpr[optimal_idx], 'ro', markersize=10, 
             label=f'Optimal threshold = {optimal_threshold:.3f}')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve - DoodleHunter Model', fontsize=14, fontweight='bold')
    plt.legend(loc = 'lower right', fontsize=11)
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ ROC curve saved to {output_path}")
    plt.close()


def plot_confusion_matrix(y_true, y_pred, output_path):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    
    # Normalize for better visualization
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Plot
    im = plt.imshow(cm_normalized, cmap=plt.cm.Blues, aspect='auto')
    
    # Labels
    classes = ['Out-of-distribution', 'In-distribution (QuickDraw)']
    plt.xticks([0, 1], classes, rotation=45, ha='right')
    plt.yticks([0, 1], classes)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.title('Confusion Matrix (Normalized)', fontsize=14, fontweight='bold')
    
    # Add text annotations
    for i in range(2):
        for j in range(2):
            plt.text(
                j,
                i,
                f'{cm[i, j]}\n({cm_normalized[i, j]:.1%})',
                ha='center',
                va='center',
                color='black' if cm_normalized[i, j] < 0.5 else 'white',
                fontsize=12
            )
    
    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Confusion matrix saved to {output_path}")
    plt.close()


def plot_threshold_analysis(y_true, y_prob, output_path):
    """Plot metrics vs threshold."""
    thresholds = np.linspace(0, 1, 101)
    precisions = []
    recalls = []
    f1_scores = []
    accuracies = []
    
    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        precisions.append(precision_score(y_true, y_pred, zero_division=0))
        recalls.append(recall_score(y_true, y_pred, zero_division=0))
        f1_scores.append(f1_score(y_true, y_pred, zero_division=0))
        accuracies.append(np.mean(y_pred == y_true))
    
    plt.figure(figsize=(12, 6))
    
    plt.plot(thresholds, precisions, label='Precision', marker='o', markersize=3, alpha=0.7)
    plt.plot(thresholds, recalls, label='Recall', marker='s', markersize=3, alpha=0.7)
    plt.plot(thresholds, f1_scores, label='F1-Score', marker='^', markersize=3, alpha=0.7)
    plt.plot(thresholds, accuracies, label='Accuracy', marker='d', markersize=3, alpha=0.7)
    
    plt.xlabel('Decision Threshold', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.title('Metrics vs Decision Threshold', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Threshold analysis saved to {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description = 'Evaluate DoodleHunter model')
    parser.add_argument("--model", default = 'models/quickdraw_model.h5',
                       help = 'Path to trained model')
    parser.add_argument("--data-dir", default = 'data/processed',
                       help = 'Directory with test data')
    parser.add_argument("--output-dir", default = 'models',
                       help = 'Output directory for plots')
    parser.add_argument("--threshold", type=float, default=None,
                       help = 'Use specific threshold (if None, finds optimal)')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print('DOODLEHUNTER MODEL EVALUATION')
    print("=" * 70)
    
    # Load model and data
    model, X_test, y_test = load_model_and_data(args.model, args.data_dir)
    
    # Get predictions
    y_prob = evaluate_model(model, X_test, y_test)
    
    # Find optimal threshold
    print('\nAnalyzing ROC curve...')
    roc_info = find_optimal_threshold(y_test, y_prob)
    optimal_threshold = roc_info['optimal_threshold']
    
    # Use specified threshold or optimal
    threshold = args.threshold if args.threshold is not None else optimal_threshold
    
    print("\n" + "=" * 70)
    print('EVALUATION RESULTS')
    print("=" * 70)
    
    # Calculate metrics
    metrics, y_pred = calculate_metrics(y_test, y_prob, threshold=threshold)
    
    print(f"\nThreshold: {threshold:.4f}")
    print(f"  Optimal threshold from ROC: {optimal_threshold:.4f}")
    print(f"  ROC-AUC: {roc_info['roc_auc']:.4f}")
    print(f"\nMetrics at threshold {threshold:.4f}:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1-Score:  {metrics['f1']:.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\nConfusion Matrix:")
    print(f"  True Negatives:  {cm[0, 0]:6d}  |  False Positives: {cm[0, 1]:6d}")
    print(f"  False Negatives: {cm[1, 0]:6d}  |  True Positives:  {cm[1, 1]:6d}")
    
    # Classification report
    print(f"\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, 
                               target_names=['Out-of-distribution', 'QuickDraw']))
    
    # Generate plots
    print("\n" + "=" * 70)
    print('GENERATING PLOTS')
    print("=" * 70)
    
    plot_roc_curve(roc_info['fpr'], roc_info['tpr'], roc_info['roc_auc'], 
                   optimal_threshold, output_dir / "roc_curve.png")
    
    plot_confusion_matrix(y_test, y_pred, output_dir / "confusion_matrix_evaluation.png")
    
    plot_threshold_analysis(y_test, y_prob, output_dir / "threshold_analysis.png")
    
    # Save metrics
    metrics_file = output_dir / "evaluation_metrics.pkl"
    with open(metrics_file, 'wb') as f:
        pickle.dump({
            'metrics': metrics,
            'roc_auc': roc_info['roc_auc'],
            'optimal_threshold': optimal_threshold,
            'confusion_matrix': cm,
            'test_samples': len(y_test)
        }, f)
    print(f"✓ Metrics saved to {metrics_file}")
    
    print("\n" + "=" * 70)
    print('✓ Evaluation complete!')
    print("=" * 70)


if __name__ == '__main__':
    main()
