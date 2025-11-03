"""
Threshold optimization for binary classification.

Analyzes ROC curve and finds optimal decision threshold
based on business requirements (precision vs recall trade-off).
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import pickle
from sklearn.metrics import roc_curve, auc, f1_score, precision_score, recall_score
import tensorflow as tf


def analyze_thresholds(y_true, y_prob, output_dir=None):
    """
    Analyze different thresholds and their metrics.
    
    Returns:
        Dictionary with comprehensive threshold analysis
    """
    thresholds = np.linspace(0, 1, 1001)
    metrics_by_threshold = []
    
    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        
        tn = np.sum((y_pred == 0) & (y_true == 0))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))
        tp = np.sum((y_pred == 1) & (y_true == 1))
        
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        sensitivity = recall
        
        metrics_by_threshold.append({
            'threshold': threshold,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'tp': tp,
            'fp': fp,
            'tn': tn,
            'fn': fn
        })
    
    return metrics_by_threshold


def find_optimal_thresholds(y_true, y_prob):
    """Find optimal thresholds for different objectives."""
    metrics = analyze_thresholds(y_true, y_prob)
    
    # Find maximum for each metric
    optimal = {
        'max_f1': max(metrics, key=lambda x: x['f1']),
        'max_accuracy': max(metrics, key=lambda x: x['accuracy']),
        'balanced': min(metrics, key=lambda x: abs(x['precision'] - x['recall'])),
    }
    
    # ROC curve for reference
    fpr, tpr, roc_thresholds = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    
    return metrics, optimal, (fpr, tpr, roc_auc)


def print_threshold_recommendations(optimal, y_true, y_prob):
    """Print recommendations for different use cases."""
    print("\n" + "=" * 80)
    print("THRESHOLD RECOMMENDATIONS FOR DIFFERENT USE CASES")
    print("=" * 80)
    
    print("\n1. MAXIMIZE F1-SCORE (Default - balanced performance)")
    t = optimal['max_f1']
    print(f"   Threshold: {t['threshold']:.4f}")
    print(f"   Accuracy: {t['accuracy']:.4f} | Precision: {t['precision']:.4f} | Recall: {t['recall']:.4f}")
    print(f"   F1-Score: {t['f1']:.4f}")
    print(f"   Confusion Matrix: TP={t['tp']}, FP={t['fp']}, TN={t['tn']}, FN={t['fn']}")
    
    print("\n2. MAXIMIZE ACCURACY")
    t = optimal['max_accuracy']
    print(f"   Threshold: {t['threshold']:.4f}")
    print(f"   Accuracy: {t['accuracy']:.4f} | Precision: {t['precision']:.4f} | Recall: {t['recall']:.4f}")
    print(f"   F1-Score: {t['f1']:.4f}")
    
    print("\n3. BALANCED PRECISION-RECALL (Equal False Positives and False Negatives)")
    t = optimal['balanced']
    print(f"   Threshold: {t['threshold']:.4f}")
    print(f"   Precision: {t['precision']:.4f} | Recall: {t['recall']:.4f}")
    print(f"   F1-Score: {t['f1']:.4f}")
    
    print("\n" + "=" * 80)
    print("USE CASE SPECIFIC RECOMMENDATIONS")
    print("=" * 80)
    
    print("\n• HIGH PRECISION (minimize false positives):")
    print("  When cost of false positive > cost of false negative")
    print("  Examples: Spam detection, Medical screening")
    high_precision = max(
        [m for m in analyze_thresholds(y_true, y_prob) if m['precision'] > 0.95],
        key=lambda x: x['f1'],
        default=None
    )
    if high_precision:
        print(f"  → Threshold: {high_precision['threshold']:.4f}")
        print(f"    Precision: {high_precision['precision']:.4f}, Recall: {high_precision['recall']:.4f}")
    
    print("\n• HIGH RECALL (minimize false negatives):")
    print("  When cost of false negative > cost of false positive")
    print("  Examples: Fraud detection, Disease diagnosis")
    high_recall = max(
        [m for m in analyze_thresholds(y_true, y_prob) if m['recall'] > 0.95],
        key=lambda x: x['f1'],
        default=None
    )
    if high_recall:
        print(f"  → Threshold: {high_recall['threshold']:.4f}")
        print(f"    Precision: {high_recall['precision']:.4f}, Recall: {high_recall['recall']:.4f}")


def plot_threshold_comparison(metrics, optimal, output_file):
    """Create comprehensive threshold comparison plots."""
    thresholds = [m['threshold'] for m in metrics]
    accuracies = [m['accuracy'] for m in metrics]
    precisions = [m['precision'] for m in metrics]
    recalls = [m['recall'] for m in metrics]
    f1_scores = [m['f1'] for m in metrics]
    specificities = [m['specificity'] for m in metrics]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: All metrics vs threshold
    ax = axes[0, 0]
    ax.plot(thresholds, accuracies, label='Accuracy', alpha=0.8)
    ax.plot(thresholds, precisions, label='Precision', alpha=0.8)
    ax.plot(thresholds, recalls, label='Recall', alpha=0.8)
    ax.plot(thresholds, f1_scores, label='F1-Score', alpha=0.8, linewidth=2)
    ax.axvline(optimal['max_f1']['threshold'], color='red', linestyle='--', 
               label=f"Optimal (F1): {optimal['max_f1']['threshold']:.3f}")
    ax.set_xlabel('Threshold')
    ax.set_ylabel('Score')
    ax.set_title('All Metrics vs Threshold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    
    # Plot 2: Precision vs Recall (PR curve)
    ax = axes[0, 1]
    ax.plot(recalls, precisions, 'b-', linewidth=2)
    ax.scatter([optimal['max_f1']['recall']], [optimal['max_f1']['precision']], 
               color='red', s=100, label='Max F1', zorder=5)
    ax.scatter([optimal['balanced']['recall']], [optimal['balanced']['precision']], 
               color='green', s=100, label='Balanced', zorder=5)
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Plot 3: Sensitivity vs Specificity
    ax = axes[1, 0]
    ax.plot(thresholds, [m['sensitivity'] for m in metrics], label='Sensitivity (Recall)', alpha=0.8)
    ax.plot(thresholds, specificities, label='Specificity', alpha=0.8)
    ax.axvline(optimal['max_f1']['threshold'], color='red', linestyle='--', alpha=0.5)
    ax.set_xlabel('Threshold')
    ax.set_ylabel('Score')
    ax.set_title('Sensitivity vs Specificity')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Plot 4: Confusion Matrix Components
    ax = axes[1, 1]
    optimal_m = optimal['max_f1']
    components = ['TP', 'FP', 'TN', 'FN']
    values = [optimal_m['tp'], optimal_m['fp'], optimal_m['tn'], optimal_m['fn']]
    colors = ['green', 'red', 'blue', 'orange']
    bars = ax.bar(components, values, color=colors, alpha=0.7)
    ax.set_ylabel('Count')
    ax.set_title(f'Confusion Matrix at Optimal Threshold ({optimal_m["threshold"]:.3f})')
    ax.grid(alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(val)}',
                ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ Threshold comparison plot saved to {output_file}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Optimize decision threshold")
    parser.add_argument("--model", default="models/quickdraw_model.h5",
                       help="Path to trained model")
    parser.add_argument("--data-dir", default="data/processed",
                       help="Directory with test data")
    parser.add_argument("--output-dir", default="models",
                       help="Output directory for results")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("THRESHOLD OPTIMIZATION")
    print("=" * 80)
    
    # Load model and data
    print("\nLoading model and data...")
    model = tf.keras.models.load_model(args.model)
    X_test = np.load(Path(args.data_dir) / "X_test.npy")
    y_test = np.load(Path(args.data_dir) / "y_test.npy")
    
    # Normalize
    if X_test.max() > 1.0:
        X_test = X_test / 255.0
    
    # Get predictions
    print("Generating predictions...")
    y_prob = model.predict(X_test, verbose=0).flatten()
    
    # Analyze thresholds
    print("Analyzing thresholds...")
    metrics, optimal, roc_info = find_optimal_thresholds(y_test, y_prob)
    
    # Print recommendations
    print_threshold_recommendations(optimal, y_test, y_prob)
    
    # Generate plots
    print("\nGenerating plots...")
    plot_threshold_comparison(metrics, optimal, output_dir / "threshold_analysis.png")
    
    # Save results
    results = {
        'metrics': metrics,
        'optimal': optimal,
        'roc_auc': roc_info[2],
        'recommended_threshold': optimal['max_f1']['threshold']
    }
    
    with open(output_dir / "threshold_optimization.pkl", 'wb') as f:
        pickle.dump(results, f)
    
    print(f"\n✓ Results saved to {output_dir / 'threshold_optimization.pkl'}")
    
    print("\n" + "=" * 80)
    print("✓ Optimization complete!")
    print("=" * 80)
    print(f"\nRecommended threshold: {optimal['max_f1']['threshold']:.4f}")
    print(f"Use this in production: model.predict(x) >= {optimal['max_f1']['threshold']:.4f}")


if __name__ == "__main__":
    main()
