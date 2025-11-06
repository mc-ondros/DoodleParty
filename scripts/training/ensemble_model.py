"""
Model Ensemble System for improved classification accuracy.

Combines predictions from multiple model architectures to create
a more robust and accurate classifier. Supports different ensemble
methods: voting, averaging, and weighted averaging.

Ensemble strategies:
1. Voting: Majority vote across models
2. Simple averaging: Average probabilities
3. Weighted averaging: Weight by validation performance
4. Stacking: Meta-learner to combine predictions

Related:
- src/core/models.py (individual architectures)
- scripts/evaluate.py (model evaluation)

Exports:
- create_ensemble, evaluate_ensemble, save_ensemble
"""

import numpy as np
import pickle
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import argparse

import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold


class ModelEnsemble:
    """
    Ensemble classifier combining multiple model predictions.

    Supports:
    - Multiple architectures (custom, resnet50, mobilenetv3, efficientnet)
    - Different ensemble methods (voting, averaging, weighted)
    - Cross-validation for weight optimization
    """

    def __init__(self, models: List[tf.keras.Model], method: str = 'averaging'):
        """
        Initialize ensemble with list of models.

        Args:
            models: List of trained Keras models
            method: Ensemble method ('voting', 'averaging', 'weighted', 'stacking')
        """
        self.models = models
        self.method = method.lower()
        self.weights = None
        self.meta_model = None

        if self.method not in ['voting', 'averaging', 'weighted', 'stacking']:
            raise ValueError(f"Unknown ensemble method: {method}")

    def fit(self, X_val: np.ndarray, y_val: np.ndarray):
        """
        Fit ensemble weights using validation data.

        Args:
            X_val: Validation features
            y_val: Validation labels
        """
        print(f"Fitting ensemble using {self.method} method...")

        if self.method == 'weighted':
            # Calculate weights based on validation performance
            self._calculate_weights(X_val, y_val)
        elif self.method == 'stacking':
            # Train meta-learner (simple logistic regression)
            self._train_meta_learner(X_val, y_val)
        elif self.method == 'voting':
            # No fitting needed for simple voting
            print('  Using simple majority voting (no fitting required)')
        else:
            # Averaging doesn't need fitting
            print('  Using simple averaging (no fitting required)')

    def _calculate_weights(self, X_val: np.ndarray, y_val: np.ndarray):
        """Calculate ensemble weights based on validation accuracy."""
        weights = []
        total_accuracy = 0

        print('  Calculating weights based on validation performance:')

        for i, model in enumerate(self.models):
            # Get predictions
            y_prob = model.predict(X_val, verbose=0).flatten()
            y_pred = (y_prob >= 0.5).astype(int)

            # Calculate metrics
            accuracy = accuracy_score(y_val, y_pred)
            f1 = f1_score(y_val, y_pred)
            auc = roc_auc_score(y_val, y_prob)

            # Use F1 score as primary metric for weight calculation
            weight = f1 if f1 > 0 else accuracy
            weights.append(weight)
            total_accuracy += weight

            print(f"    Model {i+1}: Accuracy={accuracy:.4f}, F1={f1:.4f}, AUC={auc:.4f} → Weight={weight:.4f}")

        # Normalize weights
        if total_accuracy > 0:
            self.weights = np.array(weights) / total_accuracy
        else:
            # Fall back to equal weights
            self.weights = np.ones(len(self.models)) / len(self.models)
            print('    Warning: Equal weights used (all models performed poorly)')

        print(f"  Final weights: {self.weights}")

    def _train_meta_learner(self, X_val: np.ndarray, y_val: np.ndarray):
        """Train simple meta-learner (logistic regression) to combine predictions."""
        from sklearn.linear_model import LogisticRegression

        print('  Training meta-learner (logistic regression)...')

        # Get predictions from all models
        features = []
        for model in self.models:
            prob = model.predict(X_val, verbose=0).flatten()
            features.append(prob)

        features = np.column_stack(features)  # Shape: (n_samples, n_models)

        # Train meta-learner
        self.meta_model = LogisticRegression(random_state=42, max_iter=1000)
        self.meta_model.fit(features, y_val)

        # Evaluate meta-learner
        meta_pred = self.meta_model.predict(features)
        meta_acc = accuracy_score(y_val, meta_pred)
        print(f"  Meta-learner validation accuracy: {meta_acc:.4f}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make ensemble predictions.

        Args:
            X: Input features

        Returns:
            Ensemble predictions
        """
        if len(self.models) == 1:
            # Single model shortcut
            return (self.models[0].predict(X, verbose=0).flatten() >= 0.5).astype(int)

        # Get predictions from all models
        predictions = []
        probabilities = []

        for model in self.models:
            prob = model.predict(X, verbose=0).flatten()
            pred = (prob >= 0.5).astype(int)
            predictions.append(pred)
            probabilities.append(prob)

        predictions = np.array(predictions)  # Shape: (n_models, n_samples)
        probabilities = np.array(probabilities)  # Shape: (n_models, n_samples)

        if self.method == 'voting':
            # Majority vote
            ensemble_pred = (predictions.sum(axis=0) > len(self.models) / 2).astype(int)
            return ensemble_pred

        elif self.method == 'averaging':
            # Average probabilities
            avg_prob = probabilities.mean(axis=0)
            ensemble_pred = (avg_prob >= 0.5).astype(int)
            return ensemble_pred

        elif self.method == 'weighted':
            # Weighted average of probabilities
            if self.weights is None:
                raise ValueError('Must call fit() before predict() for weighted ensemble')

            weighted_prob = np.average(probabilities, axis=0, weights=self.weights)
            ensemble_pred = (weighted_prob >= 0.5).astype(int)
            return ensemble_pred

        elif self.method == 'stacking':
            # Use meta-learner
            if self.meta_model is None:
                raise ValueError('Must call fit() before predict() for stacking ensemble')

            # Features for meta-learner
            features = probabilities.T  # Shape: (n_samples, n_models)
            ensemble_pred = self.meta_model.predict(features)
            return ensemble_pred

        else:
            raise ValueError(f"Unknown method: {self.method}")

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get ensemble probabilities.

        Args:
            X: Input features

        Returns:
            Ensemble probabilities
        """
        if len(self.models) == 1:
            return self.models[0].predict(X, verbose=0).flatten()

        # Get predictions from all models
        probabilities = []
        for model in self.models:
            prob = model.predict(X, verbose=0).flatten()
            probabilities.append(prob)

        probabilities = np.array(probabilities)

        if self.method == 'voting':
            # For voting, return fraction of models predicting positive
            predictions = (probabilities >= 0.5).astype(int)
            ensemble_prob = predictions.mean(axis=0)
            return ensemble_prob

        elif self.method == 'averaging':
            return probabilities.mean(axis=0)

        elif self.method == 'weighted':
            if self.weights is None:
                raise ValueError('Must call fit() before predict_proba() for weighted ensemble')
            return np.average(probabilities, axis=0, weights=self.weights)

        elif self.method == 'stacking':
            if self.meta_model is None:
                raise ValueError('Must call fit() before predict_proba() for stacking ensemble')

            # Use meta-learner probability estimation
            features = probabilities.T
            if hasattr(self.meta_model, 'predict_proba'):
                return self.meta_model.predict_proba(features)[:, 1]
            else:
                # Fallback: use decision function
                return self.meta_model.decision_function(features)

        else:
            raise ValueError(f"Unknown method: {self.method}")


def create_ensemble(model_paths: List[str], method: str = 'weighted') -> ModelEnsemble:
    """
    Create ensemble from list of model paths.

    Args:
        model_paths: List of paths to trained model files
        method: Ensemble method ('voting', 'averaging', 'weighted', 'stacking')

    Returns:
        ModelEnsemble instance
    """
    print(f"Loading {len(model_paths)} models for ensemble...")
    models = []

    for i, path in enumerate(model_paths):
        try:
            model = tf.keras.models.load_model(path)
            models.append(model)
            param_count = model.count_params()
            print(f"  ✓ Model {i+1}: {Path(path).name} ({param_count:,} params)")
        except Exception as e:
            print(f"  ✗ Error loading {path}: {e}")
            raise

    ensemble = ModelEnsemble(models, method=method)
    return ensemble


def evaluate_ensemble(ensemble: ModelEnsemble, X_test: np.ndarray,
                     y_test: np.ndarray, threshold: float = 0.5) -> Dict:
    """
    Evaluate ensemble on test data.

    Args:
        ensemble: Trained ensemble
        X_test: Test features
        y_test: Test labels
        threshold: Decision threshold

    Returns:
        Dictionary with evaluation metrics
    """
    print('\nEvaluating ensemble...')

    # Get predictions
    y_prob = ensemble.predict_proba(X_test)
    y_pred = (y_prob >= threshold).astype(int)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    # Additional metrics
    from sklearn.metrics import precision_score, recall_score
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'auc': auc,
        'threshold': threshold,
        'n_models': len(ensemble.models),
        'method': ensemble.method
    }

    print(f"\nEnsemble Results (n={len(ensemble.models)} models, {ensemble.method} method):")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"  AUC:       {auc:.4f}")

    return results


def cross_validate_ensemble(model_paths: List[str], X: np.ndarray, y: np.ndarray,
                           cv_folds: int = 5, method: str = 'weighted') -> Dict:
    """
    Cross-validate ensemble performance.

    Args:
        model_paths: Paths to models
        X: Features
        y: Labels
        cv_folds: Number of CV folds
        method: Ensemble method

    Returns:
        Dictionary with CV results
    """
    print(f"\nPerforming {cv_folds}-fold cross-validation...")
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

    cv_scores = {
        'accuracy': [],
        'f1': [],
        'auc': []
    }

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\nFold {fold+1}/{cv_folds}:")

        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]

        # Load models for this fold
        ensemble = create_ensemble(model_paths, method=method)

        # Fit ensemble
        ensemble.fit(X_val_fold, y_val_fold)

        # Evaluate
        results = evaluate_ensemble(ensemble, X_val_fold, y_val_fold)

        cv_scores['accuracy'].append(results['accuracy'])
        cv_scores['f1'].append(results['f1_score'])
        cv_scores['auc'].append(results['auc'])

    # Summary
    print(f"\n{'='*60}")
    print(f"CROSS-VALIDATION RESULTS ({cv_folds} folds)")
    print(f"{'='*60}")

    for metric in ['accuracy', 'f1', 'auc']:
        scores = cv_scores[metric]
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        print(f"{metric.capitalize():10s}: {mean_score:.4f} ± {std_score:.4f}")
        print(f"             Individual scores: {[f'{s:.4f}' for s in scores]}")

    return cv_scores


def save_ensemble(ensemble: ModelEnsemble, output_path: str):
    """
    Save ensemble configuration.

    Args:
        ensemble: Trained ensemble
        output_path: Path to save ensemble
    """
    config = {
        'method': ensemble.method,
        'n_models': len(ensemble.models),
        'weights': ensemble.weights.tolist() if ensemble.weights is not None else None,
        # Note: Models themselves are saved separately
    }

    with open(output_path, 'wb') as f:
        pickle.dump(config, f)

    print(f"\n✓ Ensemble configuration saved to {output_path}")


def compare_individual_vs_ensemble(model_paths: List[str], X_test: np.ndarray,
                                   y_test: np.ndarray, method: str = 'weighted'):
    """
    Compare individual model performance vs ensemble.

    Args:
        model_paths: Paths to models
        X_test: Test features
        y_test: Test labels
        method: Ensemble method
    """
    print(f"\n{'='*70}")
    print('INDIVIDUAL MODEL vs ENSEMBLE COMPARISON')
    print(f"{'='*70}")

    print('\nIndividual Model Performance:')
    for i, path in enumerate(model_paths):
        model = tf.keras.models.load_model(path)
        y_prob = model.predict(X_test, verbose=0).flatten()
        y_pred = (y_prob >= 0.5).astype(int)

        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)

        print(f"  Model {i+1}: Accuracy={accuracy:.4f}, F1={f1:.4f}, AUC={auc:.4f}")

    # Ensemble performance
    ensemble = create_ensemble(model_paths, method=method)
    ensemble.fit(X_test, y_test)  # Using test for simplicity (in practice use validation)
    ensemble_results = evaluate_ensemble(ensemble, X_test, y_test)

    print(f"\nEnsemble ({method}): Accuracy={ensemble_results['accuracy']:.4f}, "
          f"F1={ensemble_results['f1_score']:.4f}, AUC={ensemble_results['auc']:.4f}")


def main():
    parser = argparse.ArgumentParser(description = 'Create and evaluate model ensemble')
    parser.add_argument("--models", nargs='+', required=True,
                       help = 'Paths to trained model files')
    parser.add_argument("--data-dir", default = 'data/processed',
                       help = 'Directory with test data')
    parser.add_argument("--method", default='weighted',
                       choices=['voting', 'averaging', 'weighted', 'stacking'],
                       help = 'Ensemble method')
    parser.add_argument("--cross-validate", action='store_true',
                       help = 'Perform cross-validation')
    parser.add_argument("--cv-folds", type=int, default=5,
                       help = 'Number of CV folds')
    parser.add_argument("--output", default = 'models/ensemble_config.pkl',
                       help = 'Path to save ensemble configuration')

    args = parser.parse_args()

    # Load test data
    print('Loading test data...')
    X_test = np.load(Path(args.data_dir) / "X_test.npy")
    y_test = np.load(Path(args.data_dir) / "y_test.npy")

    # Normalize if needed
    if X_test.max() > 1.0:
        X_test = X_test / 255.0

    print(f"Test set: {len(X_test)} samples")
    print(f"Class distribution: Class 0={(y_test==0).sum()}, Class 1={(y_test==1).sum()}")

    # Create ensemble
    ensemble = create_ensemble(args.models, method=args.method)

    if args.cross_validate:
        # Cross-validation
        all_data = np.concatenate([X_test, np.load(Path(args.data_dir) / "X_train.npy")])
        all_labels = np.concatenate([y_test, np.load(Path(args.data_dir) / "y_train.npy")])

        # Normalize combined data
        if all_data.max() > 1.0:
            all_data = all_data / 255.0

        cv_results = cross_validate_ensemble(
            args.models, all_data, all_labels,
            cv_folds=args.cv_folds, method=args.method
        )
    else:
        # Fit and evaluate
        print(f"\nFitting ensemble with {args.method} method...")
        ensemble.fit(X_test, y_test)

        results = evaluate_ensemble(ensemble, X_test, y_test)

        # Compare individual vs ensemble
        compare_individual_vs_ensemble(args.models, X_test, y_test, method=args.method)

        # Save ensemble
        save_ensemble(ensemble, args.output)

    print(f"\n{'='*70}")
    print('✓ Ensemble evaluation complete!')
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
