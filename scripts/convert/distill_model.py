"""
Knowledge distillation: Train a smaller student model from a larger teacher.

This script implements knowledge distillation to create a more compact model
that maintains the performance of the original. The student model learns from
both the ground truth labels and the soft targets from the teacher model.

Usage:
    # Basic distillation with smaller model
    python scripts/convert/distill_model.py --teacher models/quickdraw_model.h5

    # Custom student architecture and temperature
    python scripts/convert/distill_model.py --teacher models/quickdraw_model.h5 --student-filters 16,32 --temperature 3.0

    # Extended training for better performance
    python scripts/convert/distill_model.py --teacher models/quickdraw_model.h5 --epochs 30 --alpha 0.3

Features:
    - Knowledge distillation with temperature scaling
    - Customizable student architecture
    - Combined loss (soft targets + hard labels)
    - Performance comparison with teacher

Note:
    Requires training data. The script will load data from data/processed/
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train student model via knowledge distillation'
    )
    parser.add_argument(
        '--teacher', '-t',
        required=True,
        help='Path to teacher model (.h5 or .keras)'
    )
    parser.add_argument(
        '--output', '-o',
        default=None,
        help='Path to output student model (default: {teacher}_student.keras)'
    )
    parser.add_argument(
        '--student-filters',
        type=str,
        default='16,32,64',
        help='Conv filters for student (comma-separated, default: 16,32,64)'
    )
    parser.add_argument(
        '--student-dense',
        type=int,
        default=64,
        help='Dense layer size for student (default: 64)'
    )
    parser.add_argument(
        '--temperature', '-T',
        type=float,
        default=3.0,
        help='Distillation temperature (default: 3.0)'
    )
    parser.add_argument(
        '--alpha',
        type=float,
        default=0.1,
        help='Weight for hard label loss (1-alpha for soft, default: 0.1)'
    )
    parser.add_argument(
        '--epochs', '-e',
        type=int,
        default=20,
        help='Training epochs (default: 20)'
    )
    parser.add_argument(
        '--batch-size', '-b',
        type=int,
        default=64,
        help='Batch size (default: 64)'
    )
    parser.add_argument(
        '--data-dir', '-d',
        default='data/processed',
        help='Directory containing training data (default: data/processed)'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output'
    )
    return parser.parse_args()


def load_training_data(data_dir: Path):
    """Load training data."""
    print(f"\nLoading training data from: {data_dir}")
    
    X_train_path = data_dir / 'X_train.npy'
    y_train_path = data_dir / 'y_train.npy'
    X_val_path = data_dir / 'X_test.npy'
    y_val_path = data_dir / 'y_test.npy'
    
    if not all([X_train_path.exists(), y_train_path.exists(),
                X_val_path.exists(), y_val_path.exists()]):
        print('✗ Error: Training data not found')
        sys.exit(1)
    
    X_train = np.load(X_train_path)
    y_train = np.load(y_train_path)
    X_val = np.load(X_val_path)
    y_val = np.load(y_val_path)
    
    print(f"✓ Loaded data:")
    print(f"  Train: {X_train.shape}, {y_train.shape}")
    print(f"  Val:   {X_val.shape}, {y_val.shape}")
    
    return X_train, y_train, X_val, y_val


def create_student_model(input_shape, filters, dense_units):
    """Create a smaller student model."""
    print(f"\nCreating student model:")
    print(f"  Conv filters: {filters}")
    print(f"  Dense units:  {dense_units}")
    
    model = keras.Sequential([
        keras.layers.Input(shape=input_shape),
    ])
    
    # Convolutional layers
    for i, num_filters in enumerate(filters):
        model.add(keras.layers.Conv2D(
            num_filters, (3, 3),
            activation='relu',
            padding='same',
            name=f'conv{i+1}'
        ))
        model.add(keras.layers.MaxPooling2D((2, 2), name=f'pool{i+1}'))
    
    # Dense layers
    model.add(keras.layers.Flatten(name='flatten'))
    model.add(keras.layers.Dense(dense_units, activation='relu', name='dense'))
    model.add(keras.layers.Dropout(0.5, name='dropout'))
    model.add(keras.layers.Dense(1, activation='sigmoid', name='output'))
    
    print('✓ Student model created')
    model.summary()
    
    return model


class DistillationLoss(keras.losses.Loss):
    """Custom loss for knowledge distillation."""
    
    def __init__(self, teacher_model, temperature=3.0, alpha=0.1, **kwargs):
        super().__init__(**kwargs)
        self.teacher_model = teacher_model
        self.temperature = temperature
        self.alpha = alpha
    
    def call(self, y_true, y_pred):
        # Hard label loss (standard binary crossentropy)
        hard_loss = keras.losses.binary_crossentropy(y_true, y_pred)
        
        return hard_loss  # We'll handle soft loss in training loop


class Distiller(keras.Model):
    """Custom Keras model for knowledge distillation."""
    
    def __init__(self, student, teacher, temperature=3.0, alpha=0.1):
        super().__init__()
        self.student = student
        self.teacher = teacher
        self.temperature = temperature
        self.alpha = alpha
    
    def compile(self, optimizer, metrics):
        super().compile(optimizer=optimizer, metrics=metrics)
        self.student_loss_fn = keras.losses.BinaryCrossentropy()
        self.distillation_loss_fn = keras.losses.KLDivergence()
    
    def call(self, inputs):
        return self.student(inputs)
    
    def train_step(self, data):
        x, y = data
        
        # Get teacher predictions
        teacher_predictions = self.teacher(x, training=False)
        
        with tf.GradientTape() as tape:
            # Student predictions
            student_predictions = self.student(x, training=True)
            
            # Hard loss (regular cross-entropy with true labels)
            student_loss = self.student_loss_fn(y, student_predictions)
            
            # Soft loss (distillation from teacher)
            # Apply temperature scaling for softer distributions
            soft_teacher = tf.nn.sigmoid(teacher_predictions / self.temperature)
            soft_student = tf.nn.sigmoid(student_predictions / self.temperature)
            
            distillation_loss = tf.reduce_mean(
                tf.square(soft_teacher - soft_student)
            ) * (self.temperature ** 2)
            
            # Combined loss
            loss = self.alpha * student_loss + (1 - self.alpha) * distillation_loss
        
        # Update weights
        trainable_vars = self.student.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        
        # Update metrics
        self.compiled_metrics.update_state(y, student_predictions)
        
        # Return metrics
        results = {m.name: m.result() for m in self.metrics}
        results['loss'] = loss
        results['student_loss'] = student_loss
        results['distillation_loss'] = distillation_loss
        
        return results
    
    def test_step(self, data):
        x, y = data
        
        # Student predictions
        student_predictions = self.student(x, training=False)
        
        # Update metrics
        self.compiled_metrics.update_state(y, student_predictions)
        
        results = {m.name: m.result() for m in self.metrics}
        return results


def count_parameters(model):
    """Count number of parameters in model."""
    return sum([tf.size(w).numpy() for w in model.trainable_weights])


def compare_models(teacher_model, student_model, X_val, y_val):
    """Compare teacher and student model performance."""
    print("\n" + "=" * 70)
    print('Model Comparison')
    print("=" * 70)
    
    # Evaluate teacher
    teacher_metrics = teacher_model.evaluate(X_val, y_val, verbose=0)
    teacher_acc = teacher_metrics[1]
    
    # Evaluate student
    student_metrics = student_model.evaluate(X_val, y_val, verbose=0)
    student_acc = student_metrics[1]
    
    # Count parameters
    teacher_params = count_parameters(teacher_model)
    student_params = count_parameters(student_model)
    
    param_reduction = ((teacher_params - student_params) / teacher_params) * 100
    acc_diff = student_acc - teacher_acc
    
    print(f"{'Metric':<25} {'Teacher':<20} {'Student':<20} {'Difference':<15}")
    print("-" * 70)
    print(f"{'Parameters':<25} {teacher_params:<20,} {student_params:<20,} {-param_reduction:<14.1f}%")
    print(f"{'Validation accuracy':<25} {teacher_acc:<20.4f} {student_acc:<20.4f} {acc_diff:<+14.4f}")
    print("=" * 70)
    
    if acc_diff > -0.01:
        print('\n✓ Excellent: Student matches teacher performance')
    elif acc_diff > -0.02:
        print('\n✓ Good: Minimal accuracy loss')
    elif acc_diff > -0.05:
        print('\n✓ Acceptable: Small accuracy tradeoff for size reduction')
    else:
        print('\n⚠ Warning: Significant accuracy loss')
    
    print(f"\nModel size reduction: {param_reduction:.1f}%")


def main():
    args = parse_args()
    
    # Validate input
    teacher_path = Path(args.teacher)
    if not teacher_path.exists():
        print(f"Error: Teacher model not found: {teacher_path}")
        sys.exit(1)
    
    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = teacher_path.parent / f"{teacher_path.stem}_student.keras"
    
    data_dir = Path(args.data_dir)
    
    # Parse student architecture
    filters = [int(f) for f in args.student_filters.split(',')]
    
    print("=" * 70)
    print('Knowledge Distillation')
    print("=" * 70)
    print(f"Teacher model:  {teacher_path}")
    print(f"Student model:  {output_path}")
    print(f"Temperature:    {args.temperature}")
    print(f"Alpha:          {args.alpha} (hard) / {1-args.alpha:.1f} (soft)")
    print(f"Epochs:         {args.epochs}")
    print("=" * 70)
    
    # Load teacher model
    print('\nLoading teacher model...')
    teacher_model = keras.models.load_model(teacher_path)
    print('✓ Teacher model loaded')
    
    # Load data
    X_train, y_train, X_val, y_val = load_training_data(data_dir)
    
    # Create student model
    input_shape = X_train.shape[1:]
    student_model = create_student_model(input_shape, filters, args.student_dense)
    
    # Create distiller
    print('\nInitializing distiller...')
    distiller = Distiller(
        student=student_model,
        teacher=teacher_model,
        temperature=args.temperature,
        alpha=args.alpha
    )
    
    # Compile
    distiller.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        metrics=[
            keras.metrics.BinaryAccuracy(name='accuracy'),
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall')
        ]
    )
    
    print('✓ Distiller ready')
    
    # Train
    print(f"\nTraining student model for {args.epochs} epochs...")
    history = distiller.fit(
        X_train, y_train,
        batch_size=args.batch_size,
        epochs=args.epochs,
        validation_data=(X_val, y_val),
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=5,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-6
            )
        ],
        verbose=1 if args.verbose else 2
    )
    
    print('✓ Training complete')
    
    # Save student model
    print(f"\nSaving student model to: {output_path}")
    student_model.save(output_path)
    print('✓ Student model saved')
    
    # Compare models
    compare_models(teacher_model, student_model, X_val, y_val)
    
    print('\n✓ Distillation complete')
    print('\nNext steps:')
    print('  1. Convert to TFLite for deployment')
    print('  2. Apply INT8 quantization')
    print('  3. Benchmark inference speed')
    print('  4. Test on full validation set')


if __name__ == '__main__':
    main()
