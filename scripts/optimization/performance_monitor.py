#!/usr/bin/env python3
"""
Real-time Performance Monitor for DoodleHunter

Continuously monitors model performance and provides alerts for optimization opportunities.
Designed for production environments to enable data-driven optimization decisions.

Usage:
    python scripts/optimization/performance_monitor.py --model models/current.h5 --interval 300
    python scripts/optimization/performance_monitor.py --model models/current.h5 --alert-latency 100
"""

import argparse
import sys
import time
import json
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import tensorflow as tf
from typing import Dict, List

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core.optimization import PerformanceMonitor


class RealTimePerformanceMonitor:
    """Real-time performance monitoring with alerting."""

    def __init__(self, data_flywheel_dir: str = 'data_flywheel'):
        self.monitor = PerformanceMonitor(data_flywheel_dir)
        self.models_dir = Path('models')
        self.alert_history = []
        self.performance_history = []

    def monitor_performance(
        self,
        model_path: str,
        test_data,
        monitoring_interval: int = 300,  # 5 minutes
        max_duration: int = 3600,  # 1 hour
        alert_latency_ms: float = 100.0,
        alert_accuracy_drop: float = 0.05
    ) -> Dict:
        """Monitor performance with alerting."""
        print("="*60)
        print("REAL-TIME PERFORMANCE MONITORING")
        print("="*60)
        print(f"Model: {model_path}")
        print(f"Monitoring interval: {monitoring_interval} seconds")
        print(f"Max duration: {max_duration} seconds")
        print(f"Latency alert threshold: {alert_latency_ms}ms")
        print(f"Accuracy drop alert: {alert_accuracy_drop:.1%}")
        print()

        start_time = datetime.now()
        end_time = start_time + timedelta(seconds=max_duration)
        check_count = 0

        while datetime.now() < end_time:
            check_count += 1
            current_time = datetime.now()

            print(f"[{current_time.strftime('%H:%M:%S')}] Performance check #{check_count}")

            try:
                # Load model and collect metrics
                model = tf.keras.models.load_model(model_path)
                metrics = self.monitor.collect_inference_metrics(
                    model, test_data, f'monitor_check_{check_count}'
                )

                # Check for alerts
                alerts = self._check_alerts(metrics, alert_latency_ms, alert_accuracy_drop)

                if alerts:
                    print(f"  ⚠️  ALERTS: {', '.join(alerts)}")
                    self.alert_history.append({
                        'timestamp': current_time.isoformat(),
                        'check_number': check_count,
                        'alerts': alerts,
                        'metrics': metrics
                    })

                    # Save alert data
                    alert_file = self.monitor.feedback_dir / f'alert_{current_time.strftime("%Y%m%d_%H%M%S")}.json'
                    with open(alert_file, 'w') as f:
                        json.dump({
                            'timestamp': current_time.isoformat(),
                            'alerts': alerts,
                            'metrics': metrics
                        }, f, indent=2, default=str)
                else:
                    print("  ✓ Performance normal")

                # Record performance
                self.performance_history.append({
                    'timestamp': current_time.isoformat(),
                    'check_number': check_count,
                    'metrics': metrics
                })

                # Print summary
                latency = metrics['latency']['batch_1']['avg_latency_ms']
                accuracy = metrics['accuracy']['batch_1']['accuracy']
                print(f"  Latency: {latency:.1f}ms, Accuracy: {accuracy:.3f}")

            except Exception as e:
                print(f"  ❌ Error in performance check: {e}")

            # Wait for next check
            if datetime.now() < end_time:
                print(f"  Next check in {monitoring_interval} seconds...")
                time.sleep(monitoring_interval)
                print()

        # Generate monitoring report
        return self._generate_monitoring_report(check_count)

    def _check_alerts(
        self,
        metrics: Dict,
        alert_latency_ms: float,
        alert_accuracy_drop: float
    ) -> List[str]:
        """Check for performance alerts."""
        alerts = []

        # Check latency
        latency = metrics['latency']['batch_1']['avg_latency_ms']
        if latency > alert_latency_ms:
            alerts.append(f"High latency: {latency:.1f}ms")

        # Check accuracy (compare to recent history or threshold)
        accuracy = metrics['accuracy']['batch_1']['accuracy']
        if len(self.performance_history) > 0:
            # Compare to most recent measurement
            recent_accuracy = self.performance_history[-1]['metrics']['accuracy']['batch_1']['accuracy']
            accuracy_drop = recent_accuracy - accuracy
            if accuracy_drop > alert_accuracy_drop:
                alerts.append(f"Accuracy drop: {accuracy_drop:.1%}")
        elif accuracy < 0.8:  # Absolute threshold
            alerts.append(f"Low accuracy: {accuracy:.3f}")

        return alerts

    def _generate_monitoring_report(self, total_checks: int) -> Dict:
        """Generate monitoring summary report."""
        if not self.performance_history:
            return {'error': 'No performance data collected'}

        # Calculate statistics
        latencies = [
            h['metrics']['latency']['batch_1']['avg_latency_ms']
            for h in self.performance_history
        ]
        accuracies = [
            h['metrics']['accuracy']['batch_1']['accuracy']
            for h in self.performance_history
        ]

        report = {
            'monitoring_period': {
                'start_time': self.performance_history[0]['timestamp'],
                'end_time': self.performance_history[-1]['timestamp'],
                'total_checks': total_checks,
                'alerts_generated': len(self.alert_history)
            },
            'performance_summary': {
                'latency': {
                    'avg_ms': np.mean(latencies),
                    'min_ms': np.min(latencies),
                    'max_ms': np.max(latencies),
                    'std_ms': np.std(latencies)
                },
                'accuracy': {
                    'avg': np.mean(accuracies),
                    'min': np.min(accuracies),
                    'max': np.max(accuracies),
                    'std': np.std(accuracies)
                }
            },
            'alerts': self.alert_history,
            'recommendations': self._generate_recommendations(latencies, accuracies)
        }

        # Save report
        report_file = self.monitor.feedback_dir / f'monitoring_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        print("="*60)
        print("MONITORING SUMMARY")
        print("="*60)
        print(f"Total checks: {total_checks}")
        print(f"Alerts generated: {len(self.alert_history)}")
        print(f"Average latency: {np.mean(latencies):.1f}ms")
        print(f"Average accuracy: {np.mean(accuracies):.3f}")
        print(f"Report saved: {report_file}")

        return report

    def _generate_recommendations(self, latencies: List[float], accuracies: List[float]) -> List[str]:
        """Generate optimization recommendations based on monitoring data."""
        recommendations = []

        # Analyze latency trends
        avg_latency = np.mean(latencies)
        if avg_latency > 100:
            recommendations.append("Consider model optimization for latency reduction")
        elif avg_latency > 50:
            recommendations.append("Monitor latency trends - optimization may be beneficial")

        # Analyze accuracy trends
        avg_accuracy = np.mean(accuracies)
        if avg_accuracy < 0.85:
            recommendations.append("Accuracy below target - consider model retraining")
        elif len(accuracies) > 1 and np.std(accuracies) > 0.02:
            recommendations.append("High accuracy variance - investigate stability")

        # Performance stability
        if len(latencies) > 1 and np.std(latencies) > 20:
            recommendations.append("High latency variance - check for performance issues")

        if not recommendations:
            recommendations.append("Performance metrics within acceptable ranges")

        return recommendations


def main():
    parser = argparse.ArgumentParser(description='Real-time Performance Monitor')
    parser.add_argument('--model', required=True, help='Path to model to monitor')
    parser.add_argument('--data-dir', default='data/processed', help='Data directory')
    parser.add_argument('--output-dir', default='data_flywheel', help='Data-flywheel directory')
    parser.add_argument('--interval', type=int, default=300, help='Monitoring interval (seconds)')
    parser.add_argument('--duration', type=int, default=3600, help='Total monitoring duration (seconds)')
    parser.add_argument('--alert-latency', type=float, default=100.0, help='Latency alert threshold (ms)')
    parser.add_argument('--alert-accuracy', type=float, default=0.05, help='Accuracy drop alert threshold')
    parser.add_argument('--max-samples', type=int, default=200, help='Max samples per check')

    args = parser.parse_args()

    # Check if model exists
    if not Path(args.model).exists():
        print(f"Error: Model not found: {args.model}")
        sys.exit(1)

    try:
        # Load test data
        print("Loading test data...")
        data_dir = Path(args.data_dir)

        if not (data_dir / 'X_test.npy').exists():
            print(f"Error: Test data not found in {data_dir}")
            sys.exit(1)

        X_test = np.load(data_dir / 'X_test.npy')
        y_test = np.load(data_dir / 'y_test.npy')

        # Limit samples for faster monitoring
        if len(X_test) > args.max_samples:
            indices = np.random.choice(len(X_test), args.max_samples, replace=False)
            X_test = X_test[indices]
            y_test = y_test[indices]

        test_data = (X_test, y_test)
        print(f"Loaded {len(X_test)} test samples\n")

        # Start monitoring
        monitor = RealTimePerformanceMonitor(args.output_dir)
        report = monitor.monitor_performance(
            args.model,
            test_data,
            args.interval,
            args.duration,
            args.alert_latency,
            args.alert_accuracy
        )

        print("\n" + "="*60)
        print("MONITORING COMPLETE")
        print("="*60)
        print("Check the data_flywheel/feedback/ directory for detailed reports and alerts")

    except KeyboardInterrupt:
        print("\nMonitoring interrupted by user")
    except Exception as e:
        print(f"Error during monitoring: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()