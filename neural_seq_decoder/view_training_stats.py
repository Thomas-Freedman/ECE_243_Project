#!/usr/bin/env python3
"""
View training statistics from the neural decoder models.
Usage: python view_training_stats.py <path_to_trainingStats_file>
"""

import pickle
import sys
import os
import numpy as np

def view_stats(stats_file):
    """Load and display training statistics."""
    if not os.path.exists(stats_file):
        print(f"Error: File not found: {stats_file}")
        return

    print(f"\nLoading stats from: {stats_file}")
    print("=" * 70)

    try:
        with open(stats_file, 'rb') as f:
            stats = pickle.load(f)

        print(f"\nAvailable metrics: {list(stats.keys())}")
        print("=" * 70)

        for key, value in stats.items():
            print(f"\n{key}:")
            print("-" * 70)

            if isinstance(value, np.ndarray):
                print(f"  Shape: {value.shape}")
                print(f"  Data type: {value.dtype}")
                print(f"  Min: {np.min(value):.6f}")
                print(f"  Max: {np.max(value):.6f}")
                print(f"  Mean: {np.mean(value):.6f}")
                print(f"  Last value: {value[-1]:.6f}")

                # Show first few and last few values
                if len(value) > 10:
                    print(f"\n  First 5 values:")
                    for i, val in enumerate(value[:5]):
                        print(f"    [{i}]: {val:.6f}")
                    print(f"\n  Last 5 values:")
                    for i, val in enumerate(value[-5:], start=len(value)-5):
                        print(f"    [{i}]: {val:.6f}")
                else:
                    print(f"\n  All values:")
                    for i, val in enumerate(value):
                        print(f"    [{i}]: {val:.6f}")
            else:
                print(f"  Value: {value}")

        print("\n" + "=" * 70)
        print("Summary:")
        print("=" * 70)

        if 'testLoss' in stats:
            test_loss = stats['testLoss']
            print(f"Test Loss: {test_loss[-1]:.6f} (started at {test_loss[0]:.6f})")
            improvement = ((test_loss[0] - test_loss[-1]) / test_loss[0]) * 100
            print(f"Improvement: {improvement:.2f}%")

    except Exception as e:
        print(f"\nError loading file: {e}")
        return


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python view_training_stats.py <path_to_trainingStats_file>")
        print("\nExample:")
        print("  python view_training_stats.py results/baseline/trainingStats")
        print("  python view_training_stats.py ~/results/baseline/trainingStats")
        sys.exit(1)

    stats_file = sys.argv[1]

    # Expand ~ to home directory
    stats_file = os.path.expanduser(stats_file)

    view_stats(stats_file)
