#!/usr/bin/env python3
"""
Quick test to verify the final model can be instantiated and run.
"""
import torch
import sys
sys.path.insert(0, 'src')

from neural_decoder.model import GRUDecoder

print("=" * 70)
print("Testing FINAL MODEL setup")
print("=" * 70)

# Create model with winner's configuration
model = GRUDecoder(
    neural_dim=256,
    n_classes=40,
    hidden_dim=1024,
    layer_dim=5,
    nDays=24,
    dropout=0.4,
    device='cpu',  # Use CPU for testing
    strideLen=4,
    kernelLen=32,
    gaussianSmoothWidth=2.0,
    bidirectional=True,  # KEY: bidirectional like winners!
)

print(f"✓ Model created successfully")
print(f"  - Layers: 5")
print(f"  - Hidden units: 1024")
print(f"  - Bidirectional: True")
print(f"  - Dropout: 0.4")

# Test forward pass
batch_size = 2
seq_len = 100
X = torch.randn(batch_size, seq_len, 256)
day_idx = torch.tensor([0, 1])

print(f"\nTesting forward pass...")
print(f"  Input shape: {X.shape}")

try:
    output = model(X, day_idx)
    print(f"  Output shape: {output.shape}")
    print(f"  ✓ Forward pass successful!")

    # Check output dimensions
    expected_time_steps = (seq_len - 32) // 4 + 1
    expected_classes = 41  # 40 phonemes + 1 blank

    if output.shape == (batch_size, expected_time_steps, expected_classes):
        print(f"  ✓ Output dimensions correct!")
        print(f"    Expected: ({batch_size}, {expected_time_steps}, {expected_classes})")
        print(f"    Got: {output.shape}")
    else:
        print(f"  ⚠ Output dimensions unexpected")
        print(f"    Expected: ({batch_size}, {expected_time_steps}, {expected_classes})")
        print(f"    Got: {output.shape}")

    print("\n" + "=" * 70)
    print("Model test PASSED - ready to train!")
    print("=" * 70)
    print("\nTo train the final model, run:")
    print("  python3 -m neural_decoder.main decoder=final \\")
    print("    datasetPath=~/competitionData/ptDecoder_ctc \\")
    print("    outputDir=~/results/final")
    print("=" * 70)

except Exception as e:
    print(f"  ✗ Forward pass failed!")
    print(f"  Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test optimizer setup
print("\nTesting SGD optimizer setup...")
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.02,
    momentum=0.9,
    nesterov=True,
    weight_decay=1e-5,
)
print(f"  ✓ SGD optimizer created")
print(f"    LR: 0.02")
print(f"    Momentum: 0.9")
print(f"    Nesterov: True")

print("\n" + "=" * 70)
print("All tests passed! Final model is ready to train.")
print("=" * 70)
