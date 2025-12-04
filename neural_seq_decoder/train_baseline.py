#!/usr/bin/env python3
"""
Simple baseline GRU training script - NO HYDRA
"""
import sys
sys.path.insert(0, 'src')

import os
import pickle
import time
import numpy as np
import torch
from edit_distance import SequenceMatcher
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from neural_decoder.dataset import SpeechDataset
from neural_decoder.model import GRUDecoder

# ============================================================================
# CONFIGURATION - EDIT THESE
# ============================================================================
DATASET_PATH = os.path.expanduser("~/competitionData/ptDecoder_ctc")
OUTPUT_DIR = os.path.expanduser("~/results/baseline_simple")
DEVICE = "cuda"

# Training params
BATCH_SIZE = 64
N_BATCH = 10000
LEARNING_RATE = 0.02
L2_DECAY = 1e-5
SEED = 0

# Model params
N_UNITS = 1024
N_LAYERS = 5
N_CLASSES = 40
N_INPUT_FEATURES = 256
DROPOUT = 0.4
BIDIRECTIONAL = True

# Preprocessing
STRIDE_LEN = 4
KERNEL_LEN = 32
GAUSSIAN_SMOOTH_WIDTH = 2.0

# Augmentation
WHITE_NOISE_SD = 0.8
CONSTANT_OFFSET_SD = 0.2

# ============================================================================
# TRAINING CODE
# ============================================================================

def collate_fn(batch):
    X, y, X_lens, y_lens, days = zip(*batch)
    X_padded = pad_sequence(X, batch_first=True, padding_value=0)
    y_padded = pad_sequence(y, batch_first=True, padding_value=0)
    return (
        X_padded,
        y_padded,
        torch.stack(X_lens),
        torch.stack(y_lens),
        torch.stack(days),
    )

print("=" * 70)
print("BASELINE MODEL TRAINING")
print("=" * 70)

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Set seed
torch.manual_seed(SEED)
np.random.seed(SEED)

# Load data
print(f"\nLoading data from: {DATASET_PATH}")
with open(DATASET_PATH, "rb") as f:
    data = pickle.load(f)

train_ds = SpeechDataset(data["train"])
test_ds = SpeechDataset(data["test"])

train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=True,
    collate_fn=collate_fn,
)
test_loader = DataLoader(
    test_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=True,
    collate_fn=collate_fn,
)

print(f"Train samples: {len(train_ds)}")
print(f"Test samples: {len(test_ds)}")

# Create model
print(f"\nCreating model...")
model = GRUDecoder(
    neural_dim=N_INPUT_FEATURES,
    n_classes=N_CLASSES,
    hidden_dim=N_UNITS,
    layer_dim=N_LAYERS,
    nDays=len(data["train"]),
    dropout=DROPOUT,
    device=DEVICE,
    strideLen=STRIDE_LEN,
    kernelLen=KERNEL_LEN,
    gaussianSmoothWidth=GAUSSIAN_SMOOTH_WIDTH,
    bidirectional=BIDIRECTIONAL,
).to(DEVICE)

print(f"Model created: {N_LAYERS} layers, {N_UNITS} units, bidirectional={BIDIRECTIONAL}")

# Loss and optimizer
loss_ctc = torch.nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
    betas=(0.9, 0.999),
    eps=0.1,
    weight_decay=L2_DECAY,
)
scheduler = torch.optim.lr_scheduler.LinearLR(
    optimizer,
    start_factor=1.0,
    end_factor=0.25,
    total_iters=N_BATCH,
)

# Training loop
print(f"\nStarting training for {N_BATCH} batches...")
print("=" * 70)

test_loss_list = []
test_cer_list = []
best_cer = None
start_time = time.time()

for batch_idx in range(N_BATCH):
    model.train()

    # Get batch
    X, y, X_len, y_len, day_idx = next(iter(train_loader))
    X, y, X_len, y_len, day_idx = (
        X.to(DEVICE),
        y.to(DEVICE),
        X_len.to(DEVICE),
        y_len.to(DEVICE),
        day_idx.to(DEVICE),
    )

    # Augmentation
    if WHITE_NOISE_SD > 0:
        X += torch.randn(X.shape, device=DEVICE) * WHITE_NOISE_SD
    if CONSTANT_OFFSET_SD > 0:
        X += torch.randn([X.shape[0], 1, X.shape[2]], device=DEVICE) * CONSTANT_OFFSET_SD

    # Forward
    pred = model.forward(X, day_idx)
    loss = loss_ctc(
        torch.permute(pred.log_softmax(2), [1, 0, 2]),
        y,
        ((X_len - KERNEL_LEN) / STRIDE_LEN).to(torch.int32),
        y_len,
    )
    loss = torch.sum(loss)

    # Backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()

    # Evaluation every 100 batches
    if batch_idx % 100 == 0:
        with torch.no_grad():
            model.eval()
            all_loss = []
            total_edit = 0
            total_len = 0

            # Evaluate on subset to save time
            for eval_idx, (X, y, X_len, y_len, test_day_idx) in enumerate(test_loader):
                if eval_idx >= 20:  # Only eval on first 20 batches
                    break

                X, y, X_len, y_len, test_day_idx = (
                    X.to(DEVICE),
                    y.to(DEVICE),
                    X_len.to(DEVICE),
                    y_len.to(DEVICE),
                    test_day_idx.to(DEVICE),
                )

                pred = model.forward(X, test_day_idx)
                loss = loss_ctc(
                    torch.permute(pred.log_softmax(2), [1, 0, 2]),
                    y,
                    ((X_len - KERNEL_LEN) / STRIDE_LEN).to(torch.int32),
                    y_len,
                )
                loss = torch.sum(loss)
                all_loss.append(loss.cpu().detach().numpy())

                adjusted_lens = ((X_len - KERNEL_LEN) / STRIDE_LEN).to(torch.int32)
                for i in range(pred.shape[0]):
                    logits = pred[i, :adjusted_lens[i], :]
                    decoded = torch.argmax(logits, dim=-1)
                    decoded = torch.unique_consecutive(decoded)
                    decoded = decoded.cpu().detach().numpy()
                    decoded = decoded[decoded != 0]

                    target = y[i, :y_len[i]].cpu().detach().numpy()

                    matcher = SequenceMatcher(a=target.tolist(), b=decoded.tolist())
                    total_edit += matcher.distance()
                    total_len += len(target)

            avg_loss = np.sum(all_loss) / len(all_loss)
            cer = total_edit / total_len

            elapsed = (time.time() - start_time) / 100 if batch_idx > 0 else 0.0
            print(f"batch {batch_idx:5d}, ctc loss: {avg_loss:.4f}, cer: {cer:.4f}, time/batch: {elapsed:.3f}s")
            start_time = time.time()

            # Save stats
            test_loss_list.append(avg_loss)
            test_cer_list.append(cer)

            # Save best model
            if best_cer is None or cer < best_cer:
                best_cer = cer
                torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "modelWeights.pt"))
                print(f"  → New best CER: {cer:.4f}, model saved!")

            # Save stats
            stats = {
                "testLoss": np.array(test_loss_list),
                "testCER": np.array(test_cer_list),
            }
            with open(os.path.join(OUTPUT_DIR, "trainingStats.pkl"), "wb") as f:
                pickle.dump(stats, f)

print("\n" + "=" * 70)
print("TRAINING COMPLETE!")
print("=" * 70)
print(f"Best CER: {best_cer:.4f} ({best_cer*100:.2f}%)")
print(f"Model saved to: {OUTPUT_DIR}/modelWeights.pt")
print(f"Stats saved to: {OUTPUT_DIR}/trainingStats.pkl")
print("=" * 70)
