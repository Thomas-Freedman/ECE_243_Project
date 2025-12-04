#!/usr/bin/env python3
"""
Simple final model training script - reads from config file
Optimized GRU with SGD
"""
import sys
sys.path.insert(0, 'src')

import os
import pickle
import time
import yaml
import numpy as np
import torch
from edit_distance import SequenceMatcher
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from neural_decoder.dataset import SpeechDataset
from neural_decoder.model import GRUDecoder

# ============================================================================
# LOAD CONFIGURATION
# ============================================================================
CONFIG_PATH = "src/neural_decoder/conf/decoder/final.yaml"
DATASET_PATH = os.path.expanduser("~/competitionData/ptDecoder_ctc")
OUTPUT_DIR = os.path.expanduser("~/results/final_simple")
DEVICE = "cuda"

print("=" * 70)
print("FINAL MODEL TRAINING (Optimized GRU + SGD)")
print("=" * 70)

# Load config
with open(CONFIG_PATH, 'r') as f:
    config = yaml.safe_load(f)

print(f"\nLoaded configuration from: {CONFIG_PATH}")
print(f"Model: {config['nLayers']} layers, {config['nUnits']} units, bidirectional={config['bidirectional']}")
print(f"Training: {config['nBatch']} batches, batch_size={config['batchSize']}")
print(f"Optimizer: SGD (momentum={config['momentum']}, nesterov={config['useNesterov']})")
print(f"Learning rate: {config['lrStart']} → {config['lrEnd']}")

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

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Set seed
torch.manual_seed(config['seed'])
np.random.seed(config['seed'])

# Load data
print(f"\nLoading data from: {DATASET_PATH}")
with open(DATASET_PATH, "rb") as f:
    data = pickle.load(f)

train_ds = SpeechDataset(data["train"])
test_ds = SpeechDataset(data["test"])

train_loader = DataLoader(
    train_ds,
    batch_size=config['batchSize'],
    shuffle=True,
    num_workers=0,
    pin_memory=True,
    collate_fn=collate_fn,
)
test_loader = DataLoader(
    test_ds,
    batch_size=config['batchSize'],
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
    neural_dim=config['nInputFeatures'],
    n_classes=config['nClasses'],
    hidden_dim=config['nUnits'],
    layer_dim=config['nLayers'],
    nDays=len(data["train"]),
    dropout=config['dropout'],
    device=DEVICE,
    strideLen=config['strideLen'],
    kernelLen=config['kernelLen'],
    gaussianSmoothWidth=config['gaussianSmoothWidth'],
    bidirectional=config['bidirectional'],
).to(DEVICE)

print(f"Using SGD with Nesterov momentum")

# Loss and optimizer
loss_ctc = torch.nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=config['lrStart'],
    momentum=config['momentum'],
    nesterov=config['useNesterov'],
    weight_decay=config['l2_decay'],
)
scheduler = torch.optim.lr_scheduler.LinearLR(
    optimizer,
    start_factor=1.0,
    end_factor=config['lrEnd'] / config['lrStart'],
    total_iters=config['nBatch'],
)

# Training loop
print(f"\nStarting training for {config['nBatch']} batches...")
print("=" * 70)

test_loss_list = []
test_cer_list = []
best_cer = None
start_time = time.time()

for batch_idx in range(config['nBatch']):
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
    if config['whiteNoiseSD'] > 0:
        X += torch.randn(X.shape, device=DEVICE) * config['whiteNoiseSD']
    if config['constantOffsetSD'] > 0:
        X += torch.randn([X.shape[0], 1, X.shape[2]], device=DEVICE) * config['constantOffsetSD']
    if config.get('featureMaskProb', 0) > 0:
        # Feature masking: randomly zero out individual feature values
        mask = torch.rand_like(X) < config['featureMaskProb']
        X = X.masked_fill(mask, 0)

    # Forward
    pred = model.forward(X, day_idx)
    loss = loss_ctc(
        torch.permute(pred.log_softmax(2), [1, 0, 2]),
        y,
        ((X_len - config['kernelLen']) / config['strideLen']).to(torch.int32),
        y_len,
    )
    loss = torch.sum(loss)

    # Backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()

    # Evaluation every 500 batches (less frequent to save time)
    if batch_idx % 500 == 0:
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
                    ((X_len - config['kernelLen']) / config['strideLen']).to(torch.int32),
                    y_len,
                )
                loss = torch.sum(loss)
                all_loss.append(loss.cpu().detach().numpy())

                adjusted_lens = ((X_len - config['kernelLen']) / config['strideLen']).to(torch.int32)
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

            elapsed = (time.time() - start_time) / 500 if batch_idx > 0 else 0.0
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
