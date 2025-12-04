#!/usr/bin/env python3
"""
Training script for the final2 model using NeuralDecoder architecture
"""
import sys
sys.path.insert(0, 'src')

import os
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
from omegaconf import OmegaConf
import torch

from neural_decoder.neural_decoder_trainer import DataModule
from neural_decoder.final2_model import NeuralDecoder

CONFIG_PATH = "src/neural_decoder/conf/decoder/final2.yaml"
DATASET_NAME = "competition_data"
OUTPUT_DIR = "results/final2_training"

print("=" * 70)
print("FINAL2 MODEL TRAINING")
print("=" * 70)

config = OmegaConf.load(CONFIG_PATH)
print(f"\nLoaded configuration from: {CONFIG_PATH}")
print(f"Model variant: {config.variant}")

os.makedirs(OUTPUT_DIR, exist_ok=True)

torch.manual_seed(config.get('seed', 0))

datamodule = DataModule(
    dataset_name=DATASET_NAME,
    batch_size=config.batchSize,
    num_workers=4,
)

model = NeuralDecoder(
    conv_size=config.get('conv_size', 1024),
    conv_kernel1=config.get('conv_kernel1', 7),
    conv_kernel2=config.get('conv_kernel2', 3),
    conv_g1=config.get('conv_g1', 256),
    conv_g2=config.get('conv_g2', 1),
    hidden_size=config.get('hidden_size', 512),
    encoder_n_layer=config.get('encoder_n_layer', 5),
    decoder_n_layer=config.get('decoder_n_layer', 5),
    decoders=config.get('decoders', ['al', 'ph']),
    update_probs=config.get('update_probs', 0.7),
    al_loss_weight=config.get('al_loss_weight', 0.5),
    peak_lr=config.get('peak_lr', 1e-4),
    last_lr=config.get('last_lr', 1e-6),
    beta_1=config.get('beta_1', 0.9),
    beta_2=config.get('beta_2', 0.95),
    weight_decay=config.get('weight_decay', 0.1),
    eps=config.get('eps', 1e-08),
    lr_warmup_perc=config.get('lr_warmup_perc', 0.1),
)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nModel parameters:")
print(f"  Total: {total_params:,}")
print(f"  Trainable: {trainable_params:,}")

checkpoint_callback = ModelCheckpoint(
    dirpath=OUTPUT_DIR,
    filename="final2-{epoch:02d}-{wer:.4f}",
    monitor="wer",
    mode="min",
    save_top_k=3,
    save_last=True,
)

lr_monitor = LearningRateMonitor(logging_interval='step')

logger = TensorBoardLogger(
    save_dir=OUTPUT_DIR,
    name="final2_logs",
)

trainer = L.Trainer(
    max_epochs=config.get('max_epochs', 100),
    accelerator="auto",
    devices=1,
    logger=logger,
    callbacks=[checkpoint_callback, lr_monitor],
    gradient_clip_val=config.get('gradient_clip_val', 1.0),
    accumulate_grad_batches=config.get('accumulate_grad_batches', 1),
    precision=config.get('precision', '32'),
    log_every_n_steps=10,
)

print(f"\nStarting training for {config.get('max_epochs', 100)} epochs...")
print("=" * 70)

trainer.fit(model, datamodule=datamodule)

print("\n" + "=" * 70)
print("TRAINING COMPLETE!")
print("=" * 70)
print(f"Best model saved to: {checkpoint_callback.best_model_path}")
print("=" * 70)
