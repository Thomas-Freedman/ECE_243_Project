"""
Unified training module for all neural decoder models
"""
import os
import pickle
import time
import random
import numpy as np
import torch
from edit_distance import SequenceMatcher
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from neural_decoder.dataset import SpeechDataset
from neural_decoder.model import GRUDecoder
from neural_decoder.final2_model import NeuralDecoder


def collate_fn(batch):
    """Collate function for DataLoader"""
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


def apply_time_masking(x, num_masks, max_mask_fraction, patch_size):
    """
    Apply time masking augmentation (for transformer models)

    Args:
        x: (batch, time, features)
        num_masks: Number of mask spans to apply
        max_mask_fraction: Maximum fraction of sequence to mask in one span
        patch_size: Size of each patch (for patch-level masking)

    Returns:
        Masked x: (batch, time, features)
    """
    B, T, C = x.shape
    x_masked = x.clone()

    num_patches = T // patch_size

    for b in range(B):
        for _ in range(num_masks):
            max_mask_len = max(1, int(num_patches * max_mask_fraction))
            mask_len = random.randint(1, max_mask_len)

            if num_patches > mask_len:
                start_patch = random.randint(0, num_patches - mask_len)
            else:
                start_patch = 0
                mask_len = num_patches

            start_time = start_patch * patch_size
            end_time = min((start_patch + mask_len) * patch_size, T)

            x_masked[b, start_time:end_time, :] = 0

    return x_masked


def create_model(config, nDays, device):
    """
    Factory function to create model based on config variant

    Args:
        config: Configuration dictionary
        nDays: Number of recording days
        device: Device to place model on

    Returns:
        model: Neural decoder model
    """
    variant = config['variant']

    if variant in ['baseline', 'final']:
        # GRU-based models
        model = GRUDecoder(
            neural_dim=config['nInputFeatures'],
            n_classes=config['nClasses'],
            hidden_dim=config['nUnits'],
            layer_dim=config['nLayers'],
            nDays=nDays,
            dropout=config['dropout'],
            device=device,
            strideLen=config['strideLen'],
            kernelLen=config['kernelLen'],
            gaussianSmoothWidth=config['gaussianSmoothWidth'],
            bidirectional=config['bidirectional'],
        ).to(device)

    elif variant == 'final2':
        # NeuralDecoder with Mamba encoders and dual CTC decoders
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
        ).to(device)

    else:
        raise ValueError(f"Unknown variant: {variant}")

    return model


def create_optimizer(model, config):
    """
    Create optimizer based on config

    Args:
        model: Model to optimize
        config: Configuration dictionary

    Returns:
        optimizer: PyTorch optimizer
    """
    if config.get('useAdamW', False):
        # AdamW (for transformer)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config['lr'],
            weight_decay=config['l2_decay'],
        )
    elif config.get('useSGD', False):
        # SGD with momentum (for final model)
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=config['lrStart'],
            momentum=config['momentum'],
            nesterov=config['useNesterov'],
            weight_decay=config['l2_decay'],
        )
    else:
        # Adam (for baseline)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.get('lrStart', config.get('lr', 0.02)),
            betas=(0.9, 0.999),
            eps=0.1,
            weight_decay=config['l2_decay'],
        )

    return optimizer


def create_scheduler(optimizer, config, batches_per_epoch):
    """
    Create learning rate scheduler based on config

    Args:
        optimizer: Optimizer to schedule
        config: Configuration dictionary
        batches_per_epoch: Number of batches per epoch

    Returns:
        scheduler: PyTorch scheduler
    """
    if config.get('useAdamW', False) and 'lrDropEpoch' in config:
        # Step decay for transformer
        lr_drop_step = int(config['lrDropEpoch'] * batches_per_epoch / config['nBatch'] * config['nBatch'])
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=lr_drop_step,
            gamma=config.get('lrDropFactor', 0.1),
        )
    elif config.get('useSGD', False):
        # Linear decay for SGD
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=config['lrEnd'] / config['lrStart'],
            total_iters=config['nBatch'],
        )
    else:
        # Linear decay for baseline Adam
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=0.25,
            total_iters=config['nBatch'],
        )

    return scheduler


def apply_augmentation(X, config, device):
    """
    Apply data augmentation to input batch

    Args:
        X: Input tensor (batch, time, features)
        config: Configuration dictionary
        device: Device tensor is on

    Returns:
        Augmented X
    """
    # White noise
    if config.get('whiteNoiseSD', 0) > 0:
        X = X + torch.randn(X.shape, device=device) * config['whiteNoiseSD']

    # Constant offset / baseline shift
    if config.get('constantOffsetSD', 0) > 0:
        X = X + torch.randn([X.shape[0], 1, X.shape[2]], device=device) * config['constantOffsetSD']

    # Feature masking
    if config.get('featureMaskProb', 0) > 0:
        mask = torch.rand_like(X) < config['featureMaskProb']
        X = X.masked_fill(mask, 0)

    # Time masking (for transformer)
    if config.get('numTimeMasks', 0) > 0 and config['variant'] == 'final2':
        X = apply_time_masking(
            X,
            config['numTimeMasks'],
            config['maxMaskLengthFraction'],
            config['patchSize']
        )

    return X


def compute_sequence_lengths(X_len, config):
    """
    Compute output sequence lengths based on model architecture

    Args:
        X_len: Input sequence lengths (batch,)
        config: Configuration dictionary

    Returns:
        Output sequence lengths (batch,)
    """
    variant = config['variant']

    if variant in ['baseline', 'final']:
        # GRU: strided convolution
        return ((X_len - config['kernelLen']) / config['strideLen']).to(torch.int32)
    elif variant == 'final2':
        # NeuralDecoder: 2 conv layers with stride 2 each
        # First conv: stride 2
        # Second conv: stride 2
        # Total: divide by 4
        return (X_len // 4).to(torch.int32)
    else:
        raise ValueError(f"Unknown variant: {variant}")


def evaluate_model(model, test_loader, loss_ctc, config, device, max_batches=20):
    """
    Evaluate model on test set

    Args:
        model: Model to evaluate
        test_loader: Test data loader
        loss_ctc: CTC loss function
        config: Configuration dictionary
        device: Device to run on
        max_batches: Maximum number of batches to evaluate

    Returns:
        avg_loss: Average CTC loss
        per: Phoneme error rate
    """
    model.eval()
    all_loss = []
    total_edit = 0
    total_len = 0

    for eval_idx, (X, y, X_len, y_len, test_day_idx) in enumerate(test_loader):
        if eval_idx >= max_batches:
            break

        X, y, X_len, y_len, test_day_idx = (
            X.to(device),
            y.to(device),
            X_len.to(device),
            y_len.to(device),
            test_day_idx.to(device),
        )

        # Forward pass
        pred = model.forward(X, test_day_idx)

        # Get actual output lengths from model for final2
        if config['variant'] == 'final2':
            out_lens = model._last_output_lens
        else:
            out_lens = compute_sequence_lengths(X_len, config)

        # CTC loss
        loss = loss_ctc(
            torch.permute(pred.log_softmax(2), [1, 0, 2]),
            y,
            out_lens,
            y_len,
        )
        loss = torch.sum(loss)
        all_loss.append(loss.cpu().detach().numpy())

        # Decode and compute PER (Phoneme Error Rate)
        for i in range(pred.shape[0]):
            logits = pred[i, :out_lens[i], :]
            decoded = torch.argmax(logits, dim=-1)
            decoded = torch.unique_consecutive(decoded)
            decoded = decoded.cpu().detach().numpy()
            decoded = decoded[decoded != 0]

            target = y[i, :y_len[i]].cpu().detach().numpy()

            matcher = SequenceMatcher(a=target.tolist(), b=decoded.tolist())
            total_edit += matcher.distance()
            total_len += len(target)

    avg_loss = np.sum(all_loss) / len(all_loss)
    per = total_edit / total_len

    return avg_loss, per


def train_model(config, dataset_path, output_dir, device='cuda'):
    """
    Main training loop for any model variant

    Args:
        config: Configuration dictionary
        dataset_path: Path to dataset pickle file
        output_dir: Directory to save results
        device: Device to train on
    """
    variant = config['variant']

    print("=" * 70)
    print(f"TRAINING MODEL: {variant.upper()}")
    print("=" * 70)

    # Set seeds
    seed = config.get('seed', 0)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    print(f"\nLoading data from: {dataset_path}")
    with open(dataset_path, "rb") as f:
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
    print(f"\nCreating {variant} model...")
    model = create_model(config, len(data["train"]), device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Loss and optimizer
    loss_ctc = torch.nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)
    optimizer = create_optimizer(model, config)
    scheduler = create_scheduler(optimizer, config, len(train_loader))

    print(f"\nOptimizer: {type(optimizer).__name__}")
    print(f"Learning rate: {optimizer.param_groups[0]['lr']}")

    # Training loop
    print(f"\nStarting training for {config['nBatch']} batches...")
    print("=" * 70)

    test_loss_list = []
    test_per_list = []
    best_per = None
    start_time = time.time()

    for batch_idx in range(config['nBatch']):
        model.train()

        # Get batch
        X, y, X_len, y_len, day_idx = next(iter(train_loader))
        X, y, X_len, y_len, day_idx = (
            X.to(device),
            y.to(device),
            X_len.to(device),
            y_len.to(device),
            day_idx.to(device),
        )

        # Apply augmentation
        X = apply_augmentation(X, config, device)

        # Forward pass
        pred = model.forward(X, day_idx)

        # Get actual output lengths from model for final2
        if config['variant'] == 'final2':
            out_lens = model._last_output_lens
        else:
            out_lens = compute_sequence_lengths(X_len, config)

        # CTC loss
        loss = loss_ctc(
            torch.permute(pred.log_softmax(2), [1, 0, 2]),
            y,
            out_lens,
            y_len,
        )
        loss = torch.sum(loss)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping (important for transformers to prevent NaN)
        if variant == 'final2':
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step()

        # Evaluation
        eval_freq = 500 if variant in ['final', 'final2'] else 100
        if batch_idx % eval_freq == 0:
            with torch.no_grad():
                avg_loss, per = evaluate_model(
                    model, test_loader, loss_ctc, config, device
                )

                elapsed = (time.time() - start_time) / eval_freq if batch_idx > 0 else 0.0
                current_lr = optimizer.param_groups[0]['lr']

                print(f"batch {batch_idx:5d}, loss: {avg_loss:.4f}, PER: {per:.4f}, "
                      f"lr: {current_lr:.6f}, time/batch: {elapsed:.3f}s")
                start_time = time.time()

                # Save stats
                test_loss_list.append(avg_loss)
                test_per_list.append(per)

                # Save best model
                if best_per is None or per < best_per:
                    best_per = per
                    torch.save(model.state_dict(), os.path.join(output_dir, "modelWeights.pt"))
                    print(f"  → New best PER: {per:.4f}, model saved!")

                # Save stats
                stats = {
                    "testLoss": np.array(test_loss_list),
                    "testPER": np.array(test_per_list),
                }
                with open(os.path.join(output_dir, "trainingStats.pkl"), "wb") as f:
                    pickle.dump(stats, f)

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)
    print(f"Best PER: {best_per:.4f} ({best_per*100:.2f}%)")
    print(f"Model saved to: {output_dir}/modelWeights.pt")
    print(f"Stats saved to: {output_dir}/trainingStats.pkl")
    print("=" * 70)
