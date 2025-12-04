#!/usr/bin/env python3
"""
Standalone training script for Advanced Model (StreamingTransformerDecoder)
Can be run directly from command line: python train_advanced_script.py
"""

import pickle
import random
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from edit_distance import SequenceMatcher
import os
import sys

print("=" * 70)
print("ADVANCED MODEL TRAINING SCRIPT")
print("=" * 70)
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")
print("=" * 70)

# ============================================================================
# DATASET CLASS
# ============================================================================
class SpeechDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.n_days = len(data)

        self.neural_feats = []
        self.phone_seqs = []
        self.neural_time_bins = []
        self.phone_seq_lens = []
        self.days = []

        for day_idx in range(self.n_days):
            day_data = data[day_idx]
            for trial_idx in range(len(day_data["sentenceDat"])):
                self.neural_feats.append(day_data["sentenceDat"][trial_idx])
                self.phone_seqs.append(day_data["phonemes"][trial_idx])
                self.neural_time_bins.append(day_data["sentenceDat"][trial_idx].shape[0])
                self.phone_seq_lens.append(day_data["phoneLens"][trial_idx])
                self.days.append(day_idx)

        self.n_trials = len(self.neural_feats)
        print(f"Dataset initialized: {self.n_trials} trials from {self.n_days} days")

    def __len__(self):
        return self.n_trials

    def __getitem__(self, idx):
        return (
            torch.tensor(self.neural_feats[idx], dtype=torch.float32),
            torch.tensor(self.phone_seqs[idx], dtype=torch.int64),
            torch.tensor(self.neural_time_bins[idx], dtype=torch.int64),
            torch.tensor(self.phone_seq_lens[idx], dtype=torch.int64),
            torch.tensor(self.days[idx], dtype=torch.int64),
        )


def collate_fn(batch):
    X, y, X_lens, y_lens, days = zip(*batch)
    X_padded = pad_sequence(X, batch_first=True, padding_value=0.0)
    y_padded = pad_sequence(y, batch_first=True, padding_value=0)
    return (
        X_padded,
        y_padded,
        torch.stack(X_lens),
        torch.stack(y_lens),
        torch.stack(days),
    )


# ============================================================================
# MODEL COMPONENTS
# ============================================================================
class GaussianSmoothing(nn.Module):
    def __init__(self, channels, kernel_size, sigma, dim=1):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = [kernel_size]
        if isinstance(sigma, float):
            sigma = [sigma]

        kernel = 1
        meshgrids = torch.meshgrid(
            [torch.arange(size, dtype=torch.float32) for size in kernel_size],
            indexing='ij'
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * torch.exp(-(((mgrid - mean) / std) ** 2) / 2)

        kernel = kernel / torch.sum(kernel)
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels
        self.conv = F.conv1d
        padding = []
        for k in kernel_size:
            padding.append(k // 2)
        self.padding = padding[0]

    def forward(self, input):
        return self.conv(input, weight=self.weight, groups=self.groups, padding=self.padding)


class PatchEmbedding(nn.Module):
    def __init__(self, Tin: int, feat_dim: int, hdim: int, eps: float = 1e-6):
        super().__init__()
        self.Tin = Tin
        self.feat_dim = feat_dim
        self.in_dim = Tin * feat_dim
        self.ln1 = nn.LayerNorm(self.in_dim, eps=eps)
        self.proj = nn.Linear(self.in_dim, hdim)
        self.ln2 = nn.LayerNorm(hdim, eps=eps)

    def forward(self, x):
        bsz, T, F = x.shape
        n_patch = T // self.Tin
        x = x[:, : n_patch * self.Tin, :]
        x = x.view(bsz, n_patch, self.Tin * F)
        x = self.ln1(x)
        x = self.proj(x)
        x = self.ln2(x)
        return x


class Attention(nn.Module):
    def __init__(self, hdim: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        if hdim % n_heads != 0:
            raise ValueError("hdim must be divisible by number of heads")
        self.hdim = hdim
        self.n_heads = n_heads
        self.head_dim = hdim // n_heads
        self.qkv = nn.Linear(hdim, 3 * hdim)
        self.out = nn.Linear(hdim, hdim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, causal_mask=None, rel_bias=None, key_padding_mask=None):
        bsz, seq_len, dim = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.split(dim, dim=-1)

        def reshape_heads(tensor):
            return tensor.view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        q = reshape_heads(q)
        k = reshape_heads(k)
        v = reshape_heads(v)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if rel_bias is not None:
            scores = scores + rel_bias.unsqueeze(0)
        if causal_mask is not None:
            if causal_mask.dim() == 2:
                scores = scores + causal_mask.unsqueeze(0).unsqueeze(0)
        if key_padding_mask is not None:
            scores = scores.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf")
            )

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(bsz, seq_len, dim)
        out = self.out(out)
        if key_padding_mask is not None:
            out = out.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)
        return out


class FeedForward(nn.Module):
    def __init__(self, hdim: int, ff_dim: int, dropout: float = 0.0):
        super().__init__()
        self.ln = nn.LayerNorm(hdim)
        self.fc1 = nn.Linear(hdim, ff_dim)
        self.act = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(ff_dim, hdim)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        x = self.ln(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, hdim: int, n_heads: int, ff_dim: int,
                 attn_dropout: float = 0.0, dropout: float = 0.0):
        super().__init__()
        self.ln_attn = nn.LayerNorm(hdim)
        self.attn = Attention(hdim, n_heads, attn_dropout)
        self.ff = FeedForward(hdim, ff_dim, dropout)

    def forward(self, x, causal_mask=None, rel_bias=None, key_padding_mask=None):
        attn_in = self.ln_attn(x)
        attn_out = self.attn(attn_in, causal_mask=causal_mask,
                            rel_bias=rel_bias, key_padding_mask=key_padding_mask)
        x = x + attn_out
        ff_out = self.ff(x)
        x = x + ff_out
        return x


class StreamingTransformerDecoder(nn.Module):
    def __init__(self, neural_dim: int, n_phonemes: int, d_model: int, nhead: int,
                 num_layers: int, dropout: float, stride_len: int, kernel_len: int,
                 gaussian_smooth_width: float, intermediate_layer: int, day_count: int,
                 diphone_context: int = None, time_mask_prob: float = 0.0,
                 rel_pos_max_dist: int = None, rel_bias_by_head: bool = True,
                 ff_mult: int = 4):
        super().__init__()
        self.neural_dim = neural_dim
        self.n_phonemes = n_phonemes
        self.blank_count = n_phonemes + 1
        self.kernel_len = kernel_len
        self.stride_len = stride_len
        self.gaussian = GaussianSmoothing(neural_dim, 20, gaussian_smooth_width, dim=1)
        self.time_mask_prob = time_mask_prob

        self.day_weights = nn.Parameter(torch.randn(day_count, neural_dim, neural_dim))
        self.day_bias = nn.Parameter(torch.zeros(day_count, 1, neural_dim))
        for d in range(day_count):
            with torch.no_grad():
                self.day_weights[d] = torch.eye(neural_dim)

        self.input_nonlin = nn.Softsign()
        self.patch_embed = PatchEmbedding(kernel_len, neural_dim, d_model)
        self.input_dropout = nn.Dropout(dropout)

        if intermediate_layer >= num_layers:
            raise ValueError("intermediate_layer must be < num_layers")
        self.intermediate_layer = intermediate_layer

        ff_dim = ff_mult * d_model
        self.layers = nn.ModuleList([
            TransformerBlock(hdim=d_model, n_heads=nhead, ff_dim=ff_dim,
                           attn_dropout=dropout, dropout=dropout)
            for _ in range(num_layers)
        ])

        self.rel_pos_bias = None
        self.final_ln = nn.LayerNorm(d_model)

        self.intermediate_head = nn.Linear(d_model, self.blank_count)
        self.feedback_proj = nn.Linear(self.blank_count, d_model)
        self.final_head = nn.Linear(d_model, self.blank_count)

        diphone_dim = diphone_context or n_phonemes
        self.diphone_dim = diphone_dim
        self.diphone_head = nn.Linear(d_model, diphone_dim * diphone_dim)

    def _apply_preproc(self, neural, lengths, day_idx):
        neural = torch.permute(neural, (0, 2, 1))
        neural = self.gaussian(neural)
        neural = torch.permute(neural, (0, 2, 1))

        weights = torch.index_select(self.day_weights, 0, day_idx)
        transformed = torch.einsum("btd,bdk->btk", neural, weights) + torch.index_select(
            self.day_bias, 0, day_idx
        )
        transformed = self.input_nonlin(transformed)
        max_len = transformed.shape[1]
        mask = torch.arange(max_len, device=transformed.device).unsqueeze(0) >= lengths.unsqueeze(1)
        transformed = transformed.masked_fill(mask.unsqueeze(-1), 0.0)
        return transformed

    @staticmethod
    def _causal_mask(seq_len, device):
        mask = torch.zeros((seq_len, seq_len), device=device, dtype=torch.float32)
        mask = torch.triu(mask, diagonal=1)
        mask = mask * -1e9
        return mask

    @staticmethod
    def _padding_mask(lengths, max_len):
        positions = torch.arange(max_len, device=lengths.device).unsqueeze(0)
        return positions >= lengths.unsqueeze(1)

    def _apply_time_mask(self, x):
        if not self.training or self.time_mask_prob <= 0.0:
            return x
        mask = torch.rand(x.shape[0], x.shape[1], device=x.device) < self.time_mask_prob
        return x.masked_fill(mask.unsqueeze(-1), 0.0)

    def marginalize_diphone(self, diphone_logits):
        bsz, steps, _ = diphone_logits.shape
        logits = diphone_logits.view(bsz, steps, self.diphone_dim, self.diphone_dim)
        log_probs = F.log_softmax(logits.view(bsz, steps, -1), dim=-1).view(
            bsz, steps, self.diphone_dim, self.diphone_dim
        )
        left = torch.logsumexp(log_probs, dim=3)
        right = torch.logsumexp(log_probs, dim=2)
        combined = torch.logaddexp(left, right) - math.log(2.0)
        combined = combined - torch.logsumexp(combined, dim=-1, keepdim=True)
        return combined

    def forward(self, neural, lengths, day_idx):
        transformed = self._apply_preproc(neural, lengths, day_idx)
        patches = self.patch_embed(transformed)
        patches = self.input_dropout(patches)
        patches = self._apply_time_mask(patches)

        seq_len = patches.shape[1]
        device = patches.device
        causal_mask = self._causal_mask(seq_len, device)
        rel_bias = self.rel_pos_bias(seq_len, device) if self.rel_pos_bias else None

        eff_lengths = torch.div(lengths, self.kernel_len, rounding_mode="floor").clamp(min=1)
        pad_mask = self._padding_mask(eff_lengths, seq_len)

        hidden = patches
        intermediate_logits = None
        intermediate_log_probs = None
        for idx, layer in enumerate(self.layers):
            hidden = layer(hidden, causal_mask=causal_mask, rel_bias=rel_bias,
                         key_padding_mask=pad_mask)
            if idx == self.intermediate_layer:
                intermediate_logits = self.intermediate_head(hidden)
                intermediate_log_probs = F.log_softmax(intermediate_logits, dim=-1)
                feedback = self.feedback_proj(intermediate_logits)
                hidden = hidden + feedback

        hidden = self.final_ln(hidden)
        final_logits = self.final_head(hidden)
        final_log_probs = F.log_softmax(final_logits, dim=-1)

        diphone_logits = self.diphone_head(hidden)
        diphone_log_probs = self.marginalize_diphone(diphone_logits)

        blank = final_log_probs[..., :1]
        phone_log_probs = final_log_probs[..., 1:]
        valid_dim = min(self.diphone_dim, self.n_phonemes)
        diphone_slice = diphone_log_probs[..., :valid_dim]
        phone_slice = phone_log_probs[..., :valid_dim]
        if valid_dim < self.n_phonemes:
            pad_width = self.n_phonemes - valid_dim
            diphone_slice = F.pad(diphone_slice, (0, pad_width), value=-100)
            phone_slice = F.pad(phone_slice, (0, pad_width), value=-100)

        fused_phone = torch.logsumexp(
            torch.stack([phone_slice, diphone_slice], dim=0), dim=0
        ) - math.log(2.0)
        fused = torch.cat([blank, fused_phone], dim=-1)
        fused = fused - torch.logsumexp(fused, dim=-1, keepdim=True)

        return {
            "log_probs": fused,
            "intermediate_log_probs": intermediate_log_probs,
            "intermediate_logits": intermediate_logits,
            "raw_logits": final_logits,
            "diphone_logits": diphone_logits,
            "diphone_log_probs": diphone_log_probs,
            "eff_lengths": eff_lengths.to(torch.int32),
        }


# ============================================================================
# AUGMENTATION
# ============================================================================
class SequenceAugmenter:
    def __init__(self, time_mask_ratio: float, min_time_span: int,
                 channel_drop_prob: float, feature_mask_prob: float):
        self.time_mask_ratio = time_mask_ratio
        self.min_time_span = max(1, min_time_span)
        self.channel_drop_prob = channel_drop_prob
        self.feature_mask_prob = feature_mask_prob

    def _apply_time_mask(self, seq):
        total_bins = seq.shape[0]
        target = int(total_bins * self.time_mask_ratio)
        masked = 0
        while masked < target and total_bins > 0:
            span = random.randint(self.min_time_span, max(self.min_time_span, total_bins))
            start = random.randint(0, max(0, total_bins - span))
            seq[start : start + span] = 0
            masked += span
        return seq

    def _apply_channel_mask(self, seq):
        if self.channel_drop_prob <= 0:
            return seq
        mask = torch.rand(seq.shape[-1], device=seq.device) < self.channel_drop_prob
        if mask.any():
            seq[:, mask] = 0
        return seq

    def _apply_feature_mask(self, seq):
        if self.feature_mask_prob <= 0:
            return seq
        feat_mask = torch.rand_like(seq) < self.feature_mask_prob
        seq = seq.masked_fill(feat_mask, 0)
        return seq

    def __call__(self, batch):
        augmented = batch.clone()
        for idx in range(augmented.shape[0]):
            augmented[idx] = self._apply_feature_mask(
                self._apply_channel_mask(self._apply_time_mask(augmented[idx]))
            )
        return augmented


def _length_mask(lengths, max_len):
    rng = torch.arange(max_len, device=lengths.device).unsqueeze(0)
    return rng >= lengths.unsqueeze(1)


def _symmetric_kl(log_probs_p, log_probs_q, pad_mask):
    probs_q = log_probs_q.exp()
    probs_p = log_probs_p.exp()
    kl_pq = F.kl_div(log_probs_p, probs_q, reduction="none").sum(-1)
    kl_qp = F.kl_div(log_probs_q, probs_p, reduction="none").sum(-1)
    keep = (~pad_mask).float()
    denom = keep.sum().clamp(min=1.0)
    return ((kl_pq + kl_qp) * keep).sum() / denom


def evaluate(model, loader, ctc_loss, device, max_batches=20):
    model.eval()
    losses = []
    total_edit = 0
    total_len = 0

    for idx, (X, y, X_len, y_len, day_idx) in enumerate(loader):
        if idx >= max_batches:
            break

        X, y, X_len, y_len, day_idx = (
            X.to(device), y.to(device), X_len.to(device),
            y_len.to(device), day_idx.to(device)
        )

        with torch.no_grad():
            out = model(X, X_len, day_idx)
            eff_len = out['eff_lengths']
            loss = ctc_loss(out['log_probs'].permute(1, 0, 2), y, eff_len, y_len)
            losses.append(loss.item())

            for i in range(out['log_probs'].shape[0]):
                logits = out['log_probs'][i, :eff_len[i], :]
                decoded = torch.argmax(logits, dim=-1)
                decoded = torch.unique_consecutive(decoded, dim=-1)
                decoded = decoded.cpu().detach().numpy()
                decoded = decoded[decoded != 0]
                target = y[i][:y_len[i]].cpu().detach().numpy()
                matcher = SequenceMatcher(a=target.tolist(), b=decoded.tolist())
                total_edit += matcher.distance()
                total_len += len(target)

    avg_loss = sum(losses) / max(len(losses), 1)
    cer = total_edit / max(total_len, 1)
    return avg_loss, cer


# ============================================================================
# MAIN TRAINING
# ============================================================================
def main():
    # Configuration
    config = {
        'batch_size': 128,
        'n_batch': 15000,
        'lr': 0.0005,
        'weight_decay': 0.0001,
        'n_classes': 40,
        'n_input_features': 256,
        'kernel_len': 8,
        'stride_len': 4,
        'gaussian_smooth_width': 2.0,
        'd_model': 512,
        'num_layers': 6,
        'nhead': 8,
        'dropout': 0.2,
        'intermediate_layer': 3,
        'time_mask_ratio': 0.3,
        'channel_drop_prob': 0.1,
        'feature_mask_prob': 0.05,
        'min_time_mask': 16,
        'consistency_weight': 0.2,
        'intermediate_loss_weight': 0.3,
        'diphone_context': 40,
        'transformer_time_mask_prob': 0.0,
        'rel_pos_max_dist': None,
        'ff_mult': 4,
        'seed': 0,
    }

    print("\nConfiguration:")
    for k, v in config.items():
        print(f"  {k}: {v}")
    print()

    # Set seeds
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    random.seed(config['seed'])

    # Load data
    dataset_path = '/home/ivansit1214/competitionData/ptDecoder_ctc'
    if not os.path.exists(dataset_path):
        print(f"ERROR: Dataset not found at {dataset_path}")
        print("Please ensure ptDecoder_ctc file is in the current directory")
        sys.exit(1)

    print(f"Loading data from: {dataset_path}")
    with open(dataset_path, "rb") as f:
        data = pickle.load(f)

    train_ds = SpeechDataset(data["train"])
    test_ds = SpeechDataset(data["test"])

    train_loader = DataLoader(
        train_ds, batch_size=config['batch_size'], shuffle=True,
        num_workers=0, pin_memory=False, collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_ds, batch_size=config['batch_size'], shuffle=False,
        num_workers=0, pin_memory=False, collate_fn=collate_fn
    )

    print(f"Number of days: {len(data['train'])}\n")

    # Create model
    print("Creating model...")
    model = StreamingTransformerDecoder(
        neural_dim=config['n_input_features'],
        n_phonemes=config['n_classes'],
        d_model=config['d_model'],
        nhead=config['nhead'],
        num_layers=config['num_layers'],
        dropout=config['dropout'],
        stride_len=config['stride_len'],
        kernel_len=config['kernel_len'],
        gaussian_smooth_width=config['gaussian_smooth_width'],
        intermediate_layer=config['intermediate_layer'],
        day_count=len(data['train']),
        diphone_context=config['diphone_context'],
        time_mask_prob=config['transformer_time_mask_prob'],
        rel_pos_max_dist=config['rel_pos_max_dist'],
        ff_mult=config['ff_mult'],
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}\n")

    # Setup training
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['n_batch'])
    ctc_loss = torch.nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)
    augmenter = SequenceAugmenter(
        time_mask_ratio=config['time_mask_ratio'],
        min_time_span=config['min_time_mask'],
        channel_drop_prob=config['channel_drop_prob'],
        feature_mask_prob=config['feature_mask_prob'],
    )

    # Training loop
    print("=" * 70)
    print("STARTING TRAINING")
    print("=" * 70)

    train_iter = iter(train_loader)
    best_cer = None
    val_losses = []
    val_cers = []
    start_time = time.time()

    for step in range(config['n_batch']):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        X, y, X_len, y_len, day_idx = batch
        X, y, X_len, y_len, day_idx = (
            X.to(device), y.to(device), X_len.to(device),
            y_len.to(device), day_idx.to(device)
        )

        view_a = augmenter(X)
        view_b = augmenter(X)

        out_a = model(view_a, X_len, day_idx)
        out_b = model(view_b, X_len, day_idx)

        eff_len = out_a['eff_lengths']

        loss_a = ctc_loss(out_a['log_probs'].permute(1, 0, 2), y, eff_len, y_len)
        loss_b = ctc_loss(out_b['log_probs'].permute(1, 0, 2), y, eff_len, y_len)

        pad_mask = _length_mask(eff_len, out_a['log_probs'].shape[1])
        consistency = _symmetric_kl(out_a['log_probs'], out_b['log_probs'], pad_mask)

        loss = loss_a + loss_b + config['consistency_weight'] * consistency

        if out_a['intermediate_log_probs'] is not None:
            inter_loss = ctc_loss(
                out_a['intermediate_log_probs'].permute(1, 0, 2), y, eff_len, y_len
            )
            loss = loss + config['intermediate_loss_weight'] * inter_loss

        if torch.isnan(loss):
            print(f"\n⚠️ NaN loss at step {step}!")
            break

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        scheduler.step()

        if step % 100 == 0:
            val_loss, cer = evaluate(model, test_loader, ctc_loss, device)
            elapsed = (time.time() - start_time) / 100 if step > 0 else 0.0
            current_lr = optimizer.param_groups[0]['lr']

            print(f"step {step:5d}, val_loss: {val_loss:.4f}, CER: {cer:.4f}, "
                  f"lr: {current_lr:.6f}, time/100: {elapsed:.2f}s")
            start_time = time.time()

            val_losses.append(val_loss)
            val_cers.append(cer)

            if best_cer is None or cer < best_cer:
                best_cer = cer
                print(f"  → New best CER: {cer:.4f}")
                torch.save(model.state_dict(), 'advanced_model_best.pt')

            model.train()

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)
    print(f"Best CER: {best_cer:.4f} ({best_cer*100:.2f}%)")
    print("=" * 70)

    # Save final model
    torch.save(model.state_dict(), 'advanced_model_final.pt')
    print("\nModel saved to: advanced_model_final.pt")


if __name__ == "__main__":
    main()
