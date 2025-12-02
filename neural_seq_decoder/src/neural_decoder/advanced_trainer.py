import os
import pickle
import random
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from edit_distance import SequenceMatcher
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from .advanced_models import StreamingTransformerDecoder
from .dataset import SpeechDataset


def _seed_all(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _length_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
    rng = torch.arange(max_len, device=lengths.device).unsqueeze(0)
    return rng >= lengths.unsqueeze(1)


class SequenceAugmenter:
    """Applies heavy time + channel masking similar to Feghhi et al."""

    def __init__(
        self,
        time_mask_ratio: float,
        min_time_span: int,
        channel_drop_prob: float,
        feature_mask_prob: float,
    ):
        self.time_mask_ratio = time_mask_ratio
        self.min_time_span = max(1, min_time_span)
        self.channel_drop_prob = channel_drop_prob
        self.feature_mask_prob = feature_mask_prob

    def _apply_time_mask(self, seq: torch.Tensor) -> torch.Tensor:
        total_bins = seq.shape[0]
        target = int(total_bins * self.time_mask_ratio)
        masked = 0
        while masked < target and total_bins > 0:
            span = random.randint(self.min_time_span, max(self.min_time_span, total_bins))
            start = random.randint(0, max(0, total_bins - span))
            seq[start : start + span] = 0
            masked += span
        return seq

    def _apply_channel_mask(self, seq: torch.Tensor) -> torch.Tensor:
        if self.channel_drop_prob <= 0:
            return seq
        mask = torch.rand(seq.shape[-1], device=seq.device) < self.channel_drop_prob
        if mask.any():
            seq[:, mask] = 0
        return seq

    def _apply_feature_mask(self, seq: torch.Tensor) -> torch.Tensor:
        if self.feature_mask_prob <= 0:
            return seq
        feat_mask = torch.rand_like(seq) < self.feature_mask_prob
        seq = seq.masked_fill(feat_mask, 0)
        return seq

    def __call__(self, batch: torch.Tensor) -> torch.Tensor:
        augmented = batch.clone()
        for idx in range(augmented.shape[0]):
            augmented[idx] = self._apply_feature_mask(
                self._apply_channel_mask(self._apply_time_mask(augmented[idx]))
            )
        return augmented


def _collate(batch):
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


def _build_loaders(dataset_path: str, batch_size: int):
    with open(dataset_path, "rb") as handle:
        payload = pickle.load(handle)
    train_ds = SpeechDataset(payload["train"])
    val_ds = SpeechDataset(payload["test"])
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        collate_fn=_collate,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=_collate,
    )
    return train_loader, val_loader, payload


def _symmetric_kl(
    log_probs_p: torch.Tensor, log_probs_q: torch.Tensor, pad_mask: torch.Tensor
) -> torch.Tensor:
    probs_q = log_probs_q.exp()
    probs_p = log_probs_p.exp()
    kl_pq = F.kl_div(log_probs_p, probs_q, reduction="none").sum(-1)
    kl_qp = F.kl_div(log_probs_q, probs_p, reduction="none").sum(-1)
    keep = (~pad_mask).float()
    denom = keep.sum().clamp(min=1.0)
    return ((kl_pq + kl_qp) * keep).sum() / denom


class TestTimeAdaptor:
    """One-step adaptation on masked copies, following Feghhi et al."""

    def __init__(
        self,
        model: StreamingTransformerDecoder,
        lr: float,
        augmenter: SequenceAugmenter,
    ):
        self.model = model
        self.augmenter = augmenter
        self.lr = lr

    def step(self, inputs, lengths, day_idx):
        opt = torch.optim.SGD(self.model.parameters(), lr=self.lr)
        self.model.train()
        with torch.no_grad():
            teacher_out = self.model(inputs, lengths, day_idx)
        teacher = teacher_out["log_probs"].detach()
        eff_lengths = teacher_out["eff_lengths"]

        masked_inputs = self.augmenter(inputs)
        student_out = self.model(masked_inputs, lengths, day_idx)
        student = student_out["log_probs"]

        pad_mask = _length_mask(eff_lengths, student.shape[1])
        loss = _symmetric_kl(student, teacher, pad_mask)
        opt.zero_grad()
        loss.backward()
        opt.step()
        self.model.eval()


class OnlineAdaptor:
    """
    Online supervised adaptation following Card et al. 2024 and Willett et al. 2023.
    When a decoded sentence is confirmed correct, perform one gradient step to
    rapidly calibrate the model on recent correct predictions.
    """

    def __init__(
        self,
        model: StreamingTransformerDecoder,
        lr: float,
        ctc_loss,
    ):
        self.model = model
        self.lr = lr
        self.ctc_loss = ctc_loss
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        self.adaptation_count = 0

    def adapt_on_correct(self, inputs, targets, input_lengths, target_lengths, day_idx):
        """
        Perform one gradient step on a confirmed correct example.

        Args:
            inputs: Neural features [batch, time, channels]
            targets: Ground truth phoneme sequence [batch, seq_len]
            input_lengths: Actual lengths of inputs
            target_lengths: Actual lengths of targets
            day_idx: Day indices for day-specific transformation
        """
        self.model.train()

        # Forward pass
        out = self.model(inputs, input_lengths, day_idx)
        eff_len = out["eff_lengths"]

        # Compute CTC loss on confirmed correct example
        loss = self.ctc_loss(
            out["log_probs"].permute(1, 0, 2),  # [T, B, C]
            targets,
            eff_len,
            target_lengths
        )

        # Add intermediate loss if available
        if out["intermediate_log_probs"] is not None:
            inter_loss = self.ctc_loss(
                out["intermediate_log_probs"].permute(1, 0, 2),
                targets,
                eff_len,
                target_lengths
            )
            loss = loss + 0.3 * inter_loss  # Same weight as training

        # Single gradient step
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
        self.optimizer.step()

        self.model.eval()
        self.adaptation_count += 1

        return loss.item()

    def get_adaptation_count(self):
        return self.adaptation_count


@dataclass
class AdvancedArgs:
    variant: str
    outputDir: str
    datasetPath: str
    batchSize: int
    nBatch: int
    seed: int
    lr: float
    weightDecay: float
    nInputFeatures: int
    nClasses: int
    strideLen: int
    kernelLen: int
    gaussianSmoothWidth: float
    modelDim: int
    modelLayers: int
    modelHeads: int
    dropout: float
    intermediateLayer: int
    timeMaskRatio: float
    channelDropProb: float
    featureMaskProb: float
    minTimeMask: int
    consistencyWeight: float
    intermediateLossWeight: float
    testTimeLR: float
    enableTestTimeAdaptation: bool
    enableOnlineAdaptation: bool = False
    onlineAdaptationLR: float = 0.00001
    diphoneContext: Optional[int] = None
    transformerTimeMaskProb: float = 0.0
    relPosMaxDist: Optional[int] = None
    relBiasByHead: bool = True
    ffMult: int = 4


def train_advanced_model(args: Dict):
    parsed = AdvancedArgs(**args)
    os.makedirs(parsed.outputDir, exist_ok=True)
    _seed_all(parsed.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    with open(os.path.join(parsed.outputDir, "advanced_args.pkl"), "wb") as f:
        pickle.dump(args, f)

    train_loader, val_loader, payload = _build_loaders(
        parsed.datasetPath, parsed.batchSize
    )

    model = StreamingTransformerDecoder(
        neural_dim=parsed.nInputFeatures,
        n_phonemes=parsed.nClasses,
        d_model=parsed.modelDim,
        nhead=parsed.modelHeads,
        num_layers=parsed.modelLayers,
        dropout=parsed.dropout,
        stride_len=parsed.strideLen,
        kernel_len=parsed.kernelLen,
        gaussian_smooth_width=parsed.gaussianSmoothWidth,
        intermediate_layer=parsed.intermediateLayer,
        day_count=len(payload["train"]),
        diphone_context=parsed.diphoneContext or parsed.nClasses,
        time_mask_prob=parsed.transformerTimeMaskProb,
        rel_pos_max_dist=parsed.relPosMaxDist,
        rel_bias_by_head=parsed.relBiasByHead,
        ff_mult=parsed.ffMult,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=parsed.lr, weight_decay=parsed.weightDecay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=parsed.nBatch
    )
    ctc_loss = torch.nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)

    augmenter = SequenceAugmenter(
        time_mask_ratio=parsed.timeMaskRatio,
        min_time_span=parsed.minTimeMask,
        channel_drop_prob=parsed.channelDropProb,
        feature_mask_prob=parsed.featureMaskProb,
    )
    adaptor = (
        TestTimeAdaptor(model, parsed.testTimeLR, augmenter)
        if parsed.enableTestTimeAdaptation
        else None
    )

    online_adaptor = (
        OnlineAdaptor(model, parsed.onlineAdaptationLR, ctc_loss)
        if parsed.enableOnlineAdaptation
        else None
    )

    train_iter = iter(train_loader)
    best_cer = None
    metrics = {"valLoss": [], "valCER": []}
    start = time.time()
    for step in range(parsed.nBatch):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        X, y, X_len, y_len, day_idx = batch
        X, y, X_len, y_len, day_idx = (
            X.to(device),
            y.to(device),
            X_len.to(device),
            y_len.to(device),
            day_idx.to(device),
        )

        view_a = augmenter(X)
        view_b = augmenter(X)

        out_a = model(view_a, X_len, day_idx)
        out_b = model(view_b, X_len, day_idx)

        eff_len = out_a["eff_lengths"]
        loss_a = ctc_loss(
            out_a["log_probs"].permute(1, 0, 2), y, eff_len, y_len
        )
        loss_b = ctc_loss(
            out_b["log_probs"].permute(1, 0, 2), y, eff_len, y_len
        )

        pad_mask = _length_mask(eff_len, out_a["log_probs"].shape[1])
        consistency = _symmetric_kl(out_a["log_probs"], out_b["log_probs"], pad_mask)

        loss = loss_a + loss_b + parsed.consistencyWeight * consistency

        if out_a["intermediate_log_probs"] is not None:
            inter_loss = ctc_loss(
                out_a["intermediate_log_probs"].permute(1, 0, 2), y, eff_len, y_len
            )
            loss = loss + parsed.intermediateLossWeight * inter_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        scheduler.step()

        if step % 100 == 0:
            val_loss, cer = _evaluate(
                model, val_loader, ctc_loss, parsed, device, adaptor, online_adaptor
            )
            elapsed = (time.time() - start) / 100 if step > 0 else 0.0
            adapt_msg = (
                f" online_adapt={online_adaptor.get_adaptation_count()}"
                if online_adaptor is not None
                else ""
            )
            print(
                f"[advanced] step={step} val_ctc={val_loss:.4f} cer={cer:.4f}{adapt_msg} time/100={elapsed:.2f}s"
            )
            start = time.time()
            metrics["valLoss"].append(val_loss)
            metrics["valCER"].append(cer)
            with open(os.path.join(parsed.outputDir, "advanced_stats.pkl"), "wb") as f:
                pickle.dump(metrics, f)

            if best_cer is None or cer < best_cer:
                best_cer = cer
                torch.save(
                    model.state_dict(), os.path.join(parsed.outputDir, "advanced_weights.pt")
                )


def _evaluate(
    model: StreamingTransformerDecoder,
    loader: DataLoader,
    ctc_loss,
    args: AdvancedArgs,
    device: str,
    adaptor: Optional[TestTimeAdaptor],
    online_adaptor: Optional[OnlineAdaptor] = None,
) -> Tuple[float, float]:
    model.eval()
    losses = []
    total_edit = 0
    total_len = 0

    for X, y, X_len, y_len, day_idx in loader:
        X, y, X_len, y_len, day_idx = (
            X.to(device),
            y.to(device),
            X_len.to(device),
            y_len.to(device),
            day_idx.to(device),
        )

        if adaptor is not None:
            adaptor.step(X, X_len, day_idx)

        with torch.no_grad():
            out = model(X, X_len, day_idx)
            eff_len = out["eff_lengths"]
            loss = ctc_loss(out["log_probs"].permute(1, 0, 2), y, eff_len, y_len)
            losses.append(loss.item())

            adjusted = eff_len
            # Track which examples in batch are correct for online adaptation
            correct_mask = []
            for idx in range(out["log_probs"].shape[0]):
                logits = out["log_probs"][idx, : adjusted[idx], :]
                decoded = torch.argmax(logits, dim=-1)
                decoded = torch.unique_consecutive(decoded, dim=-1)
                decoded = decoded.cpu().detach().numpy()
                decoded = decoded[decoded != 0]
                target = y[idx][: y_len[idx]].cpu().detach().numpy()
                matcher = SequenceMatcher(a=target.tolist(), b=decoded.tolist())
                edit_dist = matcher.distance()
                total_edit += edit_dist
                total_len += len(target)

                # Mark as correct if edit distance is 0 (perfect match)
                correct_mask.append(edit_dist == 0)

        # Online supervised adaptation: fine-tune on correctly decoded examples
        if online_adaptor is not None and any(correct_mask):
            # Filter to only correct examples
            correct_indices = [i for i, c in enumerate(correct_mask) if c]
            if len(correct_indices) > 0:
                X_correct = X[correct_indices]
                y_correct = y[correct_indices]
                X_len_correct = X_len[correct_indices]
                y_len_correct = y_len[correct_indices]
                day_idx_correct = day_idx[correct_indices]

                online_adaptor.adapt_on_correct(
                    X_correct, y_correct, X_len_correct, y_len_correct, day_idx_correct
                )

    avg_loss = sum(losses) / max(len(losses), 1)
    cer = total_edit / max(total_len, 1)
    return avg_loss, cer


