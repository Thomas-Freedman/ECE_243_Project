import math
from typing import Dict, Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from .augmentations import GaussianSmoothing


def _compute_strided_lengths(lengths: torch.Tensor, kernel: int, stride: int) -> torch.Tensor:
    """Replicates the baseline unfold-based receptive field logic."""
    eff = (lengths - kernel).clamp(min=0)
    return torch.div(eff, stride, rounding_mode="floor")


class StreamingTransformerBlock(nn.Module):
    """Causal encoder block used inside the streaming Transformer decoder."""

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(
        self,
        src: torch.Tensor,
        attn_mask: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        attn_out, _ = self.self_attn(
            src, src, src, attn_mask=attn_mask, key_padding_mask=padding_mask
        )
        src = src + self.dropout(attn_out)
        src = self.norm1(src)

        ff = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout(ff)
        src = self.norm2(src)
        return src


class StreamingTransformerDecoder(nn.Module):
    """
    Streaming Transformer with diphone and intermediate CTC heads.

    The model keeps the baseline preprocessing (day specific transforms,
    gaussian smoothing, unfold-based receptive fields) but swaps the GRU
    for a unidirectional Transformer stack. It exposes intermediate logits
    for auxiliary CTC supervision + feedback, and a diphone head whose
    marginals are fused with the phoneme predictions.
    """

    def __init__(
        self,
        neural_dim: int,
        n_phonemes: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dropout: float,
        stride_len: int,
        kernel_len: int,
        gaussian_smooth_width: float,
        intermediate_layer: int,
        day_count: int,
        diphone_context: Optional[int] = None,
    ):
        super().__init__()
        self.neural_dim = neural_dim
        self.n_phonemes = n_phonemes
        self.blank_count = n_phonemes + 1  # + blank
        self.kernel_len = kernel_len
        self.stride_len = stride_len
        self.gaussian = GaussianSmoothing(neural_dim, 20, gaussian_smooth_width, dim=1)

        self.day_weights = nn.Parameter(torch.randn(day_count, neural_dim, neural_dim))
        self.day_bias = nn.Parameter(torch.zeros(day_count, 1, neural_dim))
        for d in range(day_count):
            with torch.no_grad():
                self.day_weights[d] = torch.eye(neural_dim)

        self.input_nonlin = nn.Softsign()
        self.unfolder = nn.Unfold((kernel_len, 1), dilation=1, padding=0, stride=stride_len)
        self.input_proj = nn.Linear(neural_dim * kernel_len, d_model)
        self.pos_proj = nn.Linear(1, d_model)
        self.drop_in = nn.Dropout(dropout)

        self.layers = nn.ModuleList(
            [
                StreamingTransformerBlock(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=d_model * 4,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        if intermediate_layer >= num_layers:
            raise ValueError("intermediate_layer must be < num_layers")
        self.intermediate_layer = intermediate_layer

        self.intermediate_head = nn.Linear(d_model, self.blank_count)
        self.feedback_proj = nn.Linear(self.blank_count, d_model)
        self.final_head = nn.Linear(d_model, self.blank_count)

        # Diphone head only predicts contextualized phonemes (no blank)
        diphone_dim = diphone_context or n_phonemes
        self.diphone_dim = diphone_dim
        self.diphone_head = nn.Linear(d_model, diphone_dim * diphone_dim)

    def _apply_preproc(self, neural: torch.Tensor, day_idx: torch.Tensor) -> torch.Tensor:
        neural = torch.permute(neural, (0, 2, 1))
        neural = self.gaussian(neural)
        neural = torch.permute(neural, (0, 2, 1))

        weights = torch.index_select(self.day_weights, 0, day_idx)
        transformed = torch.einsum("btd,bdk->btk", neural, weights) + torch.index_select(
            self.day_bias, 0, day_idx
        )
        transformed = self.input_nonlin(transformed)

        # Unfold into overlapping windows, matching the GRU baseline
        unfolded = torch.permute(
            self.unfolder(torch.unsqueeze(torch.permute(transformed, (0, 2, 1)), 3)),
            (0, 2, 1),
        )
        return unfolded

    @staticmethod
    def _causal_mask(length: int, device: torch.device) -> torch.Tensor:
        mask = torch.full((length, length), float("-inf"), device=device)
        return torch.triu(mask, diagonal=1)

    @staticmethod
    def _padding_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
        range_row = torch.arange(max_len, device=lengths.device).unsqueeze(0)
        return range_row >= lengths.unsqueeze(1)

    def marginalize_diphone(self, diphone_logits: torch.Tensor) -> torch.Tensor:
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

    def forward(
        self,
        neural: torch.Tensor,
        lengths: torch.Tensor,
        day_idx: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        unfolded = self._apply_preproc(neural, day_idx)
        projected = self.input_proj(unfolded)

        # Add a simple positional encoding via learned linear ramp.
        seq_len = projected.shape[1]
        pos = torch.linspace(0, 1, seq_len, device=projected.device).view(1, seq_len, 1)
        projected = projected + self.pos_proj(pos)
        projected = self.drop_in(projected)

        eff_lengths = _compute_strided_lengths(lengths, self.kernel_len, self.stride_len)
        attn_mask = self._causal_mask(seq_len, projected.device)
        pad_mask = self._padding_mask(eff_lengths, seq_len)

        intermediate_logits = None
        intermediate_log_probs = None
        hidden = projected
        for idx, layer in enumerate(self.layers):
            hidden = layer(hidden, attn_mask=attn_mask, padding_mask=pad_mask)
            if idx == self.intermediate_layer:
                intermediate_logits = self.intermediate_head(hidden)
                intermediate_log_probs = F.log_softmax(intermediate_logits, dim=-1)
                feedback = self.feedback_proj(intermediate_logits)
                hidden = hidden + feedback

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
            diphone_slice = F.pad(diphone_slice, (0, pad_width), value=-math.inf)
            phone_slice = F.pad(phone_slice, (0, pad_width), value=-math.inf)

        fused_phone = torch.logsumexp(
            torch.stack(
                [
                    phone_slice,
                    diphone_slice,
                ],
                dim=0,
            ),
            dim=0,
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
