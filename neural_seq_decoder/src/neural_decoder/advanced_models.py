import math
from typing import Dict, Optional

import torch
from torch import nn
from torch.nn import functional as F

from .augmentations import GaussianSmoothing


class PatchEmbedding(nn.Module):
    """
    Segments neural sequences into non-overlapping patches of length Tin and
    projects each patch into the model dimension.
    """

    def __init__(self, Tin: int, feat_dim: int, hdim: int, eps: float = 1e-6):
        super().__init__()
        self.Tin = Tin
        self.feat_dim = feat_dim
        self.in_dim = Tin * feat_dim
        self.ln1 = nn.LayerNorm(self.in_dim, eps=eps)
        self.proj = nn.Linear(self.in_dim, hdim)
        self.ln2 = nn.LayerNorm(hdim, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F)
        bsz, T, F = x.shape
        if F != self.feat_dim:
            raise ValueError(f"expected feature dim {self.feat_dim}, got {F}")

        n_patch = T // self.Tin
        if n_patch == 0:
            raise ValueError("input sequence shorter than Tin")

        x = x[:, : n_patch * self.Tin, :]
        x = x.view(bsz, n_patch, self.Tin * F)
        x = self.ln1(x)
        x = self.proj(x)
        x = self.ln2(x)
        return x


class RelativePositionBias(nn.Module):
    """Relative attention bias shared with the teammate's implementation."""

    def __init__(self, n_heads: int, max_dist: int, by_head: bool = True):
        super().__init__()
        self.n_heads = n_heads
        self.max_dist = max_dist
        self.L = 2 * max_dist + 1
        self.by_head = by_head
        if by_head:
            self.relative_bias = nn.Parameter(torch.zeros(n_heads, self.L))
        else:
            self.relative_bias = nn.Parameter(torch.zeros(1, self.L))

    def forward(self, seq_len: int, device=None) -> torch.Tensor:
        positions = torch.arange(seq_len, device=device)
        rel = positions[None, :] - positions[:, None]
        clipped = rel.clamp(-self.max_dist, self.max_dist) + self.max_dist
        table = self.relative_bias
        bias = table[..., clipped]
        if not self.by_head:
            bias = bias.expand(self.n_heads, -1, -1)
        return bias


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
            elif causal_mask.dim() == 4:
                scores = scores + causal_mask
            else:
                raise ValueError("Unsupported causal_mask rank")

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


class MambaBlock(nn.Module):
    """
    Simplified Mamba-style Selective State Space Model (SSM) block.
    Provides efficient O(L) sequence modeling as an alternative to O(L^2) attention.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand_factor: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = d_model * expand_factor

        # Input projection
        self.in_proj = nn.Linear(d_model, self.d_inner * 2)

        # Depthwise convolution for local context
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner,
        )

        # SSM parameters (input-dependent)
        self.x_proj = nn.Linear(self.d_inner, d_state * 2)  # B and C
        self.dt_proj = nn.Linear(self.d_inner, self.d_inner)

        # State space parameters (learned)
        self.A_log = nn.Parameter(torch.randn(self.d_inner, d_state))
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        """
        x: (B, L, D) where B=batch, L=length, D=d_model
        """
        residual = x
        x = self.norm(x)

        B, L, D = x.shape

        # Input projection: split into x and z for gating
        xz = self.in_proj(x)  # (B, L, 2*d_inner)
        x, z = xz.chunk(2, dim=-1)  # each (B, L, d_inner)

        # Causal convolution
        x = x.transpose(1, 2)  # (B, d_inner, L)
        x = self.conv1d(x)[:, :, :L]  # Truncate to original length for causality
        x = x.transpose(1, 2)  # (B, L, d_inner)
        x = F.silu(x)

        # SSM computation
        # Compute input-dependent B, C, and delta
        x_db = self.x_proj(x)  # (B, L, 2*d_state)
        B_ssm, C_ssm = x_db.chunk(2, dim=-1)  # each (B, L, d_state)

        delta = F.softplus(self.dt_proj(x))  # (B, L, d_inner)

        # Discretize A (continuous to discrete)
        A = -torch.exp(self.A_log)  # (d_inner, d_state)

        # Selective scan (simplified version)
        y = self._selective_scan(x, delta, A, B_ssm, C_ssm)

        # Skip connection
        y = y + x * self.D.view(1, 1, -1)

        # Gating with z
        y = y * F.silu(z)

        # Output projection
        output = self.out_proj(y)
        output = self.dropout(output)

        return residual + output

    def _selective_scan(self, u, delta, A, B, C):
        """
        Simplified selective scan using parallel scan.
        u: (B, L, d_inner)
        delta: (B, L, d_inner)
        A: (d_inner, d_state)
        B: (B, L, d_state)
        C: (B, L, d_state)
        """
        B_batch, L, d_inner = u.shape
        d_state = A.shape[1]

        # Discretize: A_bar = exp(delta * A), B_bar = delta * B
        deltaA = torch.exp(delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))  # (B, L, d_inner, d_state)
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(2)  # (B, L, d_inner, d_state)

        # Initialize state
        h = torch.zeros(B_batch, d_inner, d_state, device=u.device, dtype=u.dtype)

        # Scan through sequence (can be parallelized with associative scan)
        ys = []
        for t in range(L):
            h = deltaA[:, t] * h + deltaB[:, t] * u[:, t:t+1].transpose(1, 2)  # (B, d_inner, d_state)
            y_t = torch.einsum('bds,bs->bd', h, C[:, t])  # (B, d_inner)
            ys.append(y_t)

        y = torch.stack(ys, dim=1)  # (B, L, d_inner)
        return y


class TransformerBlock(nn.Module):
    def __init__(
        self,
        hdim: int,
        n_heads: int,
        ff_dim: int,
        attn_dropout: float = 0.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.ln_attn = nn.LayerNorm(hdim)
        self.attn = Attention(hdim, n_heads, attn_dropout)
        self.ff = FeedForward(hdim, ff_dim, dropout)

    def forward(self, x, causal_mask=None, rel_bias=None, key_padding_mask=None):
        attn_in = self.ln_attn(x)
        attn_out = self.attn(
            attn_in, causal_mask=causal_mask, rel_bias=rel_bias, key_padding_mask=key_padding_mask
        )
        x = x + attn_out
        ff_out = self.ff(x)
        x = x + ff_out
        return x


class StreamingTransformerDecoder(nn.Module):
    """
    Advanced decoder that applies day-specific preprocessing followed by the
    teammate's causal Transformer architecture, while retaining intermediate
    CTC feedback and diphone marginalization.
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
        time_mask_prob: float = 0.0,
        rel_pos_max_dist: Optional[int] = None,
        rel_bias_by_head: bool = True,
        ff_mult: int = 4,
    ):
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
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    hdim=d_model,
                    n_heads=nhead,
                    ff_dim=ff_dim,
                    attn_dropout=dropout,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

        self.rel_pos_bias = (
            RelativePositionBias(nhead, rel_pos_max_dist, rel_bias_by_head)
            if rel_pos_max_dist is not None
            else None
        )
        self.final_ln = nn.LayerNorm(d_model)

        self.intermediate_head = nn.Linear(d_model, self.blank_count)
        self.feedback_proj = nn.Linear(self.blank_count, d_model)
        self.final_head = nn.Linear(d_model, self.blank_count)

        diphone_dim = diphone_context or n_phonemes
        self.diphone_dim = diphone_dim
        self.diphone_head = nn.Linear(d_model, diphone_dim * diphone_dim)

    def _apply_preproc(
        self, neural: torch.Tensor, lengths: torch.Tensor, day_idx: torch.Tensor
    ) -> torch.Tensor:
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
    def _causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
        mask = torch.zeros((seq_len, seq_len), device=device, dtype=torch.float32)
        mask = torch.triu(mask, diagonal=1)
        mask = mask * float("-inf")
        return mask

    @staticmethod
    def _padding_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
        positions = torch.arange(max_len, device=lengths.device).unsqueeze(0)
        return positions >= lengths.unsqueeze(1)

    def _apply_time_mask(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.time_mask_prob <= 0.0:
            return x
        mask = torch.rand(x.shape[0], x.shape[1], device=x.device) < self.time_mask_prob
        return x.masked_fill(mask.unsqueeze(-1), 0.0)

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
            hidden = layer(
                hidden, causal_mask=causal_mask, rel_bias=rel_bias, key_padding_mask=pad_mask
            )
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
            diphone_slice = F.pad(diphone_slice, (0, pad_width), value=-math.inf)
            phone_slice = F.pad(phone_slice, (0, pad_width), value=-math.inf)

        fused_phone = torch.logsumexp(
            torch.stack([phone_slice, diphone_slice], dim=0),
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


class HybridPOSSMDecoder(nn.Module):
    """
    POSSM-style hybrid architecture combining Transformer tokenization with SSM sequence modeling.

    Architecture:
    - Initial Transformer layers for learning rich token representations
    - Middle SSM (Mamba) layers for efficient O(L) sequence modeling
    - Final layers can be Transformer or SSM
    - Maintains intermediate CTC and diphone heads like StreamingTransformerDecoder
    """

    def __init__(
        self,
        neural_dim: int,
        n_phonemes: int,
        d_model: int,
        nhead: int,
        num_transformer_layers: int = 2,
        num_ssm_layers: int = 4,
        dropout: float = 0.2,
        stride_len: int = 4,
        kernel_len: int = 32,
        gaussian_smooth_width: float = 2.0,
        intermediate_layer: int = 3,
        day_count: int = 1,
        diphone_context: Optional[int] = None,
        time_mask_prob: float = 0.0,
        rel_pos_max_dist: Optional[int] = None,
        rel_bias_by_head: bool = True,
        ff_mult: int = 4,
        ssm_d_state: int = 16,
        ssm_d_conv: int = 4,
    ):
        super().__init__()
        self.neural_dim = neural_dim
        self.n_phonemes = n_phonemes
        self.d_model = d_model
        self.kernel_len = kernel_len
        self.stride_len = stride_len
        self.intermediate_layer = intermediate_layer
        self.diphone_dim = diphone_context or n_phonemes
        self.time_mask_prob = time_mask_prob

        # Day-specific preprocessing (same as StreamingTransformerDecoder)
        self.day_count = day_count
        self.day_weights = nn.Parameter(torch.ones(day_count, neural_dim))
        self.day_biases = nn.Parameter(torch.zeros(day_count, neural_dim))
        self.gaussian_smooth = GaussianSmoothing(
            neural_dim, kernel_size=20, sigma=gaussian_smooth_width
        )

        # Patch embedding
        self.patch_embed = PatchEmbedding(kernel_len, neural_dim, d_model)
        self.input_dropout = nn.Dropout(dropout)

        # Relative position bias (only for Transformer layers)
        self.rel_pos_bias = (
            RelativePositionBias(nhead, rel_pos_max_dist, rel_bias_by_head)
            if rel_pos_max_dist
            else None
        )

        # Hybrid layer stack: Transformer -> SSM -> Optional final layers
        self.layers = nn.ModuleList()
        ff_dim = d_model * ff_mult

        # Initial Transformer layers for tokenization
        for _ in range(num_transformer_layers):
            self.layers.append(
                TransformerBlock(d_model, nhead, ff_dim, attn_dropout=dropout, dropout=dropout)
            )

        # SSM layers for efficient sequence modeling
        for _ in range(num_ssm_layers):
            self.layers.append(
                MambaBlock(d_model, d_state=ssm_d_state, d_conv=ssm_d_conv, dropout=dropout)
            )

        self.num_layers = len(self.layers)

        # Output heads
        self.final_head = nn.Linear(d_model, n_phonemes)
        self.diphone_head = nn.Linear(d_model, diphone_context * diphone_context)

        # Intermediate CTC head
        if 0 <= intermediate_layer < self.num_layers:
            self.intermediate_head = nn.Linear(d_model, n_phonemes)
            self.intermediate_feedback = nn.Linear(n_phonemes, d_model)
        else:
            self.intermediate_head = None
            self.intermediate_feedback = None

    def _apply_preproc(
        self, neural: torch.Tensor, lengths: torch.Tensor, day_idx: torch.Tensor
    ) -> torch.Tensor:
        """Apply day-specific transformation and Gaussian smoothing"""
        bsz, T, F = neural.shape
        day_w = self.day_weights[day_idx]
        day_b = self.day_biases[day_idx]
        x = neural * day_w.unsqueeze(1) + day_b.unsqueeze(1)
        x = torch.nn.functional.softsign(x)
        x = self.gaussian_smooth(x, lengths)
        return x

    def _apply_time_mask(self, x: torch.Tensor) -> torch.Tensor:
        """Apply element-wise time masking during training"""
        if self.training and self.time_mask_prob > 0.0:
            mask = torch.bernoulli(
                torch.full(x.shape, 1.0 - self.time_mask_prob, device=x.device)
            )
            x = x * mask
        return x

    def _causal_mask(self, seq_len: int, device) -> torch.Tensor:
        """Create causal attention mask"""
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float("-inf"))
        return mask

    def _padding_mask(self, eff_lengths: torch.Tensor, seq_len: int) -> torch.Tensor:
        """Create padding mask from effective lengths"""
        B = eff_lengths.shape[0]
        arange = torch.arange(seq_len, device=eff_lengths.device)[None, :]
        mask = arange >= eff_lengths[:, None]
        return mask

    def _marginalize_diphone(self, logits: torch.Tensor) -> torch.Tensor:
        """Marginalize diphone predictions to phoneme predictions"""
        bsz, steps = logits.shape[:2]
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
        # Preprocessing and embedding
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

        # Forward through hybrid layers
        for i, layer in enumerate(self.layers):
            if isinstance(layer, TransformerBlock):
                # Transformer layer: use causal mask and relative position bias
                hidden = layer(hidden, causal_mask=causal_mask, rel_bias=rel_bias, key_padding_mask=pad_mask)
            else:
                # SSM layer: just forward (already causal)
                hidden = layer(hidden)

            # Intermediate CTC supervision
            if i == self.intermediate_layer and self.intermediate_head is not None:
                intermediate_logits = self.intermediate_head(hidden)
                feedback = self.intermediate_feedback(intermediate_logits)
                hidden = hidden + feedback

        # Final predictions
        final_logits = self.final_head(hidden)
        diphone_logits = self.diphone_head(hidden)

        # Marginalize diphone to phoneme
        diphone_log_probs = self._marginalize_diphone(diphone_logits)

        # Prepare final output with CTC blank
        phone_log_probs = F.log_softmax(final_logits, dim=-1)
        blank = torch.full(
            (phone_log_probs.shape[0], phone_log_probs.shape[1], 1),
            -math.inf,
            device=phone_log_probs.device,
            dtype=phone_log_probs.dtype,
        )

        # Fuse phoneme and diphone predictions
        valid_dim = min(self.diphone_dim, self.n_phonemes)
        diphone_slice = diphone_log_probs[..., :valid_dim]
        phone_slice = phone_log_probs[..., :valid_dim]
        if valid_dim < self.n_phonemes:
            pad_width = self.n_phonemes - valid_dim
            diphone_slice = F.pad(diphone_slice, (0, pad_width), value=-math.inf)
            phone_slice = F.pad(phone_slice, (0, pad_width), value=-math.inf)

        fused_phone = torch.logsumexp(
            torch.stack([phone_slice, diphone_slice], dim=0),
            dim=0,
        ) - math.log(2.0)
        fused = torch.cat([blank, fused_phone], dim=-1)
        fused = fused - torch.logsumexp(fused, dim=-1, keepdim=True)

        # Intermediate log probs (if available)
        intermediate_log_probs = None
        if intermediate_logits is not None:
            inter_phone = F.log_softmax(intermediate_logits, dim=-1)
            inter_blank = torch.full(
                (inter_phone.shape[0], inter_phone.shape[1], 1),
                -math.inf,
                device=inter_phone.device,
                dtype=inter_phone.dtype,
            )
            intermediate_log_probs = torch.cat([inter_blank, inter_phone], dim=-1)

        return {
            "log_probs": fused,
            "intermediate_log_probs": intermediate_log_probs,
            "intermediate_logits": intermediate_logits,
            "raw_logits": final_logits,
            "diphone_logits": diphone_logits,
            "diphone_log_probs": diphone_log_probs,
            "eff_lengths": eff_lengths.to(torch.int32),
        }


class SpikeTokenizer(nn.Module):
    """
    POYO-style spike tokenizer that converts continuous neural activity into discrete tokens.

    This module:
    1. Bins neural activity into temporal windows
    2. Projects binned activity through a learned codebook
    3. Outputs discrete token indices (soft or hard quantization)
    """

    def __init__(
        self,
        neural_dim: int,
        n_tokens: int = 1024,
        d_model: int = 256,
        bin_size: int = 4,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.neural_dim = neural_dim
        self.n_tokens = n_tokens
        self.d_model = d_model
        self.bin_size = bin_size
        self.temperature = temperature

        # Input projection
        self.input_proj = nn.Linear(neural_dim * bin_size, d_model)

        # Codebook for tokenization
        self.codebook = nn.Parameter(torch.randn(n_tokens, d_model) / math.sqrt(d_model))

        # Output projection (from token to embedding)
        self.output_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: Neural activity tensor [B, T, neural_dim]

        Returns:
            Dictionary with:
            - token_embeddings: Soft token embeddings [B, T', d_model]
            - token_indices: Hard token indices [B, T']
            - token_probs: Token probabilities [B, T', n_tokens]
        """
        B, T, D = x.shape

        # Bin the neural activity
        n_bins = T // self.bin_size
        x = x[:, :n_bins * self.bin_size, :]
        x = x.view(B, n_bins, self.bin_size * D)

        # Project to embedding space
        x = self.input_proj(x)  # [B, n_bins, d_model]
        x = F.normalize(x, dim=-1)

        # Compute similarity to codebook
        codebook_norm = F.normalize(self.codebook, dim=-1)
        logits = torch.einsum('btd,nd->btn', x, codebook_norm) / self.temperature  # [B, T', n_tokens]

        # Soft tokenization (differentiable)
        token_probs = F.softmax(logits, dim=-1)
        token_embeddings = torch.einsum('btn,nd->btd', token_probs, self.codebook)
        token_embeddings = self.output_proj(token_embeddings)

        # Hard tokenization (for analysis)
        token_indices = torch.argmax(logits, dim=-1)

        return {
            "token_embeddings": token_embeddings,
            "token_indices": token_indices,
            "token_probs": token_probs,
        }


class SessionEmbedding(nn.Module):
    """
    Session-specific embeddings for multi-session data.

    Following POYO, each session has learned embeddings that capture
    session-specific characteristics (electrode drift, behavioral differences).
    """

    def __init__(
        self,
        n_sessions: int,
        d_model: int,
        n_channels: int,
    ):
        super().__init__()
        self.n_sessions = n_sessions
        self.d_model = d_model
        self.n_channels = n_channels

        # Session-level embedding
        self.session_embed = nn.Embedding(n_sessions, d_model)

        # Channel-level embedding per session (models electrode drift)
        self.channel_embed = nn.Parameter(torch.zeros(n_sessions, n_channels, d_model))

        # Initialization
        nn.init.normal_(self.session_embed.weight, std=0.02)
        nn.init.normal_(self.channel_embed, std=0.02)

    def forward(self, session_idx: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            session_idx: Session indices [B]

        Returns:
            Dictionary with:
            - session_embed: Session embedding [B, d_model]
            - channel_embed: Channel-specific embedding [B, n_channels, d_model]
        """
        session_embed = self.session_embed(session_idx)
        channel_embed = self.channel_embed[session_idx]

        return {
            "session_embed": session_embed,
            "channel_embed": channel_embed,
        }


class POYODecoder(nn.Module):
    """
    POYO-style decoder with spike tokenization and session embeddings.

    Architecture:
    1. Spike tokenization to convert raw activity to tokens
    2. Session embeddings to handle multi-session data
    3. Transformer encoder for sequence modeling
    4. CTC output head for phoneme prediction
    """

    def __init__(
        self,
        neural_dim: int,
        n_phonemes: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dropout: float,
        n_sessions: int,
        n_tokens: int = 1024,
        bin_size: int = 4,
        stride_len: int = 4,
        kernel_len: int = 32,
        gaussian_smooth_width: float = 2.0,
        intermediate_layer: int = 3,
        diphone_context: Optional[int] = None,
        ff_mult: int = 4,
    ):
        super().__init__()
        self.neural_dim = neural_dim
        self.n_phonemes = n_phonemes
        self.d_model = d_model
        self.kernel_len = kernel_len
        self.intermediate_layer = intermediate_layer
        self.diphone_dim = diphone_context or n_phonemes

        # Spike tokenizer
        self.spike_tokenizer = SpikeTokenizer(
            neural_dim=neural_dim,
            n_tokens=n_tokens,
            d_model=d_model,
            bin_size=bin_size,
        )

        # Session embedding
        self.session_embed = SessionEmbedding(
            n_sessions=n_sessions,
            d_model=d_model,
            n_channels=neural_dim,
        )

        # Gaussian smoothing
        self.gaussian_smooth = GaussianSmoothing(
            neural_dim, kernel_size=20, sigma=gaussian_smooth_width
        )

        # Positional encoding
        self.pos_embed = nn.Parameter(torch.zeros(1, 1000, d_model))
        nn.init.normal_(self.pos_embed, std=0.02)

        # Input dropout
        self.input_dropout = nn.Dropout(dropout)

        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, nhead, d_model * ff_mult, attn_dropout=dropout, dropout=dropout)
            for _ in range(num_layers)
        ])

        self.num_layers = num_layers

        # Output heads
        self.final_head = nn.Linear(d_model, n_phonemes)
        self.diphone_head = nn.Linear(d_model, self.diphone_dim * self.diphone_dim)

        # Intermediate CTC head
        if 0 <= intermediate_layer < num_layers:
            self.intermediate_head = nn.Linear(d_model, n_phonemes)
            self.intermediate_feedback = nn.Linear(n_phonemes, d_model)
        else:
            self.intermediate_head = None
            self.intermediate_feedback = None

    def _causal_mask(self, seq_len: int, device) -> torch.Tensor:
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float("-inf"))
        return mask

    def _padding_mask(self, lengths: torch.Tensor, max_len: int) -> torch.Tensor:
        arange = torch.arange(max_len, device=lengths.device)[None, :]
        return arange >= lengths[:, None]

    def _marginalize_diphone(self, logits: torch.Tensor) -> torch.Tensor:
        bsz, steps = logits.shape[:2]
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
        session_idx: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            neural: Raw neural activity [B, T, neural_dim]
            lengths: Sequence lengths [B]
            session_idx: Session indices [B]
        """
        B, T, D = neural.shape
        device = neural.device

        # Apply Gaussian smoothing
        neural = self.gaussian_smooth(neural, lengths)

        # Get session embeddings
        session_data = self.session_embed(session_idx)
        session_embed = session_data["session_embed"]  # [B, d_model]
        channel_embed = session_data["channel_embed"]  # [B, n_channels, d_model]

        # Apply channel-specific transformation
        channel_weight = channel_embed.mean(dim=-1)  # [B, D]
        neural = neural * (1 + channel_weight.unsqueeze(1))  # [B, T, D]

        # Tokenize
        token_data = self.spike_tokenizer(neural)
        hidden = token_data["token_embeddings"]  # [B, T', d_model]

        # Add session embedding and positional encoding
        seq_len = hidden.shape[1]
        hidden = hidden + session_embed.unsqueeze(1)
        hidden = hidden + self.pos_embed[:, :seq_len, :]

        hidden = self.input_dropout(hidden)

        # Calculate effective lengths
        bin_size = self.spike_tokenizer.bin_size
        eff_lengths = torch.div(lengths, bin_size, rounding_mode="floor").clamp(min=1)

        causal_mask = self._causal_mask(seq_len, device)
        pad_mask = self._padding_mask(eff_lengths, seq_len)

        intermediate_logits = None

        # Forward through Transformer layers
        for i, layer in enumerate(self.layers):
            hidden = layer(hidden, causal_mask=causal_mask, key_padding_mask=pad_mask)

            if i == self.intermediate_layer and self.intermediate_head is not None:
                intermediate_logits = self.intermediate_head(hidden)
                feedback = self.intermediate_feedback(intermediate_logits)
                hidden = hidden + feedback

        # Final predictions
        final_logits = self.final_head(hidden)
        diphone_logits = self.diphone_head(hidden)

        # Marginalize diphone
        diphone_log_probs = self._marginalize_diphone(diphone_logits)

        # Output with CTC blank
        phone_log_probs = F.log_softmax(final_logits, dim=-1)
        blank = torch.full(
            (phone_log_probs.shape[0], phone_log_probs.shape[1], 1),
            -math.inf,
            device=phone_log_probs.device,
            dtype=phone_log_probs.dtype,
        )

        # Fuse predictions
        valid_dim = min(self.diphone_dim, self.n_phonemes)
        diphone_slice = diphone_log_probs[..., :valid_dim]
        phone_slice = phone_log_probs[..., :valid_dim]
        if valid_dim < self.n_phonemes:
            pad_width = self.n_phonemes - valid_dim
            diphone_slice = F.pad(diphone_slice, (0, pad_width), value=-math.inf)
            phone_slice = F.pad(phone_slice, (0, pad_width), value=-math.inf)

        fused_phone = torch.logsumexp(
            torch.stack([phone_slice, diphone_slice], dim=0), dim=0
        ) - math.log(2.0)
        fused = torch.cat([blank, fused_phone], dim=-1)
        fused = fused - torch.logsumexp(fused, dim=-1, keepdim=True)

        # Intermediate log probs
        intermediate_log_probs = None
        if intermediate_logits is not None:
            inter_phone = F.log_softmax(intermediate_logits, dim=-1)
            inter_blank = torch.full(
                (inter_phone.shape[0], inter_phone.shape[1], 1),
                -math.inf, device=inter_phone.device, dtype=inter_phone.dtype
            )
            intermediate_log_probs = torch.cat([inter_blank, inter_phone], dim=-1)

        return {
            "log_probs": fused,
            "intermediate_log_probs": intermediate_log_probs,
            "intermediate_logits": intermediate_logits,
            "raw_logits": final_logits,
            "diphone_logits": diphone_logits,
            "diphone_log_probs": diphone_log_probs,
            "eff_lengths": eff_lengths.to(torch.int32),
            "token_indices": token_data["token_indices"],
            "token_probs": token_data["token_probs"],
        }
