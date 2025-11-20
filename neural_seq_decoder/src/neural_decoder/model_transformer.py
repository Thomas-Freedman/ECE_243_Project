import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class PatchEmbedding(nn.Module):
    """
    In: Neural data x (B, T, F)
    Out: Segmented patches (B, n_patch, hdim)
    """
    def __init__(self, Tin, feat_dim, hdim, eps=1e-6):
        super().__init__()
        self.Tin = Tin
        self.feat_dim = feat_dim
        self.in_dim = Tin * feat_dim
        self.ln1 = nn.LayerNorm(self.in_dim, eps=eps)
        self.proj = nn.Linear(self.in_dim, hdim)
        self.ln2 = nn.LayerNorm(hdim, eps=eps)

    def forward(self, x: torch.Tensor):
        # x: (B, T, F)
        B, T, F = x.shape
        assert F == self.feat_dim, f"expected feature dim {self.feat_dim}, got {F}" # Remove

        n_patch = T // self.Tin
        if n_patch == 0:
            raise ValueError("Input T smaller than Tin") # Remove
        
        x = x[:, :n_patch * self.Tin, :] 
        x = x.view(B, n_patch, self.Tin * F) 
        x = self.ln1(x)
        x = self.proj(x)
        x = self.ln2(x)
        return x


class RelativePositionBias(nn.Module):
    """
    Incorporation of relative query-key position for attention using additive bias.
    """
    def __init__(self, n_heads, max_dist, by_head=True):
        super().__init__()
        self.n_heads = n_heads
        self.max_dist = max_dist
        self.L = 2 * max_dist + 1
        self.by_head = by_head
        if by_head: 
            self.relative_bias = nn.Parameter(torch.zeros(n_heads, self.L))
        else:
            self.relative_bias = nn.Parameter(torch.zeros(1, self.L))

    def forward(self, seq_len: int, device=None):
        positions = torch.arange(seq_len, device=device)
        rel = positions[None, :] - positions[:, None]  
        clipped = rel.clamp(-self.max_dist, self.max_dist) + self.max_dist

        table = self.relative_bias  # H or 1, V depending on by_head T/F
        bias = table[..., clipped] 
        if not self.by_head:
            bias = bias.expand(self.n_heads, -1, -1)
        return bias  


class Attention(nn.Module):
    def __init__(self, hdim, n_heads, dropout=0.0):
        super().__init__()
        assert hdim % n_heads == 0
        self.hdim = hdim
        self.n_heads = n_heads
        self.head_dim = hdim // n_heads

        self.qkv = nn.Linear(hdim, 3 * hdim)
        self.out = nn.Linear(hdim, hdim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, causal_mask=None, rel_bias=None):
        B, S, D = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.split(D, dim=-1)
        def reshape_heads(t):
            return t.view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        q = reshape_heads(q)
        k = reshape_heads(k)
        v = reshape_heads(v)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim) # B, H, S, S

        if rel_bias is not None:
            scores = scores + rel_bias.unsqueeze(0)

        if causal_mask is not None:
            if causal_mask.dim() == 2:
                scores = scores + causal_mask.unsqueeze(0).unsqueeze(0)
            elif causal_mask.dim() == 4:
                scores = scores + causal_mask
            else:
                raise ValueError("causal_mask dim not supported") # RM

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v) 
        out = out.transpose(1, 2).contiguous().view(B, S, D)
        out = self.out(out)
        return out


class FeedForward(nn.Module):
    """
    Feedforward for transformer blocks
    """
    def __init__(self, hdim, ff_dim, dropout=0.0):
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
    """
    Transformer block with:
      LayerNorm -> Self-attention -> Residual -> Feed Forward -> residual
    """
    def __init__(self, hdim, n_heads, ff_dim, attn_dropout=0.0, dropout=0.0):
        super().__init__()
        self.ln_attn = nn.LayerNorm(hdim)
        self.attn = Attention(hdim, n_heads, attn_dropout)
        self.ff = FeedForward(hdim, ff_dim, dropout)

    def forward(self, x, causal_mask=None, rel_bias=None):
        attn_in = self.ln_attn(x)
        attn_out = self.attn(attn_in, causal_mask=causal_mask, rel_bias=rel_bias)
        x = x + attn_out

        ff_out = self.ff(x)
        x = x + ff_out
        return x


class CausalTransformer(nn.Module):
    def __init__(
        self,
        Tin: int = 5,
        feat_dim: int = 256,
        hdim: int = 512,
        n_heads: int = 8,
        n_layers: int = 5,
        ff_dim: Optional[int] = None,
        dropout: float = 0.1,
        attn_dropout: float = 0.0,
        time_mask_prob: float = 0.0,
        max_rel_distance: Optional[int] = None,
        rel_bias_by_head: bool = True,
        num_classes: int = 100
    ):
        super().__init__()
        self.patch_emb = PatchEmbedding(Tin, feat_dim, hdim)
        self.input_dropout = nn.Dropout(dropout)
        self.time_mask_prob = time_mask_prob

        if ff_dim is None: # Remove
            ff_dim = 4 * hdim

        self.layers = nn.ModuleList(
            [TransformerBlock(hdim, n_heads, ff_dim, attn_dropout=attn_dropout, dropout=dropout)
             for _ in range(n_layers)]
        )
   
        self.max_rel_distance = max_rel_distance
        self.rel_bias_by_head = rel_bias_by_head
        if max_rel_distance is not None:
            self.rel_pos_bias = RelativePositionBias(n_heads, max_rel_distance, by_head=rel_bias_by_head)
        else:
            self.rel_pos_bias = None

        self.final_ln = nn.LayerNorm(hdim)
        self.output_head = nn.Linear(hdim, num_classes)

    def _causal_mask(self, seq_len, device):
        mask = torch.zeros((seq_len, seq_len), device=device, dtype=torch.float32)
        mask = torch.triu(mask, diagonal=1) 
        mask = mask * float("-inf") 
        return mask 

    def _time_mask(self, x):
        if not self.training or self.time_mask_prob <= 0.0:
            return x
        B, S, D = x.shape
        mask = torch.rand(B, S, device=x.device) < self.time_mask_prob  
        x = x.masked_fill(mask.unsqueeze(-1), 0.0)
        return x
    
    # TODO: Implement frequency masking

    def forward(self, x):
        """
        Full transformer model pass from neural activity to phonemes
        """
        B, T, F = x.shape
        z = self.patch_emb(x)  
        z = self.input_dropout(z)
        z = self._time_mask(z)

        S = z.shape[1]
        device = z.device

        causal_mask = self._causal_mask(S, device=device)  

        # Defining bias vector length
        if self.rel_pos_bias is not None:
            rel_bias = self.rel_pos_bias(S, device=device) 
        else:
            rel_bias = None

        for layer in self.layers:
            z = layer(z, causal_mask=causal_mask, rel_bias=rel_bias)

        z = self.final_ln(z)
        logits = self.output_head(z) 
        return logits
