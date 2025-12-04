import torch
import torch.nn as nn


class MambaLayer(nn.Module):
    """
    Simplified Mamba layer for sequence modeling
    Mamba is a state-space model architecture
    """
    def __init__(self, d_model, expand_factor=2):
        super(MambaLayer, self).__init__()
        self.d_model = d_model
        self.d_inner = d_model * expand_factor

        # Projections
        self.in_proj = nn.Linear(d_model, self.d_inner * 2)
        self.conv1d = nn.Conv1d(
            self.d_inner,
            self.d_inner,
            kernel_size=3,
            padding=1,
            groups=self.d_inner
        )
        self.out_proj = nn.Linear(self.d_inner, d_model)

        # Activation
        self.activation = nn.SiLU()

        # Normalization
        self.norm = nn.LayerNorm(d_model)

        # Initialize weights properly to prevent NaN
        nn.init.xavier_uniform_(self.in_proj.weight, gain=0.01)
        nn.init.xavier_uniform_(self.out_proj.weight, gain=0.01)
        nn.init.zeros_(self.in_proj.bias)
        nn.init.zeros_(self.out_proj.bias)
        nn.init.xavier_uniform_(self.conv1d.weight, gain=0.01)

    def forward(self, x):
        """
        Args:
            x: (batch, time, d_model)
        Returns:
            x: (batch, time, d_model)
        """
        residual = x
        x = self.norm(x)

        # Project and split
        x_proj = self.in_proj(x)
        x, gate = x_proj.chunk(2, dim=-1)

        # Conv1d expects (batch, channels, time)
        x = x.transpose(1, 2)
        x = self.conv1d(x)
        x = x.transpose(1, 2)

        # Activation and gating
        x = self.activation(x)
        x = x * torch.sigmoid(gate)

        # Output projection
        x = self.out_proj(x)

        return x + residual


class mamba_block(nn.Module):
    """
    Stack of Mamba layers with optional bidirectionality
    """
    def __init__(self, d_model, n_layer=1, bidirectional=False, update_probs=0.7):
        super(mamba_block, self).__init__()
        self.d_model = d_model
        self.n_layer = n_layer
        self.bidirectional = bidirectional

        # Forward layers
        self.forward_layers = nn.ModuleList([
            MambaLayer(d_model) for _ in range(n_layer)
        ])

        # Backward layers if bidirectional
        if bidirectional:
            self.backward_layers = nn.ModuleList([
                MambaLayer(d_model) for _ in range(n_layer)
            ])

        # Dropout (update_probs is dropout probability)
        self.dropout = nn.Dropout(1.0 - update_probs) if update_probs < 1.0 else None

    def forward(self, x, lens):
        """
        Args:
            x: (batch, time, d_model)
            lens: (batch,) sequence lengths
        Returns:
            x: (batch, time, d_model)
            lens: unchanged
        """
        # Forward direction
        for layer in self.forward_layers:
            x = layer(x)
            if self.dropout is not None:
                x = self.dropout(x)

        # Backward direction if bidirectional
        if self.bidirectional:
            x_backward = torch.flip(x, dims=[1])
            for layer in self.backward_layers:
                x_backward = layer(x_backward)
                if self.dropout is not None:
                    x_backward = self.dropout(x_backward)
            x_backward = torch.flip(x_backward, dims=[1])

            # Combine forward and backward (concatenation would double dims, so we average)
            x = (x + x_backward) / 2

        return x, lens
