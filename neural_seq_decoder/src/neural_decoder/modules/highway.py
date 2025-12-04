import torch
import torch.nn as nn


class Highway(nn.Module):
    """Highway network layer"""
    def __init__(self, input_dim, num_layers=1):
        super(Highway, self).__init__()
        self.input_dim = input_dim
        self.num_layers = num_layers

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            layer = nn.ModuleDict({
                'transform': nn.Linear(input_dim, input_dim),
                'gate': nn.Linear(input_dim, input_dim),
            })
            self.layers.append(layer)

    def forward(self, x, lens):
        """
        Args:
            x: (batch, time, features)
            lens: (batch,) sequence lengths
        Returns:
            x: transformed (batch, time, features)
            lens: unchanged
        """
        for layer in self.layers:
            transform_gate = torch.sigmoid(layer['gate'](x))
            transform = torch.relu(layer['transform'](x))
            x = transform_gate * transform + (1 - transform_gate) * x

        return x, lens
