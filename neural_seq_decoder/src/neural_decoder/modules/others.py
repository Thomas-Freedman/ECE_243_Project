import torch
import torch.nn as nn


class Pack(nn.Module):
    """Pack operation for sequence processing"""
    def __init__(self):
        super(Pack, self).__init__()

    def forward(self, x, lens):
        """
        Args:
            x: (batch, time, features)
            lens: (batch,) sequence lengths
        Returns:
            x: packed sequences
            lens: unchanged
        """
        return x, lens


class UnPack(nn.Module):
    """Unpack operation - converts (batch, time, features) to proper format"""
    def __init__(self):
        super(UnPack, self).__init__()

    def forward(self, x, lens):
        """
        Args:
            x: (batch, time, features)
            lens: (batch,) sequence lengths
        Returns:
            x: unpacked sequences
            lens: unchanged
        """
        return x, lens
