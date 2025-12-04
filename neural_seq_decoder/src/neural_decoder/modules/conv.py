import torch
import torch.nn as nn


class conv_block(nn.Module):
    """1D Convolutional block for sequence processing"""
    def __init__(self, input_dims, output_dims, kernel_size, stride, groups=1):
        super(conv_block, self).__init__()
        self.stride = stride
        self.kernel_size = kernel_size

        padding = (kernel_size - 1) // 2

        self.conv = nn.Conv1d(
            input_dims,
            output_dims,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups
        )
        self.bn = nn.BatchNorm1d(output_dims)
        self.relu = nn.ReLU()

    def forward(self, x, lens):
        """
        Args:
            x: (batch, time, features)
            lens: (batch,) sequence lengths
        Returns:
            x: convolved (batch, new_time, new_features)
            lens: updated sequence lengths
        """
        # Conv1d expects (batch, features, time)
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = x.transpose(1, 2)

        # Update lengths after striding
        new_lens = (lens + 2 * ((self.kernel_size - 1) // 2) - self.kernel_size) // self.stride + 1
        new_lens = new_lens.clamp(min=1)

        return x, new_lens
