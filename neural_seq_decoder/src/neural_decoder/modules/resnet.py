import torch
import torch.nn as nn


class resnet_block(nn.Module):
    """ResNet-style residual block for sequence processing"""
    def __init__(self, input_dims, output_dims, stride=1, hidden_size=None):
        super(resnet_block, self).__init__()
        self.stride = stride

        if hidden_size is None:
            hidden_size = output_dims

        self.conv1 = nn.Conv1d(input_dims, hidden_size, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.relu = nn.ReLU()

        self.conv2 = nn.Conv1d(hidden_size, output_dims, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(output_dims)

        # Shortcut connection
        if stride != 1 or input_dims != output_dims:
            self.shortcut = nn.Sequential(
                nn.Conv1d(input_dims, output_dims, kernel_size=1, stride=stride),
                nn.BatchNorm1d(output_dims)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x, lens):
        """
        Args:
            x: (batch, time, features)
            lens: (batch,) sequence lengths
        Returns:
            x: transformed (batch, new_time, new_features)
            lens: updated sequence lengths
        """
        # Conv1d expects (batch, features, time)
        identity = x.transpose(1, 2)

        out = x.transpose(1, 2)
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += self.shortcut(identity)
        out = self.relu(out)

        out = out.transpose(1, 2)

        # Update lengths after striding
        if self.stride > 1:
            new_lens = (lens - 1) // self.stride + 1
            new_lens = new_lens.clamp(min=1)
        else:
            new_lens = lens

        return out, new_lens
