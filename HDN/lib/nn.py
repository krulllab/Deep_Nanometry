import torch
from torch import nn


class ResidualBlock(nn.Module):
    """
    Residual block with 2 convolutional layers.
    Input, intermediate, and output channels are the same. Padding is always
    'same'. The 2 convolutional layers have the same groups. No stride allowed,
    and kernel sizes have to be odd.

    The result is:
        out = gate(f(x)) + x
    where an argument controls the presence of the gating mechanism, and f(x)
    has different structures depending on the argument block_type.
    block_type is a string specifying the structure of the block, where:
        a = activation
        b = batch norm
        c = conv layer
        d = dropout.
    For example, bacdbacd has 2x (batchnorm, activation, conv, dropout).
    """

    default_kernel_size = 3

    def __init__(self,
                 channels,
                 kernel=None,
                 groups=1,
                 batchnorm=True,
                 block_type=None,
                 dropout=None,
                 gated=None):
        super().__init__()
        if kernel is None:
            kernel = self.default_kernel_size
        elif isinstance(kernel, int):
            kernel = (kernel, kernel)
        elif len(kernel) != 1:
            raise ValueError(
                "kernel has to be None, or int")
        assert kernel % 2 == 1, "kernel sizes have to be odd"
        pad = kernel // 2
        self.gated = gated

        modules = []
        if block_type == 'cabdcabd':
            for i in range(2):
                conv = nn.Conv1d(channels,
                                 channels,
                                 kernel,
                                 padding=pad,
                                 groups=groups)
                modules.append(conv)
                modules.append(nn.ELU())
                if batchnorm:
                    modules.append(nn.BatchNorm1d(channels))
                if dropout is not None:
                    modules.append(nn.Dropout1d(dropout))

        elif block_type == 'bacdbac':
            for i in range(2):
                if batchnorm:
                    modules.append(nn.BatchNorm1d(channels))
                modules.append(nn.ELU())
                conv = nn.Conv1d(channels,
                                 channels,
                                 kernel,
                                 padding=pad,
                                 groups=groups)
                modules.append(conv)
                if dropout is not None and i == 0:
                    modules.append(nn.Dropout1d(dropout))

        elif block_type == 'bacdbacd':
            for i in range(2):
                if batchnorm:
                    modules.append(nn.BatchNorm1d(channels))
                modules.append(nn.ELU())
                conv = nn.Conv1d(channels,
                                 channels,
                                 kernel,
                                 padding=pad,
                                 groups=groups)
                modules.append(conv)
                modules.append(nn.Dropout1d(dropout))

        else:
            raise ValueError("unrecognized block type '{}'".format(block_type))

        if gated:
            modules.append(GateLayer1d(channels, 1))
        self.block = nn.Sequential(*modules)

    def forward(self, x):
        return self.block(x) + x


class GateLayer1d(nn.Module):
    """
    Double the number of channels through a convolutional layer, then use
    half the channels as gate for the other half.
    """

    def __init__(self, channels, kernel_size):
        super().__init__()
        assert kernel_size % 2 == 1
        pad = kernel_size // 2
        self.conv = nn.Conv1d(channels, 2 * channels, kernel_size, padding=pad)
        self.nonlin = nn.ELU()

    def forward(self, x):
        x = self.conv(x)
        x, gate = torch.chunk(x, 2, dim=1)
        x = self.nonlin(x)  # TODO remove this?
        gate = torch.sigmoid(gate)
        return x * gate
