import torch
import torch.nn as nn
import torch.nn.functional as F
from noise_model.GMM import GMM


class ShiftedConvolution(nn.Module):
    """Implements a 1D convolution with a shifted kernel.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int
        Length of the convolutional kernel.
    dilation : int
        Dilation factor.
    first : bool
        Whether this is the first convolution in the network.
        
    """
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, first=False):
        super().__init__()

        shift = dilation * (kernel_size - 1)
        self.pad = nn.ConstantPad1d((shift, 0), 0)

        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation)

        self.first = first
        if self.first:
            mask = torch.ones(kernel_size)
            mask[-1] = 0
            self.register_buffer("mask", mask[None, None])

    def forward(self, x):
        x = self.pad(x)
        if self.first:
            self.conv.weight.data *= self.mask
        x = self.conv(x)
        return x


class GatedBlock(nn.Module):
    """A gated activation unit.


    Parameters
    ----------
    n_filters : int
        Number of hidden channels.
    **kwargs
        Additional arguments for the convolutions.

    """

    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, first=False):
        super().__init__()

        self.in_conv = ShiftedConvolution(
            in_channels, 2 * out_channels, kernel_size, dilation, first
        )
        self.out_conv = nn.Conv1d(out_channels, out_channels, 1)
        if in_channels == out_channels:
            self.do_skip = True
        else:
            self.do_skip = False

    def forward(self, x):
        feat = self.in_conv(x)
        tan, sig = torch.chunk(feat, 2, dim=1)
        out = torch.tanh(tan) * torch.sigmoid(sig)
        out = self.out_conv(out)
        if self.do_skip:
            out = out + x
        return out


class PixelCNN(GMM):
    """A CNN with attention gates and autoregressive convolutions

    Parameters
    ----------
    in_channels : int, optional
        The number of input channels. The default is 1.
    n_filters : int, optional
        The number of hidden channels. The default is 128.
    kernel_size : int, optional
        Side length of the convolutional kernel. The default is 5.
    n_gaussians : int, optional
        Number of components in the Gaussian mixture model. The default is 10.
    noise_mean : Float, optional
        Mean of the noise samples, used for normalisation of the data. The default is 0.
    noise_std : Float, optional
        Standard deviation of the noise samples, used for normalisation of the data. The default is 1.

    """

    def __init__(
        self,
        n_filters=8,
        kernel_size=11,
        n_gaussians=2,
        noise_mean=0,
        noise_std=1,
        lr=2e-3,
    ):
        self.save_hyperparameters()
        super().__init__(n_gaussians, noise_mean, noise_std, lr)

        out_channels = n_gaussians * 3

        self.gatedconvs = nn.Sequential(
            GatedBlock(1, n_filters, kernel_size, first=True),
            GatedBlock(n_filters, n_filters, kernel_size, dilation=2),
            GatedBlock(n_filters, n_filters, kernel_size),
            GatedBlock(n_filters, n_filters, kernel_size, dilation=4),
            GatedBlock(n_filters, n_filters, kernel_size),
            GatedBlock(n_filters, n_filters, kernel_size, dilation=2),
            GatedBlock(n_filters, out_channels, kernel_size),
        )

    def forward(self, x):
        return self.gatedconvs(x)
