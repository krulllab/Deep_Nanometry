import torch
from torch import nn
import io
import matplotlib.pyplot as plt
from torchvision import transforms


def crop_img_tensor(x, size) -> torch.Tensor:
    """Crops a tensor.
    Crops a tensor of shape (batch, channels, h, w) to new height and width
    given by a tuple.
    Args:
        x (torch.Tensor): Input image
        size (list or tuple): Desired size (height, width)
    Returns:
        The cropped tensor
    """
    return _pad_crop_img(x, size, 'crop')


def _pad_crop_img(x, size, mode) -> torch.Tensor:
    """ Pads or crops a tensor.
    Pads or crops a tensor of shape (batch, channels, w) to new width given by an int.
    Args:
        x (torch.Tensor): Input image
        size (int): Desired size (width)
        mode (str): Mode, either 'pad' or 'crop'
    Returns:
        The padded or cropped tensor
    """

    assert x.dim() == 3
    x_size = x.size()[2]
    if mode == 'pad':
        cond = x_size > size
    elif mode == 'crop':
        cond = x_size < size
    else:
        raise ValueError("invalid mode '{}'".format(mode))
    if cond:
        raise ValueError('trying to {} from size {} to size {}'.format(
            mode, x_size, size))
    dr = abs(x_size - size)
    dr1, dr2 = dr // 2, dr - (dr // 2)
    if mode == 'pad':
        return nn.functional.pad(x, [dr1, dr2])
    elif mode == 'crop':
        return x[:, :, dr1:x_size - dr2]


def pad_img_tensor(x, size) -> torch.Tensor:
    """Pads a tensor.
    Pads a tensor of shape (batch, channels, w) to new width
    given by an int.
    Args:
        x (torch.Tensor): Input image
        size (int): Desired size (width)
    Returns:
        The padded tensor
    """

    return _pad_crop_img(x, size, 'pad')


def plot_to_image(figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call."""
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to PyTorch tensor
    image = transforms.ToTensor()(plt.imread(buf))
    return image
