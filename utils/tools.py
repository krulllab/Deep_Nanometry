import matplotlib.pyplot as plt
import numpy as np
import torch


def show_center_recep_field(arr, out):
    """Calculates the gradients of the input with respect to the output rightmost pixel, and visualizes the overall
    receptive field.

    Args:
        arr: Input array for which we want to calculate the receptive field on.
        out: Output features/loss which is used for backpropagation, and should be
              the output of the network/computation graph.
    """
    # Determine gradients
    loss = out[..., arr.shape[-1] - 1].sum()
    # Retain graph as we want to stack multiple layers and show the receptive field of all of them
    loss.backward(retain_graph=True)
    arr_grads = arr.grad.abs()
    arr.grad.fill_(0)  # Reset grads

    # Plot receptive field
    arr = arr_grads.squeeze().cpu().numpy()
    _, ax = plt.subplots()
    ax.plot(arr > 0, "o")
    ax.set_xlabel("Time")
    ax.set_ylabel("Binary receptive field")
    plt.show()
    plt.close()


def view_receptive_field(noise_model, img_shape):
    inp_img = torch.zeros(1, 1, *img_shape).requires_grad_()
    out = noise_model(inp_img)
    show_center_recep_field(inp_img, out[:, [0]])


def autocorrelation(arrs, max_lag=100, titles=None):
    """Calculates the autocorrelation of a list of 1D arrays.

    Args:
        a: List of input arrays.
        max_lag: Maximum lag to calculate the autocorrelation for.
    """
    for i, a in enumerate(arrs):
        a = a-a.mean()
        results = np.zeros((max_lag,))
        for j in range(max_lag):
            if j == 0:
                covar = np.mean(a**2)
            else:
                covar = np.mean(a[...,j:]*a[...,:-j])
            results[j] = covar

        ac = results/(a**2).mean()
        plt.plot(ac, "--o", label=titles[i] if titles is not None else None)
    plt.legend()
    plt.xlabel("Lag")
    plt.ylabel("Autocorrelation")
    plt.show()


def plot(arr, titles=None, figsize=(10, 5)):
    """Plots a list of 1D arrays.

    Args:
        arr: List of arrays to plot.
        title: List of titles for the plot.
    """
    arr = [a.squeeze() for a in arr]
    plt.figure(figsize=figsize)
    for i, a in enumerate(arr):
        plt.plot(a, label=titles[i])
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()
    plt.show()
