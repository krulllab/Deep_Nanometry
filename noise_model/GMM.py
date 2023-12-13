import torch
import torch.optim as optim
from torch.distributions import Normal, Categorical
import pytorch_lightning as pl
from tqdm import tqdm


def get_gaussian_params(params):
    # Make weights positive and sum to 1
    weights = params[:, 0::3]
    weights = torch.softmax(weights, dim=1)

    means = params[:, 1::3]

    # Make scales positive
    stds = params[:, 2::3]
    stds = stds.exp()
    return weights, means, stds


def sampleFromMix(weights, means, stds):
    mixture_samples = Normal(means, stds).rsample()
    component_idx = Categorical(weights.moveaxis(1, 2)).sample()

    return torch.gather(mixture_samples, dim=1, index=component_idx.unsqueeze(dim=1))


class GMM(pl.LightningModule):
    """Gaussian mixture model.

    Contains functions for calculating Gaussian mixture model
    loglikelihood and for autoregressive image sampling.

    Attributes:
        n_gaussians: An integer for the number of components in the Gaussian
        mixture model.
        noise_mean: Float for the mean of the noise samples, used to normalise
        data.
        noise_std: Float for the standard deviation of the noise samples, also
        used to normalise the data

    """

    def __init__(self, n_gaussians, noise_mean, noise_std, lr):
        super().__init__()
        self.n_gaussians = n_gaussians
        self.noise_mean = noise_mean
        self.noise_std = noise_std
        self.lr = lr

    def loglikelihood(self, x):
        """Calculates loglikelihood of noise input.

        Passes noise image through network to obtain Gaussian mixture model
        parameters. Uses those parameters to calculate loglikelihood of noise
        image.


        Parameters
        ----------
        x : torch.FloatTensor
            Noise samples

        Returns
        -------
        loglikelihoods : torch.FloatTensor
            The elementwise loglikelihood of the input.

        """

        x = (x - self.noise_mean) / self.noise_std

        params = self.forward(x)

        weights, means, stds = get_gaussian_params(params)

        loglikelihoods = Normal(means, stds).log_prob(x)
        temp = loglikelihoods.max(dim=1, keepdim=True)[0]
        loglikelihoods = loglikelihoods - temp
        loglikelihoods = loglikelihoods.exp()
        loglikelihoods = loglikelihoods * weights
        loglikelihoods = loglikelihoods.sum(dim=1, keepdim=True)
        loglikelihoods = loglikelihoods.log()
        loglikelihoods = loglikelihoods + temp

        return loglikelihoods

    @torch.no_grad()
    def sample(self, arr_shape):
        """Samples time series from the trained autoregressive model.

        Parameters
        ----------
        arr_shape : List or tuple
            The shape of the array with format [N, W], where N is the number
            of arrays, and W is the width.

        Returns
        -------
        torch.FloatTensor
            The generated noise.

        """
        # Create empty image
        arr = torch.zeros((arr_shape[0], 1, arr_shape[1]), dtype=torch.float).to(
            self.device
        )
        # Generation loop
        for w in tqdm(range(arr.shape[2])):
            params = self.forward(arr[..., : w + 1])
            weights, means, stds = get_gaussian_params(params)
            samp = sampleFromMix(weights, means, stds)
            arr[..., w] = samp[..., w]

        return arr * self.noise_std + self.noise_mean

    def training_step(self, batch, _):
        loss = -torch.mean(self.loglikelihood(batch))
        self.log("train/nll", loss)
        return loss

    def validation_step(self, batch, _):
        loss = -torch.mean(self.loglikelihood(batch))
        self.log("val/nll", loss, prog_bar=True)

    def configure_optimizers(self):
        optimizer = optim.Adamax(self.parameters(), lr=self.lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.5, patience=10
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val/nll"}
