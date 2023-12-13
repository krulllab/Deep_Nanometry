import numpy as np
import torch
from torch import nn
from torch import optim
from pytorch_lightning import LightningModule
import matplotlib.pyplot as plt

from ..lib.utils import crop_img_tensor, pad_img_tensor, plot_to_image
from .lvae_layers import TopDownLayer, BottomUpLayer, TopDownDeterministicResBlock, BottomUpDeterministicResBlock


class LadderVAE(LightningModule):

    def __init__(self,
                 data_mean,
                 data_std,
                 img_width,
                 noise_model,
                 z_dims=None,
                 blocks_per_layer=1,
                 n_filters=64,
                 res_block_type='bacdbacd',
                 merge_type='residual',
                 stochastic_skip=True,
                 gated=True,
                 batchnorm=True,
                 dropout=0,
                 downsample=True,
                 mode_pred=False):
        if z_dims is None:
            z_dims = [32] * 8
        self.save_hyperparameters()
        super().__init__()
        self.data_mean = data_mean
        self.data_std = data_std
        self.img_width = img_width
        self.noise_model = noise_model
        self.z_dims = z_dims
        self.n_layers = len(self.z_dims)
        self.blocks_per_layer = blocks_per_layer
        self.n_filters = n_filters
        self.stochastic_skip = stochastic_skip
        self.gated = gated
        self.dropout = dropout
        self.mode_pred = mode_pred

        # Number of downsampling steps per layer
        if downsample:
            downsampling = [1] * self.n_layers
        else:
            downsampling = [0] * self.n_layers
            
        # Downsample by a factor of 2 at each downsampling operation
        self.overall_downscale_factor = np.power(2, sum(downsampling))

        assert max(downsampling) <= self.blocks_per_layer
        assert len(downsampling) == self.n_layers

        # First bottom-up layer: change num channels
        self.first_bottom_up = nn.Sequential(
            nn.Conv1d(1, n_filters, 5, padding=2, padding_mode='replicate'),
            nn.ELU(),
            BottomUpDeterministicResBlock(
                c_in=n_filters,
                c_out=n_filters,
                batchnorm=batchnorm,
                dropout=dropout,
                res_block_type=res_block_type,
            ))

        # Init lists of layers
        self.top_down_layers = nn.ModuleList([])
        self.bottom_up_layers = nn.ModuleList([])

        for i in range(self.n_layers):
            # Whether this is the top layer
            is_top = i == self.n_layers - 1

            # Add bottom-up deterministic layer at level i.
            # It's a sequence of residual blocks (BottomUpDeterministicResBlock)
            # possibly with downsampling between them.
            self.bottom_up_layers.append(
                BottomUpLayer(
                    n_res_blocks=self.blocks_per_layer,
                    n_filters=n_filters,
                    downsampling_steps=downsampling[i],
                    batchnorm=batchnorm,
                    dropout=dropout,
                    res_block_type=res_block_type,
                    gated=gated,
                ))

            # Add top-down stochastic layer at level i.
            # The architecture when doing inference is roughly as follows:
            #    p_params = output of top-down layer above
            #    bu = inferred bottom-up value at this layer
            #    q_params = merge(bu, p_params)
            #    z = stochastic_layer(q_params)
            #    possibly get skip connection from previous top-down layer
            #    top-down deterministic ResNet
            #
            # When doing generation only, the value bu is not available, the
            # merge layer is not used, and z is sampled directly from p_params.
            self.top_down_layers.append(
                TopDownLayer(
                    z_dim=z_dims[i],
                    n_res_blocks=blocks_per_layer,
                    n_filters=n_filters,
                    is_top_layer=is_top,
                    downsampling_steps=downsampling[i],
                    merge_type=merge_type,
                    batchnorm=batchnorm,
                    dropout=dropout,
                    stochastic_skip=stochastic_skip,
                    top_prior_param_shape=self.get_top_prior_param_shape(),
                    res_block_type=res_block_type,
                    gated=gated,
                ))

        # Final top-down layer
        modules = list()
        for i in range(blocks_per_layer):
            modules.append(
                TopDownDeterministicResBlock(
                    c_in=n_filters,
                    c_out=n_filters,
                    batchnorm=batchnorm,
                    dropout=dropout,
                    res_block_type=res_block_type,
                    gated=gated,
                ))
        modules.append(
            nn.Conv1d(n_filters,
                      1,
                      kernel_size=3,
                      padding=1,
                      padding_mode='replicate'))
        self.final_top_down = nn.Sequential(*modules)

    def forward(self, x):
        img_size = x.size()[2]

        # Pad x to have base 2 side lengths to make resampling steps simpler
        x_pad = self.pad_input(x)

        # Bottom-up inference: return list of length n_layers (bottom to top)
        bu_values = self.bottomup_pass(x_pad)

        # Top-down inference/generation
        out, kl = self.topdown_pass(bu_values)

        if not self.mode_pred:
            kl_sums = [torch.sum(layer) for layer in kl]
            kl_loss = sum(kl_sums) / float(x.shape[0] * x.shape[1] * x.shape[2])
        else:
            kl_loss = None

        # Restore original image size
        predicted_signal = crop_img_tensor(out, img_size)

        x_denormalised = x * self.data_std + self.data_mean
        predicted_signal_denormalised = predicted_signal * self.data_std + self.data_mean

        predicted_noise = x_denormalised - predicted_signal_denormalised

        if not self.mode_pred:
            # Noise model returns log[p(x|predicted_s)]
            ll = self.noise_model.loglikelihood(predicted_noise)
            ll = ll.mean()
        else:
            ll = None

        output = {
            'll': ll,
            'kl_loss': kl_loss,
            'predicted_signal': predicted_signal,
        }
        return output

    def bottomup_pass(self, x):
        # Bottom-up initial layer
        x = self.first_bottom_up(x)

        # Loop from bottom to top layer, store all deterministic nodes we
        # need in the top-down pass
        bu_values = []
        for i in range(self.n_layers):
            x = self.bottom_up_layers[i](x)
            bu_values.append(x)

        return bu_values

    def topdown_pass(self,
                     bu_values=None,
                     n_img_prior=None,
                     mode_layers=None,
                     constant_layers=None,
                     forced_latent=None):

        # Default: no layer is sampled from the distribution's mode
        if mode_layers is None:
            mode_layers = []
        if constant_layers is None:
            constant_layers = []

        # If the bottom-up inference values are not given, don't do
        # inference, sample from prior instead
        inference_mode = bu_values is not None

        # Check consistency of arguments
        if inference_mode != (n_img_prior is None):
            msg = ("Number of images for top-down generation has to be given "
                   "if and only if we're not doing inference")
            raise RuntimeError(msg)

        # KL divergence of each layer
        kl = [None] * self.n_layers

        if forced_latent is None:
            forced_latent = [None] * self.n_layers

        # Top-down inference/generation loop
        out = None
        for i in reversed(range(self.n_layers)):
            # If available, get deterministic node from bottom-up inference
            try:
                bu_value = bu_values[i]
            except TypeError:
                bu_value = None

            # Whether the current layer should be sampled from the mode
            use_mode = i in mode_layers
            constant_out = i in constant_layers

            # Input for skip connection
            skip_input = out  # TODO or out_pre_residual? or both?

            # Full top-down layer, including sampling and deterministic part
            out, kl_elementwise = self.top_down_layers[i](
                out,
                skip_connection_input=skip_input,
                inference_mode=inference_mode,
                bu_value=bu_value,
                n_img_prior=n_img_prior,
                use_mode=use_mode,
                force_constant_output=constant_out,
                forced_latent=forced_latent[i],
                mode_pred=self.mode_pred)
            kl[i] = kl_elementwise  # (batch, ch, w)

        # Final top-down layer
        out = self.final_top_down(out)

        return out, kl

    def pad_input(self, x):
        """
        Pads input x so that its sizes are powers of 2
        :param x:
        :return: Padded tensor
        """
        size = self.get_padded_size(x.size()[-1])
        x = pad_img_tensor(x, size)
        return x

    def get_padded_size(self, size):
        # Overall downscale factor from input to top layer (power of 2)
        dwnsc = self.overall_downscale_factor

        # Output smallest powers of 2 that are larger than current sizes
        padded_size = ((size - 1) // dwnsc + 1) * dwnsc

        return padded_size

    def get_top_prior_param_shape(self, n_imgs=1):
        # TODO num channels depends on random variable we're using
        dwnsc = self.overall_downscale_factor
        sz = self.get_padded_size(self.img_width)
        w = sz // dwnsc
        c = self.z_dims[-1] * 2  # mu and logvar
        top_layer_shape = (n_imgs, c, w)
        return top_layer_shape

    def training_step(self, batch, _):
        # Normalise data
        x = (batch - self.data_mean) / self.data_std
        # Returns dictionary containing predicted signal and loss terms
        model_out = self.forward(x)

        recons_loss = -model_out['ll']
        kl_loss = model_out['kl_loss']
        elbo = recons_loss + kl_loss

        self.log_dict({
            'train/elbo': elbo,
            'train/kl_divergence': kl_loss,
            'train/reconstruction_loss': recons_loss
        })

        return elbo

    def log_images_for_tensorboard(self, x, samples, median):
        figure = plt.figure()
        plt.plot(x[0, 0].cpu(), color="blue", label="Attenuated")
        plt.plot(median[0, 0].cpu(), color="orange", label="Denoised")
        for i in range(len(samples)):
            plt.plot(samples[i, 0].cpu(), color="orange", alpha=0.2)
        plt.legend()

        self.trainer.logger.experiment.add_image("Validation_plot",
                                                 plot_to_image(figure),
                                                 self.current_epoch)

    def validation_step(self, batch, batch_idx):
        x = (batch - self.data_mean) / self.data_std
        model_out = self.forward(x)

        recons_loss = -model_out['ll']
        kl_loss = model_out['kl_loss']
        elbo = recons_loss + kl_loss

        self.log_dict({
            'val/elbo': elbo,
            'val/reconstruction_loss': recons_loss,
            'val/kl_divergence': kl_loss
        })

        if batch_idx == 0:
            # Display validation set results on tensorboard
            # One noisy array, 10 predictions of its signal and their median
            idx = np.random.randint(len(x))
            samples = self.forward(
                torch.repeat_interleave(x[idx:idx + 1], 10, 0))["predicted_signal"]
            median = torch.quantile(samples, 0.5, dim=0, keepdim=True)
            self.log_images_for_tensorboard(x[idx:idx + 1], samples, median)

    def predict_step(self, batch, _):
        # Don't calculate loss or carry out masking
        self.mode_pred = True
        x = (batch - self.data_mean) / self.data_std
        out = self.forward(x)['predicted_signal']
        out = out * self.data_std + self.data_mean
        return out

    def configure_optimizers(self):
        optimizer = optim.Adamax(self.parameters(), lr=3e-4)
        scheduler = {
            'scheduler':
                optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                     factor=0.5,
                                                     verbose=True),
            'monitor':
                'val/elbo'
        }

        return [optimizer], [scheduler]
