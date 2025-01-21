"""
Module containing the main VAE class.
"""
import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F

from models.initialization import weights_init
from models.regression import WTPregression

class VAE(nn.Module):
    def __init__(self, img_size, encoder, decoder, regression, latent_dim, model_type, threshold_val,sup_signal1,sup_signal2,sup_signal3,sup_signal4,sup_signal5):
        """
        Class which defines model and forward pass.

        Parameters
        ----------
        img_size : tuple of ints
        """
        super(VAE, self).__init__()

        self.latent_dim = latent_dim
        self.img_size = img_size
        self.num_pixels = self.img_size[1] * self.img_size[2]
        self.encoder = encoder(self.img_size, self.latent_dim)
        self.model_type = model_type
        self.threshold_val = threshold_val
        self.sup_signal1 = sup_signal1
        self.sup_signal2 = sup_signal2
        self.sup_signal3 = sup_signal3
        self.sup_signal4 = sup_signal4
        self.sup_signal5 = sup_signal5

        if self.sup_signal1 == "make":
            self.regression1 = regression(self.latent_dim,9)
        elif self.sup_signal1 == "color":
            self.regression1 = regression(self.latent_dim,6)
        elif self.sup_signal1 == "region":
            self.regression1 = regression(self.latent_dim,6)

        # self.regression1 = regression(self.latent_dim,43)
        self.regression2 = regression(self.latent_dim)
        self.regression3 = regression(self.latent_dim)
        self.regression4 = regression(self.latent_dim)
        self.regression5 = regression(self.latent_dim)

        self.decoder = decoder(self.img_size, self.latent_dim)

        self.reset_parameters()

    def reparameterize(self, mean, logvar):
        """
        Samples from a normal distribution using the reparameterization trick.

        Parameters
        ----------
        mean : torch.Tensor
            Mean of the normal distribution. Shape (batch_size, latent_dim)

        logvar : torch.Tensor
            Diagonal log variance of the normal distribution. Shape (batch_size,
            latent_dim)
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mean + std * eps
        else:
            # Reconstruction mode
            return mean

    def meaningful_visual_attributes(self, mean, logvar, threshold_val):
        """
        """
        latent_dim = mean.size(1)
        batch_size = mean.size(0)
        latent_kl = 0.5 * (-1 - logvar + mean.pow(2) + logvar.exp()).mean(dim=0)
        zeros = torch.zeros([mean.size(0),mean.size(1)])
        ones = torch.ones([mean.size(0),mean.size(1)])

        for i in range(latent_dim):
            if latent_kl[i].item() < threshold_val:
                ones[:,i] = zeros[:,i]

        return ones

    def forward(self, x, make, makemodel, color, firm, region, price, hp, mpg, mpd, hpwt, space, wt, length, wid, ht, wb, xi_fe, shares, wph):
        """
        Forward pass of model.

        Parameters
        ----------
        x : torch.Tensor
            Batch of data. Shape (batch_size, n_chan, height, width)
        """
        price_s = price.shape[0]
        hp_s = hp.shape[0]
        mpg_s = mpg.shape[0]
        mpd_s = mpd.shape[0]
        xi_fe_s = xi_fe.shape[0]
        shares_s = shares.shape[0]
        wph_s = wph.shape[0]
        hpwt_s = hpwt.shape[0]
        space_s = space.shape[0]
        wt_s = wt.shape[0]
        length_s = length.shape[0]
        wid_s = wid.shape[0]
        ht_s = ht.shape[0]
        wb_s = wb.shape[0]

        price = torch.reshape(price,(price_s,1))
        hp = torch.reshape(hp,(hp_s,1))
        mpg = torch.reshape(mpg,(mpg_s,1))
        mpd = torch.reshape(mpd,(mpd_s,1))
        xi_fe = torch.reshape(xi_fe,(xi_fe_s,1))
        shares = torch.reshape(shares,(shares_s,1))
        wph = torch.reshape(wph,(wph_s,1))
        hpwt = torch.reshape(hpwt,(hpwt_s,1))
        space = torch.reshape(space,(space_s,1))
        wt = torch.reshape(wt,(wt_s,1))
        length = torch.reshape(length,(length_s,1))
        wid = torch.reshape(wid,(wid_s,1))
        ht = torch.reshape(ht,(ht_s,1))
        wb = torch.reshape(wb,(wb_s,1))

        latent_dist = self.encoder(x)
        latent_sample = self.reparameterize(*latent_dist)
        reconstruct = self.decoder(latent_sample)
        visual_attributes = self.meaningful_visual_attributes(*latent_dist, self.threshold_val)

        wtp_pred1 = self.regression1(torch.mul(latent_dist[0],visual_attributes.cuda()))
        wtp_pred2 = self.regression2(torch.mul(latent_dist[0],visual_attributes.cuda()))
        wtp_pred3 = self.regression3(torch.mul(latent_dist[0],visual_attributes.cuda()))
        wtp_pred4 = self.regression4(torch.mul(latent_dist[0],visual_attributes.cuda()))
        wtp_pred5 = self.regression5(torch.mul(latent_dist[0],visual_attributes.cuda()))

        return reconstruct, latent_dist, latent_sample, wtp_pred1, wtp_pred2, wtp_pred3, wtp_pred4, wtp_pred5, visual_attributes

    def reset_parameters(self):
        self.apply(weights_init)

    def sample_latent(self, x):
        """
        Returns a sample from the latent distribution.

        Parameters
        ----------
        x : torch.Tensor
            Batch of data. Shape (batch_size, n_chan, height, width)
        """
        latent_dist = self.encoder(x)
        latent_sample = self.reparameterize(*latent_dist)
        return latent_sample

class Encoder(nn.Module):
    def __init__(self,
                 img_size,
                 latent_dim=10):
        r"""Encoder of the model proposed in [1].

        Parameters
        ----------
        img_size : tuple of ints

        latent_dim : int
            Dimensionality of latent output.

        Model Architecture (transposed for decoder)
        ------------
        - 4 convolutional layers (each with 32 channels), (4 x 4 kernel), (stride of 2)
        - 2 fully connected layers (each of 256*2 units)
        - Latent distribution:
            - 1 fully connected layer of 20 units (log variance and mean for 10 Gaussians)

        """
        super(Encoder, self).__init__()

        # Layer parameters
        hid_channels = 32
        kernel_size = 4
        hidden_dim = 256*2
        self.latent_dim = latent_dim
        self.img_size = img_size
        # Shape required to start transpose convs
        self.reshape = (hid_channels, kernel_size, kernel_size)
        n_chan = self.img_size[0]
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(n_chan, hid_channels, kernel_size, stride=2, padding=1, dilation=1)
        self.conv2 = nn.Conv2d(hid_channels, hid_channels, kernel_size, stride=2, padding=1, dilation=1)
        self.conv3 = nn.Conv2d(hid_channels, hid_channels, kernel_size, stride=2, padding=1, dilation=1)
        self.conv_64 = nn.Conv2d(hid_channels, hid_channels, kernel_size, stride=2, padding=1, dilation=1)
        self.conv_128=nn.Conv2d(hid_channels, hid_channels, kernel_size, stride=2, padding=1, dilation=1)
        # Fully connected layers
        self.lin1 = nn.Linear(np.product(self.reshape), hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, hidden_dim)

        # Fully connected layers for mean and variance
        self.mu_logvar_gen = nn.Linear(hidden_dim, self.latent_dim * 2)

    def forward(self, x):
        batch_size = x.size(0)

        # Convolutional layers with ReLu activations
        x = torch.nn.functional.leaky_relu(self.conv1(x))
        x = torch.nn.functional.leaky_relu(self.conv2(x))
        x = torch.nn.functional.leaky_relu(self.conv3(x))
        x = torch.nn.functional.leaky_relu(self.conv_64(x))
        x = torch.nn.functional.leaky_relu(self.conv_128(x))

        # Fully connected layers with ReLu activations
        x = x.view((batch_size, -1))
        x = torch.nn.functional.leaky_relu(self.lin1(x))
        x = torch.nn.functional.leaky_relu(self.lin2(x))

        # Fully connected layer for log variance and mean
        # Log std-dev in paper (bear in mind)
        mu_logvar = self.mu_logvar_gen(x)
        mu, logvar = mu_logvar.view(-1, self.latent_dim, 2).unbind(-1)

        return mu, logvar


class Decoder(nn.Module):
    def __init__(self, img_size,
                 latent_dim=10):
        r"""Decoder of the model proposed in [1].

        Parameters
        ----------
        img_size : tuple of ints

        latent_dim : int
            Dimensionality of latent output.

        Model Architecture (transposed for decoder)
        ------------
        - 4 convolutional layers (each with 32 channels), (4 x 4 kernel), (stride of 2)
        - 2 fully connected layers (each of 256*2 units)
        - Latent distribution:
            - 1 fully connected layer of 20 units (log variance and mean for 10 Gaussians)
        """
        super(Decoder, self).__init__()

        # Layer parameters
        hid_channels = 32
        kernel_size = 4
        hidden_dim = 256*2
        self.img_size = img_size
        # Shape required to start transpose convs
        self.reshape = (hid_channels, kernel_size, kernel_size)
        n_chan = self.img_size[0]
        self.img_size = img_size

        # Fully connected layers
        self.lin1 = nn.Linear(latent_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, hidden_dim)
        self.lin3 = nn.Linear(hidden_dim, np.product(self.reshape))

        # Convolutional layers
        self.convT_128 = nn.ConvTranspose2d(hid_channels, hid_channels, kernel_size, stride=2, padding=1, dilation=1)
        self.convT_64 = nn.ConvTranspose2d(hid_channels, hid_channels, kernel_size, stride=2, padding=1, dilation=1)
        self.convT1 = nn.ConvTranspose2d(hid_channels, hid_channels, kernel_size, stride=2, padding=1, dilation=1)
        self.convT2 = nn.ConvTranspose2d(hid_channels, hid_channels, kernel_size, stride=2, padding=1, dilation=1)
        self.convT3 = nn.ConvTranspose2d(hid_channels, n_chan, kernel_size, stride=2, padding=1, dilation=1)

    def forward(self, z):
        batch_size = z.size(0)

        # Fully connected layers with ReLu activations
        x = torch.nn.functional.leaky_relu(self.lin1(z))
        x = torch.nn.functional.leaky_relu(self.lin2(x))
        x = torch.nn.functional.leaky_relu(self.lin3(x))
        x = x.view(batch_size, *self.reshape)

        # Convolutional layers with ReLu activations
        x = torch.nn.functional.leaky_relu(self.convT_128(x))
        x = torch.nn.functional.leaky_relu(self.convT_64(x))
        x = torch.nn.functional.leaky_relu(self.convT1(x))
        x = torch.nn.functional.leaky_relu(self.convT2(x))
        # Sigmoid activation for final conv layer
        x = torch.sigmoid(self.convT3(x))

        return x

