import os
import logging
import math
from functools import reduce
from collections import defaultdict
import json
from timeit import default_timer
import pandas
import csv

from tqdm import trange, tqdm
import numpy as np
import torch

from models.math import log_density_gaussian
from models.modelIO import save_metadata_test

TEST_LOSSES_FILE = "test_losses.log"

class Evaluator:
    """
    Class to handle training of model.

    Parameters
    ----------
    model: disvae.vae.VAE

    loss_f: disvae.models.BaseLoss
        Loss function.

    device: torch.device, optional
        Device on which to run the code.

    logger: logging.Logger, optional
        Logger.

    save_dir : str, optional
        Directory for saving logs.

    is_progress_bar: bool, optional
        Whether to use a progress bar for training.
    """

    def __init__(self, model, loss_f,
                 device=torch.device("cpu"),
                 logger=logging.getLogger(__name__),
                 save_dir="results",
                 experiment_name="temp",
                 model_type='m2',
                 is_progress_bar=True,
                 file_type="temp"):

        self.device = device
        self.model = model.to(self.device)
        self.loss_f = loss_f
        self.save_dir = save_dir
        self.experiment_name = experiment_name
        self.is_progress_bar = is_progress_bar
        self.logger = logger
        self.model_type = model_type
        self.file_type = file_type
        self.logger.info("Testing Device: {}".format(self.device))

    def __call__(self, data_loader):
        """Compute all test losses.

        Parameters
        ----------
        data_loader: torch.utils.data.DataLoader

        is_losses: bool, optional
            Whether to compute and store the test losses.
        """
        start = default_timer()
        is_still_training = self.model.training
        self.model.eval()

        metric, losses = None, None
 ## Comment for 0-255 to 0-1
        self.logger.info('Computing metrics...')
        metrics = self.compute_metrics(data_loader, self.experiment_name, self.file_type)

        self.logger.info('Computing losses...')
        losses = self.compute_losses(data_loader, self.file_type)
        self.logger.info('Losses: {}'.format(losses))
        save_metadata_test(losses, self.save_dir, filename=TEST_LOSSES_FILE)
        if is_still_training:
            self.model.train()

        self.logger.info('Finished evaluating after {:.1f} min.'.format((default_timer() - start) / 60))

        return metric, losses

    def compute_losses(self, dataloader, file_type):
        """Compute all test losses.

        Parameters
        ----------
        data_loader: torch.utils.data.DataLoader
        """
        storer = defaultdict(list)

        epoch_loss = 0.
        epoch_rec_loss = 0.
        epoch_mi_loss = 0.
        epoch_tc_loss = 0.
        epoch_dw_kl_loss = 0.
        epoch_mse_loss = 0.
        epoch_rsq = 0.
        epoch_latent_kl = 0.

        with trange(len(dataloader)) as t:
            for _, (data, _, wtp1, wtp2, wtp3, wtp4, wtp5, make, makemodel, color, firm, region, price, hp, mpg, mpd, filenames, hpwt, space, wt, length, wid, ht, wb, xi_fe, shares, wph) in enumerate(dataloader):
                data = data.to(self.device)
                recon_batch, latent_dist, latent_sample,wtp_pred1, wtp_pred2, wtp_pred3, wtp_pred4, wtp_pred5, visual_attributes = self.model(data, make, makemodel, color, firm, region, price, hp, mpg, mpd, hpwt, space, wt, length, wid, ht, wb, xi_fe, shares, wph)
                loss, rec_loss, mi_loss, tc_loss, dw_kl_loss, mse_loss, rsq, latent_kl = self.loss_f(data, recon_batch, latent_dist, self.model.training,storer, wtp_pred1, wtp_pred2, wtp_pred3, wtp_pred4, wtp_pred5, wtp1, wtp2, wtp3, wtp4, wtp5, latent_sample=latent_sample)
                batch_size, channel, height, width = data.size()
                epoch_loss += loss.item() * batch_size
                epoch_rec_loss += rec_loss.item() * batch_size
                epoch_mi_loss += mi_loss.item() * batch_size
                epoch_tc_loss += tc_loss.item() * batch_size 
                epoch_dw_kl_loss += dw_kl_loss.item() * batch_size
                epoch_mse_loss += mse_loss.item() * batch_size
                epoch_rsq += rsq.item() * batch_size
                epoch_latent_kl += latent_kl * batch_size

                t.set_postfix(loss=loss)
                t.set_postfix(loss=rec_loss)
                t.set_postfix(loss=mi_loss)
                t.set_postfix(loss=tc_loss)
                t.set_postfix(loss=dw_kl_loss)
                t.set_postfix(loss=mse_loss)
                t.set_postfix(loss=rsq)
                t.set_postfix(loss=latent_kl)
                t.update()

        mean_epoch_loss = epoch_loss / len(dataloader.dataset)
        mean_rec_loss = epoch_rec_loss / len(dataloader.dataset)
        mean_mi_loss = epoch_mi_loss / len(dataloader.dataset) 
        mean_tc_loss = epoch_tc_loss / len(dataloader.dataset) 
        mean_dw_kl_loss = epoch_dw_kl_loss / len(dataloader.dataset) 
        mean_mse_loss = epoch_mse_loss / len(dataloader.dataset) 
        mean_rsq = epoch_rsq / len(dataloader.dataset)     
        mean_latent_kl = epoch_latent_kl / len(dataloader.dataset)

        storer['loss_test'].append(mean_epoch_loss) 
        storer['mi_loss_test'].append(mean_mi_loss) 
        storer['tc_loss_test'].append(mean_tc_loss) 
        storer['dw_kl_loss_test'].append(mean_dw_kl_loss) 
        storer['mse_loss_test'].append(mean_mse_loss) 
        storer['recon_loss_test'].append(mean_rec_loss) 
        storer['rsq_test'].append(mean_rsq)
        for i in range(mean_latent_kl.shape[0]):
            storer['kl_loss_test_' + str(i)].append(mean_latent_kl[i].item())

        losses = {k: sum(v) for k, v in storer.items()}
        return losses

    def compute_metrics(self, dataloader,experiment_name,file_type):
        """
        """
        mean_params, logvar_params, filenames = self._compute_q_zCx(dataloader)
        mean_params = [item for sublist in mean_params for item in sublist]
        logvar_params = [item for sublist in logvar_params for item in sublist]
        filenames = [item for sublist in filenames for item in sublist]
        np.savetxt(os.path.join(self.save_dir,experiment_name+"_mean_params_"+file_type+".csv"),mean_params,delimiter=",",fmt='%s')
        # np.savetxt(os.path.join(self.save_dir,experiment_name+"_logvar_params_"+file_type+".csv"),logvar_params,delimiter=",",fmt='%s')
        np.savetxt(os.path.join(self.save_dir,experiment_name+"_filename_"+file_type+".csv"),filenames,delimiter=",",fmt='%s')
        return 0

    def _compute_q_zCx(self, dataloader):
        """
        """
        indices = []
        mean = []
        logvar = []

        with torch.no_grad():
            for _,(data, label, wtp1, wtp2, wtp3, wtp4, wtp5, make, makemodel, color, firm, region, price, hp, mpg, mpd, filenames, hpwt, space, wt, length, wid, ht, wb, xi_fe, shares, wph) in enumerate(dataloader):
                data = data.to(self.device)
                recon_batch, latent_dist, latent_sample, wtp_pred1, wtp_pred2, wtp_pred3, wtp_pred4, wtp_pred5, visual_attributes = self.model(data, make, makemodel, color, firm, region, price, hp, mpg, mpd, hpwt, space, wt, length, wid, ht, wb, xi_fe, shares, wph)
                batch_size, channel, height, width = data.size()
                mean_val, logvar_val = latent_dist
                dim = mean_val.size(1)
                mean_val = torch.mul(mean_val,visual_attributes.cuda())
                logvar_val = torch.mul(logvar_val,visual_attributes.cuda())

                mean.append(mean_val.cpu().detach().numpy())
                logvar.append(logvar_val.cpu().detach().numpy())
                indices.append(list(filenames))

        return mean, logvar, indices

