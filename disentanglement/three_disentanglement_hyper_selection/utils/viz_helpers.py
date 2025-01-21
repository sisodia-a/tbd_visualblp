import random

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pandas as pd
import torch
import imageio

from torchvision.utils import make_grid
from dataset.datasets import get_dataloaders

FPS_GIF = 12

def get_samples(data_loader, num_samples, idcs=[]):
    """ Generate a number of samples from the dataset.

    Parameters
    ----------
    dataset : str

    num_samples : int, optional
        The number of samples to load from the dataset

    idcs : list of ints, optional
        List of indices to of images to put at the begning of the samples.
    """
    idcs += random.sample(range(len(data_loader.dataset)), num_samples - len(idcs))
    samples = torch.stack([data_loader.dataset[i][0] for i in idcs], dim=0)

#    location = torch.stack([torch.from_numpy(data_loader.dataset[i][3]) for i in idcs], dim=0)
#    brand = torch.stack([torch.from_numpy(data_loader.dataset[i][4]) for i in idcs], dim=0)
#    circa = torch.stack([torch.from_numpy(data_loader.dataset[i][5]) for i in idcs], dim=0)
#    movement = torch.stack([torch.from_numpy(data_loader.dataset[i][6]) for i in idcs], dim=0)
#    diameter = torch.stack([torch.as_tensor(data_loader.dataset[i][7]) for i in idcs], dim=0)
#    material = torch.stack([torch.from_numpy(data_loader.dataset[i][8]) for i in idcs], dim=0)
#    timetrend = torch.stack([torch.as_tensor(data_loader.dataset[i][9]) for i in idcs], dim=0)
#    filenames = torch.stack([torch.as_tensor(data_loader.dataset[i][10]) for i in idcs], dim=0)
    print("Selected idcs: {}".format(idcs))

    return samples
#    return samples, location, brand, circa, movement, diameter, material, timetrend, filenames

def sort_list_by_other(to_sort, other, reverse=True):
    """Sort a list by an other."""
    return [el for _, el in sorted(zip(other, to_sort), reverse=reverse)]

# TO-DO: clean
def read_loss_from_file(log_file_path, loss_to_fetch):
    """ Read the average KL per latent dimension at the final stage of training from the log file.
        Parameters
        ----------
        log_file_path : str
            Full path and file name for the log file. For example 'experiments/custom/losses.log'.

        loss_to_fetch : str
            The loss type to search for in the log file and return. This must be in the exact form as stored.
    """
    EPOCH = "Epoch"
    LOSS = "Loss"

    logs = pd.read_csv(log_file_path)
    df_last_epoch_loss = logs[logs.loc[:, EPOCH] == logs.loc[:, EPOCH].max()]
    df_last_epoch_loss = df_last_epoch_loss.loc[df_last_epoch_loss.loc[:, LOSS].str.startswith(loss_to_fetch), :]
    df_last_epoch_loss.loc[:, LOSS] = df_last_epoch_loss.loc[:, LOSS].str.replace(loss_to_fetch, "").astype(int)
    df_last_epoch_loss = df_last_epoch_loss.sort_values(LOSS).loc[:, "Value"]
    return list(df_last_epoch_loss)


def add_labels(input_image, labels):
    """Adds labels next to rows of an image.

    Parameters
    ----------
    input_image : image
        The image to which to add the labels
    labels : list
        The list of labels to plot
    """
    new_width = input_image.width + 100
    new_size = (new_width, input_image.height)
    new_img = Image.new("RGB", new_size, color='white')
    new_img.paste(input_image, (0, 0))
    draw = ImageDraw.Draw(new_img)

    for i, s in enumerate(labels):
        draw.text(xy=(new_width - 100 + 0.005,
                      int((i / len(labels) + 1 / (2 * len(labels))) * input_image.height)),
                  text=s,
                  fill=(0, 0, 0))

    return new_img

def make_grid_img(tensor, **kwargs):
    """Converts a tensor to a grid of images that can be read by imageio.

    Notes
    -----
    * from in https://github.com/pytorch/vision/blob/master/torchvision/utils.py

    Parameters
    ----------
    tensor (torch.Tensor or list): 4D mini-batch Tensor of shape (B x C x H x W)
        or a list of images all of the same size.

    kwargs:
        Additional arguments to `make_grid_img`.
    """
    grid = make_grid(tensor, **kwargs)
    img_grid = grid.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0)
    img_grid = img_grid.to('cpu', torch.uint8).numpy()
    return img_grid

def concatenate_pad(arrays, pad_size, pad_values, axis=0):
    """Concatenate lsit of array with padding inbetween."""
    pad = np.ones_like(arrays[0]).take(indices=range(pad_size), axis=axis) * pad_values

    new_arrays = [pad]
    for arr in arrays:
        new_arrays += [arr, pad]
    new_arrays += [pad]
    return np.concatenate(new_arrays, axis=axis)
