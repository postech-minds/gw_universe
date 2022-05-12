import os
import random

import numpy as np
import tensorflow as tf


def seed_all(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.config.experimental.enable_op_determinism()


def make_grid(x, ncols, pad=1):
    """Make a grid of images. This is the NumPy version of torchvision.utils.make_grid.

    Args:
        x (np.ndarray): 4D mini-batch of np.ndarray with shape (n_samples, height, width, n_channels)
        ncols (int): the number of images displayed in each row.
        pad (int, optional): the size of white space around each image. Defaults to 1.

    Returns:
        np.ndarray: a 3D np.ndarray containing a grid of images
    """
    k = 255.0 if x.max() > 1.0 else 1.0  # This makes padded regions white

    x = np.pad(x, [(0, 0), (pad, pad), (pad, pad), (0, 0)], constant_values=k)
    n, h, w, c = x.shape
    nrows = int(np.ceil(n / ncols))
    grid = np.zeros((h*nrows, w*ncols, c)) + k
    for row in range(nrows):
        row_images = x[row*ncols:row*ncols+ncols]
        row_images = np.hstack(row_images)
        grid[row*h:(row+1)*h, :row_images.shape[1]] = row_images

    return grid
