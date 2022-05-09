import os
from glob import glob

import numpy as np
import tensorflow as tf
from astropy.io import fits


def min_max_normalization(x):
    minimum = np.min(x)
    maximum = np.max(x)
    x_normalized = (x - minimum) / (maximum - minimum)

    return x_normalized


def load_fits(fpath):
    img = fits.getdata(fpath)         # np.ndarray of shape (38, 38)
    img = img[..., np.newaxis]        # (height, width, n_channels)
    img = min_max_normalization(img)  # MinMaxNormalization

    return img


def prepare_rb_dataset(dir_data):
    images = []
    labels = []
    for i, cls in enumerate(['bogus', 'transient']):
        fpath_list = glob(f'{os.path.join(dir_data, cls)}/*')
        labels += [i] * len(fpath_list)

        for fpath in fpath_list:
            images.append(load_fits(fpath))

    images = np.stack(images).astype(np.float32)
    labels = np.array(labels).astype(np.int64)

    return images, labels


class InMemoryDataLoader(tf.keras.utils.Sequence):
    def __init__(self, images, labels=None, batch_size=64, shuffle=False, *args, **kwargs):
        super(InMemoryDataLoader, self).__init__(*args, **kwargs)
        self.images = images
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.on_epoch_end()

    def __getitem__(self, index):
        start = index * self.batch_size
        end = (index + 1) * self.batch_size

        X = np.stack([self.images[i] for i in self.indices[start:end]])

        if self.labels is None:
            return X

        y = np.stack([self.labels[i] for i in self.indices[start:end]])

        return X, y

    def __len__(self):
        return len(self.indices) // self.batch_size

    def on_epoch_end(self):
        self.indices = np.arange(len(self.images))
        if self.shuffle:
            np.random.shuffle(self.indices)
