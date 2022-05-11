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
    def __init__(self,
                 images,
                 labels=None,
                 is_train=True,
                 resampling=None,
                 batch_size=64,
                 shuffle=False,
                 *args,
                 **kwargs):
        super(InMemoryDataLoader, self).__init__(*args, **kwargs)
        self.images = images
        self.labels = labels
        self.is_train = is_train
        if resampling not in [None, 'random_undersampling', 'random_oversampling', 'smote']:
            ValueError(f"resmpling must be 'random_undersamplig', 'random_oversampling', 'smote', or None, \
                got: {resampling}")
        self.resampling = resampling
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

        if self.is_train:
            if self.resampling is not None:
                self.indices = self.resampling_factory()

            if self.shuffle:
                np.random.shuffle(self.indices)

    def resampling_factory(self):
        if self.resampling == 'random_oversampling':
            return self.random_oversampling(self.labels)

        elif self.resampling == 'random_undersampling':
            return self.random_undersampling(self.labels)

    @staticmethod
    def random_oversampling(y):
        indices = np.arange(len(y))
        counts = np.bincount(y)
        target_nums = max(counts) - counts

        resampled_indices = indices.tolist()
        for label, target_num in enumerate(target_nums):
            pool_indices = np.where(y == label)
            resampled_indices += np.random.choice(indices[pool_indices], target_num).tolist()

        return np.array(resampled_indices)

    @staticmethod
    def random_undersampling(y):
        indices = np.arange(len(y))
        counts = np.bincount(y)
        argmin = np.argmin(counts)
        target_num = min(counts)

        resampled_indices = []
        for label in range(len(counts)):
            replace = label == argmin
            pool_indices = np.where(y != label)
            resampled_indices += np.random.choice(indices[pool_indices], target_num, replace=replace).tolist()

        return np.array(resampled_indices)
