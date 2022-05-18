import os
from glob import glob

import numpy as np
import tensorflow as tf
from astropy.io import fits
from tqdm import tqdm


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


def prepare_rb_dataset(dir_data, channels):
    images = []
    labels = []

    for label, cls in enumerate(['bogus', 'real']):
        fpath_lists = []
        for channel in channels:
            fpath_lists.append(
                np.sort(glob(os.path.join(dir_data, cls, f'*.{channel}.fits')))
            )

        for it in tqdm(zip(*fpath_lists), total=len(fpath_lists[0])):
            assert len(set(map(lambda x: ''.join(x.split('.')[:-2]), it))) == 1

            x = []
            for i in it:
                sample = fits.getdata(i)
                maximum = sample.max()
                minimum = sample.min()
                sample = (sample - minimum) / (maximum - minimum)

                sample = tf.keras.utils.array_to_img(sample[:, :, np.newaxis])
                sample = sample.resize((38, 38))
                sample = tf.keras.utils.img_to_array(sample) / 255.0

                sample = (maximum - minimum) * sample + minimum

                x.append(sample[:, :, 0])
            x = np.stack(x, axis=-1)
            x = min_max_normalization(x)

            images.append(x)
        labels += len(fpath_lists[0]) * [label]

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

        elif self.resampling == 'smote':
            if not hasattr(self, 'smote'):
                self._cache = dict(
                    images=self.images.copy(),
                    labels=self.labels.copy()
                )
                self.smote = SMOTE()
                minor_class = np.argmin(np.bincount(self.labels))
                self._cache['minor_class'] = minor_class
                self.smote.fit(self.images[np.where(self.labels == minor_class)])

            X_gen = self.smote.generate()
            self.images = np.concatenate((self._cache['images'], X_gen), axis=0)
            self.labels = np.concatenate((self._cache['labels'], np.array(len(X_gen) * [self._cache['minor_class']])))
            return np.arange(len(self.images))

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


class SMOTE:
    def __init__(self, k=5, n=11, noise_scale=0.001):
        self.k = k
        self.n = n
        self.noise_scale = noise_scale
        self.X = None
        self.neigh = None

    def fit(self, X):
        self.X = X
        n = len(X)
        D = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                D[i, j] = np.linalg.norm(X[i].flatten() - X[j].flatten())
        self.neigh = np.argsort(D, axis=1)[:, :self.k]

    def generate(self):
        X_gen = []
        for i, x in enumerate(self.X):
            indices = np.random.choice(self.neigh[i], size=self.n, replace=True)
            neighbors = self.X[indices]
            diff = neighbors - x
            e = self.noise_scale * np.random.normal(size=diff.shape)
            X_gen.append(x + diff * e)
        X_gen = np.vstack(X_gen)

        return X_gen
