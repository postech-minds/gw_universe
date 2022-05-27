import argparse
import os

import numpy as np
import pandas as pd
import tensorflow as tf
from gw_universe.datasets.rb import (InMemoryDataLoader, create_metadata,
                                     prepare_dataset_from_metadata)
from gw_universe.models.rb import get_otrain
from gw_universe.utils import seed_all
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.utils import class_weight


def get_arguments():
    parser = argparse.ArgumentParser()
    # Experiment configuration
    parser.add_argument('--dir_data', type=str, default='./data/20220209_LOAO_check')
    parser.add_argument('--seed', type=str, default=0)
    parser.add_argument('--dir_log', type=str, default='./results')
    parser.add_argument('--exp_name', type=str, default='baseline')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--num_folds', type=int, default=5)

    # Image configuration
    parser.add_argument('--img_size', type=int, default=38)
    parser.add_argument('--channels', nargs='+', type=str, default=['sub'])
    parser.add_argument('--obssites', nargs='+', type=str, default=['LOAO'])

    # Options for handling class imbalance.
    parser.add_argument('--resampling', type=str, default=None)
    parser.add_argument('--class_weight', type=bool, default=False)
    parser.add_argument('--label_smoothing', type=float, default=0.0)
    args = parser.parse_args()

    return args


def main():
    args = get_arguments()
    seed_all(args.seed)

    log_dir = os.path.join(args.dir_log, args.exp_name)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Loading metadata
    path_meta = os.path.join(args.dir_data, 'meta.csv')
    if not os.path.exists(path_meta):
        create_metadata(args.dir_data)
    meta = pd.read_csv(path_meta)

    # Preparing dataset
    X, y, groups = prepare_dataset_from_metadata(args.dir_data, meta, args.channels, args.obssites, args.img_size)
    input_shape = X.shape[1:]

    cv = StratifiedGroupKFold(n_splits=args.num_folds, shuffle=True, random_state=args.seed)
    for i, (train_index, valid_index) in enumerate(cv.split(np.arange(len(y)), y, groups=groups)):
        X_train, y_train = X[train_index], y[train_index]
        X_valid, y_valid = X[valid_index], y[valid_index]

        # Comput class weights
        # ``balanced`` compute the following: n_samples / (n_classes * np.bincount(y))
        weights = None
        if args.class_weight:
            weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
            weights = {k: v for k, v in enumerate(weights)}

        train_loader = InMemoryDataLoader(X_train, y_train, resampling=args.resampling, shuffle=True)
        valid_loader = InMemoryDataLoader(X_valid, y_valid, is_train=False, shuffle=False)

        optimizer = tf.keras.optimizers.Adam()
        loss = tf.keras.losses.BinaryCrossentropy(label_smoothing=args.label_smoothing)
        metrics = [
            tf.keras.metrics.BinaryAccuracy(),
            tf.keras.metrics.Recall(),
            tf.keras.metrics.Precision(),
            tf.keras.metrics.AUC(),
        ]
        callbacks = [
            tf.keras.callbacks.CSVLogger(f'{log_dir}/fold{i}.csv'),
        ]

        model = get_otrain(input_shape)
        model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
        model.fit(
            train_loader,
            validation_data=valid_loader,
            class_weight=weights,
            epochs=args.epochs,
            callbacks=callbacks,
            verbose=0
        )

        model.save(f'{log_dir}/fold{i}.h5')
        tf.keras.backend.clear_session()


if __name__ == '__main__':
    main()
