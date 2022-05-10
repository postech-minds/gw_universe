import argparse
import os

import numpy as np
import tensorflow as tf
from gw_universe.datasets.rb import InMemoryDataLoader, prepare_rb_dataset
from gw_universe.models.rb import get_otrain
from gw_universe.utils import seed_all
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import class_weight


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir_data', type=str, default='./data/20220209_LOAO_check')
    parser.add_argument('--exp_name', type=str, default='class_weight')
    parser.add_argument('--seed', type=str, default=0)
    parser.add_argument('--num_folds', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=30)
    args = parser.parse_args()

    return args


def main():
    args = get_arguments()
    seed_all(args.seed)

    log_dir = os.path.join('./results', args.exp_name)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    X, y = prepare_rb_dataset(dir_data=args.dir_data)
    input_shape = X.shape[1:]

    cv = StratifiedKFold(n_splits=args.num_folds, shuffle=True, random_state=args.seed)
    for i, (train_index, valid_index) in enumerate(cv.split(np.arange(len(y)), y)):
        X_train, y_train = X[train_index], y[train_index]
        X_valid, y_valid = X[valid_index], y[valid_index]

        # Comput class weights
        # ``balanced`` compute the following: n_samples / (n_classes * np.bincount(y))
        weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        weights = {k: v for k, v in enumerate(weights)}

        train_loader = InMemoryDataLoader(X_train, y_train, shuffle=True)
        valid_loader = InMemoryDataLoader(X_valid, y_valid, shuffle=True)

        optimizer = tf.keras.optimizers.Adam()
        loss = tf.keras.losses.BinaryCrossentropy()
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

        tf.keras.backend.clear_session()


if __name__ == '__main__':
    main()
