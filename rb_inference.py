import argparse
import os
from glob import glob

import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm

from gw_universe.datasets.rb import load_fits


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir_data', type=str)
    parser.add_argument('--dir_model', type=str)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--ensemble', action='store_true')
    args = parser.parse_args()

    return args


def prepare_dataset(dir_data):
    if not os.path.exists(dir_data):
        raise ValueError(f"{dir_data} does not exist.")
    img_list = glob(f'{dir_data}/*.fits')
    id = list(map(lambda x: x.split('/')[-1], img_list))

    # Loading and preprocessing dataset
    X = list(map(lambda x: load_fits(x), img_list))
    X = np.array(X)

    # Create a pd.DataFrame to store predictions
    df = pd.DataFrame({'id': id})
    df['pred'] = np.nan
    df['prob'] = np.nan

    return X, df


def inference(model, X):
    probs = np.array([model.predict(x[np.newaxis])[0] for x in tqdm(X)])

    return probs


def ensemble(dir_model, X):
    if not os.path.isdir(dir_model):
        ValueError("In ensemble mode, dir_model must be a directory")

    candidates = []
    for fpath in glob(f'{dir_model}/*.h5'):
        model = tf.keras.models.load_model(fpath)
        candidates.append(inference(model, X))

    return candidates


def main():
    args = get_arguments()

    X, df = prepare_dataset(args.dir_data)

    if not args.ensemble:
        model = tf.keras.models.load_model(args.dir_model)
        probs = inference(model, X)
        preds = (probs > 0.5).astype(np.uint8)

    else:
        candidates = ensemble(args.dir_model, X)
        candidates = np.hstack(candidates)
        probs = np.mean(candidates, axis=1)
        preds = np.mean((candidates > 0.5).astype(np.uint8), axis=1)
        preds = (preds > 0.5).astype(np.uint8)

    df['pred'] = preds
    df['prob'] = probs

    df.to_csv('./result.csv', index=False)


if __name__ == '__main__':
    main()
