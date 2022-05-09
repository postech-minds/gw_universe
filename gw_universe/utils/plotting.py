import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def get_datasets(logdir):
    """
    Recursively look through logdir for output files produced by CSVLogger.

    Refereces:
        https://github.com/openai/spinningup/blob/master/spinup/utils/plot.py#L61
    """
    datasets = []
    for root, _, files in os.walk(logdir):
        for file in files:
            if 'csv' in file:
                _, exp_name = os.path.split(root)
                fold = file.split('.')[0]

                exp_data = pd.read_csv(os.path.join(root, file))

                # Add columns indicating experiments
                exp_data.insert(len(exp_data.columns), 'exp_name', exp_name)
                exp_data.insert(len(exp_data.columns), 'run_id', fold)
                datasets.append(exp_data)

    return datasets


def plot_data(data, target='val_recall', smooth=1, hue='exp_name', **kwargs):
    if smooth > 1:
        y = np.ones(smooth)  # filter
        for datum in data:
            x = np.asarry(datum[target])
            z = np.ones(len(x))  # Dummy vector for counting instances
            smoothed_x = np.convolve(x, y, 'same') / np.convolve(z, y, 'same')
            datum[target] = smoothed_x

    if isinstance(data, list):
        data = pd.concat(data, ignore_index=True)

    # Plotting
    sns.set(style='darkgrid', font_scale=1.5)
    sns.lineplot(data=data, x='epoch', y=target, hue=hue, **kwargs)
    plt.ylim(0.0, 1.0)
    plt.legend(loc='best').set_draggable(True)
    plt.tight_layout()


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir_results', type=str, default='./results')
    parser.add_argument('--target', type=str, default='val_loss')
    parser.add_argument('--save', type=str, default=None)
    args = parser.parse_args()

    return args


def main():
    args = get_arguments()
    data = get_datasets(args.dir_results)

    # Plotting
    plt.figure(figsize=(8, 6))
    plot_data(data)
    if args.save:
        plt.savefig(args.save)
    plt.show()


if __name__ == '__main__':
    main()
