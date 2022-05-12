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
                exp_name = exp_name.replace('-', '+')
                exp_name = exp_name.replace('_', ' ')
                fold = file.split('.')[0]

                exp_data = pd.read_csv(os.path.join(root, file))

                # Add columns indicating experiments
                exp_data.insert(len(exp_data.columns), 'exp_name', exp_name)
                exp_data.insert(len(exp_data.columns), 'run_id', fold)
                datasets.append(exp_data)

    return datasets


def get_barplot_dataset(data, last_n_epochs=5, targets=['val_recall', 'val_precision']):
    data = pd.concat(data, ignore_index=True)

    new_df = pd.DataFrame()
    for target in targets:
        temp_df = data[['epoch', 'exp_name', 'run_id'] + [target]]
        temp_df = temp_df.rename(columns={target: 'score'})
        temp_df['metric'] = target

        new_df = pd.concat([new_df, temp_df], axis=0, ignore_index=True)

    epochs = [data['epoch'].max() - i for i in range(last_n_epochs)]
    new_df = new_df[new_df['epoch'].isin(epochs)].reset_index(drop=True)

    return new_df


def barplot(data, x='metric', y='score', hue='exp_name', **kwargs):
    if isinstance(data, list):
        data = pd.concat(data, ignore_index=True)

    sns.set(style='darkgrid', font_scale=1.5)
    fig = plt.figure(figsize=(12, 4))
    ax = fig.add_axes([0.1, 0.2, 0.45, 0.7])
    sns.barplot(data=data, x=x, y=y, hue=hue, ax=ax, **kwargs)

    plt.ylim(0.4, 1.01)
    plt.yticks([0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0])
    plt.legend(bbox_to_anchor=[1.0, 0.95])


def plot_data(data, target='val_recall', smooth=1, hue='exp_name', **kwargs):
    if smooth > 1:
        y = np.ones(smooth)  # filter
        for datum in data:
            x = np.asarray(datum[target])
            z = np.ones(len(x))  # Dummy vector for counting instances
            smoothed_x = np.convolve(x, y, 'same') / np.convolve(z, y, 'same')
            datum[target] = smoothed_x

    if isinstance(data, list):
        data = pd.concat(data, ignore_index=True)

    ymin = data[target].min()

    # Plotting
    sns.set(style='darkgrid', font_scale=1.5)
    sns.lineplot(data=data, x='epoch', y=target, hue=hue, **kwargs)
    plt.ylim(ymin, 1.0)
    plt.legend(loc='best').set_draggable(True)
    plt.tight_layout()


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir_results', type=str, default='./results')
    parser.add_argument('--target', type=str, default='val_loss')
    parser.add_argument('--smooth', type=int, default=1)
    parser.add_argument('--save', type=str, default=None)
    parser.add_argument('--barplot', action='store_true')
    args = parser.parse_args()

    return args


def main():
    args = get_arguments()
    data = get_datasets(args.dir_results)

    # Plotting
    if args.barplot:
        data = get_barplot_dataset(data)
        barplot(data)

    else:
        plt.figure(figsize=(12, 8))
        plot_data(data, args.target, smooth=args.smooth)

    if args.save:
        plt.savefig(args.save, dpi=300)


if __name__ == '__main__':
    main()
