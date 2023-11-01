# GW Universe project

## Installation
~~~
git clone https://github.com/postech-minds/gw_universe.git
cd gw_universe
pip install -e .
~~~

## Real vs. Bogus classification

We conducted a set of experiments with different strategies to resolve the class imbalance in the Real vs. Bogus dataset
. All experiments can be reproduced by running `run_all_experiments.sh`:
~~~
bash run_all_experiments.sh
~~~
 
 Name                    | Description 
:-----------------------:|:----------------------------------------------------------------------:
 `baseline`              | No strategy is applied
 `label smoothing`       | Label smoothing with alpha=0.2
 `class weight`          | More penalties for incorrect samples in the minority class
 `random oversampling`   | For each epoch, minor samples are oversampled to # of major samples
 `random undersampling`  | For each epoch, major samples are undersampled to # of minor samples

<br>

Once a set of experiments is finished, you can visualize the learning curves of a specific metric as follows:
~~~
python gw_universe/utils/plotting.py --target val_auc --dir_results ./results --save ./results/auc.png
python gw_universe/utils/plotting.py --target val_recall --dir_results ./results --save ./results/recall.png
python gw_universe/utils/plotting.py --target val_precision --dir_results ./results --save ./results/precision.png
~~~

This gives you a figure as below:

<p>
    <img src="results/auc.png" width="250"/>
    <img src="results/recall.png" width="250"/>
    <img src="results/precision.png" width="250"/>
</p>

Here, the solid lines indicate the average performance over 5 folds and the bands 95% confidence intervals. To compare 
the performance across experiments, you can visualize the bar graph by passing `--barplot` flags.

~~~
python gw_universe/utils/plotting.py --barplot --dir_results ./results --save ./results/barplot.png
~~~

Our implementation considers the average performance of last 5 epochs over 5 folds, i.e., the height of each bar in the 
barplot indicates the average score of 25 models and the solid line corresponds to 95% confidence interval.

<img src="results/barplot.png"/>

## Acknowledgement
This work was supported by the research grant from the NRF to the Center for the Gravitational-Wave Universe under the Grant Number 2021M3F7A1082053.
