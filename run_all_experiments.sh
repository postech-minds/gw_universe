# Baseline
python experiments/rb.py 

# Class weight
python experiments/rb.py \
    --exp_name class_weight \
    --class_weight True

# Label smoothing
python experiments/rb.py \
    --exp_name label_smoothing \
    --label_smoothing 0.2

# Class weight + label smoothing
python experiments/rb.py \
    --exp_name class_weight-label_smoothing \
    --class_weight True \
    --label_smoothing 0.2 \
                         
# Random oversampling
python experiments/rb.py \
    --exp_name random_oversampling \
    --resampling random_oversampling

# Random oversampling + label smoothing
python experiments/rb.py \
    --exp_name random_oversampling-label_smoothing \
    --label_smoothing 0.2 \
    --resampling random_oversampling

# Random undersampling
python experiments/rb.py \
    --exp_name random_underampling \
    --resampling random_undersampling

# Plotting learning curves
python gw_universe/utils/plotting.py --target val_auc --dir_results ./results --save ./results/auc.png
python gw_universe/utils/plotting.py --target val_recall --dir_results ./results --save ./results/recall.png
python gw_universe/utils/plotting.py --target val_precision --dir_results ./results --save ./results/precision.png
