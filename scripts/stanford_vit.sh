#!/bin/bash

set -e

common_args="--exp_index 1 --batch_size 512 --lr 1e-4 --epochs 300 --pretrained True --dataset stanford-multi --dataset_path ../stanford_dataset/ --wandb_log True --wandb_group stanford --predict_car_model --mixup"
debug_args="--debug --exp_index 1 --batch_size 512 --lr 1e-4 --epochs 1 --pretrained True --dataset stanford-multi --dataset_path ../stanford_dataset/ --wandb_group stanford --predict_car_model --mixup"

# Unified model
python main.py $common_args --model vit-small-unified --trainer classic --run_name stanford-vit-unified

# Separate model
python main.py $common_args --model vit-small-separate --trainer multitask --predict_full --predict_body --run_name stanford-vit-separate

# Conditional model
python main.py $common_args --model vit-small-conditional --trainer multitask --predict_full --predict_body --run_name stanford-vit-conditional

# Conditional model with consistency loss
python main.py $common_args --model vit-small-conditional --trainer multitask --predict_full --predict_body --use_consistency_loss --run_name stanford-vit-conditional-consistency
