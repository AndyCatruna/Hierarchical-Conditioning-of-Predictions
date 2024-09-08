#!/bin/bash

set -e

common_args="--exp_index 1 --batch_size 512 --lr 1e-4 --epochs 300 --pretrained True --dataset car_model_dataset-multi --dataset_path ../car_model_dataset/ --wandb_log True --wandb_group car_model_dataset-class --predict_car_model"
debug_args="--debug --exp_index 1 --batch_size 512 --lr 1e-4 --epochs 1 --pretrained True --dataset car_model_dataset-multi --dataset_path ../car_model_dataset/ --wandb_group car_model_dataset-class --predict_car_model"

# Unified model
python main.py $common_args --model vit-small-unified --trainer classic --run_name car_model-vit-unified

# Separate model
python main.py $common_args --model vit-small-separate --trainer multitask --predict_full --run_name car_model-vit-separate

# Conditional model
python main.py $common_args --model vit-small-conditional --trainer multitask --predict_full --run_name car_model-vit-conditional

# Conditional model with consistency loss
python main.py $common_args --model vit-small-conditional --trainer multitask --predict_full --use_consistency_loss --run_name car_model-vit-conditional-consistency 
